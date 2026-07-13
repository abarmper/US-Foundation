"""Phase-1 SSL domain adaptation of the DINOv2 encoder.

Two modes behind cfg.phase1.mode:
  * "sameview"  -- student/teacher patch-token cosine alignment (simple; the
                   collapse-prone variant).
  * "multicrop" -- DINO-style asymmetric multi-crop self-distillation (recommended).

Both save a bare adapted-encoder state dict (dinov2_adapted_ep{N}.pth) consumable
by Phase 2. The backbone (incl. the register variant) comes from cfg.model.backbone.
"""

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from ..models.model import build_model_from_config
from ..data.dataset import RobustBiometryDataset
from ..data.multicrop import MultiCropBiometryDataset
from ..data.transforms import get_unlabeled_transforms, get_multicrop_transforms
from .common import create_logger, get_writer, get_device, set_seed, runs_dir, resolve_amp


def _ckpt_dir(cfg):
    d = os.path.join(runs_dir(), cfg.run_name, "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def train(cfg, logger=None):
    exp_dir = os.path.join(runs_dir(), cfg.run_name)
    logger = logger or create_logger(exp_dir, "phase1")
    if cfg.phase1.mode == "sameview":
        return _train_sameview(cfg, logger)
    return _train_multicrop(cfg, logger)


# --------------------------------------------------------------------------- #
# Same-view cosine alignment
# --------------------------------------------------------------------------- #
def _train_sameview(cfg, logger):
    device = get_device()
    set_seed(cfg.seed)
    writer = get_writer(os.path.join(runs_dir(), cfg.run_name))
    ckpt_dir = _ckpt_dir(cfg)

    tf = get_unlabeled_transforms(cfg.data.canvas)
    ds = RobustBiometryDataset(cfg.data.data_root, mode="unlabeled", transforms=tf)
    loader = DataLoader(ds, batch_size=cfg.phase1.batch_size, shuffle=True,
                        num_workers=cfg.phase1.num_workers, pin_memory=True, drop_last=True)

    student = build_model_from_config(cfg, freeze_encoder=False).to(device)
    teacher = build_model_from_config(cfg, freeze_encoder=False).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    opt = AdamW(student.encoder.parameters(), lr=cfg.phase1.lr, weight_decay=cfg.phase1.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.phase1.epochs, eta_min=1e-6)
    amp_on, amp_dtype, use_scaler = resolve_amp(cfg.optim.amp_dtype)
    scaler = GradScaler(enabled=use_scaler)

    for epoch in range(cfg.phase1.epochs):
        student.train(); teacher.eval()
        losses = []
        pbar = tqdm(loader, desc=f"[sameview] Epoch {epoch+1}/{cfg.phase1.epochs}",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            imgs = batch["image"].to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                sf = student.forward_phase1(imgs)
                with torch.no_grad():
                    tf_ = teacher.forward_phase1(imgs)
                loss = 1.0 - F.cosine_similarity(sf, tf_, dim=-1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.encoder.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                a = cfg.phase1.ema_alpha
                for p, ep in zip(student.encoder.parameters(), teacher.encoder.parameters()):
                    ep.data.mul_(a).add_(p.data, alpha=1 - a)
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{sum(losses)/len(losses):.5f}")
        pbar.close()
        sched.step()
        avg = sum(losses) / max(1, len(losses))
        writer.add_scalar("Loss/Phase1_SSL", avg, epoch)
        logger.info(f"[sameview] Epoch {epoch+1}/{cfg.phase1.epochs} | SSL loss {avg:.5f}")
        if (epoch + 1) % 20 == 0 or (epoch + 1) == cfg.phase1.epochs:
            path = os.path.join(ckpt_dir, f"dinov2_adapted_ep{epoch+1}.pth")
            torch.save(student.encoder.state_dict(), path)
            logger.info(f"--> saved adapted encoder -> {path}")
    writer.close()
    logger.info("Phase 1 (sameview) finished.")


# --------------------------------------------------------------------------- #
# DINO multi-crop
# --------------------------------------------------------------------------- #
class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, ncrops=8, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=30, nepochs=100, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # clamp warmup so short runs don't build a negative-length schedule
        warmup_teacher_temp_epochs = min(warmup_teacher_temp_epochs, nepochs)
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach().chunk(2)
        total, n_terms = 0, 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                total += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                n_terms += 1
        total /= n_terms
        self._update_center(teacher_output)
        return total

    @torch.no_grad()
    def _update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True) / len(teacher_output)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class _DINOWrapper(nn.Module):
    def __init__(self, base_model, embed_dim=None, out_dim=256):
        super().__init__()
        self.model = base_model
        embed_dim = embed_dim or base_model.embed_dim   # derive -> supports vits/b/l/g
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True),
            nn.Linear(2048, out_dim))

    @property
    def encoder(self):
        return self.model.encoder

    def forward(self, x):
        feats = self.model.forward_phase1(x)      # (B, N, C)
        return self.head(feats.mean(dim=1))


def _cosine_ema(step, total, base=0.99, top=1.0):
    return top - (top - base) * (math.cos(math.pi * step / total) + 1) / 2


def _train_multicrop(cfg, logger):
    device = get_device()
    set_seed(cfg.seed)
    writer = get_writer(os.path.join(runs_dir(), cfg.run_name))
    ckpt_dir = _ckpt_dir(cfg)

    base = RobustBiometryDataset(cfg.data.data_root, mode="unlabeled", transforms=None)
    mc = get_multicrop_transforms(global_size=cfg.data.canvas, local_size=98)
    ds = MultiCropBiometryDataset(base, mc, n_local_crops=cfg.phase1.n_local_crops)
    loader = DataLoader(ds, batch_size=cfg.phase1.batch_size, shuffle=True,
                        num_workers=cfg.phase1.num_workers, pin_memory=True, drop_last=True)

    student = _DINOWrapper(build_model_from_config(cfg, freeze_encoder=False),
                           out_dim=cfg.phase1.out_dim).to(device)
    teacher = _DINOWrapper(build_model_from_config(cfg, freeze_encoder=False),
                           out_dim=cfg.phase1.out_dim).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    ncrops = 2 + cfg.phase1.n_local_crops
    opt = AdamW(student.parameters(), lr=cfg.phase1.lr, weight_decay=cfg.phase1.weight_decay)
    total_steps = len(loader) * cfg.phase1.epochs
    sched = CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=1e-6)
    amp_on, amp_dtype, use_scaler = resolve_amp(cfg.optim.amp_dtype)
    scaler = GradScaler(enabled=use_scaler)
    dino_loss = DINOLoss(out_dim=cfg.phase1.out_dim, ncrops=ncrops, nepochs=cfg.phase1.epochs).to(device)

    step = 0
    for epoch in range(cfg.phase1.epochs):
        student.train()
        losses = []
        pbar = tqdm(loader, desc=f"[multicrop] Epoch {epoch+1}/{cfg.phase1.epochs}",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            crops = [c.to(device) for c in batch["crops"]]
            opt.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                with torch.no_grad():
                    t_out = torch.stack([teacher(g) for g in crops[:2]])
                s_out = torch.stack([student(c) for c in crops])
                loss = dino_loss(s_out, t_out, epoch)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            a = _cosine_ema(step, total_steps)
            with torch.no_grad():
                for p, ep in zip(student.parameters(), teacher.parameters()):
                    ep.data.mul_(a).add_(p.data, alpha=1 - a)
            step += 1
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{sum(losses)/len(losses):.5f}")
        pbar.close()
        avg = sum(losses) / max(1, len(losses))
        writer.add_scalar("Loss/Phase1_MultiCrop", avg, epoch)
        logger.info(f"[multicrop] Epoch {epoch+1}/{cfg.phase1.epochs} | DINO loss {avg:.5f}")
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.phase1.epochs:
            path = os.path.join(ckpt_dir, f"dinov2_adapted_ep{epoch+1}.pth")
            torch.save(student.encoder.state_dict(), path)
            logger.info(f"--> saved adapted encoder -> {path}")
    writer.close()
    logger.info("Phase 1 (multicrop) finished.")
