"""Phase-1 DINOv2-faithful self-supervised pretraining (`phase1.mode == "dinov2"`).

CLS-token DINO loss (global+local) + iBOT masked-patch loss (global) + KoLeo, with the
vendored official heads/losses (gubiometry/models/dino_ssl.py), single-GPU synchronous
centering, gradient accumulation for a large effective batch, teacher.eval(), LR/WD/
momentum/teacher-temp schedules over EFFECTIVE steps, and freeze-last-layer via lr=0.

Optionally a **high-resolution adaptation tail** (DINOv2 §5): after `epochs` at the bulk
resolution, run `highres_epochs` more at `highres_crop_size` (e.g. 224 -> 518), with a
compressed warmup+cosine restart of the schedules, so the encoder adapts its features to
the downstream (Phase-2) resolution. Both phases save the same bare `encoder.state_dict()`
-> dinov2_adapted_ep{N}.pth, so Phase-2 loading is unchanged.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import RobustBiometryDataset
from ..data.multicrop import MultiCropBiometryDataset, make_dinov2_collate
from ..data.transforms import get_multicrop_transforms, get_multicrop_transforms_v2
from ..models.backbone import load_backbone
from ..models.dino_ssl import (DINOHead, DINOLossV2, iBOTPatchLossV2, KoLeoLoss,
                               MaskingGenerator, CosineParamScheduler)
from .common import (create_logger, get_writer, get_device, set_seed, runs_dir,
                     resolve_amp, save_checkpoint_atomic)

PATCH = 14


class _DINOv2Wrapper(nn.Module):
    """Backbone encoder + DINO head (+ optional separate iBOT head)."""
    def __init__(self, backbone_name, p1):
        super().__init__()
        self.encoder, self.embed_dim = load_backbone(backbone_name)
        dino_dim = p1.dino_out_dim
        ibot_dim = p1.ibot_out_dim or p1.dino_out_dim
        mk = lambda out: DINOHead(self.embed_dim, out, nlayers=p1.head_nlayers,
                                  hidden_dim=p1.head_hidden_dim, bottleneck_dim=p1.head_bottleneck_dim)
        self.dino_head = mk(dino_dim)
        self.ibot_head = mk(ibot_dim) if p1.ibot_separate_head else self.dino_head
        for h in {id(self.dino_head): self.dino_head, id(self.ibot_head): self.ibot_head}.values():
            h.last_layer.weight_g.requires_grad = False


def _build_param_groups(model, weight_decay):
    decay, no_decay, last_layer = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "last_layer" in name:
            last_layer.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "wd_sched": True, "is_last": False},
        {"params": no_decay, "weight_decay": 0.0, "wd_sched": False, "is_last": False},
        {"params": last_layer, "weight_decay": weight_decay, "wd_sched": True, "is_last": True},
    ]


def _tt_sched(warmup_temp, temp, total, warmup_iters):
    # linear warmup_temp -> temp over warmup_iters, then constant temp
    return CosineParamScheduler(base_value=temp, final_value=temp, total_iters=total,
                                warmup_iters=warmup_iters, start_warmup_value=warmup_temp)


def _build_segment(cfg, p1, global_size, batch_size, grad_accum):
    """Build the DataLoader + steps/epoch for one resolution segment."""
    if p1.aug == "us_v2":
        mc = get_multicrop_transforms_v2(global_size=global_size, local_size=p1.local_crop_size,
                                         global_scale=(p1.global_scale_min, p1.global_scale_max),
                                         local_scale=(p1.local_scale_min, p1.local_scale_max),
                                         rotate_limit=p1.rotate_limit)
    else:
        mc = get_multicrop_transforms(global_size=global_size, local_size=p1.local_crop_size)
    base = RobustBiometryDataset(cfg.data.data_root, mode="unlabeled", transforms=None)
    ds = MultiCropBiometryDataset(base, mc, n_local_crops=p1.n_local_crops,
                                  foreground_crop=p1.foreground_crop, min_local_fg_frac=p1.min_local_fg_frac)
    n_tokens = (global_size // PATCH) ** 2
    mask_gen = MaskingGenerator(input_size=global_size // PATCH, max_num_patches=int(0.5 * n_tokens))
    collate = make_dinov2_collate(PATCH, p1.mask_ratio_min, p1.mask_ratio_max, p1.mask_sample_probability,
                                  mask_gen, mask_foreground=p1.mask_foreground)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=p1.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    return loader, steps_per_epoch


def train_dinov2(cfg, logger=None):
    device = get_device()
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    p1 = cfg.phase1
    exp_dir = os.path.join(runs_dir(), cfg.run_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = logger or create_logger(exp_dir, "phase1")
    writer = get_writer(exp_dir)

    # --- segments: bulk (+ optional high-res tail) ---
    global_bulk = p1.global_crop_size or cfg.data.canvas
    bulk_loader, bulk_spe = _build_segment(cfg, p1, global_bulk, p1.batch_size, p1.grad_accum_steps)
    bulk_eff = bulk_spe * p1.epochs
    tail_epochs = p1.highres_epochs
    tail_loader = None
    if tail_epochs > 0:
        bs_t = p1.highres_batch_size or p1.batch_size
        accum_t = p1.highres_grad_accum_steps or p1.grad_accum_steps
        tail_loader, tail_spe = _build_segment(cfg, p1, p1.highres_crop_size, bs_t, accum_t)
        tail_eff = tail_spe * tail_epochs
    else:
        bs_t = accum_t = tail_spe = tail_eff = 0
    total_epochs = p1.epochs + tail_epochs

    # --- models ---
    student = _DINOv2Wrapper(cfg.model.backbone.name, p1).to(device)
    teacher = _DINOv2Wrapper(cfg.model.backbone.name, p1).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    dino_loss = DINOLossV2(p1.dino_out_dim, student_temp=p1.student_temp, center_momentum=p1.center_momentum).to(device)
    ibot_loss = iBOTPatchLossV2(p1.ibot_out_dim or p1.dino_out_dim, student_temp=p1.student_temp,
                                center_momentum=p1.center_momentum).to(device)
    koleo_loss = KoLeoLoss().to(device)

    # --- optim + schedules ---
    opt = AdamW(_build_param_groups(student, p1.weight_decay), lr=p1.lr)
    amp_on, amp_dtype, use_scaler = resolve_amp(cfg.optim.amp_dtype)
    scaler = GradScaler(enabled=use_scaler)
    wd_final = p1.weight_decay if p1.weight_decay_end == 0.0 else p1.weight_decay_end
    # bulk schedules (indexed by eff_step in [0, bulk_eff))
    sched_b = dict(
        lr=CosineParamScheduler(p1.lr, p1.min_lr, bulk_eff, warmup_iters=p1.warmup_epochs * bulk_spe),
        wd=CosineParamScheduler(p1.weight_decay, wd_final, bulk_eff),
        mom=CosineParamScheduler(p1.momentum_base, p1.momentum_final, bulk_eff),
        tt=_tt_sched(p1.warmup_teacher_temp, p1.teacher_temp, bulk_eff,
                     min(p1.warmup_teacher_temp_epochs, p1.epochs) * bulk_spe))
    # tail schedules -- a compressed warmup+cosine restart (indexed by eff_step - bulk_eff)
    if tail_epochs > 0:
        sched_t = dict(
            lr=CosineParamScheduler(p1.highres_lr or p1.lr, p1.min_lr, tail_eff,
                                    warmup_iters=p1.highres_warmup_epochs * tail_spe),
            wd=CosineParamScheduler(wd_final, wd_final, tail_eff),
            mom=CosineParamScheduler(p1.momentum_base, p1.momentum_final, tail_eff),
            tt=_tt_sched(p1.teacher_temp, p1.teacher_temp, tail_eff, 0))
    freeze_last_steps = p1.freeze_last_layer_epochs * bulk_spe

    start_epoch, eff_step = 0, 0
    if cfg.resume and os.path.isfile(cfg.resume):
        ck = torch.load(cfg.resume, map_location=device)
        if ck.get("objective") != "dinov2":
            raise ValueError(f"Refusing to resume a non-dinov2 checkpoint ({ck.get('objective')!r}) in dinov2 mode.")
        student.load_state_dict(ck["student_state_dict"]); teacher.load_state_dict(ck["teacher_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"]); scaler.load_state_dict(ck["scaler_state_dict"])
        dino_loss.center = ck["dino_center"].to(device); ibot_loss.center = ck["ibot_center"].to(device)
        start_epoch, eff_step = ck["epoch"], ck["eff_step"]
        logger.info(f"Resumed Phase 1 (dinov2) from {cfg.resume} @ epoch {start_epoch}, eff_step {eff_step}")

    logger.info(f"=== Phase 1 (dinov2): {cfg.run_name} | backbone={cfg.model.backbone.name} | "
                f"bulk global={global_bulk} eff_batch={p1.batch_size*p1.grad_accum_steps} (bs {p1.batch_size} x accum {p1.grad_accum_steps}) x {p1.epochs}ep"
                + (f" | tail global={p1.highres_crop_size} eff_batch={bs_t*accum_t} (bs {bs_t} x accum {accum_t}) x {tail_epochs}ep" if tail_epochs else "")
                + f" | dino_dim={p1.dino_out_dim} ibot_dim={p1.ibot_out_dim or p1.dino_out_dim} ===")
    logger.info(f"AMP: {cfg.optim.amp_dtype} (enabled={amp_on}, scaler={use_scaler}) | bulk steps/ep={bulk_spe}"
                + (f" tail steps/ep={tail_spe}" if tail_epochs else ""))

    n_local = p1.n_local_crops

    for epoch in range(start_epoch, total_epochs):
        in_tail = epoch >= p1.epochs
        loader = tail_loader if in_tail else bulk_loader
        accum = accum_t if in_tail else p1.grad_accum_steps
        scheds = sched_t if in_tail else sched_b
        base_eff = bulk_eff if in_tail else 0
        student.train()
        losses, dparts, iparts, kparts = [], [], [], []
        dc_sum = dc_n = ic_sum = ic_n = None
        seg = "tail@%d" % p1.highres_crop_size if in_tail else "bulk@%d" % global_bulk
        pbar = tqdm(loader, desc=f"[dinov2 {seg}] Epoch {epoch+1}/{total_epochs}", leave=False, dynamic_ncols=True)
        for micro_i, batch in enumerate(pbar):
            if cfg.optim.max_train_batches > 0 and micro_i >= cfg.optim.max_train_batches * accum:
                break
            eff = eff_step - base_eff
            g_crops = batch["collated_global_crops"].to(device, non_blocking=True)
            l_crops = batch["collated_local_crops"].to(device, non_blocking=True) if batch["collated_local_crops"] is not None else None
            masks = batch["collated_masks"].to(device, non_blocking=True)
            mask_idx = batch["mask_indices_list"].to(device, non_blocking=True)
            masks_weight = batch["masks_weight"].to(device, non_blocking=True)
            n_glob = batch["n_global_crops"]
            teacher_temp = scheds["tt"][eff]

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                with torch.no_grad():
                    t_out = teacher.encoder.forward_features(g_crops)
                    t_cls_head = teacher.dino_head(t_out["x_norm_clstoken"])
                    t_dino_soft = [dino_loss.softmax_center_teacher(c, teacher_temp) for c in t_cls_head.chunk(2)]
                    t_patch = t_out["x_norm_patchtokens"].reshape(-1, t_out["x_norm_patchtokens"].shape[-1])
                    t_masked_head = teacher.ibot_head(t_patch[mask_idx]) if mask_idx.numel() else None
                    t_ibot_soft = ibot_loss.softmax_center_teacher(t_masked_head, teacher_temp) if t_masked_head is not None else None

                s_g = student.encoder.forward_features(g_crops, masks=masks)
                s_cls_g = s_g["x_norm_clstoken"]
                s_dino = list(student.dino_head(s_cls_g).chunk(2))
                if l_crops is not None:
                    s_l = student.encoder.forward_features(l_crops)
                    s_dino += list(student.dino_head(s_l["x_norm_clstoken"]).chunk(n_local))

                dino_l = dino_loss.forward(s_dino, t_dino_soft, skip_diagonal=True)
                if t_ibot_soft is not None:
                    s_patch = s_g["x_norm_patchtokens"].reshape(-1, s_g["x_norm_patchtokens"].shape[-1])
                    ibot_l = ibot_loss.forward_masked(student.ibot_head(s_patch[mask_idx]), t_ibot_soft, masks_weight, n_glob)
                else:
                    ibot_l = torch.zeros((), device=device)
                koleo_l = sum(koleo_loss(c) for c in s_cls_g.chunk(2)) / 2.0
                total = p1.dino_loss_weight * dino_l + p1.ibot_loss_weight * ibot_l + p1.koleo_weight * koleo_l

            scaler.scale(total / accum).backward()

            dc = t_cls_head.detach().sum(0, keepdim=True).float()
            dc_sum = dc if dc_sum is None else dc_sum + dc
            dc_n = t_cls_head.shape[0] + (dc_n or 0)
            if t_masked_head is not None:
                ic = t_masked_head.detach().sum(0, keepdim=True).float()
                ic_sum = ic if ic_sum is None else ic_sum + ic
                ic_n = t_masked_head.shape[0] + (ic_n or 0)

            losses.append(float(total)); dparts.append(float(dino_l)); iparts.append(float(ibot_l)); kparts.append(float(koleo_l))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}", dino=f"{np.mean(dparts):.3f}", ibot=f"{np.mean(iparts):.3f}")

            if (micro_i + 1) % accum == 0:
                lr_now, wd_now = scheds["lr"][eff], scheds["wd"][eff]
                freeze = (not in_tail) and (eff_step < freeze_last_steps)
                for g in opt.param_groups:
                    g["lr"] = 0.0 if (g["is_last"] and freeze) else lr_now
                    if g["wd_sched"]:
                        g["weight_decay"] = wd_now
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), p1.clip_grad)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                dino_loss.update_center_ema(dc_sum / max(dc_n, 1))
                if ic_sum is not None:
                    ibot_loss.update_center_ema(ic_sum / max(ic_n, 1))
                dc_sum = dc_n = ic_sum = ic_n = None
                m = scheds["mom"][eff]
                with torch.no_grad():
                    for ps, pt in zip(student.parameters(), teacher.parameters()):
                        pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                eff_step += 1
        pbar.close()

        avg = float(np.mean(losses)) if losses else float("nan")
        with torch.no_grad():
            probs = t_dino_soft[0].float()
            entropy = float((-(probs * (probs + 1e-9).log()).sum(-1)).mean())
            occupancy = float(probs.argmax(-1).unique().numel()) / probs.shape[-1]
        eff = max(0, eff_step - base_eff - 1)
        for k, v in [("Loss/Phase1_DINOv2", avg), ("Phase1/dino", np.mean(dparts)), ("Phase1/ibot", np.mean(iparts)),
                     ("Phase1/koleo", np.mean(kparts)), ("Phase1/teacher_entropy", entropy), ("Phase1/prototype_occupancy", occupancy)]:
            writer.add_scalar(k, float(v), epoch)
        logger.info(f"[dinov2 {seg}] Epoch {epoch+1}/{total_epochs} | loss {avg:.4f} | dino {np.mean(dparts):.3f} "
                    f"ibot {np.mean(iparts):.3f} koleo {np.mean(kparts):.3f} | lr {scheds['lr'][eff]:.2e} "
                    f"mom {scheds['mom'][eff]:.4f} ttemp {scheds['tt'][eff]:.3f} | "
                    f"teacher_entropy {entropy:.3f} occupancy {occupancy:.3f}")

        src = teacher if p1.save_encoder_from == "teacher" else student
        if (epoch + 1) % 10 == 0 or (epoch + 1) == total_epochs or (epoch + 1) == p1.epochs:
            path = os.path.join(ckpt_dir, f"dinov2_adapted_ep{epoch+1}.pth")
            torch.save(src.encoder.state_dict(), path)
            logger.info(f"--> saved adapted encoder -> {path}")
        save_checkpoint_atomic({
            "objective": "dinov2", "epoch": epoch + 1, "eff_step": eff_step,
            "student_state_dict": student.state_dict(), "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": opt.state_dict(), "scaler_state_dict": scaler.state_dict(),
            "dino_center": dino_loss.center, "ibot_center": ibot_loss.center,
        }, os.path.join(ckpt_dir, "latest_checkpoint.pth"))
    writer.close()
    logger.info("Phase 1 (dinov2) finished.")
