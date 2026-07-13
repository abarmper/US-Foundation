"""Phase-2 supervised training with metric-aligned validation.

Differences from the original train_phase2_hrnet.py:
  * Validation predictions are inverse-letterboxed to ORIGINAL pixels and scored
    with the real challenge metric (metrics.challenge_score); the best checkpoint
    is selected on `cfg.optim.select_metric` (default 'challenge_blend', lower is
    better) instead of 518-canvas L1 loss.
  * Config is round-tripped to runs/<run>/config.json for later rebuild.
  * Loss/optimizer/EMA come from the shared losses/optim modules (DSNT + LLRD ready).
"""

import os
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from ..config import save_config_json
from ..constants import TASK_ORDER
from ..geometry import soft_argmax_coords, inverse_letterbox_kps
from ..losses import soft_argmax_loss, mean_radial_error, measurement_loss
from ..metrics import challenge_score
from ..models.model import build_model_from_config
from ..optim import build_optimizer, build_scheduler, update_ema_phase2
from ..data.dataset import RobustBiometryDataset
from ..data.transforms import LetterboxTransform, get_train_transforms
from ..data.samplers import (DeterministicBalancedHomogeneousSampler, HomogeneousTaskSampler,
                             TemperatureTaskSampler)
from .common import (create_logger, get_writer, get_device, set_seed, runs_dir,
                     load_phase1_encoder_weights, md5_of_file, resolve_amp)


def _make_dataset(cfg, mode, transforms):
    return RobustBiometryDataset(cfg.data.data_root, mode=mode, transforms=transforms,
                                 split_file=cfg.data.split_file, fold=cfg.data.fold,
                                 kfold_dir=cfg.data.kfold_dir)


@torch.no_grad()
def validate(teacher, loader_val, device, cfg, amp_on, amp_dtype):
    """Run the EMA teacher, score in original pixels with the challenge metric."""
    teacher.eval()
    entries = []
    for step, batch in enumerate(loader_val):
        if cfg.optim.max_val_batches > 0 and step >= cfg.optim.max_val_batches:
            break
        imgs = batch["image"].to(device)
        task = batch["task_id"][0]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
            logits = teacher.forward_phase2(imgs, task)
        coords = soft_argmax_coords(logits.float(), cfg.optim.softargmax_temp, cfg.data.canvas).cpu().numpy()
        gt_orig = batch["keypoints_orig"].numpy()
        oh, ow, tid = batch["orig_h"], batch["orig_w"], batch["task_id"]
        for b in range(imgs.shape[0]):
            pred_px = inverse_letterbox_kps(coords[b], int(oh[b]), int(ow[b]), cfg.data.canvas)
            entries.append({"task_id": tid[b], "pred_px": pred_px, "gt_px": gt_orig[b],
                            "height": float(oh[b]), "width": float(ow[b])})
    return challenge_score(entries)


def train(cfg, logger=None):
    device = get_device()
    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True   # fixed 518x518 input -> faster convs
    exp_dir = os.path.join(runs_dir(), cfg.run_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = logger or create_logger(exp_dir, "phase2")
    writer = get_writer(exp_dir)

    meta = {"phase1_weights": cfg.phase1_weights}
    if cfg.phase1_weights and os.path.isfile(cfg.phase1_weights):
        meta["phase1_md5"] = md5_of_file(cfg.phase1_weights)
    save_config_json(cfg, os.path.join(exp_dir, "config.json"), extra=meta)

    logger.info(f"=== Phase 2: {cfg.run_name} | backbone={cfg.model.backbone.name} "
                f"input_mode={cfg.model.neck.input_mode} fold={cfg.data.fold} ===")

    # --- data ---
    ds_train = _make_dataset(cfg, "train_labeled",
                             get_train_transforms(cfg.data.canvas, cfg.data.aug_strength))
    ds_val = _make_dataset(cfg, "val", LetterboxTransform(cfg.data.canvas))
    labels = [s["task_id"] for s in ds_train.samples]
    max_task = max(Counter(labels).values())
    n_tasks = len(set(labels))
    batches_per_epoch = max(1, (max_task // cfg.data.batch_size) * n_tasks)
    if cfg.data.sample_temp > 0:
        train_sampler = TemperatureTaskSampler(labels, cfg.data.batch_size, batches_per_epoch,
                                               cfg.data.sample_temp)
    else:
        train_sampler = DeterministicBalancedHomogeneousSampler(labels, cfg.data.batch_size, batches_per_epoch)
    loader_train = DataLoader(ds_train, batch_sampler=train_sampler,
                              num_workers=cfg.data.num_workers, pin_memory=True)
    val_sampler = HomogeneousTaskSampler(ds_val, cfg.data.batch_size)
    loader_val = DataLoader(ds_val, batch_sampler=val_sampler,
                            num_workers=cfg.data.num_workers, pin_memory=True)

    # --- models (student + EMA teacher) ---
    student = build_model_from_config(cfg).to(device)
    if cfg.phase1_weights and os.path.isfile(cfg.phase1_weights):
        load_phase1_encoder_weights(student.encoder, cfg.phase1_weights, device, logger)
    elif cfg.phase1_weights:
        logger.warning(f"phase1_weights not found: {cfg.phase1_weights}")

    teacher = build_model_from_config(cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    n_head = sum(p.numel() for p in student.head_trainable_parameters())
    n_enc = sum(p.numel() for p in student.encoder_trainable_parameters())
    logger.info(f"Trainable -> neck+heads: {n_head:,} | encoder(last {cfg.model.backbone.unfreeze_last_n_blocks}): {n_enc:,}")

    optimizer = build_optimizer(student, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim.warmup_epochs, cfg.optim.epochs, cfg.optim.scheduler)
    amp_on, amp_dtype, use_scaler = resolve_amp(cfg.optim.amp_dtype)
    scaler = GradScaler(enabled=use_scaler)
    logger.info(f"AMP: {cfg.optim.amp_dtype} (enabled={amp_on}, scaler={use_scaler})")
    trainable = student.head_trainable_parameters() + student.encoder_trainable_parameters()

    start_epoch, best, epochs_since_best = 0, float("inf"), 0
    if cfg.resume and os.path.isfile(cfg.resume):
        ck = torch.load(cfg.resume, map_location=device)
        student.load_state_dict(ck["student_state_dict"])
        teacher.load_state_dict(ck["teacher_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"]
        best = ck.get("best_metric", float("inf"))
        epochs_since_best = ck.get("epochs_since_best", 0)   # keep early-stopping continuous
        logger.info(f"Resumed from {cfg.resume} @ epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.optim.epochs):
        student.train()
        student.encoder.eval()   # LayerNorm-only backbone
        teacher.eval()
        train_losses = []
        for step, batch in enumerate(loader_train):
            if cfg.optim.max_train_batches > 0 and step >= cfg.optim.max_train_batches:
                break
            imgs = batch["image"].to(device)
            coords = batch["keypoints"].to(device)
            task = batch["task_id"][0]
            coord_scale = None
            if cfg.optim.loss_space == "original":
                # canvas-px residual x (max(h,w)/canvas) = original-px residual
                s = torch.maximum(batch["orig_h"], batch["orig_w"]).to(device).float() / cfg.data.canvas
                coord_scale = s.view(-1, 1, 1)   # (B,1,1) broadcasts over K and (x,y)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_on):
                logits = student.forward_phase2(imgs, task)
                loss, pred, _ = soft_argmax_loss(
                    logits, coords, cfg.optim.softargmax_temp, cfg.data.canvas,
                    cfg.optim.dsnt_lambda, cfg.optim.dsnt_type, cfg.optim.dsnt_sigma,
                    cfg.optim.coord_loss, cfg.optim.wing_omega, cfg.optim.wing_epsilon, coord_scale)
                if cfg.optim.measurement_lambda > 0.0:
                    loss = loss + cfg.optim.measurement_lambda * measurement_loss(pred, coords, task)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=cfg.optim.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema_phase2(student, teacher, alpha=cfg.optim.ema_alpha)
            train_losses.append(loss.item())
        scheduler.step()
        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        score = validate(teacher, loader_val, device, cfg, amp_on, amp_dtype)
        metric = score[cfg.optim.select_metric]
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Val/average_mre", score["average_mre"], epoch)
        writer.add_scalar("Val/average_avg_mae", score["average_avg_mae"], epoch)
        writer.add_scalar("Val/challenge_blend", score["challenge_blend"], epoch)
        for t in TASK_ORDER:
            if t in score["per_task_mre"]:
                writer.add_scalar(f"MRE_per_task/{t}", score["per_task_mre"][t], epoch)

        logger.info(
            f"Epoch {epoch+1}/{cfg.optim.epochs} | train_loss {avg_train_loss:.4f} | "
            f"MRE(orig px) {score['average_mre']:.2f} | AvgMAE {score['average_avg_mae']:.2f} | "
            f"{cfg.optim.select_metric} {metric:.4f}")

        # update best / early-stop counter BEFORE saving so the checkpoint state is
        # current (resume restores an accurate counter).
        improved = metric < best
        if improved:
            best = metric
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save({
            "epoch": epoch + 1,
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best,
            "epochs_since_best": epochs_since_best,
        }, os.path.join(ckpt_dir, "latest_checkpoint.pth"))

        if improved:
            torch.save(teacher.state_dict(), os.path.join(ckpt_dir, "best_teacher_model.pth"))
            logger.info(f"--> new BEST ({cfg.optim.select_metric}={best:.4f}, "
                        f"MRE {score['average_mre']:.2f}px)")
        elif cfg.optim.early_stop_patience > 0 and epochs_since_best >= cfg.optim.early_stop_patience:
            logger.info(f"Early stopping (no improvement for {epochs_since_best} epochs).")
            break

    writer.close()
    logger.info(f"Phase 2 finished. Best {cfg.optim.select_metric}={best:.4f}")
    return best
