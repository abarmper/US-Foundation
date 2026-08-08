#!/usr/bin/env python3
"""Phase-1 (SSL pretraining) compute-cost profiling: params / FLOPs / peak GPU
memory / throughput, for both the bulk (224px) and high-res tail (518px)
resolution segments.

Companion to scripts/profile_compute.py (Phase 2). Reviewer 3 (KKV9) asked
about "the large DINOv2-L encoder"'s compute cost broadly, not just Phase 2 --
Phase 1 is actually the more expensive stage (100+ epochs over 191K unlabeled
images, an 8-view multi-crop forward per image, and the FULL encoder trained,
not just the last 4 blocks like Phase 2).

Reuses the REAL training code paths (gubiometry.engine.phase1_dinov2's
_build_segment/_DINOv2Wrapper/_build_param_groups, gubiometry.models.dino_ssl's
loss modules) so the data pipeline (multicrop transforms, iBOT masking,
foreground-aware cropping) and model are exactly what training uses -- only a
tiny on-disk synthetic *unlabeled* image set is substituted for the real
191K-image data_root (Phase 1 doesn't need labels, so this is a much smaller
harness than gubiometry.testing.build_synthetic_dataset). Random-noise images
(not solid-color) so the foreground/iBOT masking isn't degenerate.

Usage:
    python scripts/profile_compute_phase1.py --config configs/phase1_dinov2.yaml --device cuda:0
"""
import argparse
import itertools
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from gubiometry.config import load_config
from gubiometry.constants import TASK_ORDER
from gubiometry.engine.common import resolve_amp
from gubiometry.engine.phase1_dinov2 import _build_segment, _DINOv2Wrapper, _build_param_groups
from gubiometry.models.dino_ssl import DINOLossV2, iBOTPatchLossV2, KoLeoLoss


def make_synthetic_unlabeled_root(root, n_per_task=24, size=(640, 800)):
    """images/<TASK>/unlabeled/*.png, random noise (not solid color -- degenerate
    solid-color images give zero-variance patches, so iBOT foreground masking
    would select nothing). No CSV/splits needed -- Phase 1 ignores labels."""
    from PIL import Image
    rng = np.random.default_rng(0)
    h, w = size
    root = Path(root)
    for task in TASK_ORDER[:3]:   # a few real task names is enough; content doesn't matter
        d = root / "images" / task / "unlabeled"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_task):
            arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{task}_u{i:04d}.png")
    return str(root)


def count_params(student):
    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in student.encoder.parameters())
    heads = total - encoder
    return dict(total=total, trainable=trainable, encoder=encoder, heads=heads)


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.2f}{unit}"
    return str(n)


def build_models_and_losses(cfg, p1, device):
    student = _DINOv2Wrapper(cfg.model.backbone.name, p1).to(device)
    teacher = _DINOv2Wrapper(cfg.model.backbone.name, p1).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    dino_loss = DINOLossV2(p1.dino_out_dim, student_temp=p1.student_temp,
                           center_momentum=p1.center_momentum).to(device)
    ibot_loss = iBOTPatchLossV2(p1.ibot_out_dim or p1.dino_out_dim, student_temp=p1.student_temp,
                                center_momentum=p1.center_momentum).to(device)
    koleo_loss = KoLeoLoss().to(device)
    return student, teacher, dino_loss, ibot_loss, koleo_loss


def one_step(student, teacher, dino_loss, ibot_loss, koleo_loss, batch, device,
             amp_on, amp_dtype, p1, teacher_temp=0.04):
    g_crops = batch["collated_global_crops"].to(device, non_blocking=True)
    l_crops = batch["collated_local_crops"].to(device, non_blocking=True) if batch["collated_local_crops"] is not None else None
    masks = batch["collated_masks"].to(device, non_blocking=True)
    mask_idx = batch["mask_indices_list"].to(device, non_blocking=True)
    masks_weight = batch["masks_weight"].to(device, non_blocking=True)
    n_glob = batch["n_global_crops"]
    n_local = batch["n_local_crops"]

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
    return total


def profile_segment(name, cfg, p1, global_size, batch_size, accum, lr, device, iters=16, warmup=4):
    print(f"\n{'='*70}\nSegment: {name}  (global_size={global_size}, batch_size={batch_size}, "
          f"grad_accum={accum}, eff_batch={batch_size*accum}, lr={lr})\n{'='*70}")

    loader, _ = _build_segment(cfg, p1, global_size, batch_size, accum)
    batches = itertools.cycle(loader)

    student, teacher, dino_loss, ibot_loss, koleo_loss = build_models_and_losses(cfg, p1, device)
    p = count_params(student)
    print(f"Params: total={human(p['total'])}  trainable={human(p['trainable'])} "
          f"(FULL encoder trainable in Phase 1, unlike Phase 2's partial unfreeze) "
          f"encoder={human(p['encoder'])}  heads={human(p['heads'])}")

    amp_on, amp_dtype, _ = resolve_amp(cfg.optim.amp_dtype)
    opt = AdamW(_build_param_groups(student, p1.weight_decay), lr=lr)

    # --- FLOPs: one micro-batch's forward only (teacher + student) ---
    first_batch = next(batches)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            one_step(student, teacher, dino_loss, ibot_loss, koleo_loss, first_batch,
                     device, amp_on, amp_dtype, p1)
        print(f"FLOPs (fwd only, teacher+student, one micro-batch of {batch_size} images, "
              f"{2 + p1.n_local_crops} views/image): {fc.get_total_flops()/1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"[flops] FlopCounterMode failed ({e!r}) -- skipping FLOPs for this segment.")

    # --- throughput + peak memory: real accum-cycle training steps ---
    torch.cuda.reset_peak_memory_stats(device)
    student.train()

    def step(batch, do_opt_step):
        total = one_step(student, teacher, dino_loss, ibot_loss, koleo_loss, batch,
                         device, amp_on, amp_dtype, p1)
        (total / accum).backward()
        if do_opt_step:
            torch.nn.utils.clip_grad_norm_(student.parameters(), p1.clip_grad)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(0.99).add_(ps.detach(), alpha=0.01)

    for i in range(warmup):
        step(first_batch if i == 0 else next(batches), (i + 1) % accum == 0)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for i in range(iters):
        step(next(batches), (i + 1) % accum == 0)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 2**30
    imgs_per_sec = (iters * batch_size) / (t1 - t0)
    views_per_img = 2 + p1.n_local_crops
    print(f"Training: {imgs_per_sec:.1f} img/s ({imgs_per_sec*views_per_img:.1f} crop-views/s, "
          f"{views_per_img} views/img)   Peak GPU memory: {peak_mem_gb:.2f} GB")

    del student, teacher, dino_loss, ibot_loss, koleo_loss, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1_dinov2.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=16)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "This script needs a GPU."
    device = torch.device(args.device)

    cfg = load_config(args.config)
    p1 = cfg.phase1
    print(f"Config: {args.config}")
    print(f"  backbone={cfg.model.backbone.name}  amp_dtype={cfg.optim.amp_dtype}")
    print(f"  bulk: global={p1.global_crop_size} bs={p1.batch_size} accum={p1.grad_accum_steps} x {p1.epochs}ep")
    if p1.highres_epochs > 0:
        print(f"  tail: global={p1.highres_crop_size} bs={p1.highres_batch_size} "
              f"accum={p1.highres_grad_accum_steps} x {p1.highres_epochs}ep")

    tmpdir = tempfile.mkdtemp(prefix="gubiometry_profile_p1_")
    try:
        cfg.data.data_root = make_synthetic_unlabeled_root(tmpdir)

        profile_segment("bulk", cfg, p1, p1.global_crop_size, p1.batch_size,
                        p1.grad_accum_steps, p1.lr, device, iters=args.iters)

        if p1.highres_epochs > 0:
            bs_t = p1.highres_batch_size or p1.batch_size
            accum_t = p1.highres_grad_accum_steps or p1.grad_accum_steps
            profile_segment("tail", cfg, p1, p1.highres_crop_size, bs_t, accum_t,
                            p1.highres_lr or p1.lr, device, iters=args.iters)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("\nDone. Copy the numbers above into EXPERIMENT_RESULTS.md / the paper's "
          "compute-cost discussion.")


if __name__ == "__main__":
    main()
