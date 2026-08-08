#!/usr/bin/env python3
"""Compute-cost profiling: params / FLOPs / peak GPU memory / throughput.

Reviewer 3 (KKV9): "The method uses a large DINOv2-L encoder with complex
multi-branch neck, yet computational cost, throughput, and memory footprint
are not reported." Nothing in EXPERIMENT_RESULTS.md logs this -- this script
fills that gap.

No dataset needed: builds the real model straight from a YAML config (the same
way training does) and profiles it on random tensors of the right shape.
torch.hub-downloads the real DINOv2-L/14 backbone on first run (~1.1GB) --
this is NOT the CPU dummy-backbone smoke test, it's the actual deployed model.

Single task head active per forward pass, matching the task-homogeneous batch
routing invariant (CLAUDE.md: forward_phase2(x, task_id) runs exactly one head).

Usage:
    python scripts/profile_compute.py --config configs/phase2_upgraded.yaml \
        --device cuda:0 --batch-size 64

Reports, at fp32 param-count granularity and the config's own amp_dtype for
speed/memory:
  - parameter counts (total / trainable / frozen, and encoder vs. neck vs. heads)
  - forward-pass FLOPs (single task head, batch=1, canvas resolution)
  - peak GPU memory: inference (eval, no_grad) and training (forward+backward+AdamW step)
  - throughput: images/sec, inference and training
"""
import argparse
import time

import torch
import torch.nn as nn

from gubiometry.config import load_config
from gubiometry.models.model import build_model_from_config
from gubiometry.constants import TASK_ORDER

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder = sum(p.numel() for p in model.encoder.parameters())
    encoder_trainable = sum(p.numel() for p in model.encoder_trainable_parameters())
    neck = sum(p.numel() for p in model.shared_upsampler.parameters())
    heads_total = sum(p.numel() for p in model.heads.parameters())
    one_head = sum(p.numel() for p in model.heads[TASK_ORDER[0]].parameters())
    return dict(total=total, trainable=trainable, frozen=total - trainable,
                encoder=encoder, encoder_trainable=encoder_trainable,
                neck=neck, heads_total=heads_total, one_head=one_head,
                n_heads=len(model.heads))


class _Phase2Wrapper(nn.Module):
    """Adapts forward_phase2(x, task_id) to a plain forward(x) for fvcore."""
    def __init__(self, model, task_id):
        super().__init__()
        self.model = model
        self.task_id = task_id

    def forward(self, x):
        return self.model.forward_phase2(x, self.task_id)


def count_flops(model, x, task_id):
    """Single forward pass, one task head. Tries torch's built-in FlopCounterMode
    first (dispatch-based, robust to custom architectures); falls back to fvcore."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model.forward_phase2(x, task_id)
        return fc.get_total_flops(), "torch.utils.flop_counter"
    except Exception as e:
        print(f"[flops] FlopCounterMode failed ({e!r}); trying fvcore...")
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        wrapper = _Phase2Wrapper(model, task_id)
        with torch.no_grad():
            fca = FlopCountAnalysis(wrapper, x)
            fca.unsupported_ops_warnings(False)
            return fca.total(), "fvcore"
    except Exception as e:
        print(f"[flops] fvcore also unavailable/failed ({e!r}) -- skipping FLOPs.")
        return None, None


@torch.no_grad()
def measure_inference(model, batch_size, canvas, task_id, device, amp_dtype="fp32",
                       iters=30, warmup=10):
    """amp_dtype='fp32' matches the actual predict.py/evaluate.py (no autocast there
    today); also report 'bf16' to show the (currently unused) headroom -- otherwise
    fp32 inference can look slower than bf16 training, which reads as a bug, not a
    precision artifact."""
    model.eval()
    dtype = _AMP_DTYPES[amp_dtype]
    autocast_on = dtype != torch.float32
    x = torch.randn(batch_size, 3, canvas, canvas, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=dtype, enabled=autocast_on):
        for _ in range(warmup):
            model.forward_phase2(x, task_id)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model.forward_phase2(x, task_id)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 2**30
    imgs_per_sec = (iters * batch_size) / (t1 - t0)
    return imgs_per_sec, peak_mem_gb


def measure_training(model, batch_size, canvas, task_id, device, amp_dtype, lr,
                      iters=20, warmup=5):
    model.train()
    params = model.encoder_trainable_parameters() + model.head_trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr)
    x = torch.randn(batch_size, 3, canvas, canvas, device=device)
    dtype = _AMP_DTYPES[amp_dtype]
    autocast_on = dtype != torch.float32

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=autocast_on):
            out = model.forward_phase2(x, task_id)
            loss = out.float().pow(2).mean()   # fake loss -- exercises the real backward graph
        loss.backward()
        opt.step()

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 2**30
    imgs_per_sec = (iters * batch_size) / (t1 - t0)
    return imgs_per_sec, peak_mem_gb


def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.2f}{unit}"
    return str(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase2_upgraded.yaml",
                     help="YAML config to profile (default: the paper's actual recipe)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=None,
                     help="default: the config's data.batch_size")
    ap.add_argument("--task", default="AOP", choices=TASK_ORDER,
                     help="which task head to route through (default: AOP)")
    ap.add_argument("--inference-batch-size", type=int, default=1,
                     help="separate (smaller) batch size for the inference-latency measurement (default: 1)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "This script needs a GPU."
    device = torch.device(args.device)

    cfg = load_config(args.config)
    canvas = cfg.data.canvas
    train_bs = args.batch_size or cfg.data.batch_size
    amp_dtype = cfg.optim.amp_dtype
    lr = cfg.optim.lr

    print(f"Config: {args.config}")
    print(f"  backbone={cfg.model.backbone.name}  unfreeze_last_n_blocks={cfg.model.backbone.unfreeze_last_n_blocks}")
    print(f"  neck.input_mode={cfg.model.neck.input_mode}  neck.decoder={getattr(cfg.model.neck, 'decoder', 'hrnet')}")
    print(f"  heatmap_size={cfg.model.heatmap_size}  canvas={canvas}  amp_dtype={amp_dtype}")
    print(f"  train batch_size={train_bs}  task_id={args.task!r}  device={device}")
    print()

    print("Building model (torch.hub-downloads DINOv2-L/14 on first run)...")
    model = build_model_from_config(cfg, freeze_encoder=True).to(device)

    # --- params ---
    p = count_params(model)
    print("\n=== Parameters ===")
    print(f"  Total:              {human(p['total'])} ({p['total']:,})")
    print(f"  Trainable:          {human(p['trainable'])} ({p['trainable']:,})")
    print(f"  Frozen:             {human(p['frozen'])} ({p['frozen']:,})")
    print(f"  Encoder (total):    {human(p['encoder'])}")
    print(f"  Encoder (trainable, last {cfg.model.backbone.unfreeze_last_n_blocks} blocks): {human(p['encoder_trainable'])}")
    print(f"  Neck:               {human(p['neck'])}")
    print(f"  Heads (all {p['n_heads']}):    {human(p['heads_total'])}  (~{human(p['one_head'])} each)")

    # --- FLOPs (single forward, batch=1, one task head) ---
    print("\n=== FLOPs (single forward pass, batch=1, one task head) ===")
    x1 = torch.randn(1, 3, canvas, canvas, device=device)
    flops, method = count_flops(model, x1, args.task)
    if flops is not None:
        print(f"  {flops/1e9:.2f} GFLOPs  (via {method})")
    del x1
    torch.cuda.empty_cache()

    # --- inference: memory + throughput ---
    # fp32 matches the actual predict.py/evaluate.py today (no autocast there);
    # bf16 is also reported so it's directly comparable to bf16 training throughput
    # below, rather than looking like training is mysteriously faster than inference.
    print(f"\n=== Inference, fp32 (eval, no_grad, batch={args.inference_batch_size}) ===")
    imgs_s, mem_gb = measure_inference(model, args.inference_batch_size, canvas, args.task, device, "fp32")
    print(f"  Throughput: {imgs_s:.1f} img/s   Peak GPU memory: {mem_gb:.2f} GB")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    print(f"\n=== Inference, fp32 (eval, no_grad, batch={train_bs}, matches training batch size) ===")
    imgs_s_b, mem_gb_b = measure_inference(model, train_bs, canvas, args.task, device, "fp32")
    print(f"  Throughput: {imgs_s_b:.1f} img/s   Peak GPU memory: {mem_gb_b:.2f} GB")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    print(f"\n=== Inference, {amp_dtype} autocast (eval, no_grad, batch={train_bs}) — headroom check, not used by predict.py today ===")
    imgs_s_bf, mem_gb_bf = measure_inference(model, train_bs, canvas, args.task, device, amp_dtype)
    print(f"  Throughput: {imgs_s_bf:.1f} img/s   Peak GPU memory: {mem_gb_bf:.2f} GB")
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # --- training: memory + throughput ---
    print(f"\n=== Training (forward+backward+AdamW step, batch={train_bs}, {amp_dtype}) ===")
    imgs_s_t, mem_gb_t = measure_training(model, train_bs, canvas, args.task, device, amp_dtype, lr)
    print(f"  Throughput: {imgs_s_t:.1f} img/s   Peak GPU memory: {mem_gb_t:.2f} GB")

    print("\nDone. Copy the numbers above into EXPERIMENT_RESULTS.md / the paper's")
    print("Implementation Details section.")


if __name__ == "__main__":
    main()
