"""Optimizer/scheduler/EMA builders.

`build_param_groups` implements layer-wise LR decay (LLRD) over the unfrozen
DINOv2 blocks -- the canonical ViT fine-tuning recipe. `llrd_decay=1.0` gives every
unfrozen block the same LR (= base_lr * encoder_lr_mult), i.e. the original
single-multiplier behavior, so it is a strict backward-compat escape hatch.
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR


def build_param_groups(model, base_lr, encoder_lr_mult=0.1, llrd_decay=1.0):
    """Groups: [head/neck @ base_lr] + [encoder.norm @ top_lr] + [block_i @ top_lr*decay^depth].

    depth 0 = deepest (last) unfrozen block / final norm; deeper decay for earlier
    blocks. Returns a list of {"params", "lr"} dicts (weight_decay is applied
    globally by the optimizer, matching the original).
    """
    groups = [{"params": model.head_trainable_parameters(), "lr": base_lr}]

    n = model.unfreeze_last_n_blocks
    if n > 0:
        top_lr = base_lr * encoder_lr_mult
        groups.append({"params": list(model.encoder.norm.parameters()), "lr": top_lr})
        # blocks[-n:] reversed -> deepest first (depth 0)
        for depth, blk in enumerate(reversed(list(model.encoder.blocks[-n:]))):
            groups.append({"params": list(blk.parameters()), "lr": top_lr * (llrd_decay ** depth)})
    return groups


def build_optimizer(model, optim_cfg):
    """AdamW with the standard ViT recipe: weight decay only on >=2-D weights; norms and
    biases (ndim<2) are excluded (decaying LayerNorm/GroupNorm scales is harmful). Each
    LR group is split into decay / no-decay sub-groups, preserving its learning rate."""
    lr_groups = build_param_groups(model, optim_cfg.lr, optim_cfg.encoder_lr_mult, optim_cfg.llrd_decay)
    opt_groups = []
    for g in lr_groups:
        decay = [p for p in g["params"] if p.requires_grad and p.ndim >= 2]
        no_decay = [p for p in g["params"] if p.requires_grad and p.ndim < 2]
        if decay:
            opt_groups.append({"params": decay, "lr": g["lr"], "weight_decay": optim_cfg.weight_decay})
        if no_decay:
            opt_groups.append({"params": no_decay, "lr": g["lr"], "weight_decay": 0.0})
    return AdamW(opt_groups)


def build_scheduler(optimizer, warmup_epochs, epochs, kind="warmup_cosine"):
    """`warmup_cosine` (default): linear warmup then cosine (original behavior).
    `cosine`: cosine over all epochs, no warmup. `constant`: flat LR."""
    if kind == "warmup_cosine":
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    if kind == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)
    if kind == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    raise ValueError(f"Unknown scheduler: {kind!r} (warmup_cosine|cosine|constant)")


@torch.no_grad()
def update_ema_phase2(model, ema_model, alpha=0.999):
    """EMA over the Phase-2 trainable params (head/neck + unfrozen encoder)."""
    for p, ep in zip(model.head_trainable_parameters(), ema_model.head_trainable_parameters()):
        ep.data.mul_(alpha).add_(p.data, alpha=1 - alpha)
    for p, ep in zip(model.encoder_trainable_parameters(), ema_model.encoder_trainable_parameters()):
        ep.data.mul_(alpha).add_(p.data, alpha=1 - alpha)


@torch.no_grad()
def update_ema_all(model, ema_model, alpha=0.999):
    """EMA over all parameters (Phase-1 encoder adaptation)."""
    for p, ep in zip(model.parameters(), ema_model.parameters()):
        ep.data.mul_(alpha).add_(p.data, alpha=1 - alpha)
