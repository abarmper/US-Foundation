"""Soft-argmax coordinate-regression loss (+ optional DSNT regularization) and MRE.

`dsnt_lambda=0.0` (default) makes `soft_argmax_loss` return exactly the original
loss value -- pure coordinate L1, no heatmap-amplitude target. With `dsnt_lambda>0`
a DSNT-style regularizer (Nibali et al. 2018) is added to keep each predicted
heatmap tight and unimodal, WITHOUT reintroducing a fixed heatmap-MSE target:

  * "js"  : Jensen-Shannon divergence between the predicted probability map and a
            unit-sum Gaussian centered at the GT point (amplitude-free -- both are
            normalized distributions).
  * "var" : penalize the predicted spatial variance above sigma^2 (GT-free).
"""

import math

import torch
import torch.nn.functional as F

from .metrics import MEASUREMENT_SPECS


def _grids(H, W, device):
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    return grid_x.flatten().float(), grid_y.flatten().float()   # (H*W,), (H*W,)


def _coord_term(pred, target, mask, kind, omega, eps, coord_scale=None):
    """Masked coordinate loss. `l1` reproduces the original exactly; missing points
    (mask=0) contribute 0 in every variant (their masked residual is 0, and wing(0)=0).

    `coord_scale` (B,1,2), if given, multiplies the residual per sample/axis -- used to
    convert canvas-px residuals to original-px (loss_space="original") so the loss is
    weighted like the original-pixel metric.
    """
    diff = (pred - target) * mask
    if coord_scale is not None:
        diff = diff * coord_scale
    if kind == "l1":
        per = diff.abs()
    elif kind == "smooth_l1":
        per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none", beta=1.0)
    elif kind == "wing":
        r = diff.abs()
        C = omega - omega * math.log(1.0 + omega / eps)
        per = torch.where(r < omega, omega * torch.log(1.0 + r / eps), r - C)
    else:
        raise ValueError(f"Unknown coord_loss: {kind!r}")
    return per.sum() / (mask.sum() + 1e-6)


def soft_argmax_loss(pred_logits, target_coords_518, temperature=10.0, canvas=518,
                     dsnt_lambda=0.0, dsnt_type="js", dsnt_sigma=1.0,
                     coord_loss="l1", wing_omega=10.0, wing_epsilon=2.0, coord_scale=None):
    """Return (loss, pred_coords, parts).

    pred_logits: (B, K, H, W); target_coords_518: (B, K, 2) in `canvas` px, with
    missing points encoded (-1, -1). pred_coords are in `canvas` px.
    `coord_loss` selects l1 (default, bit-identical) | smooth_l1 | wing.
    `coord_scale` (B,1,2) optionally reweights residuals per sample (loss_space="original").
    """
    B, K, H, W = pred_logits.shape
    P = F.softmax(pred_logits.reshape(B, K, -1) * temperature, dim=-1)       # (B,K,HW)

    grid_x, grid_y = _grids(H, W, pred_logits.device)
    pred_x = torch.sum(P * grid_x, dim=-1) * (float(canvas) / W)
    pred_y = torch.sum(P * grid_y, dim=-1) * (float(canvas) / H)
    pred_coords = torch.stack([pred_x, pred_y], dim=-1)                       # (B,K,2)

    mask = (target_coords_518[..., 0] >= 0).float().unsqueeze(-1)            # (B,K,1)
    coord_l = _coord_term(pred_coords, target_coords_518, mask, coord_loss,
                          wing_omega, wing_epsilon, coord_scale)

    parts = {"coord": float(coord_l.detach())}
    loss = coord_l

    if dsnt_lambda > 0.0:
        m = mask.squeeze(-1)                                                  # (B,K)
        reg = _dsnt_regularizer(P, target_coords_518, m, grid_x, grid_y,
                                H, W, canvas, dsnt_type, dsnt_sigma)
        parts["dsnt"] = float(reg.detach())
        loss = coord_l + dsnt_lambda * reg

    return loss, pred_coords, parts


def _dsnt_regularizer(P, target_518, m, grid_x, grid_y, H, W, canvas, dsnt_type, sigma):
    eps = 1e-12
    if dsnt_type == "var":
        mu_x = torch.sum(P * grid_x, dim=-1, keepdim=True)                    # (B,K,1)
        mu_y = torch.sum(P * grid_y, dim=-1, keepdim=True)
        var_x = torch.sum(P * (grid_x - mu_x) ** 2, dim=-1)                   # (B,K)
        var_y = torch.sum(P * (grid_y - mu_y) ** 2, dim=-1)
        s2 = sigma * sigma
        r = torch.relu(var_x - s2) + torch.relu(var_y - s2)
        return torch.sum(m * r) / (m.sum() + 1e-6)

    if dsnt_type == "js":
        # GT in grid-cell units
        tx = target_518[..., 0].clamp(min=0) * (W / float(canvas))           # (B,K)
        ty = target_518[..., 1].clamp(min=0) * (H / float(canvas))
        d2 = (grid_x[None, None, :] - tx[..., None]) ** 2 + \
             (grid_y[None, None, :] - ty[..., None]) ** 2                     # (B,K,HW)
        G = torch.softmax(-d2 / (2.0 * sigma * sigma), dim=-1)               # unit-sum Gaussian
        M = 0.5 * (P + G)
        js = 0.5 * torch.sum(P * (torch.log(P + eps) - torch.log(M + eps)), dim=-1) + \
             0.5 * torch.sum(G * (torch.log(G + eps) - torch.log(M + eps)), dim=-1)  # (B,K)
        return torch.sum(m * js) / (m.sum() + 1e-6)

    raise ValueError(f"Unknown dsnt_type: {dsnt_type!r}")


def mean_radial_error(pred_coords, target_coords_518):
    """Per-batch MRE (canvas px) over visible keypoints only."""
    mask = (target_coords_518[..., 0] >= 0).float()
    dist = torch.sqrt(torch.sum((pred_coords - target_coords_518) ** 2, dim=-1) + 1e-12)
    dist = dist * mask
    return dist.sum() / (mask.sum() + 1e-6)


# --------------------------------------------------------------------------- #
# Measurement-aligned auxiliary loss (optim.measurement_lambda)
# --------------------------------------------------------------------------- #
def _t_dist(p, i, j):
    return torch.linalg.norm(p[:, i] - p[:, j], dim=-1)                       # (B,)


def _t_angle_deg(p, a, b, c):
    v1 = p[:, b] - p[:, a]
    v2 = p[:, c] - p[:, a]
    n1 = torch.linalg.norm(v1, dim=-1)
    n2 = torch.linalg.norm(v2, dim=-1)
    cos = (v1 * v2).sum(-1) / (n1 * n2 + 1e-9)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)   # keep acos gradient finite
    return torch.acos(cos) * (180.0 / math.pi)


def _t_measure(p, kind, idx):
    if kind == "dist":
        return _t_dist(p, *idx)
    if kind == "angle":
        return _t_angle_deg(p, *idx)
    if kind == "pi_half_sum":
        i, j, k, l = idx
        return math.pi * (_t_dist(p, i, j) + _t_dist(p, k, l)) / 2.0
    if kind == "min_dist":
        i, j, k, l = idx
        return torch.minimum(_t_dist(p, i, j), _t_dist(p, k, l))
    raise ValueError(f"Unknown measurement kind: {kind!r}")


def measurement_loss(pred_coords, target_coords, task_id):
    """Differentiable |pred_measure - gt_measure| over the task's clinical measurements
    (distances in canvas px, the AOP angle in degrees), reusing the exact index pairs
    from metrics.MEASUREMENT_SPECS. Measurements requiring a missing GT point are
    skipped per-sample. Returns a 0 scalar if nothing is measurable.
    """
    if isinstance(task_id, (list, tuple)):
        task_id = task_id[0]
    # float32 for numerically stable acos (angles) under AMP/autocast
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    specs = MEASUREMENT_SPECS.get(task_id, [])
    K = pred_coords.shape[1]
    total = pred_coords.new_zeros(())
    n_terms = 0
    for _name, kind, idx in specs:
        if max(idx) >= K:      # defensive: keypoints must cover the measurement's indices
            continue
        vis = torch.ones(pred_coords.shape[0], dtype=torch.bool, device=pred_coords.device)
        for i in idx:
            vis = vis & (target_coords[:, i, 0] >= 0) & (target_coords[:, i, 1] >= 0)
        if vis.sum() == 0:
            continue
        err = (_t_measure(pred_coords, kind, idx) - _t_measure(target_coords, kind, idx)).abs()
        total = total + err[vis].mean()
        n_terms += 1
    return total / n_terms if n_terms > 0 else total
