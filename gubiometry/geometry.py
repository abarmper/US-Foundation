"""Canonical coordinate geometry: soft-argmax decoding and the letterbox forward /
inverse transforms. Pure numpy/torch -- no cv2/albumentations -- so inference,
evaluation and the metric-aligned validation loop have no heavy dependencies.

The letterbox convention matches the original pipeline's
`LongestMaxSize(canvas) + PadIfNeeded(canvas, canvas, center pad, fill=0)`:
    scale     = canvas / max(orig_h, orig_w)
    new_h/w   = int(orig_h/w * scale)          # truncation, matched forward & inverse
    pad_top   = (canvas - new_h) // 2          # centered
    pad_left  = (canvas - new_w) // 2
Forward and inverse here are exact inverses of each other by construction.
"""

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Soft-argmax
# --------------------------------------------------------------------------- #
def soft_argmax_coords(logits, temperature=10.0, canvas=518):
    """Continuous (x, y) per keypoint from heatmap logits, in `canvas` pixels.

    logits: (B, K, H, W). Returns (B, K, 2) with last dim = (x, y).

    Identical math to the original train/eval soft-argmax; `canvas` is now an
    explicit argument (was hardcoded 518) so multi-scale TTA can pass the view's
    own canvas -- hardcoding 518 would silently bias every rescaled view.
    """
    B, K, H, W = logits.shape
    heatmaps = F.softmax(logits.reshape(B, K, -1) * temperature, dim=-1)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=logits.device),
        torch.arange(W, device=logits.device),
        indexing="ij",
    )
    grid_x = grid_x.flatten().float()
    grid_y = grid_y.flatten().float()

    pred_x = torch.sum(heatmaps * grid_x, dim=-1) * (float(canvas) / W)
    pred_y = torch.sum(heatmaps * grid_y, dim=-1) * (float(canvas) / H)
    return torch.stack([pred_x, pred_y], dim=-1)


# --------------------------------------------------------------------------- #
# Letterbox forward / inverse
# --------------------------------------------------------------------------- #
def letterbox_params(orig_h, orig_w, canvas=518):
    """Return (scale, pad_left, pad_top, new_h, new_w) for the centered letterbox."""
    scale = canvas / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_top = (canvas - new_h) // 2
    pad_left = (canvas - new_w) // 2
    return scale, pad_left, pad_top, new_h, new_w


def forward_letterbox_kps(kps, orig_h, orig_w, canvas=518):
    """Map original-image (x, y) points into canvas coordinates.

    kps: (N, 2) array-like in original pixels. Negative (missing) points are
    passed through unchanged so the (-1, -1) padding convention survives.
    """
    kps = np.asarray(kps, dtype=np.float64)
    scale, pad_left, pad_top, _, _ = letterbox_params(orig_h, orig_w, canvas)
    out = kps.copy()
    mask = kps[:, 0] >= 0
    out[mask, 0] = kps[mask, 0] * scale + pad_left
    out[mask, 1] = kps[mask, 1] * scale + pad_top
    return out


def remove_padding_and_scale(x_padded, y_padded, orig_h, orig_w, canvas=518):
    """Invert the letterbox for a single point: canvas px -> original px (clipped).

    Verbatim behavior from the original visualization/common.py, kept as the one
    canonical inverse used by validation, prediction and evaluation.
    """
    scale = canvas / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_top = (canvas - new_h) // 2
    pad_left = (canvas - new_w) // 2
    x_orig = (x_padded - pad_left) / scale
    y_orig = (y_padded - pad_top) / scale
    return (
        float(np.clip(x_orig, 0.0, float(orig_w))),
        float(np.clip(y_orig, 0.0, float(orig_h))),
    )


def inverse_letterbox_kps(kps_canvas, orig_h, orig_w, canvas=518):
    """Vectorized inverse letterbox: (N, 2) canvas px -> (N, 2) original px, clipped."""
    kps_canvas = np.asarray(kps_canvas, dtype=np.float64)
    scale = canvas / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_top = (canvas - new_h) // 2
    pad_left = (canvas - new_w) // 2
    out = np.empty_like(kps_canvas)
    out[:, 0] = np.clip((kps_canvas[:, 0] - pad_left) / scale, 0.0, float(orig_w))
    out[:, 1] = np.clip((kps_canvas[:, 1] - pad_top) / scale, 0.0, float(orig_h))
    return out
