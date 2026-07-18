"""Vendored DINOv2 self-supervised components (single-GPU, no xFormers / torch.distributed).

Adapted from facebookresearch/DINOv2 (Apache License 2.0):
  * DINOHead            <- dinov2/layers/dino_head.py         (verbatim)
  * DINOLossV2          <- dinov2/loss/dino_clstoken_loss.py  (async/dist centering -> synchronous EMA)
  * iBOTPatchLossV2     <- dinov2/loss/ibot_patch_loss.py     (masked patch loss; sinkhorn removed)
  * KoLeoLoss           <- dinov2/loss/koleo_loss.py          (cuda-amp autocast -> .float())
  * MaskingGenerator    <- dinov2/data/masking.py             (verbatim, numpy)
  * CosineParamScheduler<- dinov2/utils/utils.py::CosineScheduler (compact port)

We VENDOR (not import) so that (a) the CPU smoke-test flow runs with no torch.hub package,
and (b) we are immune to the cached hub checkout being a fork / re-downloaded. The centering
paths are rewritten to a plain synchronous EMA -- the original defers an async all_reduce; on a
single GPU that reduces to `center = m*center + (1-m)*batch_center`. `sinkhorn_knopp_teacher`
(unconditional `dist.all_reduce`) is intentionally omitted -- softmax-centering only.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


# --------------------------------------------------------------------------- #
# Projection head (verbatim from dinov2/layers/dino_head.py)
# --------------------------------------------------------------------------- #
def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """MLP -> L2-normalized bottleneck -> weight-normed linear (no BatchNorm by default)."""
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256, mlp_bias=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


# --------------------------------------------------------------------------- #
# DINO CLS-token loss (synchronous single-GPU centering)
# --------------------------------------------------------------------------- #
class DINOLossV2(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        """Center + sharpen the teacher logits into target probabilities. Reads the current
        center; does NOT mutate it (the trainer calls update_center once per effective step)."""
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    def forward(self, student_output_list, teacher_softmaxed_list, skip_diagonal=True):
        """Cross-entropy of student log-probs against centered teacher targets.

        student_output_list: list over student crops (0,1 = global; 2.. = local) of (B, K) logits.
        teacher_softmaxed_list: list over the 2 teacher global crops of (B, K) probabilities.
        skip_diagonal: skip the (teacher crop i, student crop i) pair (standard DINO).
        """
        total, n_terms = 0.0, 0
        for iq, t in enumerate(teacher_softmaxed_list):
            for v, s in enumerate(student_output_list):
                if skip_diagonal and v == iq:
                    continue
                lsm = F.log_softmax(s / self.student_temp, dim=-1)
                total = total - torch.sum(t * lsm, dim=-1).mean()
                n_terms += 1
        return total / max(n_terms, 1)

    @torch.no_grad()
    def update_center_ema(self, batch_center):
        """batch_center: (1, K) mean of teacher outputs over the effective batch."""
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


# --------------------------------------------------------------------------- #
# iBOT masked-patch loss (synchronous single-GPU centering; sinkhorn removed)
# --------------------------------------------------------------------------- #
class iBOTPatchLossV2(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, patch_out_dim))

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        """teacher_patch_tokens: (M, K) masked-patch logits -> centered/sharpened targets."""
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward_masked(self, student_masked, teacher_softmaxed_masked, masks_weight, n_images):
        """Weighted cross-entropy over masked patches.

        student_masked: (M, K) student logits at masked positions.
        teacher_softmaxed_masked: (M, K) centered teacher targets.
        masks_weight: (M,) per-patch weight = 1 / (#masked patches in that image).
        n_images: number of images (global crops) the masked patches came from.
        """
        loss = torch.sum(teacher_softmaxed_masked * F.log_softmax(student_masked / self.student_temp, dim=-1), dim=-1)
        loss = loss * masks_weight
        return -loss.sum() / max(n_images, 1)

    @torch.no_grad()
    def update_center_ema(self, batch_center):
        """batch_center: (1, K) mean of teacher masked-patch outputs over the effective batch."""
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


# --------------------------------------------------------------------------- #
# KoLeo spread regularizer (from dinov2/loss/koleo_loss.py; .float() vs cuda-amp)
# --------------------------------------------------------------------------- #
class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic regularizer (Sablayrolles et al. 2018)."""
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)          # fill diagonal with -1
        _, I = torch.max(dots, dim=1)
        return I

    def forward(self, student_output, eps=1e-8):
        """student_output: (B, D) backbone CLS features of one crop."""
        x = student_output.float()
        x = F.normalize(x, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(x)
        distances = self.pdist(x, x[I])
        return -torch.log(distances + eps).mean()


# --------------------------------------------------------------------------- #
# iBOT block-wise mask generator (verbatim from dinov2/data/masking.py, numpy)
# --------------------------------------------------------------------------- #
import random


class MaskingGenerator:
    def __init__(self, input_size, num_masking_patches=None, min_num_patches=4,
                 max_num_patches=None, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        if self.max_num_patches is None:          # neither given -> cap at the whole grid
            self.max_num_patches = self.num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top: top + h, left: left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = min(num_masking_patches - mask_count, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask


# --------------------------------------------------------------------------- #
# Cosine schedule with linear warmup (compact port of dinov2 CosineScheduler)
# --------------------------------------------------------------------------- #
class CosineParamScheduler:
    """Precomputed per-step schedule: optional linear warmup then cosine to `final_value`.
    Index with `sched[step]`; out-of-range steps clamp to the last value."""
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0.0):
        warmup_iters = max(0, min(warmup_iters, total_iters))
        warmup = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_iters > 0 else np.array([])
        n_cos = max(1, total_iters - warmup_iters)
        it = np.arange(n_cos)
        cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * it / n_cos))
        self.schedule = np.concatenate((warmup, cosine))
        self.total_iters = len(self.schedule)

    def __getitem__(self, it):
        return float(self.schedule[min(int(it), self.total_iters - 1)])
