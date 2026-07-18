"""DINOv2 backbone loading + multi-level feature extraction, plus a DummyBackbone
for CPU tests (no torch.hub download).

`load_backbone(name)` returns (encoder, embed_dim). Supported names:
  * "dinov2_vitl14"      -- original.
  * "dinov2_vitl14_reg"  -- register variant (cleaner dense features; requires a
                            Phase-1 re-run since the checkpoint is backbone-specific).
  * "dummy"              -- shape-correct stand-in for tests.

Both real variants expose `forward_features(x)['x_norm_patchtokens']` and
`get_intermediate_layers(x, n, reshape=True, norm=True)` with CLS/register tokens
already stripped, so the neck code is backbone-agnostic.
"""

import torch
import torch.nn as nn


def load_backbone(name="dinov2_vitl14"):
    if name == "dummy":
        enc = DummyBackbone()
        return enc, enc.embed_dim
    enc = torch.hub.load("facebookresearch/dinov2", name)
    return enc, enc.embed_dim


def get_multilevel_features(encoder, x, layers=(5, 11, 17, 23)):
    """Return a tuple of (B, C, H, W) grids at the given block depths (shallow..deep)."""
    return encoder.get_intermediate_layers(x, n=list(layers), reshape=True, norm=True)


# --------------------------------------------------------------------------- #
# Dummy backbone (tests only)
# --------------------------------------------------------------------------- #
class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.scale


class DummyBackbone(nn.Module):
    """Patch-embed + per-block scalar scales; produces correctly-shaped token grids
    for any input whose side is a multiple of 14. Encoder params (blocks, norm)
    receive gradients so LLRD/unfreeze plumbing is exercised for real."""

    def __init__(self, embed_dim=1024, depth=24, patch=14, num_register_tokens=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch
        self.num_register_tokens = num_register_tokens
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)
        self.blocks = nn.ModuleList([_DummyBlock() for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # CLS / register / mask tokens -> lets the Phase-1 SSL path (which reads
        # x_norm_clstoken and passes masks) be exercised on CPU with the dummy backbone.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        # small non-zero init: the dummy's pure-multiply blocks would otherwise keep a
        # zero-init mask token exactly zero, and DINOHead's L2-normalize kills the gradient
        # at exact zero (a degeneracy the real attention+residual backbone never hits).
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def _run(self, x, masks=None):
        f = self.patch_embed(x)              # (B, C, H, W)
        B, C, H, W = f.shape
        tok = f.flatten(2).transpose(1, 2)   # (B, N, C)
        if masks is not None:                # swap in the mask token (mirrors prepare_tokens_with_masks)
            tok = torch.where(masks.unsqueeze(-1), self.mask_token.to(tok.dtype).view(1, 1, C), tok)
        feats = []
        for blk in self.blocks:
            tok = blk(tok)
            feats.append(tok)
        return feats, (B, C, H, W)

    def forward_features(self, x, masks=None):
        feats, (B, C, H, W) = self._run(x, masks=masks)
        patches = self.norm(feats[-1])                                   # (B, N, C)
        cls = self.norm(feats[-1].mean(dim=1) + self.cls_token.view(1, C))   # (B, C)
        reg = self.norm(self.register_tokens.expand(B, -1, -1))          # (B, R, C)
        return {
            "x_norm_clstoken": cls,
            "x_norm_regtokens": reg,
            "x_norm_patchtokens": patches,
            "x_prenorm": feats[-1],
            "masks": masks,
        }

    def get_intermediate_layers(self, x, n, reshape=True, norm=True, return_class_token=False):
        feats, (B, C, H, W) = self._run(x)
        out = []
        for i in n:
            t = feats[i]
            if norm:
                t = self.norm(t)
            if reshape:
                t = t.transpose(1, 2).reshape(B, C, H, W)
            out.append(t)
        return tuple(out)
