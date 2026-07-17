"""Multi-stage HRNet neck with repeated cross-resolution exchange.

Two input modes:
  * "single"     -- the original behavior: one DINOv2 last-layer 37x37 grid is
                    deconvolved outward to spawn the 74/148 branches. Submodule
                    names and numerics are byte-identical to the original
                    architecture_hrnet.TrueHRNetNeck, so old checkpoints load.
  * "multilevel" -- DPT/ViTPose "reassemble": four intermediate DINOv2 depths
                    (shallow..deep) initialize the branches. Deepest (g23)->b1@37,
                    g17->b2@74, g11 + g05 -> b3@148. The exchange units and final
                    fusion are unchanged. `stage2_b3` is not built in this mode;
                    `reassemble.*` is built instead -> disjoint checkpoint keys.
"""

import torch
import torch.nn as nn

from .heads import group_norm, conv_bn_relu


class UpsampleUnit(nn.Module):
    """1x1 conv (channel match) + bilinear upsample -- HRNet fusion-in path."""
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = group_norm(out_ch)
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def forward(self, x):
        return self.up(self.bn(self.proj(x)))


class DownsampleUnit(nn.Module):
    """Chain of stride-2 3x3 convs -- HRNet fusion-out path (capacity at each scale)."""
    def __init__(self, in_ch, out_ch, num_stride2_steps):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(num_stride2_steps):
            is_last = i == num_stride2_steps - 1
            step_out = out_ch if is_last else in_ch
            layers.append(nn.Conv2d(ch, step_out, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(group_norm(step_out))
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
            ch = step_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ExchangeUnit2Branch(nn.Module):
    """Bidirectional exchange between (37, coarse) and (74, fine)."""
    def __init__(self, w1, w2):
        super().__init__()
        self.down_2to1 = DownsampleUnit(w2, w1, num_stride2_steps=1)
        self.up_1to2 = UpsampleUnit(w1, w2, scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, b1, b2):
        new_b1 = self.relu(b1 + self.down_2to1(b2))
        new_b2 = self.relu(b2 + self.up_1to2(b1))
        return new_b1, new_b2


class ExchangeUnit3Branch(nn.Module):
    """Full all-pairs exchange between (37), (74), (148)."""
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.down_2to1 = DownsampleUnit(w2, w1, num_stride2_steps=1)
        self.down_3to1 = DownsampleUnit(w3, w1, num_stride2_steps=2)
        self.up_1to2 = UpsampleUnit(w1, w2, scale_factor=2)
        self.down_3to2 = DownsampleUnit(w3, w2, num_stride2_steps=1)
        self.up_1to3 = UpsampleUnit(w1, w3, scale_factor=4)
        self.up_2to3 = UpsampleUnit(w2, w3, scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, b1, b2, b3):
        new_b1 = self.relu(b1 + self.down_2to1(b2) + self.down_3to1(b3))
        new_b2 = self.relu(self.up_1to2(b1) + b2 + self.down_3to2(b3))
        new_b3 = self.relu(self.up_1to3(b1) + self.up_2to3(b2) + b3)
        return new_b1, new_b2, new_b3


class _DeconvX2(nn.Module):
    """embed_dim @37 -> out_ch @148 via two stride-2 ConvTranspose2d (37->74->148)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            group_norm(out_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            group_norm(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiLevelReassemble(nn.Module):
    """Projections for the finest branch in multilevel mode (b3 @148 from g11 + g05).

    b1 (from g23) reuses the neck's stage1_b1; b2 (from g17) reuses stage1_b2 --
    both already have exactly the needed (in=embed_dim, out=w1/w2, res=37/74) shape,
    so only the 148-branch needs new modules here.
    """
    def __init__(self, in_channels, w3):
        super().__init__()
        self.proj_b3 = _DeconvX2(in_channels, w3)
        self.proj_b3_aux = _DeconvX2(in_channels, w3)


class SimpleDecoder(nn.Module):
    """ViTPose-style lightweight decoder -- an alternative to TrueHRNetNeck.

    Motivation (see METHOD_CHANGES.md / EXPERIMENTS.md): the strongest finding in the
    ViT-pose literature is that with a strong backbone a *minimal* decoder matches a
    heavy multi-scale one (ViTPose), and for tiny/imbalanced datasets fewer decoder
    params overfit less -- which is exactly the regime the small cardiac tasks live in.
    So this deconvolves a DINOv2 patch grid up to out_channels @148 with two stride-2
    ConvTranspose blocks (37->74->148), replacing HRNet's parallel branches +
    cross-resolution exchange. The soft-argmax heads downstream are unchanged, so
    loss/metric/TTA are byte-identical to hrnet mode -- only the neck differs.

    Input depends on `input_mode` (the caller sizes `in_channels` accordingly):
      * single     -- forward() gets the last-layer grid (B, embed_dim, 37, 37).
      * multilevel -- forward() gets a tuple of L grids at the same 37x37 (all DINOv2
                      depths share resolution), CONCATENATED along channels here to
                      (B, embed_dim*L, 37, 37). This is the concat-based multi-level
                      fusion analogue of hrnet's sum-based reassemble.

    GroupNorm throughout (identical train/eval stats across the heterogeneous tasks,
    matching the rest of the neck/heads). Output shape matches TrueHRNetNeck's:
    (B, out_channels, 148, 148), so it is a drop-in for `shared_upsampler`.
    """
    def __init__(self, in_channels=1024, out_channels=128, deconv_channels=256, n_deconv=2):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_deconv):                      # 37 -> 74 -> 148 (two x2 upsamples)
            layers += [
                nn.ConvTranspose2d(ch, deconv_channels, kernel_size=4, stride=2, padding=1, bias=False),
                group_norm(deconv_channels),
                nn.ReLU(inplace=True),
            ]
            ch = deconv_channels
        layers += [                                    # project to the neck's output width
            nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
            group_norm(out_channels),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        if isinstance(feats, (tuple, list)):           # multilevel: concat depths on channels
            feats = torch.cat(feats, dim=1)            # (B, embed_dim*L, 37, 37)
        return self.net(feats)                         # -> (B, out_channels, 148, 148)


class TrueHRNetNeck(nn.Module):
    def __init__(self, in_channels=1024, out_channels=128, branch_width=(128, 96, 64),
                 dropout_p=0.3, input_mode="single"):
        super().__init__()
        self.input_mode = input_mode
        w1, w2, w3 = branch_width

        # ---- Stage 1: initialization (shared by both modes) ----
        self.stage1_b1 = conv_bn_relu(in_channels, w1, kernel_size=3, padding=1)
        self.stage1_b2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, w2, kernel_size=4, stride=2, padding=1, bias=False),
            group_norm(w2),
            nn.ReLU(inplace=True),
        )
        self.exchange1 = ExchangeUnit2Branch(w1, w2)

        # ---- Stage 2: expansion ----
        self.stage2_b1 = conv_bn_relu(w1, w1, kernel_size=3, padding=1)
        self.stage2_b2 = conv_bn_relu(w2, w2, kernel_size=3, padding=1)
        if input_mode == "single":
            self.stage2_b3 = nn.Sequential(
                nn.ConvTranspose2d(w2, w3, kernel_size=4, stride=2, padding=1, bias=False),
                group_norm(w3),
                nn.ReLU(inplace=True),
            )
        elif input_mode == "multilevel":
            self.reassemble = MultiLevelReassemble(in_channels, w3)
        else:
            raise ValueError(f"Unknown neck input_mode: {input_mode!r}")
        self.exchange2 = ExchangeUnit3Branch(w1, w2, w3)

        # ---- Final multi-scale fusion (shared) ----
        self.fuse_b1 = UpsampleUnit(w1, out_channels, scale_factor=4)
        self.fuse_b2 = UpsampleUnit(w2, out_channels, scale_factor=2)
        self.fuse_b3 = nn.Sequential(
            nn.Conv2d(w3, out_channels, kernel_size=1, bias=False),
            group_norm(out_channels),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.final_layer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            group_norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        if self.input_mode == "single":
            x = feats                                   # (B, C, 37, 37)
            b1 = self.stage1_b1(x)
            b2 = self.stage1_b2(x)
            b1, b2 = self.exchange1(b1, b2)
            b1 = self.stage2_b1(b1)
            b2 = self.stage2_b2(b2)
            b3 = self.stage2_b3(b2)
        else:  # multilevel: feats = (g05, g11, g17, g23) shallow..deep
            g05, g11, g17, g23 = feats
            b1 = self.stage1_b1(g23)
            b2 = self.stage1_b2(g17)
            b1, b2 = self.exchange1(b1, b2)
            b1 = self.stage2_b1(b1)
            b2 = self.stage2_b2(b2)
            b3 = self.reassemble.proj_b3(g11) + self.reassemble.proj_b3_aux(g05)

        b1, b2, b3 = self.exchange2(b1, b2, b3)
        out = self.fuse_b1(b1) + self.fuse_b2(b2) + self.fuse_b3(b3)
        out = self.dropout(out)
        return self.final_layer(out)
