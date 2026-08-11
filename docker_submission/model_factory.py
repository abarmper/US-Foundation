"""Network architecture for the FU Biometry submission.

Byte-for-byte copy of the original training-time architecture
(experiments/v4_true_hrnet_softargmax_reg/architecture_hrnet.py), with exactly
one change: the DINOv2 backbone is loaded from a LOCAL vendored copy of the
facebookresearch/dinov2 repo (./dinov2_repo, baked into the image at build
time) instead of via torch.hub's default GitHub-fetching behavior, and with
pretrained=False (Meta's pretrained weights would be immediately overwritten
by our own checkpoint's state_dict anyway, so there is no reason to download
or bundle them). This is what makes model construction work under
`--network none`.
"""
import os
import torch
import torch.nn as nn

TASK_KEYPOINTS = {
    "A4C": 16,
    "AOP": 4,
    "FA": 4,
    "FUGC": 2,
    "HC": 4,
    "IVC": 2,
    "PLAX": 22,
    "PSAX": 4,
    "fetal_femur": 2,
}

_DINOV2_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov2_repo")


# ==============================================================================
# BUILDING BLOCKS
# ==============================================================================

def group_norm(num_channels, max_groups=32):
    """
    GroupNorm instead of BatchNorm2d everywhere in this model. The neck
    (TrueHRNetNeck) is shared across all 9 tasks, whose batches are
    task-homogeneous during training (DeterministicBalancedHomogeneousSampler).
    In train() mode BatchNorm normalizes with that batch's own live
    statistics -- effectively per-task-adaptive even though the layer is
    shared -- but in eval() mode it switches to a single running mean/var
    averaged across cardiac, fetal, and vascular batches with very different
    feature statistics. That mismatch between train-time (per-batch, quasi
    per-task) and eval-time (global, domain-blended) normalization was
    causing training MRE to drop fast while validation MRE stayed flat near
    its initial value. GroupNorm computes statistics per-sample (over
    channel groups), identical in train() and eval(), so it's invariant to
    this domain heterogeneity. Picks the largest divisor of num_channels
    that is <= max_groups, so it's safe for any channel width, not just
    multiples of 32.
    """
    g = min(max_groups, num_channels)
    while num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


def conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        group_norm(out_ch),
        nn.ReLU(inplace=True),
    )


class UpsampleUnit(nn.Module):
    """1x1 conv (channel match) + bilinear upsample, the standard HRNet fusion-in path."""
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = group_norm(out_ch)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.up(self.bn(self.proj(x)))


class DownsampleUnit(nn.Module):
    """
    Chain of stride-2 3x3 convs, the standard HRNet fusion-out path. Two stride-2
    convs for a 4x reduction (instead of one stride-4 conv) matches the original
    HRNet paper and gives the downsampling path actual learnable capacity at the
    intermediate scale rather than a single large-stride jump.
    """
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
    """
    Bidirectional exchange between (37x37, coarse) and (74x74, fine) branches.
    Fusing a finer branch into a coarser one requires downsampling; fusing a
    coarser branch into a finer one requires upsampling.
    """
    def __init__(self, w1, w2):
        super().__init__()
        self.down_2to1 = DownsampleUnit(w2, w1, num_stride2_steps=1)   # 74 -> 37
        self.up_1to2 = UpsampleUnit(w1, w2, scale_factor=2)            # 37 -> 74
        self.relu = nn.ReLU(inplace=True)

    def forward(self, b1, b2):
        new_b1 = self.relu(b1 + self.down_2to1(b2))
        new_b2 = self.relu(b2 + self.up_1to2(b1))
        return new_b1, new_b2


class ExchangeUnit3Branch(nn.Module):
    """Full all-pairs exchange between (37x37), (74x74), (148x148) branches."""
    def __init__(self, w1, w2, w3):
        super().__init__()
        self.down_2to1 = DownsampleUnit(w2, w1, num_stride2_steps=1)   # 74  -> 37
        self.down_3to1 = DownsampleUnit(w3, w1, num_stride2_steps=2)   # 148 -> 37
        self.up_1to2 = UpsampleUnit(w1, w2, scale_factor=2)             # 37  -> 74
        self.down_3to2 = DownsampleUnit(w3, w2, num_stride2_steps=1)   # 148 -> 74
        self.up_1to3 = UpsampleUnit(w1, w3, scale_factor=4)             # 37  -> 148
        self.up_2to3 = UpsampleUnit(w2, w3, scale_factor=2)             # 74  -> 148
        self.relu = nn.ReLU(inplace=True)

    def forward(self, b1, b2, b3):
        new_b1 = self.relu(b1 + self.down_2to1(b2) + self.down_3to1(b3))
        new_b2 = self.relu(self.up_1to2(b1) + b2 + self.down_3to2(b3))
        new_b3 = self.relu(self.up_1to3(b1) + self.up_2to3(b2) + b3)
        return new_b1, new_b2, new_b3


# ==============================================================================
# TRUE HRNET NECK
# ==============================================================================

class TrueHRNetNeck(nn.Module):
    """
    Multi-stage HRNet neck with repeated cross-resolution exchange, built on top
    of a frozen DINOv2 37x37 patch grid:

      Stage 1 (init):      b1 @37, b2 @74 spawned from the DINOv2 grid
      Exchange 1:           b1 <-> b2
      Stage 2 (expansion): b1 @37, b2 @74 refined; b3 @148 spawned from b2
      Exchange 2:           b1 <-> b2 <-> b3 (all pairs)
      Final fusion:        1x1 conv + upsample each branch to 148x148, sum

    `branch_width` sets (w1, w2, w3) for the three resolutions; defaults follow
    the original HRNet convention of roughly doubling width as resolution halves
    is deliberately NOT used here since DINOv2 already gives us a very high base
    channel count (1024) -- a flat width keeps parameter count in check.
    """
    def __init__(self, in_channels=1024, out_channels=128,
                 branch_width=(128, 96, 64), dropout_p=0.3):
        super().__init__()
        w1, w2, w3 = branch_width

        # ---- Stage 1: initialization ----
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
        self.stage2_b3 = nn.Sequential(
            nn.ConvTranspose2d(w2, w3, kernel_size=4, stride=2, padding=1, bias=False),
            group_norm(w3),
            nn.ReLU(inplace=True),
        )
        self.exchange2 = ExchangeUnit3Branch(w1, w2, w3)

        # ---- Final multi-scale fusion ----
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

    def forward(self, x):
        # Stage 1
        b1 = self.stage1_b1(x)
        b2 = self.stage1_b2(x)
        b1, b2 = self.exchange1(b1, b2)

        # Stage 2
        b1 = self.stage2_b1(b1)
        b2 = self.stage2_b2(b2)
        b3 = self.stage2_b3(b2)
        b1, b2, b3 = self.exchange2(b1, b2, b3)

        # Final fusion (all branches resampled to b3's 148x148 grid)
        out = self.fuse_b1(b1) + self.fuse_b2(b2) + self.fuse_b3(b3)
        out = self.dropout(out)
        return self.final_layer(out)


# ==============================================================================
# HEAD (unchanged from architecture_v2.py -- same trunk shape/keys)
# ==============================================================================

def _make_spatial_reasoning_trunk(in_channels=128, heatmap_size=128, dropout_p=0.3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
        group_norm(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        group_norm(64),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_p),
        nn.Upsample(size=(heatmap_size, heatmap_size), mode='bilinear', align_corners=False),
    )


class TaskSpecificHead(nn.Module):
    def __init__(self, in_channels=128, num_keypoints=2, heatmap_size=128, dropout_p=0.3):
        super().__init__()
        self.spatial_reasoning = _make_spatial_reasoning_trunk(in_channels, heatmap_size, dropout_p)
        self.heatmap_projection = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x):
        features = self.spatial_reasoning(x)
        return self.heatmap_projection(features)


# ==============================================================================
# FULL MODEL
# ==============================================================================

class UnifiedBiometryModel(nn.Module):
    def __init__(self, freeze_encoder=True, heatmap_size=128, dropout_p=0.3,
                 unfreeze_last_n_blocks=0, neck_branch_width=(128, 96, 64), shared_head=False,
                 backbone_name='dinov2_vitl14'):
        super().__init__()
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.shared_head = shared_head

        print(f"Loading DINOv2 Large backbone ({backbone_name}) from local vendored repo (offline)...")
        self.encoder = torch.hub.load(_DINOV2_REPO_DIR, backbone_name, source="local", pretrained=False)
        self.embed_dim = self.encoder.embed_dim

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            if unfreeze_last_n_blocks > 0:
                for blk in self.encoder.blocks[-unfreeze_last_n_blocks:]:
                    for param in blk.parameters():
                        param.requires_grad = True
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True

        self.shared_upsampler = TrueHRNetNeck(in_channels=self.embed_dim, out_channels=128,
                                               branch_width=neck_branch_width, dropout_p=dropout_p)

        if shared_head:
            self.shared_trunk = _make_spatial_reasoning_trunk(in_channels=128, heatmap_size=heatmap_size,
                                                               dropout_p=dropout_p)
            self.heads = nn.ModuleDict({
                task: nn.Conv2d(64, num_keypoints, kernel_size=1)
                for task, num_keypoints in TASK_KEYPOINTS.items()
            })
        else:
            self.heads = nn.ModuleDict({
                task: TaskSpecificHead(in_channels=128, num_keypoints=num_keypoints,
                                        heatmap_size=heatmap_size, dropout_p=dropout_p)
                for task, num_keypoints in TASK_KEYPOINTS.items()
            })

    def encoder_trainable_parameters(self):
        params = []
        if self.unfreeze_last_n_blocks > 0:
            for blk in self.encoder.blocks[-self.unfreeze_last_n_blocks:]:
                params += list(blk.parameters())
            params += list(self.encoder.norm.parameters())
        return params

    def head_trainable_parameters(self):
        params = list(self.shared_upsampler.parameters()) + list(self.heads.parameters())
        if self.shared_head:
            params += list(self.shared_trunk.parameters())
        return params

    def forward_phase2(self, x, task_id):
        B, C, H, W = x.shape

        features = self.encoder.forward_features(x)['x_norm_patchtokens']

        patch_h, patch_w = H // 14, W // 14
        features = features.permute(0, 2, 1).view(B, self.embed_dim, patch_h, patch_w)

        upsampled_features = self.shared_upsampler(features)

        if isinstance(task_id, list):
            task_id = task_id[0]

        if self.shared_head:
            trunk_features = self.shared_trunk(upsampled_features)
            return self.heads[task_id](trunk_features)
        return self.heads[task_id](upsampled_features)
