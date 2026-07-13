"""Normalization helpers, the shared spatial-reasoning trunk, and TaskSpecificHead.

Ported verbatim (behavior + submodule names) from the original
architecture_hrnet.py so existing Phase-2 checkpoints keep loading.
"""

import torch.nn as nn


def group_norm(num_channels, max_groups=32):
    """GroupNorm (not BatchNorm) everywhere: identical stats in train()/eval(), which
    matters because batches are task-homogeneous and cardiac/fetal/vascular feature
    statistics differ wildly. Picks the largest divisor of num_channels <= max_groups.
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


def make_spatial_reasoning_trunk(in_channels=128, heatmap_size=128, dropout_p=0.3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
        group_norm(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        group_norm(64),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_p),
        nn.Upsample(size=(heatmap_size, heatmap_size), mode="bilinear", align_corners=False),
    )


class TaskSpecificHead(nn.Module):
    def __init__(self, in_channels=128, num_keypoints=2, heatmap_size=128, dropout_p=0.3):
        super().__init__()
        self.spatial_reasoning = make_spatial_reasoning_trunk(in_channels, heatmap_size, dropout_p)
        self.heatmap_projection = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x):
        features = self.spatial_reasoning(x)
        return self.heatmap_projection(features)
