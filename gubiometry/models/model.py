"""UnifiedBiometryModel + build_model_from_config.

Attribute names (`encoder`, `shared_upsampler`, `heads`, `shared_trunk`) are kept
identical to the original architecture_hrnet.UnifiedBiometryModel so single-mode
checkpoints load with strict=True. `input_mode`/`feature_layers` are plain
attributes (not modules) and add no state_dict keys.
"""

import torch
import torch.nn as nn

from ..constants import TASK_KEYPOINTS
from .backbone import load_backbone, get_multilevel_features
from .neck import TrueHRNetNeck
from .heads import TaskSpecificHead, make_spatial_reasoning_trunk


class UnifiedBiometryModel(nn.Module):
    def __init__(self, backbone_name="dinov2_vitl14", freeze_encoder=True, heatmap_size=128,
                 dropout_p=0.3, unfreeze_last_n_blocks=0, neck_branch_width=(128, 96, 64),
                 shared_head=False, input_mode="single", feature_layers=(5, 11, 17, 23)):
        super().__init__()
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.shared_head = shared_head
        self.input_mode = input_mode
        self.feature_layers = tuple(feature_layers)

        self.encoder, self.embed_dim = load_backbone(backbone_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            if unfreeze_last_n_blocks > 0:
                for blk in self.encoder.blocks[-unfreeze_last_n_blocks:]:
                    for param in blk.parameters():
                        param.requires_grad = True
                for param in self.encoder.norm.parameters():
                    param.requires_grad = True

        self.shared_upsampler = TrueHRNetNeck(
            in_channels=self.embed_dim, out_channels=128,
            branch_width=neck_branch_width, dropout_p=dropout_p, input_mode=input_mode,
        )

        if shared_head:
            self.shared_trunk = make_spatial_reasoning_trunk(
                in_channels=128, heatmap_size=heatmap_size, dropout_p=dropout_p)
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

    # --- parameter groups ---------------------------------------------------
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

    # --- forward ------------------------------------------------------------
    def _neck_features(self, x):
        if self.input_mode == "multilevel":
            return get_multilevel_features(self.encoder, x, self.feature_layers)
        B, C, H, W = x.shape
        features = self.encoder.forward_features(x)["x_norm_patchtokens"]
        patch_h, patch_w = H // self.encoder.patch_size, W // self.encoder.patch_size
        return features.permute(0, 2, 1).reshape(B, self.embed_dim, patch_h, patch_w)

    def forward_phase1(self, x):
        """Phase-1 SSL: return raw DINOv2 patch tokens (B, N, C)."""
        return self.encoder.forward_features(x)["x_norm_patchtokens"]

    def forward_phase2(self, x, task_id):
        feats = self._neck_features(x)
        upsampled = self.shared_upsampler(feats)
        if isinstance(task_id, (list, tuple)):
            task_id = task_id[0]
        if self.shared_head:
            return self.heads[task_id](self.shared_trunk(upsampled))
        return self.heads[task_id](upsampled)


def build_model_from_config(cfg, freeze_encoder=True):
    """Construct a model from a RunConfig (or its .model sub-config).

    Reused by training (student + teacher), prediction, evaluation and viz, so the
    model is always rebuilt from the exact config it was trained with.
    """
    model_cfg = getattr(cfg, "model", cfg)
    return UnifiedBiometryModel(
        backbone_name=model_cfg.backbone.name,
        freeze_encoder=freeze_encoder,
        heatmap_size=model_cfg.heatmap_size,
        dropout_p=model_cfg.neck.dropout_p,
        unfreeze_last_n_blocks=model_cfg.backbone.unfreeze_last_n_blocks,
        neck_branch_width=tuple(model_cfg.neck.branch_width),
        shared_head=model_cfg.neck.shared_head,
        input_mode=model_cfg.neck.input_mode,
        feature_layers=tuple(model_cfg.neck.feature_layers),
    )
