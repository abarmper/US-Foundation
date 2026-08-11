"""FU Biometry submission: class Model, as required by predict.py (organizer entry script).

Inference logic is a direct port of the already-verified, already-scored
(Codabench: 28.98) generate_submission_dv2ep104_hrnet_reg.py -- same
transforms, same soft-argmax decode, same letterbox-inverse math (including
the round()-not-int() fix for the systematic ~1px drift documented in that
script). The only things that differ here are (a) the offline-safe backbone
loading in model_factory.py, and (b) reading images/task ids from the
organizer's test_metadata.csv instead of walking a local directory tree.
"""
import os
import csv
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model_factory import UnifiedBiometryModel

CANVAS = 518

TRANSFORM = A.Compose([
    A.LongestMaxSize(max_size=CANVAS),
    A.PadIfNeeded(min_height=CANVAS, min_width=CANVAS, border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def _get_soft_argmax_coords(logits):
    """logits: (1, K, H, W) -> (pred_x, pred_y), each (K,) in CANVAS-pixel space."""
    B, K, H, W = logits.shape
    heatmaps = F.softmax(logits.view(B, K, -1) * 10.0, dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=logits.device),
        torch.arange(W, device=logits.device),
        indexing="ij",
    )
    grid_x = grid_x.flatten().float()
    grid_y = grid_y.flatten().float()
    pred_x = torch.sum(heatmaps * grid_x, dim=-1) * (CANVAS / W)
    pred_y = torch.sum(heatmaps * grid_y, dim=-1) * (CANVAS / H)
    return pred_x[0], pred_y[0]


def _remove_padding_and_scale(x_padded, y_padded, orig_h, orig_w):
    """Invert LongestMaxSize+PadIfNeeded back to original-image pixel space.

    round(), not int(): Albumentations' LongestMaxSize rounds-to-nearest when
    computing the resized target dims. Truncating here mismatches its actual
    output in ~49% of aspect ratios, producing a systematic 1px (518-canvas)
    coordinate drift for roughly half of all images.
    """
    scale = CANVAS / max(orig_h, orig_w)
    new_h, new_w = round(orig_h * scale), round(orig_w * scale)
    pad_top = (CANVAS - new_h) // 2
    pad_left = (CANVAS - new_w) // 2

    x_orig = (x_padded - pad_left) / scale
    y_orig = (y_padded - pad_top) / scale

    return (
        float(np.clip(x_orig, 0.0, float(orig_w))),
        float(np.clip(y_orig, 0.0, float(orig_h))),
    )


class Model:
    def __init__(self):
        """Load model weights once, when the container starts."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[model.py] Using device: {self.device}", flush=True)

        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
        print(f"[model.py] Loading checkpoint: {weights_path}", flush=True)
        state_dict = torch.load(weights_path, map_location=self.device)

        self.model = UnifiedBiometryModel(
            freeze_encoder=True,
            unfreeze_last_n_blocks=4,
            neck_branch_width=(128, 96, 64),
            backbone_name="dinov2_vitl14_reg",
        ).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print("[model.py] Model ready.", flush=True)

    @torch.no_grad()
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        """Read test_metadata.csv, run inference image-by-image, write regression_predictions.json.

        batch_size is accepted per the interface contract but unused: this
        model routes every forward pass through exactly one per-task head
        (see UnifiedBiometryModel.forward_phase2), so batching would require
        grouping by task_id first for no meaningful speed gain here -- the
        already-verified reference script processes one image at a time and
        comfortably finishes well inside the timeout, so we keep that exact,
        already-proven-correct behavior rather than risk a batching bug.
        """
        csv_path = os.path.join(data_root, "csv", "test_metadata.csv")
        images_root = os.path.join(data_root, "images")

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"[model.py] {len(rows)} rows in test_metadata.csv", flush=True)

        predictions = []
        for i, row in enumerate(rows):
            image_path = row["image_path"]
            task_id = row["task_id"]
            abs_path = os.path.join(images_root, image_path)

            img = cv2.imread(abs_path)
            if img is None:
                print(f"[model.py] WARNING: could not read {abs_path}, skipping", flush=True)
                continue

            orig_h, orig_w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_img = TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(self.device)

            logits = self.model.forward_phase2(tensor_img, task_id)
            pred_x_canvas, pred_y_canvas = _get_soft_argmax_coords(logits)

            num_points = logits.shape[1]
            predicted_points_pixels = []
            for k in range(num_points):
                px, py = _remove_padding_and_scale(
                    pred_x_canvas[k].item(), pred_y_canvas[k].item(), orig_h, orig_w
                )
                predicted_points_pixels.extend([px, py])

            predictions.append({
                "image_path": image_path,
                "task_id": task_id,
                "predicted_points_pixels": predicted_points_pixels,
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(rows):
                print(f"[model.py] {i + 1}/{len(rows)} done", flush=True)

        out_path = os.path.join(output_dir, "regression_predictions.json")
        with open(out_path, "w") as f:
            json.dump(predictions, f)
        print(f"[model.py] Saved {len(predictions)} predictions to {out_path}", flush=True)
