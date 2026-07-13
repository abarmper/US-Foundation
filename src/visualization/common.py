"""Shared utilities for the paper visualization scripts.

Rewired onto the `gubiometry` package: coordinate geometry, the task list, image
reading and the (albumentations-free) inference letterbox now come from the
canonical modules, and models are rebuilt from each run's config.json via
`load_model_from_run_dir` -- no more hand-matching --neck_branch_width /
--unfreeze_last_n_blocks under strict=True.
"""

import os
import sys
import json
import numpy as np

# Make `import gubiometry` work when running these scripts directly from src/visualization.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gubiometry.constants import VALID_TASKS, TASK_ORDER          # noqa: E402
from gubiometry.geometry import remove_padding_and_scale, soft_argmax_coords  # noqa: E402
from gubiometry.data.dataset import read_rgb                      # noqa: E402
from gubiometry.data.transforms import letterbox_to_tensor        # noqa: E402

_DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "data")
DATA_ROOT = os.environ.get("GU_BIOMETRY_DATA_ROOT", _DEFAULT_DATA_ROOT)
GT_JSON_PATH = os.path.join(DATA_ROOT, "splits", "local_eval_gt", "internal_ground_truth_val.json")
IMAGES_ROOT = os.path.join(DATA_ROOT, "images")


class _InferenceLetterbox:
    """albumentations-style callable: transform(image=rgb)['image'] -> CHW tensor."""

    def __init__(self, canvas=518):
        self.canvas = canvas

    def __call__(self, image):
        return {"image": letterbox_to_tensor(image, self.canvas)}


INFERENCE_TRANSFORM = _InferenceLetterbox(518)


def build_path_index(images_root=IMAGES_ROOT):
    from pathlib import Path
    index = {}
    for file_path in Path(images_root).rglob("*"):
        if file_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            if "_vis" in file_path.stem.lower():
                continue
            task_id = next((p for p in file_path.parts if p in VALID_TASKS), None)
            if task_id:
                index[(task_id, file_path.name)] = str(file_path)
    return index


def select_one_val_sample_per_task(seed=42, gt_json_path=GT_JSON_PATH):
    with open(gt_json_path) as f:
        gt_records = json.load(f)
    by_task = {}
    for rec in gt_records:
        by_task.setdefault(rec["task_id"], []).append(rec)
    rng = np.random.default_rng(seed)
    selected = {}
    for task_id in TASK_ORDER:
        candidates = by_task.get(task_id, [])
        if candidates:
            selected[task_id] = candidates[rng.integers(0, len(candidates))]
    return selected


def load_image_rgb(abs_path):
    img = read_rgb(abs_path)
    if img is None:
        raise FileNotFoundError(abs_path)
    return img


def get_soft_argmax_coords(logits, temperature=10.0, canvas=518):
    """Return (pred_x, pred_y) in canvas px -- thin wrapper over the canonical decoder."""
    coords = soft_argmax_coords(logits, temperature, canvas)
    return coords[..., 0], coords[..., 1]


def load_model_from_run_dir(run_dir, checkpoint_name="best_teacher_model.pth", device="cpu"):
    """Rebuild the exact trained model from runs/<run>/config.json and load its checkpoint."""
    import torch
    from gubiometry.config import config_from_run_dir
    from gubiometry.models import build_model_from_config
    cfg = config_from_run_dir(run_dir)
    model = build_model_from_config(cfg).to(device).eval()
    sd = torch.load(os.path.join(run_dir, "checkpoints", checkpoint_name), map_location=device)
    model.load_state_dict(sd, strict=True)
    return model, cfg
