"""Ensemble x TTA inference -> submission JSON.

Members are rebuilt from each run dir's own config.json, so folds trained with
different backbones / neck input_modes can be mixed. Predictions are averaged in
COORDINATE space (original pixels) across members x TTA views -- robust and
alignment-free. `average_space="heatmap"` averages logits across members per view
first (valid because all members see the same view).

Safe TTA only: multi-scale + optional intensity. NO naive flips (keypoints are
semantic); each view carries its own canvas so soft-argmax/inverse use the right
scale.
"""

import os
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..constants import TASK_KEYPOINTS, VALID_TASKS
from ..config import config_from_run_dir
from ..geometry import soft_argmax_coords, inverse_letterbox_kps
from ..models.model import build_model_from_config
from ..data.dataset import read_rgb
from ..data.transforms import make_tta_views
from .common import get_device

# Codabench requires this exact filename for the JSON inside the submission zip.
SUBMISSION_JSON_NAME = "regression_predictions.json"


def _build_path_index(data_root):
    index = {}
    for fp in (Path(data_root) / "images").rglob("*"):
        if fp.suffix.lower() in (".png", ".jpg", ".jpeg") and "_vis" not in fp.stem.lower():
            task = next((p for p in fp.parts if p in VALID_TASKS), None)
            if task:
                index[(task, fp.name)] = str(fp)
    return index


def _targets_from_gt(data_root, gt_json):
    recs = json.load(open(gt_json))
    idx = _build_path_index(data_root)
    out = []
    for r in recs:
        ap = idx.get((r["task_id"], os.path.basename(r["image_path"])))
        if ap:
            out.append({"task_id": r["task_id"], "image_path": r["image_path"], "abs_path": ap})
    return out


def _targets_for(data_root, split_file, fold, kfold_dir):
    from ..data.dataset import RobustBiometryDataset
    ds = RobustBiometryDataset(data_root, mode="val", transforms=None,
                               split_file=split_file, fold=fold, kfold_dir=kfold_dir)
    return [{"task_id": s["task_id"], "image_path": s["image_path"], "abs_path": s["abs_path"]}
            for s in ds.samples]


def _targets_from_split(cfg):
    return _targets_for(cfg.data.data_root, cfg.data.split_file, cfg.data.fold, cfg.data.kfold_dir)


def _targets_from_image_dir(image_dir):
    """Enumerate every image in a `<task>/<file>` folder tree (the challenge val/test
    layout). task_id is inferred from the task-named path component; image_path is the
    path relative to image_dir (e.g. "PSAX/0001.png"), matching the committed format."""
    root = Path(image_dir)
    targets = []
    for fp in sorted(root.rglob("*")):
        if fp.suffix.lower() in (".png", ".jpg", ".jpeg") and "_vis" not in fp.stem.lower():
            task = next((p for p in fp.parts if p in VALID_TASKS), None)
            if task:
                targets.append({"task_id": task, "image_path": str(fp.relative_to(root)),
                                "abs_path": str(fp)})
    return targets


def _predict_target(members, tgt, cfg, canvases, device):
    img = read_rgb(tgt["abs_path"])
    h, w = img.shape[:2]
    pred_px = _predict_one(members, tgt["task_id"], img, cfg.predict.average_space,
                           canvases, cfg.predict.tta_intensity, cfg.predict.reduce)
    norm = pred_px / np.array([w, h], dtype=np.float64)
    return {
        "image_path": tgt["image_path"],
        "task_id": tgt["task_id"],
        "predicted_points_normalized": norm.flatten().tolist(),
        "predicted_points_pixels": pred_px.flatten().tolist(),
    }


def _load_members(member_run_dirs, checkpoint_name, device):
    members = []
    for run_dir in member_run_dirs:
        mcfg = config_from_run_dir(run_dir)
        model = build_model_from_config(mcfg).to(device).eval()
        sd = torch.load(os.path.join(run_dir, "checkpoints", checkpoint_name), map_location=device)
        model.load_state_dict(sd, strict=True)
        members.append((model, mcfg))
    return members


def _reduce(coord_list, reduce):
    """Combine a list of (K,2) original-px predictions -> (K,2)."""
    stack = np.stack(coord_list, axis=0)             # (N, K, 2)
    return np.median(stack, axis=0) if reduce == "median" else np.mean(stack, axis=0)


@torch.no_grad()
def _predict_one(members, task, img, average_space, canvases, intensity, reduce="mean"):
    h, w = img.shape[:2]
    views = make_tta_views(img, canvases, intensity)
    device = next(members[0][0].parameters()).device
    coord_list = []

    if average_space == "heatmap":
        # average logits across members per view, then one soft-argmax per view
        temp = members[0][1].optim.softargmax_temp
        for v in views:
            t = v["tensor"].unsqueeze(0).to(device)
            logit_sum = None
            for model, _mcfg in members:
                lg = model.forward_phase2(t, task).float()
                logit_sum = lg if logit_sum is None else logit_sum + lg
            lg = logit_sum / len(members)
            coords = soft_argmax_coords(lg, temp, v["canvas"]).cpu().numpy()[0]
            coord_list.append(inverse_letterbox_kps(coords, h, w, v["canvas"]))
        return _reduce(coord_list, reduce)

    # coordinate-space: one prediction per (member, view)
    for model, mcfg in members:
        temp = mcfg.optim.softargmax_temp
        for v in views:
            t = v["tensor"].unsqueeze(0).to(device)
            logits = model.forward_phase2(t, task).float()
            coords = soft_argmax_coords(logits, temp, v["canvas"]).cpu().numpy()[0]
            coord_list.append(inverse_letterbox_kps(coords, h, w, v["canvas"]))
    return _reduce(coord_list, reduce)


def run(cfg, device=None):
    device = device or get_device()
    members = _load_members(cfg.predict.member_run_dirs, cfg.predict.checkpoint_name, device)
    canvases = tuple(cfg.predict.tta_canvases)
    records = []

    n_views = len(canvases) * (3 if cfg.predict.tta_intensity else 1)
    if cfg.predict.oof:
        # Out-of-fold: each member predicts ONLY its own held-out val fold, so every
        # training image is scored by the model that did not train on it -> an honest
        # CV estimate (does NOT apply to the hidden test set, where you use all members).
        for model, mcfg in members:
            if mcfg.data.fold is None:
                raise ValueError("predict.oof requires each member to have a fold in its config.json.")
            for tgt in tqdm(_targets_for(cfg.data.data_root, mcfg.data.split_file,
                                         mcfg.data.fold, mcfg.data.kfold_dir),
                            desc=f"oof fold{mcfg.data.fold}"):
                records.append(_predict_target([(model, mcfg)], tgt, cfg, canvases, device))
    else:
        if cfg.predict.image_dir:
            targets = _targets_from_image_dir(cfg.predict.image_dir)
        elif cfg.predict.gt_json:
            targets = _targets_from_gt(cfg.data.data_root, cfg.predict.gt_json)
        else:
            targets = _targets_from_split(cfg)
        print(f"predicting {len(targets)} images with {len(members)} member(s) x {n_views} TTA view(s) "
              f"[reduce={cfg.predict.reduce}]")
        for tgt in tqdm(targets, desc="predict"):
            records.append(_predict_target(members, tgt, cfg, canvases, device))

    with open(cfg.predict.out_json, "w") as f:
        json.dump(records, f, indent=2)
    if cfg.predict.zip_output:
        # Codabench expects the JSON inside the zip to be named regression_predictions.json,
        # regardless of the on-disk file name.
        zip_path = os.path.splitext(cfg.predict.out_json)[0] + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(cfg.predict.out_json, arcname=SUBMISSION_JSON_NAME)
        print(f"wrote {zip_path} (contains {SUBMISSION_JSON_NAME}, {len(records)} records)")
    return records
