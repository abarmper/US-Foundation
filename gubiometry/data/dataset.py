"""RobustBiometryDataset -- labeled / unlabeled ultrasound frames.

Changes vs the original dataset_final.RobustBiometryDataset:
  * fold-aware: accepts an explicit `split_file` or a `fold` index
    (splits/<kfold_dir>/fold_{fold}.json) instead of the hardcoded holdout file.
  * returns original-space GT (`gt_points_orig`, `orig_h`, `orig_w`, `image_path`)
    so the validation loop can inverse-letterbox predictions and score with the
    real challenge metric.
  * image reading falls back to PIL when cv2 is unavailable.
  * the unused 128x128 Gaussian heatmaps are no longer generated (soft-argmax uses
    coordinates only).
"""

import os
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..constants import VALID_TASKS

# Cache of {resolved data_root -> {(task_id, filename): abs_path}} so repeated dataset
# constructions (train/val/folds) don't each rglob the whole images/ tree.
_PATH_INDEX_CACHE = {}


def read_rgb(path):
    """Read an image as RGB uint8. Prefers cv2, falls back to PIL."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class RobustBiometryDataset(Dataset):
    def __init__(self, data_root, mode="train_labeled", transforms=None,
                 split_file="splits/train_val_split_keys.json", fold=None,
                 kfold_dir="splits/kfold_v1"):
        self.data_root = Path(data_root)
        self.mode = mode
        self.transforms = transforms
        self.valid_tasks = set(VALID_TASKS)

        if fold is not None:
            self.split_path = self.data_root / kfold_dir / f"fold_{fold}.json"
        else:
            self.split_path = self.data_root / split_file

        self.path_index = self._build_path_index()

        if self.mode in ("train_labeled", "val"):
            self.samples = self._load_labeled_data()
        elif self.mode == "unlabeled":
            self.samples = self._load_unlabeled_data()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    # ------------------------------------------------------------------ #
    def _build_path_index(self):
        # Cache per data_root: train + val + every fold share one rglob of the
        # (large) images/ tree instead of re-walking it each construction.
        key = str(self.data_root.resolve())
        cached = _PATH_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        index = {}
        images_dir = self.data_root / "images"
        for file_path in images_dir.rglob("*"):
            if file_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                if "_vis" in file_path.stem.lower():
                    continue
                task_id = next((p for p in file_path.parts if p in self.valid_tasks), None)
                if task_id:
                    index[(task_id, file_path.name)] = str(file_path)
        _PATH_INDEX_CACHE[key] = index
        return index

    def _load_labeled_data(self):
        if not self.split_path.exists():
            raise FileNotFoundError(
                f"Split JSON not found: {self.split_path}. Run `gubiometry make-splits` first.")
        with open(self.split_path) as f:
            split_data = json.load(f)
        dict_key = "train" if "train" in self.mode else "val"
        target_keys = set(tuple(k) for k in split_data[dict_key])

        csv_dir = self.data_root / "csv"
        dfs = []
        for f in csv_dir.glob("*.csv"):
            if "pseudo" in str(f).lower():
                continue
            try:
                dfs.append(pd.read_csv(f, encoding="utf-8"))
            except UnicodeDecodeError:
                dfs.append(pd.read_csv(f, encoding="gbk"))
        df_all = pd.concat(dfs, ignore_index=True)
        self.point_cols = [c for c in df_all.columns if c.startswith("point_")]

        valid = []
        for _, row in df_all.iterrows():
            task_id = row["task_id"]
            filename = Path(row["image_path"]).name
            if (task_id, filename) in target_keys:
                true_path = self.path_index.get((task_id, filename))
                if true_path:
                    valid.append({"task_id": task_id, "abs_path": true_path,
                                  "image_path": row["image_path"], "row_data": row})
        print(f"[{self.mode}] Loaded {len(valid)} valid labeled samples from {self.split_path.name}.")
        return valid

    def _load_unlabeled_data(self):
        valid = []
        for (task_id, _fname), true_path in self.path_index.items():
            if "unlabeled" in Path(true_path).parts:
                valid.append({"task_id": task_id, "abs_path": true_path})
        print(f"[{self.mode}] Loaded {len(valid)} valid unlabeled samples.")
        return valid

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def _parse_keypoints(self, row):
        kps = []
        for col in self.point_cols:
            if pd.notna(row[col]) and isinstance(row[col], str):
                try:
                    pt = ast.literal_eval(row[col])
                    kps.append((float(pt[0]), float(pt[1])))
                except Exception:
                    pass
        num_classes = int(row["num_classes"])
        while len(kps) < num_classes:
            kps.append((-1.0, -1.0))
        return kps

    def __getitem__(self, idx):
        sample = self.samples[idx]
        task_id = sample["task_id"]
        image = read_rgb(sample["abs_path"])
        if image is None:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        if self.mode == "unlabeled":
            if self.transforms:
                image = self.transforms(image=image, keypoints=[(-1.0, -1.0)])["image"]
            return {"image": image, "task_id": task_id, "abs_path": sample["abs_path"]}

        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        keypoints_orig = self._parse_keypoints(sample["row_data"])

        if self.transforms:
            out = self.transforms(image=image, keypoints=keypoints_orig)
            image = out["image"]
            keypoints_518 = out["keypoints"]
        else:
            keypoints_518 = keypoints_orig

        return {
            "image": image,
            "keypoints": torch.tensor(keypoints_518, dtype=torch.float32),
            "keypoints_orig": torch.tensor(keypoints_orig, dtype=torch.float32),
            "orig_h": orig_h,
            "orig_w": orig_w,
            "task_id": task_id,
            "image_path": sample["image_path"],
        }
