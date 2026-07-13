"""Build a throwaway data_root (tiny PNGs + CSV rows + split JSON) so the whole
pipeline runs on CPU with no real dataset and no cv2 (PIL only). Combined with the
"dummy" backbone this exercises every code path end to end.
"""

import os
import csv
import random
from pathlib import Path

from ..constants import TASK_KEYPOINTS


_SIZES = [(540, 800), (480, 640), (661, 959), (600, 600)]  # (h, w)


def build_synthetic_dataset(root, n_labeled=6, n_unlabeled=2, seed=0):
    """Create images/<task>/{labeled,unlabeled}/*.png + csv/<task>.csv, then a
    holdout split. Returns the root path (str)."""
    from PIL import Image

    rng = random.Random(seed)
    root = Path(root)
    (root / "csv").mkdir(parents=True, exist_ok=True)

    for task, K in TASK_KEYPOINTS.items():
        lab_dir = root / "images" / task / "labeled"
        unl_dir = root / "images" / task / "unlabeled"
        lab_dir.mkdir(parents=True, exist_ok=True)
        unl_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(n_labeled):
            h, w = rng.choice(_SIZES)
            fname = f"{task}_{i:03d}.png"
            Image.new("RGB", (w, h), (rng.randint(0, 255),) * 3).save(lab_dir / fname)
            pts = [(rng.uniform(5, w - 5), rng.uniform(5, h - 5)) for _ in range(K)]
            row = {"image_path": f"{task}/{fname}", "task_id": task, "task_name": task,
                   "num_classes": K, "height": h, "width": w}
            for k, (x, y) in enumerate(pts, start=1):
                row[f"point_{k}_xy"] = f"[{x:.2f}, {y:.2f}]"
            rows.append(row)

        for i in range(n_unlabeled):
            h, w = rng.choice(_SIZES)
            Image.new("RGB", (w, h), (rng.randint(0, 255),) * 3).save(
                unl_dir / f"{task}_u{i:03d}.png")

        # write per-task CSV
        cols = ["image_path", "task_id", "task_name", "num_classes", "height", "width"] + \
               [f"point_{k}_xy" for k in range(1, K + 1)]
        with open(root / "csv" / f"{task}.csv", "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=cols)
            wtr.writeheader()
            wtr.writerows(rows)

    from ..data.splits import make_holdout_split
    make_holdout_split(str(root))
    return str(root)
