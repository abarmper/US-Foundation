# Data layout

This repository does **not** ship the GU_Biometry dataset itself (images are
too large for git, and redistribution is governed by the challenge's own
terms). This folder only tracks the small, non-heavy artifacts needed to
*reproduce the exact train/val split and evaluation* used in the paper:

```
data/
├── README.md                              <- this file
└── splits/
    ├── train_val_split_keys.json          <- (task_id, filename) keys defining
    │                                          the 80%/20% train/val split
    │                                          (5,414 train / 1,354 val images)
    └── local_eval_gt/
        └── internal_ground_truth_val.json <- ground-truth keypoints for the
                                                 val split, used by the
                                                 visualization/scoring scripts
```

## Expected full layout (after you add the real dataset)

Download the GU_Biometry MICCAI 2026 Challenge dataset separately, then
arrange it under a `data_root` directory (default: this `data/` folder,
override with `--data_root /path/to/data` on any training script, or the
`GU_BIOMETRY_DATA_ROOT` environment variable for the visualization scripts)
so that it looks like:

```
data_root/
├── images/
│   ├── <TASK_ID>/                 <- one of: A4C, AOP, FA, FUGC, HC, IVC,
│   │   ├── labeled/                 PLAX, PSAX, fetal_femur
│   │   │   └── *.png / *.jpg
│   │   └── unlabeled/
│   │       └── *.png / *.jpg
│   └── ...
├── csv/
│   └── *.csv                      <- one or more CSVs with at least the
│                                      columns `task_id`, `image_path`, and
│                                      `point_*` (keypoint coordinate columns,
│                                      x/y pairs per landmark; missing/occluded
│                                      keypoints are encoded as -1, -1)
└── splits/
    ├── train_val_split_keys.json
    └── local_eval_gt/
        └── internal_ground_truth_val.json
```

Notes on how this is consumed (see `src/data/dataset_final.py`):
- `RobustBiometryDataset` recursively scans `images/` and indexes every file
  by `(task_id, filename)`, inferring `task_id` from whichever path component
  matches one of the 9 valid task names — so the exact subfolder nesting
  under `images/<TASK_ID>/` is flexible as long as the task name appears
  somewhere in the path.
- A file is treated as **unlabeled** if the literal string `unlabeled`
  appears anywhere in its path; everything else is looked up against
  `splits/train_val_split_keys.json` to decide train vs. val membership.
- CSVs with `pseudo` in the filename are ignored (reserved for a
  semi-supervised pseudo-labeling extension, not part of the core pipeline).

## Regenerating the split

`src/data/make_splits.py` regenerates `train_val_split_keys.json` from the
raw CSV annotations (80%/20% train/val, stratified per task). You should
not need to re-run this unless you want a different split — the shipped
`train_val_split_keys.json` is the exact split used for every result in the
paper.
