"""Split-file generation: a single 80/20 holdout and stratified k-fold.

Both write `{"train": [[task_id, filename], ...], "val": [...]}` -- the identical
schema the dataset consumes, so a fold file is a drop-in replacement for the
holdout file. Pseudo-label CSVs are excluded to match the dataset loader.
"""

import os
import glob
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def _load_csv_df(data_root):
    dfs = []
    for f in glob.glob(os.path.join(data_root, "csv", "*.csv")):
        if "pseudo" in os.path.basename(f).lower():
            continue
        try:
            dfs.append(pd.read_csv(f, encoding="utf-8"))
        except UnicodeDecodeError:
            dfs.append(pd.read_csv(f, encoding="gbk"))
    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {os.path.join(data_root, 'csv')}")
    return pd.concat(dfs, ignore_index=True)


def _keys(df):
    return [[row["task_id"], Path(row["image_path"]).name] for _, row in df.iterrows()]


def make_holdout_split(data_root, val_size=0.2, seed=42, out="splits/train_val_split_keys.json"):
    df = _load_csv_df(data_root)
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=df["task_id"])
    split = {"train": _keys(train_df), "val": _keys(val_df)}
    out_path = os.path.join(data_root, out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)
    return out_path, {"train": len(train_df), "val": len(val_df)}


def make_kfold_splits(data_root, n_splits=5, seed=42, out_dir="splits/kfold_v1"):
    df = _load_csv_df(data_root)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    d = os.path.join(data_root, out_dir)
    os.makedirs(d, exist_ok=True)
    paths = []
    for fold, (tr, va) in enumerate(skf.split(df, df["task_id"])):
        split = {"train": _keys(df.iloc[tr]), "val": _keys(df.iloc[va])}
        p = os.path.join(d, f"fold_{fold}.json")
        with open(p, "w") as f:
            json.dump(split, f, indent=2)
        paths.append(p)
    return paths
