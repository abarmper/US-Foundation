"""K-fold driver: train one Phase-2 model per fold, reusing engine.phase2.train.

The resulting per-fold run dirs are the ensemble members for engine.predict.
"""

import os
import glob
import copy

from . import phase2
from .common import create_logger, runs_dir


def _count_folds(cfg):
    d = os.path.join(cfg.data.data_root, cfg.data.kfold_dir)
    return len(glob.glob(os.path.join(d, "fold_*.json")))


def run(cfg):
    n = _count_folds(cfg)
    if n == 0:
        raise FileNotFoundError(
            f"No fold files in {os.path.join(cfg.data.data_root, cfg.data.kfold_dir)}. "
            f"Run `gubiometry make-splits --kfold` first.")

    logger = create_logger(os.path.join(runs_dir(), cfg.run_name), "kfold")
    logger.info(f"K-fold training: {n} folds, base run_name={cfg.run_name}")

    results = []
    for fold in range(n):
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.data.fold = fold
        fold_cfg.run_name = f"{cfg.run_name}_fold{fold}"
        logger.info(f"--- fold {fold}/{n - 1} -> run {fold_cfg.run_name} ---")
        best = phase2.train(fold_cfg)
        results.append({"run_name": fold_cfg.run_name, "fold": fold, "best_metric": best,
                        "run_dir": os.path.join(runs_dir(), fold_cfg.run_name)})
    logger.info("K-fold done. Member run dirs:")
    for r in results:
        logger.info(f"  fold {r['fold']}: {r['run_dir']}  (best={r['best_metric']:.4f})")
    return results
