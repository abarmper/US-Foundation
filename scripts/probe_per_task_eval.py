"""Re-score already-trained frozen-probe checkpoints with a per-task breakdown.

Pure inference, no retraining: rebuilds each probe's model from its own
runs/<run>/config.json, loads the already-trained
runs/<run>/checkpoints/best_teacher_model.pth (the EMA teacher, a bare
state_dict), and re-runs the exact validate() used during training -- one
forward pass over the fold-0 val split, no gradients. This recovers the
per-task MRE / AvgMAE breakdown that phase2.validate() computes every epoch
but only ever logs to TensorBoard (never persisted to disk), without
retraining anything.

Usage:
  python scripts/probe_per_task_eval.py --runs probe_nossl probe_dv2_ep10 \
      probe_dv2_ep20 probe_dv2_ep60 probe_dv2_ep100 probe_dv2_ep104 \
      --out docs/MWM_54/per_task_probe_results.md
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from gubiometry.config import config_from_run_dir
from gubiometry.constants import TASK_ORDER
from gubiometry.data.dataset import RobustBiometryDataset
from gubiometry.data.samplers import HomogeneousTaskSampler
from gubiometry.data.transforms import LetterboxTransform
from gubiometry.engine.common import get_device, resolve_amp, runs_dir
from gubiometry.engine.phase2 import validate
from gubiometry.models.model import build_model_from_config


def eval_run(run_name, device):
    run_dir = os.path.join(runs_dir(), run_name)
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_teacher_model.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found -- was {run_name} completed?")

    cfg = config_from_run_dir(run_dir)
    ds_val = RobustBiometryDataset(cfg.data.data_root, mode="val",
                                    transforms=LetterboxTransform(cfg.data.canvas),
                                    split_file=cfg.data.split_file, fold=cfg.data.fold,
                                    kfold_dir=cfg.data.kfold_dir)
    val_sampler = HomogeneousTaskSampler(ds_val, cfg.data.batch_size)
    loader_val = DataLoader(ds_val, batch_sampler=val_sampler,
                             num_workers=cfg.data.num_workers, pin_memory=True)

    model = build_model_from_config(cfg).to(device).eval()
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)

    amp_on, amp_dtype, _ = resolve_amp(cfg.optim.amp_dtype)
    score = validate(model, loader_val, device, cfg, amp_on, amp_dtype)

    out_path = os.path.join(run_dir, "per_task_eval.json")
    with open(out_path, "w") as f:
        json.dump(score, f, indent=2)
    print(f"{run_name}: blend={score['challenge_blend']:.4f} "
          f"MRE={score['average_mre']:.2f} -> wrote {out_path}")
    return score


def _table(scores, run_names, key, fmt):
    lines = ["| Task | " + " | ".join(run_names) + " |",
             "|---|" + "---|" * len(run_names)]
    for t in TASK_ORDER:
        row = [fmt(scores[r][key].get(t)) for r in run_names]
        lines.append(f"| {t} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                     help="run_name(s) under runs/, e.g. probe_nossl probe_dv2_ep104")
    ap.add_argument("--out", default=None, help="optional combined markdown table path")
    args = ap.parse_args()

    device = get_device()
    scores = {r: eval_run(r, device) for r in args.runs}
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"

    table_md = (
        "Per-task MRE (orig px, fold-0 internal validation):\n\n"
        + _table(scores, args.runs, "per_task_mre", fmt)
        + "\n\nPer-task AvgMAE (clinical measurement error):\n\n"
        + _table(scores, args.runs, "per_task_avg_mae", fmt)
    )
    print("\n" + table_md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(table_md + "\n")
        print(f"\nwrote combined table -> {args.out}")


if __name__ == "__main__":
    main()
