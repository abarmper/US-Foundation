"""Score a submission JSON against a ground-truth JSON with the challenge metric.

Either scores an existing submission (cfg.predict.out_json) or runs prediction
first, then evaluates against cfg.predict.gt_json.
"""

import json

from ..metrics import score_json
from . import predict as predict_engine


def evaluate_submission(pred_records, gt_json):
    gt = json.load(open(gt_json))
    return score_json(pred_records, gt)


def run(cfg, device=None):
    if not cfg.predict.gt_json:
        raise ValueError("evaluate requires predict.gt_json (ground-truth JSON path).")
    import os
    if cfg.predict.member_run_dirs:
        records = predict_engine.run(cfg, device=device)
    elif os.path.isfile(cfg.predict.out_json):
        records = json.load(open(cfg.predict.out_json))
    else:
        raise ValueError("Provide predict.member_run_dirs (to predict) or an existing predict.out_json.")

    score = evaluate_submission(records, cfg.predict.gt_json)
    print(f"Average MRE (orig px):     {score['average_mre']:.4f}")
    print(f"Average task AvgMAE:       {score['average_avg_mae']:.4f}")
    print(f"challenge_blend (proxy):   {score['challenge_blend']:.4f}")
    print("Per-task MRE:", {k: round(v, 2) for k, v in score["per_task_mre"].items()})
    return score
