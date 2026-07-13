"""Challenge scorer -- the metric the leaderboard actually uses.

Two components per task, macro-averaged (each task weighted equally regardless of
sample count -- so the tiny cardiac tasks matter as much as AOP):

  * MRE  : mean per-point radial error, ORIGINAL-image pixels, over visible points.
  * AvgMAE: mean absolute error of the derived clinical measurements
            (distances in pixels, the AOP angle in DEGREES), from the exact
            index pairs in task_measurement_table.csv.

`challenge_score` reproduces the committed local-eval numbers (Average MRE and
Average task AvgMAE) exactly, and additionally returns `challenge_blend`, a
normalized 0.5/0.5 scalar used for checkpoint selection (documented as a proxy --
the official normalizer, clinical-tolerance/IQR, is not published).

Missing points are encoded (-1, -1) and masked, matching the training loss mask
`target[..., 0] >= 0`.
"""

import math
import numpy as np

from .constants import TASK_ORDER


# --------------------------------------------------------------------------- #
# Measurement primitives (operate on original-pixel points, 0-indexed)
# --------------------------------------------------------------------------- #
def _dist(p, i, j):
    return float(np.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]))


def _angle_deg(p, a, b, c):
    """Angle (degrees) between vectors (p[b]-p[a]) and (p[c]-p[a]). Pixel space."""
    v1 = p[b] - p[a]
    v2 = p[c] - p[a]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _pi_half_sum(p, i, j, k, l):
    return math.pi * (_dist(p, i, j) + _dist(p, k, l)) / 2.0


def _min_dist(p, i, j, k, l):
    return min(_dist(p, i, j), _dist(p, k, l))


# name, kind, indices  -- transcribed from task_measurement_table.csv (1-indexed
# there, 0-indexed here). Distances in px, the one angle in degrees.
MEASUREMENT_SPECS = {
    "A4C": [
        ("LV_horizontal", "dist", (0, 1)), ("LV_vertical", "dist", (2, 3)),
        ("RV_horizontal", "dist", (4, 5)), ("RV_vertical", "dist", (6, 7)),
        ("LA_horizontal", "dist", (8, 9)), ("LA_vertical", "dist", (10, 11)),
        ("RA_horizontal", "dist", (12, 13)), ("RA_vertical", "dist", (14, 15)),
    ],
    "AOP": [
        ("HSD", "dist", (0, 2)),
        ("AOP", "angle", (0, 1, 3)),   # angle(p2-p1, p4-p1)
    ],
    "FA": [("FA", "pi_half_sum", (0, 1, 2, 3))],
    "FUGC": [("CL", "dist", (0, 1))],
    "HC": [
        ("BPD", "min_dist", (0, 1, 2, 3)),
        ("HC", "pi_half_sum", (0, 1, 2, 3)),
    ],
    "IVC": [("IVC", "dist", (0, 1))],
    "PLAX": [
        ("LV", "dist", (0, 1)), ("RV", "dist", (2, 3)), ("IVS", "dist", (4, 5)),
        ("LVPW", "dist", (6, 7)), ("VAO", "dist", (8, 9)), ("STJ", "dist", (10, 11)),
        ("AAO", "dist", (12, 13)), ("AV", "dist", (14, 15)), ("LVOT", "dist", (16, 17)),
        ("LA", "dist", (18, 19)), ("RVOT", "dist", (20, 21)),
    ],
    "PSAX": [("RVOT", "dist", (0, 1)), ("PA", "dist", (2, 3))],
    "fetal_femur": [("FL", "dist", (0, 1))],
}

_KIND = {"dist": _dist, "angle": _angle_deg, "pi_half_sum": _pi_half_sum, "min_dist": _min_dist}


def measurements_from_points(task_id, pts):
    """Return {name: value or None}. None if any required index is missing (<0)."""
    pts = np.asarray(pts, dtype=np.float64)
    out = {}
    for name, kind, idx in MEASUREMENT_SPECS.get(task_id, []):
        if any(pts[i, 0] < 0 or pts[i, 1] < 0 for i in idx):
            out[name] = None
        else:
            out[name] = _KIND[kind](pts, *idx)
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _sample_mre(pred, gt):
    """Mean radial error over visible GT points (original px), or None if none visible."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    visible = (gt[:, 0] >= 0) & (gt[:, 1] >= 0)
    if visible.sum() == 0:
        return None
    d = np.hypot(pred[visible, 0] - gt[visible, 0], pred[visible, 1] - gt[visible, 1])
    return float(d.mean())


def challenge_score(entries):
    """Score a list of entries.

    entries: list of dicts with keys
        task_id, pred_px (K,2), gt_px (K,2), height, width
    (pred_px/gt_px in ORIGINAL image pixels; missing GT points are (-1,-1)).

    Returns a dict with per-task MRE / AvgMAE (reproducing the committed local
    eval), macro averages, and the normalized `challenge_blend` selection scalar.
    """
    by_task = {}
    for e in entries:
        by_task.setdefault(e["task_id"], []).append(e)

    per_task_mre = {}
    per_task_measurements = {}   # task -> {name: mae}
    per_task_avg_mae = {}
    mre_norm = {}                # per-task normalizer (median image diagonal)
    mae_norm = {}                # per-task normalizer (median |gt measurement|)

    for task, es in by_task.items():
        # --- MRE ---
        sample_mres = [m for m in (_sample_mre(e["pred_px"], e["gt_px"]) for e in es) if m is not None]
        per_task_mre[task] = float(np.mean(sample_mres)) if sample_mres else float("nan")
        diags = [math.hypot(e["height"], e["width"]) for e in es]
        mre_norm[task] = float(np.median(diags)) if diags else 1.0

        # --- measurement MAE (per measurement name, then mean over names) ---
        names = [n for (n, _, _) in MEASUREMENT_SPECS.get(task, [])]
        abs_err = {n: [] for n in names}
        gt_mag = {n: [] for n in names}
        for e in es:
            gm = measurements_from_points(task, e["gt_px"])
            pm = measurements_from_points(task, e["pred_px"])
            for n in names:
                if gm[n] is not None and pm[n] is not None:
                    abs_err[n].append(abs(pm[n] - gm[n]))
                    gt_mag[n].append(abs(gm[n]))
        meas_mae = {n: (float(np.mean(abs_err[n])) if abs_err[n] else float("nan")) for n in names}
        per_task_measurements[task] = meas_mae
        valid = [v for v in meas_mae.values() if not math.isnan(v)]
        per_task_avg_mae[task] = float(np.mean(valid)) if valid else float("nan")
        all_mag = [m for n in names for m in gt_mag[n]]
        mae_norm[task] = float(np.median(all_mag)) if all_mag else 1.0

    tasks = [t for t in TASK_ORDER if t in by_task]
    average_mre = float(np.nanmean([per_task_mre[t] for t in tasks])) if tasks else float("nan")
    average_avg_mae = float(np.nanmean([per_task_avg_mae[t] for t in tasks])) if tasks else float("nan")

    # Normalized selection blend: divide each task's error by its own scale so no
    # single task/measurement dominates, then macro-average the two halves 50/50.
    mre_terms, mae_terms = [], []
    for t in tasks:
        if mre_norm[t] > 0 and not math.isnan(per_task_mre[t]):
            mre_terms.append(per_task_mre[t] / mre_norm[t])
        if mae_norm[t] > 0 and not math.isnan(per_task_avg_mae[t]):
            mae_terms.append(per_task_avg_mae[t] / mae_norm[t])
    blend = 0.5 * (np.mean(mre_terms) if mre_terms else 0.0) + \
            0.5 * (np.mean(mae_terms) if mae_terms else 0.0)

    return {
        "average_mre": average_mre,
        "average_avg_mae": average_avg_mae,
        "challenge_blend": float(blend),
        "per_task_mre": per_task_mre,
        "per_task_avg_mae": per_task_avg_mae,
        "per_task_measurements": per_task_measurements,
        "num_tasks": len(tasks),
    }


# --------------------------------------------------------------------------- #
# JSON adapters (submission / GT schema <-> entries)
# --------------------------------------------------------------------------- #
def _basename(path):
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def entries_from_json(pred_records, gt_records):
    """Match predicted records to GT by (task_id, basename(image_path))."""
    gt_index = {(r["task_id"], _basename(r["image_path"])): r for r in gt_records}
    entries = []
    for pr in pred_records:
        key = (pr["task_id"], _basename(pr["image_path"]))
        gr = gt_index.get(key)
        if gr is None:
            continue
        pred_px = np.asarray(pr["predicted_points_pixels"], dtype=np.float64).reshape(-1, 2)
        gt_px = np.asarray(gr["ground_truth_points_pixels"], dtype=np.float64).reshape(-1, 2)
        entries.append({
            "task_id": pr["task_id"],
            "pred_px": pred_px,
            "gt_px": gt_px,
            "height": float(gr["height"]),
            "width": float(gr["width"]),
        })
    return entries


def score_json(pred_records, gt_records):
    return challenge_score(entries_from_json(pred_records, gt_records))
