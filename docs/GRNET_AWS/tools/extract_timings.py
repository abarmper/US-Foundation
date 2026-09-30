"""Measure per-phase wall-clock on our A100-SXM4-80GB node from the training logs.

Reads the timestamped epoch lines that gubiometry writes to runs/<run>/phase{1,2}_*.log
and the tqdm totals in runs/_launch_predict_*.out, and writes
docs/GRNET_AWS/data/measured_timings.json -- the measured half of the budget model
(budget.py applies the slack on top). Run from the repository root:

    python docs/GRNET_AWS/tools/extract_timings.py
"""
import datetime as dt
import glob
import json
import os
import re
import statistics as st

RUNS = "runs"
OUT = "docs/GRNET_AWS/data/measured_timings.json"
TS = r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[INFO\] "
P1_EPOCH = re.compile(TS + r"\[dinov2 (bulk|tail)@(\d+)\] Epoch (\d+)/(\d+)")
P2_EPOCH = re.compile(TS + r"Epoch (\d+)/(\d+)")
START = re.compile(TS + r"AMP")
t = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def phase1_epochs(run):
    """[(resolution, hours)] for every Phase-1 epoch, measured between consecutive log
    lines of the same process (the first epoch of a process is timed from its AMP line)."""
    out = []
    for log in sorted(glob.glob(f"{RUNS}/{run}/phase1_2*.log")):
        prev = None
        for line in open(log, errors="ignore"):
            if m := START.match(line):
                prev = t(m.group(1))
            elif (m := P1_EPOCH.match(line)) and prev is not None:
                now = t(m.group(1))
                out.append((int(m.group(3)), (now - prev).total_seconds() / 3600))
                prev = now
    return out


def phase2_run(run):
    """(epochs trained, median minutes/epoch) of one FINISHED Phase-2 run (median =
    uncontended); None for runs that were aborted before 'Phase 2 finished'."""
    logs = sorted(glob.glob(f"{RUNS}/{run}/phase2_2*.log"))
    if not any("Phase 2 finished" in open(log, errors="ignore").read() for log in logs):
        return None
    per, n_ep = [], 0
    for log in logs:
        prev = None
        for line in open(log, errors="ignore"):
            if m := START.match(line):
                prev = t(m.group(1))
            elif (m := P2_EPOCH.match(line)) and prev is not None:
                now = t(m.group(1))
                per.append((now - prev).total_seconds() / 60)
                prev, n_ep = now, max(n_ep, int(m.group(2)))
    return (n_ep, st.median(per)) if per else None


def summarize(runs):
    rows = {r: phase2_run(r) for r in runs}
    rows = {r: v for r, v in rows.items() if v}
    hours = [n * m / 60 for n, m in rows.values()]
    return {
        "runs": sorted(rows),
        "n": len(rows),
        "epochs_mean": st.mean(n for n, _ in rows.values()),
        "epochs_min": min(n for n, _ in rows.values()),
        "epochs_max": max(n for n, _ in rows.values()),
        "min_per_epoch_median": st.median(m for _, m in rows.values()),
        "run_hours_mean": st.mean(hours),
        "run_hours_median": st.median(hours),
        "run_hours_max": max(hours),
    }


def full_recipe_hours(run="phase1_dinov2"):
    log = open(f"{RUNS}/{run}/phase1_latest.log", errors="ignore").read().splitlines()
    start = next(t(m.group(1)) for line in log if (m := START.match(line)))
    end = next(t(line[:19]) for line in log if "Phase 1 (dinov2) finished" in line)
    return (end - start).total_seconds() / 3600


def predict_minutes():
    out = {}
    for f in sorted(glob.glob(f"{RUNS}/_launch_predict_*.out")):
        s = open(f, errors="ignore").read()
        members = re.search(r"predicting (\d+) images with (\d+) member\(s\) x (\d+) TTA", s)
        total = re.findall(r"(\d+)/\1 \[(\d+):(\d+)", s)
        if members and total:
            n, mm, ss = total[-1]
            out[os.path.basename(f)] = {"images": int(members.group(1)), "members": int(members.group(2)),
                                        "tta_views": int(members.group(3)), "minutes": int(mm) + int(ss) / 60}
    return out


if __name__ == "__main__":
    bulk = [h for res, h in phase1_epochs("phase1_dinov2") if res == 224]
    hi = [h for run in ("phase1_dinov2", "phase1_dinov2_tail_from_ep60", "phase1_dinov2_fullres")
          for res, h in phase1_epochs(run) if res == 518]
    imgs_per_epoch = 746 * 256  # 746 effective steps x effective batch 256 (log header)
    p2 = sorted(os.path.basename(d) for d in glob.glob(f"{RUNS}/abl_ep20_*") if "fullft" not in d) + \
        ["phase2_upgraded_fold0"]
    res = {
        "hardware": "1x NVIDIA A100-SXM4-80GB per job (shared 8-GPU node, 2x AMD EPYC 7J13)",
        "ssl_vitl_224_epoch_h": {"mean": st.mean(bulk), "median": st.median(bulk), "n": len(bulk),
                                 "img_per_s": imgs_per_epoch / (st.mean(bulk) * 3600)},
        "ssl_vitl_518_epoch_h": {"mean": st.mean(hi), "median": st.median(hi), "n": len(hi),
                                 "img_per_s": imgs_per_epoch / (st.mean(hi) * 3600)},
        "ssl_paper_recipe_h": full_recipe_hours(),
        "phase2_unfreeze4": summarize(p2),
        "phase2_fullft": summarize(sorted(os.path.basename(d) for d in glob.glob(f"{RUNS}/abl_baseline_*fullft*"))),
        "probe_frozen": summarize(sorted(os.path.basename(d) for d in glob.glob(f"{RUNS}/probe_*"))),
        "predict": predict_minutes(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res, indent=1))
