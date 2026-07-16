# RUNBOOK — training & submission on this machine

Practical, copy-paste steps for the **default (recommended) path** to a competitive
submission, plus the **most promising alternative paths** to try. See
[METHOD_CHANGES.md](METHOD_CHANGES.md) for what each upgrade is and
[EXPERIMENTS.md](EXPERIMENTS.md) for the full knob list.

---

## Machine quick-reference

| Thing | Value |
|---|---|
| Python env | `/data/abar/alexenv` (torch 2.11 + albumentations + cv2 + tensorboard installed) |
| Data root | **`data/data/train_data`** (already the default in every config) |
| Labeled / unlabeled | 5414 train + 1354 val / ~191k unlabeled |
| GPUs | 8× A100-80GB, **shared** — always pick a free one |
| Throughput | legacy single-mode baseline ran ~6 min/ep @ bs64 (~29 GB, ~4.6 h for 46 epochs); the **default** multilevel + heatmap-148 recipe is heavier — measure your first epoch |
| Recipe defaults | the recommended recipe is now baked into `phase2_upgraded.yaml` **and** the config defaults: multilevel neck, `dinov2_vitl14_reg` backbone, LLRD 0.75, DSNT 0.1, **`sample_temp` 0.5**, `loss_space` original, heatmap 148, bf16, WD-excluded norms. `phase2_baseline.yaml` pins the legacy values. |
| Splits | holdout `splits/train_val_split_keys.json` + 5-fold `splits/kfold_v1/fold_{0..4}.json` (done) |

**Find a free GPU:**
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
 | awk -F',' '{gsub(/ /,"",$2); if($2+0<2000) print "GPU "$1" FREE"}'
```

**Logging is handled in-code — no shell redirects.** Just run the command (e.g. in a
tmux pane) and it both prints live and writes `runs/<run_name>/<phase>_<timestamp>.log`,
with a stable `<phase>_latest.log` symlink next to it. Monitor from another pane with
`tail -f runs/<run_name>/phase2_latest.log` (per-epoch line shows `MRE(orig px)` and
`challenge_blend`). Or TensorBoard: `tensorboard --logdir runs/ --port 6006`. A crash
is captured in the log file too.

All runs write to `runs/<run_name>/` (config.json + text log + tfevents +
`checkpoints/best_teacher_model.pth`). Rebuild any trained model automatically from
its `config.json` — no need to re-specify hyperparameters.

---

## DEFAULT PATH (recommended → submission)

### Step 0 — setup ✅ done
Env has deps, data is in place, holdout + 5-fold splits generated. (To regenerate
folds: `python -m gubiometry make-splits --data-root data/data/train_data --kfold --n-splits 5`.)

### Step 1 — legacy baseline reference ✅ done
A fast Phase-2 baseline on the **legacy** recipe (single-mode, no SSL, single holdout)
established the number to beat: **MRE 23.8 px / measurement-MAE 16.5** on the 1354-image
holdout (`runs/p2_baseline_nossl`) — already ahead of the reference `local_eval_v7`
(32.6 / 26.4). The recommended recipe (Steps 2–3) should beat it further. For reference,
the command was:
```bash
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry phase2 --config configs/phase2_baseline.yaml \
  -o phase1_weights= -o run_name=p2_baseline_nossl -o data.batch_size=64 -o optim.epochs=80 -o optim.early_stop_patience=20
```

### Step 2 — recommended recipe, 5 folds (the ensemble members)
`configs/phase2_upgraded.yaml` **is** the recommended recipe — multilevel features, reg
backbone, LLRD, DSNT, `sample_temp 0.5`, original-px loss, heatmap 148, bf16 (all now
defaults) plus the A100 run-scale (bs64 / lr2e-4 / 150 ep). **Run the 5 folds in parallel
on 5 GPUs**, each started from your Phase-1 encoder (the **Phase 1** section below) — or
`-o phase1_weights=` to skip SSL. Cleanest in tmux: one window per fold (foreground, live log). Or launch all five
at once from one pane — logging is in-code, so each fold writes its own
`runs/phase2_upgraded_fold<f>/phase2_latest.log`; the `>/dev/null` below only stops the
five live streams interleaving in the launching pane (it is *not* the logging):
```bash
PHASE1=runs/phase1_multicrop/checkpoints/dinov2_adapted_ep<N>.pth   # or set to "" for no SSL
for f in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$f python -m gubiometry phase2 --config configs/phase2_upgraded.yaml \
    -o phase1_weights=$PHASE1 -o data.fold=$f -o run_name=phase2_upgraded_fold$f >/dev/null 2>&1 &
done
# then e.g.  tail -f runs/phase2_upgraded_fold0/phase2_latest.log
```
Run names `phase2_upgraded_fold{0..4}` match `configs/predict_ensemble.yaml` out of the
box. **CV estimate = mean of the 5 folds' best val `average_mre`** (each on its own
held-out fold → honest, no leakage).

### Step 3 — Codabench validation submission (committed format, ensemble + TTA)
```bash
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry predict --config configs/submit_val.yaml   # -> regression_predictions.zip
```
Predicts **every image in `data/data/val_data/`** (the challenge's 619 val images,
`<task>/<file>` layout) with the 5-fold ensemble × multi-scale/intensity TTA and writes
`regression_predictions.json` (+`.zip`) in the committed format
`{image_path, task_id, predicted_points_normalized, predicted_points_pixels}`. The zip's
inner file is always named `regression_predictions.json` (Codabench requires that exact
name), regardless of `out_json`. Until the folds exist, submit the current single model:
```bash
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry predict --config configs/submit_val.yaml \
  -o predict.member_run_dirs=runs/p2_baseline_nossl -o predict.out_json=submission_baseline.json
```
Upload the `.zip` to Codabench (validation phase: **2/day, 10 total per team**; it is a
**format sanity check only** — the real ranking is the Docker test phase). Try
`-o predict.reduce=median`. `configs/submit_val.yaml` is the one blessed submission path
(always ensemble + TTA).

### Step 4 — local score
```bash
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry evaluate --config configs/predict_ensemble.yaml \
  -o predict.gt_json=data/data/train_data/splits/local_eval_gt/internal_ground_truth_val.json
```
Prints per-task MRE (orig px) + measurement MAE, matching the challenge's own scorer.
Note: the 1354-holdout images also appear in some fold's training, so this number is
mildly optimistic for the *ensemble* — trust the **per-fold val** numbers (Step 2) as
the honest CV, and this scorer for absolute per-task/measurement diagnostics.

---

## Phase 1 — SSL domain adaptation (do this *before* Step 2)

Adapt the register-ViT encoder on the ~191k unlabeled frames; Step 2's folds start
from it. **Expensive** (many hours–days; the multi-crop objective runs 8 crops/image).
The backbone is already the config default — don't override it unless you also change
Phase 2's backbone to match.
```bash
# run in a tmux pane -- logging is in-code (writes runs/phase1_multicrop/phase1_*.log)
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry phase1 --config configs/phase1_multicrop.yaml
# checkpoints every 10 epochs -> runs/phase1_multicrop/checkpoints/dinov2_adapted_ep{10,20,...}.pth
```
No epoch cap by default (falls back to 100). Watch
`tail -f runs/phase1_multicrop/phase1_latest.log` (`DINO loss` per epoch) and judge
convergence from the per-10-epoch checkpoints; you can start a Step-2 fold from an early
checkpoint (e.g. `ep30`) to gauge the trend before the full run finishes. Then set
`PHASE1=…/dinov2_adapted_ep<N>.pth` in Step 2. If you want to know whether SSL is worth
the cost, also run the folds with `PHASE1=""` (no SSL) and compare the CV.

**If it stops (crash, preemption, `Ctrl-C`):** a full resumable state is written every
epoch to `runs/phase1_multicrop/checkpoints/latest_checkpoint.pth`. Re-run the *same*
command with `-o resume=...`:
```bash
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry phase1 --config configs/phase1_multicrop.yaml \
  -o resume=runs/phase1_multicrop/checkpoints/latest_checkpoint.pth
```
It restores model/EMA-teacher/optimizer/scheduler/AMP-scaler/DINO-center and continues
from the next epoch. Don't change `phase1.epochs`/`batch_size` on resume — the teacher-temp
and cosine schedules are indexed by epoch count. Phase 2 folds resume the same way
(`-o resume=runs/<fold_run>/checkpoints/latest_checkpoint.pth`).

---

## PROMISING ALTERNATIVE PATHS (ablations)

Most best-practice knobs are **already the defaults** in the Step-2 recipe
(`sample_temp 0.5`, `loss_space original`, LLRD, DSNT, multilevel neck, reg backbone,
heatmap 148). What's left to explore are the *uncertain* knobs — run each as a single
change vs the recommended recipe on the holdout, compare on best val `average_mre`, keep
the winners. Template:
```bash
CUDA_VISIBLE_DEVICES=<free> python -m gubiometry phase2 --config configs/phase2_upgraded.yaml \
  -o phase1_weights= -o run_name=<name> <ONE KNOB>
```
- **Measurement-aligned aux loss** — `-o optim.measurement_lambda=0.05` (directly
  supervises the clinical measurements = 50% of the score). `exp_measurement_aux.yaml`.
- **Wing loss** — `-o optim.coord_loss=wing` (canonical landmark loss). `exp_wing_loss.yaml`.
- **Stronger augmentation** — `-o data.aug_strength=strong` (may help the small tasks).
- **Median ensembling** — at *predict* time, `-o predict.reduce=median` (robust to an
  outlier fold on the tiny high-variance tasks).
- **Bigger / finer** — `-o model.backbone.name=dinov2_vitg14_reg` (giant; needs a matching
  Phase-1 run) or `-o model.heatmap_size=160` (finer sub-pixel).
- **Sanity checks** — confirm a new default actually helps by turning it *off*:
  `-o data.sample_temp=0.0`, `-o optim.loss_space=canvas`, or `-o model.neck.input_mode=single`.

---

## Suggested order

1. **Legacy baseline** (Step 1) → ✅ done (MRE 23.8 px — the number to beat).
2. **Phase 1 SSL** (section above) → the adapted register-ViT encoder.
3. **Recommended 5-fold** (Step 2) from the Phase-1 encoder → **predict + submit** (Steps 3–4).
4. Optionally, **ablations** (measurement / Wing / strong-aug / median / bigger backbone) as single-holdout runs on spare GPUs → fold the winners into a final ensemble.

## Housekeeping / gotchas
- Checkpoints are large (~1.2 GB `best`, ~2.9 GB `latest` per run × 5 folds). Watch disk; delete `latest_checkpoint.pth` once a run is done (resume needs it, the final model doesn't — see "If it stops" above).
- `phase2_upgraded.yaml` already pairs bs64 with lr2e-4 (sqrt-scaled) — no manual LR bump needed. AMP is **bf16** on the A100 (no gradient scaler); a leftover `GradScaler` import may still print one harmless deprecation line.
- `sample_temp=0.5` (default) means the tiny tasks are seen less often than fully-balanced — intended (less overfit), but glance at their per-task val MRE to be sure it's not regressing.
- Always set `CUDA_VISIBLE_DEVICES` to a **free** GPU — this box is shared.
