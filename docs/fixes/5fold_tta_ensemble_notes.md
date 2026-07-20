# 5-Fold Ensemble + TTA — deferred work, resume notes

Not run for the current paper draft. Results (Table `tab:codabench-results`) reports
a **single model** (best Phase-1 checkpoint × upgraded Phase-2 recipe) submitted
directly to Codabench, since there wasn't time to complete the full ensemble before
submission. Named as Future Work in the Conclusion. This file has the full technical
recipe so it can be picked back up later without re-deriving anything.

## The procedure (A6, `METHOD_CHANGES.md`)

- Train **five** Phase-2 models via stratified 5-fold cross-validation (task-stratified,
  seed 42) — one per fold:
  `gubiometry kfold --config configs/phase2_upgraded.yaml -o phase1_weights=<winning checkpoint>`
- At inference, ensemble across all 5 fold models' **EMA teachers**.
- Apply safe TTA per view:
  - Multi-scale canvases: **490, 518, 546px** (all divisible by the 14-pixel patch stride)
  - Intensity: gamma jitter (0.8, 1.2)
  - **No flips** — keypoints are semantically ordered, not flip-invariant
- Average predictions across folds × TTA views directly in **original-pixel coordinate
  space** (not heatmap space) — robust even if ensemble members differ in backbone/neck
  config.
- `gubiometry predict --config configs/predict_ensemble.yaml` (`member_run_dirs` = the
  5 fold run dirs, `tta_canvases=[490,518,546]`, `tta_intensity=true`,
  `average_space=coord`).

## Cost estimate (colleague's empirical measurements, 2026-07-17)

- ~6 min/epoch, upgraded recipe, bs64, A100-80GB.
- Historical early-stops landed at epoch 72–117 of 150 (patience=40).
- Full 5-fold: ~5 × (7–15h) sequential on one GPU, or ~7–15h if parallelized across 5 GPUs.

## To resume later

1. Confirm the winning Phase-1 checkpoint from Table `tab:ssl-duration` (ep10 vs ep20).
2. `python -m gubiometry kfold --config configs/phase2_upgraded.yaml -o phase1_weights=<winner>`
3. `python -m gubiometry predict --config configs/predict_ensemble.yaml -o predict.member_run_dirs=[...]`
4. `python -m gubiometry evaluate --config configs/predict_ensemble.yaml -o predict.gt_json=...`
   for local verification before re-submitting officially.
5. Update `tab:codabench-results`' caption/prose back to "5-fold ensemble + TTA," and
   restore the Implementation Details paragraph describing the procedure (cut from
   `main.tex` when this was deferred — see git history around this file's addition).
