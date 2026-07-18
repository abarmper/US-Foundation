# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Training/inference code for a two-phase ultrasound landmark-detection pipeline for
the **GU/FU Biometry MICCAI 2026 challenge** (9 tasks: `A4C, AOP, FA, FUGC, HC, IVC,
PLAX, PSAX, fetal_femur`): Phase 1 domain-adapts a DINOv2-L/14 encoder with SSL;
Phase 2 adds a multi-stage HRNet neck + 9 per-task soft-argmax heads.

The code lives in the installable **`gubiometry/`** package driven by one CLI and
YAML configs. `METHOD_CHANGES.md` documents the challenge-specific upgrades layered
on the original method (and which are paper-affecting). `docs/ARCHITECTURE.md` covers
the neck design. Git repo (remote: `abarmper/US-Foundation`, private); no test suite (verification is via
CPU smoke tests with a dummy backbone — see below).

## Commands

```bash
pip install -e .

python -m gubiometry make-splits [--kfold --n-splits 5]      # writes splits/*.json
python -m gubiometry phase1   --config configs/phase1_multicrop.yaml
python -m gubiometry phase2   --config configs/phase2_baseline.yaml   # or phase2_upgraded
python -m gubiometry kfold    --config configs/phase2_upgraded.yaml   # 5 fold models
python -m gubiometry predict  --config configs/predict_ensemble.yaml  # submission.zip
python -m gubiometry evaluate --config configs/predict_ensemble.yaml -o predict.gt_json=...
```

- **Config = one YAML** (`gubiometry/config.py` dataclasses) + dotted `--override`
  (`-o optim.epochs=1 -o model.backbone.name=dummy`). `scripts/*.sh` are thin wrappers.
- **CPU smoke test (no GPU/dataset/torch.hub):** set `model.backbone.name=dummy` and
  build a throwaway dataset with `gubiometry.testing.build_synthetic_dataset(...)`;
  `optim.max_train_batches/max_val_batches` cap work. This is how every gate is
  verified — the venv here lacks `cv2/albumentations/tensorboard`, and the code
  degrades gracefully (numpy letterbox, PIL image read, optional TB, no-aug fallback).

## The challenge metric drives the design (read `METHOD_CHANGES.md`)

Score = **50% MRE (original-image px) + 50% clinical-measurement error** (AOP angle in
degrees, other lengths in px), **macro-averaged with equal weight per task**. Two
consequences that pervade the code:
1. Tiny cardiac tasks (IVC 38, PSAX 49 imgs) count as much as AOP (4000) → the levers
   are **cross-validation, ensembling, TTA** (`kfold`/`predict`).
2. Training/selection must use the real metric. `gubiometry/metrics.py` (data-driven
   from `data/.../task_measurement_table.csv`) computes it in original px and
   **exactly reproduces the committed `local_eval` numbers** — it's the correctness
   anchor; if you touch it, re-run that reproduction test. Phase-2 selects the best
   checkpoint on `challenge_blend`, **not** 518-canvas loss.

## Architecture invariants (do not break)

- **Per-batch task routing:** `forward_phase2(x, task_id)` runs exactly one head
  (`task_id[0]`). This only works because batches are **task-homogeneous** — enforced
  by the samplers in `gubiometry/data/samplers.py`. Never use a plain shuffling loader.
- **Soft-argmax is the method** (`gubiometry/losses.py`), a coordinate-regression L1;
  missing keypoints are `(-1,-1)` and masked. DSNT (`dsnt_lambda`) is an optional
  regularizer, **off by default** (λ=0 = original loss bit-identical). Heatmaps are
  never MSE-supervised.
- **GroupNorm everywhere** in the neck/heads (identical train/eval stats across the
  heterogeneous task domains) — keep it.
- **EMA teacher is the artifact:** validation runs the teacher; `best_teacher_model.pth`
  is a bare model `state_dict`.
- **`canvas` is explicit** in `geometry.soft_argmax_coords` / letterbox inverse — multi-
  scale TTA passes each view's own canvas. Never hardcode 518 in decode/inverse.

## Method upgrades (now the defaults; each preserves the DINO approach)

The recommended settings are the **dataclass defaults** now; `configs/phase2_baseline.yaml`
pins the legacy recipe as a reference. See `METHOD_CHANGES.md` for details and paper impact.
Defaults (override to get the legacy value):
- `model.neck.input_mode: multilevel` — 4 intermediate DINOv2 depths feed the neck
  (**shape-breaking**: `reassemble.*` keys replace `stage2_b3`; `single` reproduces the
  original graph & checkpoint keys exactly).
- `model.backbone.name: dinov2_vitl14_reg` — register backbone (the Phase-2 backbone must
  match the Phase-1 one; a different backbone **requires re-running Phase 1**).
- `optim.llrd_decay: 0.75` (`1.0` = legacy) · `optim.dsnt_lambda: 0.1` (`0.0` = legacy) ·
  `optim.loss_space: original` (`canvas` = legacy) · `data.sample_temp: 0.5` (`0.0` = legacy) ·
  `model.heatmap_size: 148` (was 128) · `optim.amp_dtype: bf16` · weight-decay excluded from norms/biases.

**A6 (5-fold CV + ensemble + TTA)** remains a workflow opt-in: `gubiometry kfold` then `predict`.

Additional **ablation knobs** (all default to current behavior) are in
`EXPERIMENTS.md`: `optim.coord_loss` (l1|smooth_l1|wing), `optim.measurement_lambda`
(clinical-measurement aux loss), `data.aug_strength` (none|light|medium|strong),
`predict.reduce` (mean|median), `optim.scheduler`, and already-wired capacity sweeps
(backbone size, `model.heatmap_size`, `model.neck.feature_layers`, unfreeze depth,
`optim.select_metric`).

## Package map

```
gubiometry/
  config.py          nested dataclasses + YAML + legacy config.json adapter
  constants.py       TASK_KEYPOINTS / TASK_ORDER (single source of truth)
  geometry.py        soft_argmax_coords, letterbox forward/inverse (pure numpy/torch)
  metrics.py         challenge scorer (reproduces official local_eval)
  losses.py          soft_argmax_loss (+DSNT), mean_radial_error
  optim.py           LLRD param groups, scheduler, EMA
  models/            backbone.py (hub + DummyBackbone), neck.py (single|multilevel),
                     heads.py, model.py (build_model_from_config), dino_ssl.py (vendored
                     DINOv2 heads/losses: DINOHead, DINO/iBOT/KoLeo, MaskingGenerator)
  data/              dataset.py, samplers.py, transforms.py, multicrop.py, splits.py (kfold)
  engine/            phase1 (multicrop|sameview), phase1_dinov2 (DINOv2-faithful SSL),
                     phase2 (metric-aligned), kfold, predict (ensemble×TTA), evaluate
  testing/           synthetic.py (tiny data_root + dummy-backbone harness)
configs/*.yaml       phase1_multicrop, phase1_dinov2, probe_phase1, phase2_baseline, phase2_upgraded, predict_ensemble
src/visualization/   paper-figure scripts (rewired onto gubiometry via common.py)
```

Old Phase-2 `best_teacher_model.pth` from the original code still loads: the
single-mode neck keys are identical (verified). Rebuild any trained model with
`build_model_from_config(config_from_run_dir(run_dir))` — no hand-matched hyperparameters.

## Data layout

Dataset not shipped (only `data/splits/`). Arrange the real data under `data_root`
per `data/README.md`. `RobustBiometryDataset` indexes `images/<TASK>/...` recursively,
joins to `csv/*.csv` (excluding `pseudo`) by `(task_id, basename)`, and returns
original-space GT (`keypoints_orig`, `orig_h/w`) for metric scoring plus the 518-canvas
`keypoints` for the loss. First run pulls DINOv2 from `torch.hub` into
`~/.cache/torch_gu_biometry`.
