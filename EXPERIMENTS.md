# EXPERIMENTS.md

Config knobs for alternative experiments.

> **Note — recommended defaults changed (design audit, METHOD_CHANGES.md §A7):**
> `heatmap_size=148`, `llrd_decay=0.75`, `sample_temp=0.5`, `loss_space=original`,
> `dsnt_lambda=0.1`, `amp_dtype=bf16`, plus weight-decay exclusion. The "default"
> column in the tables below marks the *original* value; the recommended config
> `configs/phase2_upgraded.yaml` uses the new defaults, and
> `configs/phase2_baseline.yaml` pins the old ones as a reference.

Two ways to run an experiment:
- **Override on a base config** (isolates the variable):
  `python -m gubiometry phase2 --config configs/phase2_upgraded.yaml -o optim.coord_loss=wing`
- **Ready-made example config** under `configs/experiments/` (upgraded recipe +
  one experimental change): `python -m gubiometry phase2 --config configs/experiments/exp_wing_loss.yaml`

For a clean ablation, change **one** knob at a time and compare on the same fold /
holdout with `python -m gubiometry evaluate`.

---

## 1. Coordinate loss  (`optim.coord_loss`) — code-backed

| Value | Meaning |
|---|---|
| `l1` (default) | Original masked-L1 on soft-argmax coordinates. |
| `smooth_l1` | Huber (β=1) — less sensitive to large outlier residuals. |
| `wing` | Wing loss (Feng CVPR2018), the canonical landmark loss; amplifies small/medium errors. Tunable: `optim.wing_omega` (10.0), `optim.wing_epsilon` (2.0). |

`-o optim.coord_loss=wing` · `-o optim.coord_loss=smooth_l1`

## 2. Measurement-aligned auxiliary loss  (`optim.measurement_lambda`) — code-backed

Adds `lambda · |pred_measure − gt_measure|` over the task's clinical measurements
(distances in canvas px, AOP angle in degrees), reusing the metric's exact index
pairs. Directly targets the half of the challenge score that is measurement error.

| Value | Meaning |
|---|---|
| `0.0` (default) | Off — pure coordinate loss. |
| `0.02–0.1` | Auxiliary measurement supervision. Start ~`0.05`. |

`-o optim.measurement_lambda=0.05`  (the most challenge-specific, most experimental knob)

## 3. Augmentation strength  (`data.aug_strength`) — code-backed

| Value | Meaning |
|---|---|
| `none` | Letterbox + normalize only (no aug). |
| `light` | Small affine + mild brightness. |
| `medium` (default) | The original fixed pipeline. |
| `strong` | Wider affine/rotate + stronger color/noise/blur. |

`-o data.aug_strength=strong` · `-o data.aug_strength=light`
(Small-data cardiac tasks may benefit from `strong`; near-solved AOP may prefer `light`.)

## 4. Ensemble aggregation  (`predict.reduce`) — code-backed

| Value | Meaning |
|---|---|
| `mean` (default) | Average predictions across members × TTA views. |
| `median` | Median — robust to an outlier fold/view (useful for tiny high-variance tasks). |

`predict … -o predict.reduce=median`

To generate a **Codabench validation submission** (committed format, ensemble + TTA over
`data/data/val_data/`): `python -m gubiometry predict --config configs/submit_val.yaml`
(see RUNBOOK Step 3). `predict.image_dir` predicts any `<task>/<file>` image tree.

## 5. LR schedule  (`optim.scheduler`) — code-backed

| Value | Meaning |
|---|---|
| `warmup_cosine` (default) | Linear warmup → cosine (original). |
| `cosine` | Cosine over all epochs, no warmup. |
| `constant` | Flat LR (debugging / short runs). |

`-o optim.scheduler=cosine`

## 6. Capacity / already-wired knobs (documentation only)

These need **no code change** — set them directly.

| Knob | Default | Alternatives | Notes |
|---|---|---|---|
| `model.backbone.name` | `dinov2_vitl14` | `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitg14` (+`_reg`) | Any DINOv2 size; embed dim auto-detected (Phase 1 & 2). A different size/variant **requires re-running Phase 1**. |
| `model.heatmap_size` | `128` | `96`, `160` | Higher → finer soft-argmax sub-pixel precision (more memory). |
| `model.neck.feature_layers` | `[5,11,17,23]` | e.g. `[2,5,8,11]` (shallower), `[11,17,20,23]` (deeper) | Which DINOv2 depths feed the neck (only when `input_mode=multilevel`). |
| `model.backbone.unfreeze_last_n_blocks` | `4` | `0`, `6`, `8` | How much of the encoder to fine-tune. Pairs with `optim.llrd_decay`. |
| `optim.encoder_lr_mult` | `0.1` | `0.05`, `0.2` | Encoder LR = `lr × mult` (top of encoder). |
| `optim.softargmax_temp` | `10.0` | `5.0`, `20.0` | Higher = sharper argmax. |
| `optim.select_metric` | `challenge_blend` | `average_mre`, `average_avg_mae` | Which validation metric selects the best checkpoint. |

Example: `-o model.backbone.name=dinov2_vitg14 -o model.heatmap_size=160`

## 7. Pipeline alignment & robustness — code-backed

| Knob | Default | Alternatives | Notes |
|---|---|---|---|
| `data.sample_temp` | `0.0` | `0.5` (sqrt), `1.0` (natural) | Task sampling frequency ∝ `n_task**temp`. `0` = fully balanced (a tiny task like IVC is replayed ~100×/epoch → overfitting); `0.5` is the usual middle ground. **The highest-value knob for the small cardiac tasks.** |
| `optim.loss_space` | `canvas` | `original` | Scales each sample's residual by `max(h,w)/canvas` so the loss is weighted like the **original-pixel** metric. Modest effect (checkpoint selection is already metric-aligned). ⚠️ `wing` ω/ε are tuned for canvas-px residuals — retune if combining `coord_loss=wing` with `loss_space=original`. |
| `predict.oof` | `false` | `true` | **Out-of-fold** evaluation: each fold member predicts only its held-out val, so every training image is scored by a model that didn't see it — an honest local CV estimate. Not for the hidden test (there you use all members). |

`-o data.sample_temp=0.5` · `-o optim.loss_space=original` · `predict … -o predict.oof=true`

**Comparing runs/folds:** `challenge_blend` normalizers are computed from whatever
val set is passed, so the blend is only comparable **within a run** (epoch selection).
To compare across folds/experiments, use `average_mre` / `average_avg_mae` (raw px /
degrees, printed by `gubiometry evaluate`) or set `optim.select_metric=average_mre`.

---

## Example configs (`configs/experiments/`)

Each is `phase2_upgraded` + one experimental change, ready to run as-is:

| File | Change |
|---|---|
| `exp_wing_loss.yaml` | `optim.coord_loss: wing` |
| `exp_measurement_aux.yaml` | `optim.measurement_lambda: 0.05` |
| `exp_strong_aug.yaml` | `data.aug_strength: strong` |
| `exp_sample_temp.yaml` | `data.sample_temp: 0.5` (sqrt task sampling) |
| `exp_median_ensemble.yaml` | prediction with `predict.reduce: median` |

```bash
python -m gubiometry phase2 --config configs/experiments/exp_wing_loss.yaml \
    -o phase1_weights=pretrained/dinov2_reg_adapted_ep20.pth
```

All new knobs are also captured in each run's `runs/<run>/config.json`, so
`predict`/`evaluate` rebuild the exact model automatically.
