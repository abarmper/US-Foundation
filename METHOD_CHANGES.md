# METHOD_CHANGES.md

Architectural upgrades + pipeline refactor for the **GU/FU Biometry MICCAI 2026**
challenge, layered on the original two-phase DINOv2-HRNet method. The general DINO
approach is **preserved**: every method change is additive/opt-in, and the default
config reproduces the original graph, optimizer and loss.

This file is split into **(1) method changes that affect the paper** and
**(2) pipeline/engineering changes that do not**. It also flags which changes are
*shape-breaking* (old checkpoints won't load) or require *re-running Phase 1*.

Challenge facts that motivate the changes (from Codabench comp. 15590):
- Score = **50% MRE (original-image pixels) + 50% clinical-measurement error**
  (AOP angle in degrees, other lengths in px/mm), **macro-averaged with equal
  weight per task**.
- Because of the macro-average, the tiny cardiac tasks dominate: v7 per-task MRE is
  AOP 4.3 (4000 imgs) vs **PSAX 66.8 (49), IVC 48.2 (38), A4C 46.5 (108)**. Variance
  reduction (CV/ensemble/TTA) is the biggest lever.
- Keypoints are **strictly ordered/semantic** (measurements use fixed index pairs)
  → naive flip augmentation is unsafe.

---

## 1. METHOD changes (update the paper)

### A1 — Metric-aligned validation + challenge scorer  *(changes reported numbers)*
The original loop measured validation in **518-canvas pixels** and selected
`best_teacher_model.pth` on 518-canvas L1 loss — neither is the challenge metric.
Now a data-driven scorer (`gubiometry/metrics.py`, from
`task_measurement_table.csv`) computes, in **original pixels**, per-task MRE and the
derived clinical measurements (distances; the AOP angle in **degrees**, computed in
pixel space so W≠H doesn't distort it), macro-averaged. Validation inverse-
letterboxes predictions to original pixels and the best checkpoint is selected on
`challenge_blend` (a normalized 0.5·MRE + 0.5·measurement-MAE proxy; lower is
better). **Verified: the scorer reproduces the committed local-eval v7 numbers
exactly** (Average MRE 32.6425, Average AvgMAE 26.4186, and every per-task value).
- *Paper impact:* checkpoint selection now optimizes the real objective; expect all
  reported numbers to change (and to correlate better with the leaderboard).
- *Shape-breaking:* no. Old checkpoints can be re-scored/re-selected.

### A2 — Multi-level DINOv2 features into the HRNet neck  *(DPT/ViTPose "reassemble")*
Instead of deconvolving a **single** last-layer 37×37 grid, four intermediate ViT
depths `get_intermediate_layers(x, n=(5,11,17,23))` initialize the neck branches
(deepest→coarsest: g23→b1@37, g17→b2@74, g11+g05→b3@148). Exchange units and fusion
are unchanged. Enabled by `model.neck.input_mode: multilevel`.
- *Hyperparameters:* `feature_layers=(5,11,17,23)`.
- *Paper impact:* a genuine architecture change (multi-scale ViT feature
  aggregation). **Shape-breaking:** multilevel checkpoints have `reassemble.*` keys
  and no `stage2_b3`; `input_mode: single` reproduces the original graph/keys exactly.

### A3 — Layer-wise LR decay (LLRD) for encoder fine-tuning
The single `encoder_lr_mult` for all unfrozen blocks is replaced by depth-decayed
LRs (`optim.llrd_decay`, default **0.75**): deepest unfrozen block least decayed.
- *Backward compat:* `llrd_decay: 1.0` gives every unfrozen block the same LR — the
  original single-multiplier behavior (used by `configs/phase2_baseline.yaml`).
- *Paper impact:* optimization recipe change (canonical ViT transfer); no shape change.

### A4 — DSNT regularization on the soft-argmax heatmaps  *(Nibali et al. 2018)*
An optional regularizer keeps each predicted heatmap tight/unimodal **without a
fixed heatmap-MSE target** (the pathology the original rejected): `js` (default) =
Jensen–Shannon divergence to a unit-sum Gaussian centered at GT — amplitude-free;
`var` = penalize spatial variance above σ². Added as `L = L_coord + dsnt_lambda·R`.
- *Hyperparameters:* `dsnt_lambda` (default **0.0** = original loss bit-identical;
  recommended 0.1), `dsnt_type=js`, `dsnt_sigma=1.0`.
- *Paper impact:* loss change; no shape change.

### A5 — Register backbone `dinov2_vitl14_reg`
`model.backbone.name` selects the backbone; the register variant yields cleaner
dense feature maps. Neck code is backbone-agnostic (registers stripped by
`forward_features`/`get_intermediate_layers`).
- *Requires re-running Phase 1* on the reg backbone (checkpoint is backbone-specific).
- *Paper impact:* backbone change. **Shape-breaking** relative to `vitl14` Phase-1.

### A6 — 5-fold cross-validation + ensemble + TTA  *(biggest, most reliable lever)*
Stratified 5-fold splits (`StratifiedKFold`, seed 42); one Phase-2 model per fold;
inference averages predictions across folds × TTA views. **Safe TTA only:**
multi-scale (`tta_canvases=(490,518,546)`, all %14==0) + optional intensity (gamma);
**no naive flips** (keypoints are semantic — a per-task index permutation would be
required, left unimplemented). Averaging is in **coordinate space** (robust; members
may mix single/multilevel/reg), with a `heatmap` option for same-canvas fold averaging.
- *Paper impact:* results reported as a 5-model ensemble + TTA; expected largest gain
  on the small-data cardiac tasks that dominate the macro-average.

**Default vs legacy summary.** The recommended settings are now the **defaults**
(the dataclass defaults in `gubiometry/config.py`); `configs/phase2_baseline.yaml`
explicitly pins the legacy values as a reference point.

| Change | Default now | Legacy (pinned by `phase2_baseline.yaml`) |
|---|---|---|
| A1 metric-aligned selection | **on** | — (always on) |
| A2 multi-level features | **on** (`neck.input_mode: multilevel`) | `single` |
| A3 LLRD | **on** (`optim.llrd_decay: 0.75`) | `1.0` |
| A4 DSNT | **on** (`optim.dsnt_lambda: 0.1`) | `0.0` |
| A5 register backbone | **on** (`backbone.name: dinov2_vitl14_reg`) | `dinov2_vitl14` |
| A6 CV+ensemble+TTA | opt-in (`gubiometry kfold` + `predict`) | single holdout |

Plus the §A7 best-practice defaults (heatmap 148, WD-exclusion, bf16, sqrt sampling,
original-px loss). `configs/phase2_upgraded.yaml` is the full recommended recipe
(these defaults + the A100 run-scale: bs64 / lr2e-4 / 150 ep). A run needing the
exact original method uses `configs/phase2_baseline.yaml`.

**Further ablation knobs** (alternative losses incl. Wing, a measurement-aligned
auxiliary loss, augmentation-strength presets, median ensembling, LR-schedule
choice, and capacity sweeps) are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

### A7 — Best-practice default fixes (design audit)

A review of the pipeline against general best practice (independent of the paper)
changed several **defaults**. These move reported numbers; `configs/phase2_baseline.yaml`
now explicitly pins the pre-change recipe as a reference.

| Fix | Before | After (default) | Why |
|---|---|---|---|
| Heatmap resolution | 128 (downsized from the neck's 148) | **148** | match the neck; finer soft-argmax (cell 4.05→3.5 px). Param-free — old checkpoints still load. |
| Weight decay on norms/biases | decayed | **excluded (ndim<2 → wd 0)** | standard ViT recipe; decaying LayerNorm/GroupNorm scales is harmful. |
| AMP dtype | fp16 + GradScaler | **bf16** on Ampere (fp16 fallback) | wider range, no loss-scaling, more stable on A100. |
| Layer-wise LR decay | off (1.0) | **0.75** | canonical ViT fine-tuning recipe. |
| Task sampling | fully balanced (0.0) | **sqrt (0.5)** | stop replaying IVC/PSAX (30-39 imgs) ~100×/epoch → less small-task overfit. |
| Loss space | 518-canvas px | **original px** | align the loss with the original-px metric. |
| Soft-argmax spread | none (dsnt 0) | **dsnt 0.1** | keep heatmaps tight/unbiased. |
| Fine-tune scope | partial unfreeze (last 4) | **unchanged** | partial > freeze/full for a foundation backbone + small data (the recommended answer). |
| `phase2_upgraded.yaml` | bs16 / lr1e-4 / 500 ep | **bs64 / lr2e-4 / 150 ep / patience40** | use the A100; sqrt-scale LR; let the cosine anneal finish before early-stop. |

`configs/phase2_baseline.yaml` reproduces the pre-change recipe (`dinov2_vitl14`
backbone, single-mode neck, heatmap 128, llrd 1.0, sample_temp 0, canvas loss, dsnt 0,
fp16) — it still inherits the two strict code fixes (WD-exclusion, bf16 availability),
which are corrections, not tunables. The already-trained `runs/p2_baseline_nossl`
checkpoint is the exact legacy reference. All other configs (and bare `RunConfig()`)
now default to the recommended recipe, including the multi-level neck and the register
backbone.

---

## 2. Pipeline / engineering changes (no paper impact)

- **`src/` → installable `gubiometry/` package** (`pyproject.toml`, real imports;
  the `sys.path.append` hacks are gone). Superseded `src/{models,data,training}`
  were deleted; only the paper-figure scripts remain under `src/visualization`,
  rewired onto the package.
- **Single config system** (`gubiometry/config.py`): nested dataclasses + YAML
  (`configs/*.yaml`), dotted `--override`, and a legacy `config.json` adapter.
  Replaces scattered argparse across three scripts.
- **Unified CLI:** `python -m gubiometry {phase1|phase2|kfold|predict|evaluate|make-splits}`.
- **`build_model_from_config` reused everywhere** (training, prediction, evaluation,
  visualization) and models auto-rebuild from `runs/<run>/config.json` — no more
  hand-matching `--neck_branch_width/--unfreeze_last_n_blocks` under `strict=True`.
- **De-duplication:** one source of truth for the task list/keypoints
  (`constants.py`), soft-argmax + letterbox math (`geometry.py`), losses, EMA,
  logging, checkpointing — each was previously copied 3–5×.
- **New capabilities that were missing:** in-repo challenge scorer, submission
  generator (`predict`), evaluator (`evaluate`), k-fold split generator.
- **Robustness:** inference/validation letterbox is pure numpy/PIL (no
  albumentations/cv2) — lighter Docker inference; training augmentation still uses
  albumentations (lazy, with a no-aug fallback if unavailable); TensorBoard optional;
  image reading falls back to PIL. Phase-1 checkpoints now save a **bare** encoder
  dict (no manual key-stripping needed for Phase 2).
- Small fixes from `FIXES.md` are carried in (correct split JSON output, robust
  Phase-1 weight loading, repo-root run dirs, DSNT/DINO warmup crash guard).

---

## Reproducibility notes for the paper

- **Re-run Phase 1** only for A5 (register backbone).
- **Shape-breaking** for checkpoints: A2 (multilevel) and A5 (register).
- The scorer is validated against the challenge's own local evaluator (exact match),
  so `gubiometry evaluate` numbers are directly comparable to the committed
  `local_eval` outputs.

## Quickstart

```bash
pip install -e .
python -m gubiometry make-splits --kfold --n-splits 5        # fold_0..4.json
python -m gubiometry phase1  --config configs/phase1_multicrop.yaml
python -m gubiometry kfold   --config configs/phase2_upgraded.yaml \
    -o phase1_weights=pretrained/dinov2_reg_adapted_ep20.pth  # 5 fold models
python -m gubiometry predict --config configs/predict_ensemble.yaml   # submission.zip
python -m gubiometry evaluate --config configs/predict_ensemble.yaml \
    -o predict.gt_json=data/splits/local_eval_gt/internal_ground_truth_val.json
```
