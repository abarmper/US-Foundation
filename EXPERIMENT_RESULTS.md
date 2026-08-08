# Phase-2 Experiment Results

GU/FU Biometry — Phase-2 landmark runs. Metric: **`challenge_blend`** (0.5·MRE + 0.5·measurement-MAE proxy, **lower = better**) and **MRE** (mean radial error, original px). Best checkpoint selected on `challenge_blend`. Snapshot: 2026-07-28.

> ⚠️ **Read §5 + findings #16–18 first.** Internal fold-0 `challenge_blend` is both **leaky**
> (video-frame overlap in cardiac tasks) and a **proxy** (unofficial normalizer), and it does **not**
> transfer to the Codabench leaderboard — the best *external* model is the plain no-SSL HRNet. Treat
> all §1–§3 rankings as internal-only until splits are loop-level and results are leaderboard-checked.

## Legend

**SSL encoder** (what Phase-1 produced the encoder weights):
- **none** — off-the-shelf DINOv2 (`dinov2_vitl14_reg`, no Phase-1 adaptation)
- **old ep{N}** — legacy `multicrop` DINO (`runs/phase1_multicrop`) — the weak recipe (toy head, no iBOT/KoLeo)
- **NEW ep{N}** — DINOv2-faithful (`runs/phase1_dinov2`) — iBOT + KoLeo + proper DINOHead, 224 bulk

**Protocol:**
- **full FT** — unfreeze last 4 blocks, 150 ep (early-stop patience 40) — the real pipeline
- **frozen probe** — unfreeze 0, 25 ep — isolates *representation quality* of the encoder

**Recipe:**
- **simple** — single neck, HRNet, heatmap 128, llrd 1.0, sample_temp 0, canvas loss, dsnt 0, fp16
- **upgraded** — multilevel neck, HRNet, heatmap 148, llrd 0.75, sample_temp 0.5, original loss, dsnt 0.1, bf16

---

## 1. Fold-0, full fine-tune (the decisive comparisons)

Sorted by `challenge_blend`. All fold 0, unfreeze 4, 150 ep.

| Run | SSL | recipe | neck | decoder | s_temp | blend ↓ | MRE | status |
|---|---|---|---|---|---|---|---|---|
| `abl_ep20_simplehead_ml_dv2ep20` | **NEW** ep20 | upgraded | multilevel-**concat** | **simple** | 0.5 | **0.0696** | **25.98** | ✅ done — **best blend** |
| `abl_ep20_simplehead_ml_dv2ep104` | **NEW** ep104 (224 bulk + 518 tail) | upgraded | multilevel-**concat** | **simple** | 0.5 | 0.0717 | 24.61 | ✅ done — best@ep68, early-stopped@ep108 — **best MRE**, 2nd-best blend |
| `abl_ep20_simplehead_ml_dv2tailep60ep5` | **NEW** ep60+5-ep 518 tail (cheap) | upgraded | multilevel-**concat** | **simple** | 0.5 | 0.0722 | 24.86 | ✅ done — ties `ep104` at ~40 fewer Phase-1 bulk epochs |
| `abl_ep20_simplehead_dv2ep20` | **NEW** ep20 | upgraded | single | **simple** | 0.5 | 0.0721 | 27.69 | ✅ done (early-stop ep62; best@23) |
| `abl_nossl_fold0` | none | simple | single | hrnet | 0.0 | 0.0740 | 28.98 | ✅ done |
| `phase2_simple_dv2ep20` | **NEW** ep20 | simple | single | hrnet | 0.0 | 0.0791 | 28.85 | ✅ done |
| `abl_ep20_upgraded_dv2ep20` | **NEW** ep20 | upgraded | multilevel | hrnet | 0.5 | 0.0842 | 26.70 | ✅ done |
| `abl_ep20_upgraded` | old ep20 | upgraded | multilevel | hrnet | 0.5 | 0.0946 | 31.41 | ✅ done |
| `phase2_baseline_fold0_ssl20` | old ep20 | simple | single | hrnet | 0.0 | 0.0973 | 33.95 | ✅ done |
| `phase2_upgraded_fold0` | old ep10 | upgraded | multilevel | hrnet | 0.5 | 0.1006 | 31.12 | ✅ done |
| `phase2_baseline_fold0_ssl10` | old ep10 | simple | single | hrnet | 0.0 | 0.1107 | 40.32 | ✅ done |
| `abl_ep20_sampletemp0` | old ep20 | upgraded | multilevel | hrnet | **0.0** | 0.1135 | 30.05 | ✅ done |
| `abl_ep20_simplehead` | old ep20 | upgraded | single | **simple** | 0.5 | 0.1214 | 47.54 | ✅ done |

Note: `abl_ep20_simplehead_dv2ep20`'s best blend (0.0721) was set at ep23 and selected on
`challenge_blend`; its **MRE kept improving afterward** (27.69 → 26.78 by ep62) while the AvgMAE
half drifted up, so the ep23 "best" caught an AvgMAE-lucky epoch. On `average_mre` it would score
better/later — a reminder that `challenge_blend` selection is noisier than MRE alone.

---

## 2. Fold-0, frozen-encoder probe (isolates SSL representation quality)

All fold 0, unfreeze **0**, 25 ep, upgraded-recipe knobs (multilevel/heatmap148/etc.). Differ only in the encoder.

| Probe | SSL | blend ↓ | MRE |
|---|---|---|---|
| `probe_dv2_ep104` | **NEW ep104 — final, 224 bulk(100) + 518 TAIL** | **0.0747** | 25.09 |
| `probe_dv2_tail_ep60_ep5` | **NEW — 224 bulk(60) + 5-ep 518 tail (fresh heads)** | **0.0748** | 25.23 |
| `probe_dv2_tail_ep60_ep10` | **NEW — 224 bulk(60) + 10-ep 518 tail (fresh heads)** | 0.0755 | 25.50 |
| `probe_dv2_ep20` | **NEW** ep20, 224px (bulk only) | 0.0782 | **24.70** |
| `probe_dv2_ep60` | **NEW** ep60, 224px (bulk only) | 0.0836 | 26.43 |
| `probe_dv2_ep10` | **NEW** ep10, 224px (bulk only) | 0.0872 | 26.88 |
| `probe_nossl` | none | 0.0890 | 31.27 |
| `probe_dv2fullres_ep10` | **NEW** ep10, 518px (full-res, no downsampling) | ⏳ queued | ⏳ queued |
| `probe_dv2fullres_ep20` | **NEW** ep20, 518px (full-res, no downsampling) | 0.0901 | 25.96 |
| `probe_legacy_ep20` | old ep20 | 0.0965 | 33.11 |
| `probe_dv2_ep100` | **NEW ep100 — end of 224 bulk, NO tail** | 0.0974 | 25.95 |
| `probe_dv2fullres_ep30` | **NEW** ep30, 518px (full-res, no downsampling) | 0.0934 | 26.92 |
| `probe_legacy_ep10` | old ep10 | 0.1453 | 47.08 |
| `probe_noreg_nossl` | none, **non-register** backbone (`dinov2_vitl14`) | 0.0976 | 29.34 |

**Bulk-only quality declines monotonically with more 224 training:** ep20 0.0782 < ep60 0.0836 <
ep100 0.0974 — earlier bulk stop = better representation. **But a short 518 tail rescues an early
checkpoint to match the full one:** ep60 + 5-epoch tail (0.0748) ≈ ep104 = ep100 + tail (0.0747),
identical within noise, *despite* the ep60 tail using fresh heads. So the **tail is the dominant
factor, not bulk length** — ~40 bulk epochs can be cut with no downstream loss.

**Longer tails don't help further, and may mildly hurt:** ep60 + 10-ep tail (0.0755) is *not* better
than ep60 + 5-ep tail (0.0748) or the fully-bulk-trained ep104 (0.0747) — all three sit within noise
of each other. So ~5 epochs at 518 is already enough to capture the tail's benefit; epochs 6-10 add
cost with no measurable further gain.

**The 518 high-res tail is a large, clean win:** `ep104` (with tail) vs `ep100` (same run, no tail) —
**0.0747 vs 0.0974 (~23% better)**, isolating exactly the 4-epoch tail's effect. `ep104` is now the
**best frozen probe of any encoder**, ahead of `ep20`. Curiously `ep100` (100 bulk epochs, no tail) is
*worse* than `ep20` (20 bulk epochs) — more 224-only bulk training alone doesn't help past a point;
only the resolution-matched tail does.

**The full-res-only control shows the same non-monotonic bulk pattern:** ep30 (0.0934) is worse than
ep20 (0.0901) — consistent with finding #12 (more bulk-only epochs alone don't help, downsampled or
not) and still well behind the downsampled-bulk-then-tail design at any matched checkpoint.

**Register-backbone ablation (`configs/probe_noreg_nossl.yaml`):** the register backbone
(`dinov2_vitl14_reg`, METHOD_CHANGES.md A5) was adopted on literature rationale (Darcet et al.,
*Vision Transformers Need Registers*, ICLR 2024), not an in-project ablation — the only non-register
run anywhere in the project (`p2_baseline_nossl`) differs in split, batch size, LR, and epoch budget,
so it cannot isolate the backbone choice (see that run's own config comment). `probe_noreg_nossl`
is `probe_nossl` with only `model.backbone.name` swapped to `dinov2_vitl14` — same fold-0 split,
same upgraded-recipe knobs, no SSL, frozen encoder (25 ep, cheap, no Phase-1 re-run needed) — a
clean, matched pair against `probe_nossl` (0.0890 / 31.27) to actually isolate the register effect.
Result: `probe_noreg_nossl` reaches 0.0976 / 29.34px (best at epoch 7, early-stopped) — *worse* than
the register-backbone `probe_nossl` (0.0890 / 31.27) on blend, but the non-register run posts a
noticeably better MRE (29.34px vs 31.27px), so on this matched-pair evidence the register backbone
is not a clean win for this frozen-probe setting; the two metrics disagree on which is "better," consistent with finding #17 (challenge_blend and MRE don't always rank models the same way).

---

## 3. 5-fold CV — upgraded recipe, old ep20 SSL (colleague's sweep)

Upgraded recipe, unfreeze 4, 150 ep, old `multicrop` ep20 encoder.

| Fold | Run | blend ↓ | MRE |
|---|---|---|---|
| 0 | `abl_ep20_upgraded` | 0.0946 | 31.41 |
| 1 | `abl_ep20_upgraded_fold1` | 0.1029 | 30.21 |
| 2 | `abl_ep20_upgraded_fold2` | 0.0829 | 26.32 |
| 3 | `abl_ep20_upgraded_fold3` | 0.0851 | 26.63 |
| 4 | `abl_ep20_upgraded_fold4` | 0.1070 | 36.00 |
| **mean** | | **0.0945** | **30.11** |

---

## 3b. 5-fold CV — champion recipe, NEW ep104 SSL (in progress)

Champion recipe (multilevel-concat simple decoder, unfreeze 4, 150 ep), NEW `ep104` encoder.
This is the "confirm across folds before ensembling/submitting" sweep.

| Fold | Run | blend ↓ | MRE | status |
|---|---|---|---|---|
| 0 | `abl_ep20_simplehead_ml_dv2ep104` | 0.0717 | 24.61 | ✅ done |
| 1 | `abl_ep20_simplehead_ml_dv2ep104_fold1` | 0.0846 | 23.64 | ✅ done |
| 2 | `abl_ep20_simplehead_ml_dv2ep104_fold2` | 0.0665 (so far) | 23.45 | 🔄 running — epoch 124/150, still setting new bests, train_loss not plateaued (may not fully converge by the 150-ep cap) |
| 3 | `abl_ep20_simplehead_ml_dv2ep104_fold3` | — | — | queued |
| 4 | `abl_ep20_simplehead_ml_dv2ep104_fold4` | — | — | queued |

Spread so far 0.067–0.085 (±13%) — the fold-to-fold noise that makes any single-fold ranking
unreliable. Trust the 5-fold **mean**, not fold 0.

Fold 2, still running, is already the best single-fold result in the project (0.0665 <
the fold-0 champion's 0.0696) — same leaky-split caveat applies (finding #18): comparable
internally, not proof of anything vs. Codabench.

---

## 4. Reference (NOT fold-0 comparable)

| Run | note | blend | MRE |
|---|---|---|---|
| `p2_baseline_nossl` | **holdout split** (not fold 0; 81% of its val is in fold-0 train) — do not rank against §1 | 0.0770 | 23.85 |
| `smoke_p2` | CPU smoke test (dummy) — ignore | 0.2656 | 289.81 |

---

## 5. External (Codabench) submissions & the internal↔external mismatch ⚠️

Submissions built on the real challenge val set (`data/data/val_data`, unlabeled), single-model +
multi-scale/intensity TTA, via `gubiometry predict`:

| Zip | model | note |
|---|---|---|
| `regression_predictions.zip` (Jul 13) | old **no-SSL HRNet** | **best external score so far** |
| `submission_fold0_intermediate.zip` (Jul 16) | fold-0 intermediate | — |
| `submission_dv2ep104.zip` (Jul 27) | `abl_ep20_simplehead_ml_dv2ep104` | worse externally than the no-SSL zip |
| `submission_dv2ep20.zip` (Jul 27) | `abl_ep20_simplehead_ml_dv2ep20` | worse than the ep104 zip externally |

**The internal ranking does NOT transfer to Codabench.** Internally (fold 0) the order is
SSL-ep20 > SSL-ep104 > no-SSL; externally it is roughly **no-SSL > SSL-ep104 > SSL-ep20** — both
optimism *and* a reordering. Diagnosed causes (see findings #16–18): video-frame **leakage** in the
splits, a **proxy** selection metric, and SSL **domain over-specialization**. Submission format is
clean (identical schema across all four zips), so this is a generalization/objective problem, not a
pipeline bug.

---

## Key findings

1. **The old SSL hurts.** Every legacy-`multicrop` full-FT run loses to no-SSL (`abl_nossl` 0.0740). The frozen probe shows why: the legacy encoder is *worse than off-the-shelf* (probe legacy ep10 0.145 / ep20 0.097, both above no-SSL 0.089) — the weak head degraded DINOv2's features.
2. **The new SSL fixed the encoder.** Frozen probe: NEW ep20 (0.0782, MRE **24.70**) is the **best representation of all** — beats off-the-shelf (0.0890) and improves ep10→ep20 (0.0872→0.0782). Its frozen MRE even beats the best *fine-tuned* model.
3. **But full fine-tuning erases most of that edge.** With unfreeze-4 + 150 ep, NEW-ep20 in the *simple* recipe (0.0791) lands just behind no-SSL (0.0740). A better starting point ≠ better fine-tuned optimum here.
4. **Champion (fold 0, blend): `abl_ep20_simplehead_ml_dv2ep20` — 0.0696 / MRE 25.98.**
   NEW ep20 + **simple (ViTPose) decoder that CONCATENATES 4 DINOv2 depths** (multi-level concat).
   Best `challenge_blend` of everything, and clearly beats no-SSL (0.0740 / 28.98). The same recipe
   on the later `ep104` encoder (224 bulk + 518 tail) is very close behind on blend (0.0717) and
   actually **wins on MRE** (24.61 vs 25.98) — the two encoders are close to a wash on full fine-tune,
   unlike the much larger gap seen frozen (finding #13).
5. **The winning architecture = multi-level features + a *simple concat* decoder, NOT the heavy HRNet
   sum.** The three multi-level runs on the new encoder rank: HRNet-sum 0.0842 > simple-single 0.0721
   > **simple-concat 0.0696**. So (a) multi-level depth helps, (b) a minimal decoder consuming it
   (ViTPose lesson) beats HRNet's fusion, and (c) **concatenation beats summation** for the fusion.
6. **New SSL beats old SSL, cleanly confirmed in the full pipeline:** identical upgraded-HRNet recipe,
   `abl_ep20_upgraded_dv2ep20` (new ep20) **0.0842 / MRE 26.70** vs `abl_ep20_upgraded` (old ep20)
   0.0946 / 31.41. The redesign pays off with everything else held fixed.
7. **Old SSL was worse than no-SSL** (frozen probe: legacy ep10/ep20 both above off-the-shelf) — the
   weak legacy head degraded DINOv2's features; the *same* simple decoder was **worst** with old SSL
   (0.1214) and **best** with new SSL (0.0696). The earlier "simple decoder loses" verdict was purely
   a bad-encoder confound.
8. **`sample_temp`** is a tradeoff, not a harm: balanced (0.0) gives best MRE (30.05) but worse blend.
9. **Metric nuance:** `challenge_blend` selection is noisier than MRE (it compounds MRE + a shakier
   AvgMAE half) — several runs' best-blend epoch caught an AvgMAE-lucky point while MRE kept improving.
10. **224px-only bulk SSL beats 518px-only full-res SSL at equal epoch count (ep20 vs ep20).** 224px
    `probe_dv2_ep20` **0.0782/24.70** vs 518px `probe_dv2fullres_ep20` 0.0901/25.96 — so the
    224-bulk-then-tail *design* isn't only ~4x cheaper per epoch, it's also more sample-efficient
    early on.
11. **The 518 high-res tail is a large, clean win — do not skip it.** Same run, only the last 4 epochs
    differ: `probe_dv2_ep104` (224 bulk + 518 tail) **0.0747** vs `probe_dv2_ep100` (100 bulk epochs,
    no tail) 0.0974 — **~23% better** from just 4 epochs at 518. `ep104` is now the **best frozen
    probe of any encoder tested**, ahead of `ep20` (0.0782). This matches DINOv2's own recipe design
    (brief high-res tail after extensive low-res pretraining) and confirms implementing it was worth it.
12. **More 224-only bulk epochs alone stop helping (and can hurt) past a point.** `ep100` (100 bulk
    epochs, no tail) scores *worse* (0.0974) than `ep20` (20 bulk epochs, 0.0782) — bulk-only training
    plateaus/wobbles; only the resolution-matched tail reliably improves it further (finding #11).
13. **Full fine-tuning compresses the frozen-probe gap between checkpoints.** Frozen, `ep104` beats
    `ep20` by a lot (0.0747 vs 0.0782). Fully fine-tuned (champion recipe, unfreeze 4, 150 ep), the
    two are close: `ep20` 0.0696/25.98 vs `ep104` 0.0717/24.61 — `ep104` wins MRE, `ep20` wins blend,
    both within noise of each other. Encoder-quality differences that show up frozen partly wash out
    once the backbone itself is allowed to adapt during Phase 2.
14. **Longer 518 tails (5→10 epochs) don't help further, and may mildly hurt.** Frozen probe:
    ep60+5-ep tail (0.0748) ≈ ep60+10-ep tail (0.0755) ≈ the fully-bulk-trained ep104 (0.0747) — all
    within noise. ~5 epochs at 518 already captures the tail's benefit.
15. **The full-res-only control (`phase1_dinov2_fullres`) shows the same non-monotonic bulk pattern**
    as the downsampled run: frozen probe ep30 (0.0934) is worse than ep20 (0.0901) — more bulk-only
    epochs alone don't help regardless of resolution, and this line remains behind the
    downsampled-bulk-then-tail design at every matched checkpoint so far.
16. **⚠️ Internal validation is LEAKY (biggest external-transfer problem).** Splits are per-image
    `StratifiedKFold` (task-stratified only, `data/splits.py`) but the cardiac data is video frames
    (`DCM_IM_0008_frame061.png`): A4C/IVC/PLAX/PSAX have 1.8–2.9 frames per loop. Measured in fold 0:
    **45% of A4C and 24% of PLAX val images share a loop with a training frame.** Because the metric
    is macro-averaged (equal weight per task), those ~2 tasks are internally inflated. An SSL encoder
    pretrained on the internal set memorizes loop/patient appearance → gains more from the leak
    internally, over-specializes → loses externally. **Fix: loop-level `GroupKFold`.** (Literature:
    image-level vs case-level splits are the canonical cause of over-optimistic medical-imaging val.)
17. **⚠️ `challenge_blend` is a PROXY, not the official metric.** It normalizes each task by a
    home-made scale (median image diagonal for MRE, median |gt-measurement| for the measurement half);
    `metrics.py` states the official normalizer (clinical-tolerance/IQR) is unpublished. So per-task
    weighting differs from Codabench — a model can win our blend and lose officially with zero leakage.
    `average_mre` (raw px) IS reproduced exactly and is more trustworthy than the blend for ranking.
18. **⚠️ SSL wins internally / loses externally is consistent across three independent axes:** leakage
    (#16), proxy metric (#17), and domain over-specialization (finding #3 already showed full-FT erases
    most of SSL's edge). The best *external* model remains the plain **no-SSL HRNet** with the legacy
    loss (canvas L1, `dsnt=0`, `measurement_lambda=0` — i.e. only the MRE half of the metric is
    optimized). Do not rank encoders/recipes on fold-0 blend until splits are loop-level AND validated
    on the leaderboard.

## Recommended recipe so far (fold 0)
**NEW-SSL encoder + multilevel `input_mode` + `decoder: simple` (concat) + full fine-tune (unfreeze 4)
+ upgraded knobs (heatmap 148, llrd 0.75, sample_temp 0.5, original loss, dsnt 0.1, bf16).** =
`abl_ep20_simplehead_ml` config. Two full-FT results now exist and are close: `ep20` **0.0696 / 25.98**
(best blend) and `ep104` (224 bulk + 518 tail) **0.0717 / 24.61** (best MRE) — pick by which metric
the submission should optimize; `ep104` is the more expensive checkpoint to produce (104 vs 20 Phase-1
epochs) for a blend result that's currently *slightly worse*, so `ep20` remains the practical default
until this is confirmed across folds.

## Currently running / idle (as of 2026-07-28)
- 🔄 `abl_ep20_simplehead_ml_dv2ep104_fold{1..4}` sweep (§3b) — fold1 done, fold2 ~ep90/150, fold3/4
  queued. GPU 2.
- ✅ `abl_ep20_simplehead_ml_dv2tailep60ep5` finished — 0.0722 / 24.86 (now in §1).
- ✅ `predict_challenge_dv2ep104` / `_dv2ep20` finished — zips written (§5).
- `phase1_dinov2_fullres` (518-throughout control) **not running** (evicted mid-ep31); last checkpoint
  `dinov2_adapted_ep30.pth`. Resume with `-o resume=…/checkpoints/latest_checkpoint.pth` if still wanted.

## Untested / next
- **Fix the splits first: loop-level `GroupKFold`** (finding #16) — prerequisite for any trustworthy
  internal ranking; then re-score existing models on leak-free folds.
- **Loss on the no-SSL HRNet base** (external winner): the measurement half of the metric gets zero
  gradient today → try `measurement_lambda>0` (primary), `coord_loss=wing` (MRE half), `loss_space=original`.
  Validate on group splits + leaderboard, NOT fold-0 blend.
- **Ensemble across leak-free folds + TTA** for the real submission (not single fold-0).
- Finish the ep104 5-fold sweep (§3b) and report the mean.
- NEW-**fullres** encoder — Phase-1 stalled at ep30; no Phase-2 uses it yet.
