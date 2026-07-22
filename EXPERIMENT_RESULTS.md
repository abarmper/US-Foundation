# Phase-2 Experiment Results

GU/FU Biometry — Phase-2 landmark runs. Metric: **`challenge_blend`** (0.5·MRE + 0.5·measurement-MAE proxy, **lower = better**) and **MRE** (mean radial error, original px). Best checkpoint selected on `challenge_blend`. Snapshot: 2026-07-20.

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
| `abl_ep20_simplehead_ml_dv2ep20` | **NEW** ep20 | upgraded | multilevel-**concat** | **simple** | 0.5 | **0.0696** | **25.98** | ✅ done — **best (both metrics)** |
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
| `probe_dv2_ep104` | **NEW ep104 — final, 224 bulk + 518 TAIL** | **0.0747** | 25.09 |
| `probe_dv2_ep20` | **NEW** ep20, 224px (bulk only) | 0.0782 | **24.70** |
| `probe_dv2_ep10` | **NEW** ep10, 224px (bulk only) | 0.0872 | 26.88 |
| `probe_nossl` | none | 0.0890 | 31.27 |
| `probe_dv2fullres_ep20` | **NEW** ep20, 518px (full-res, no downsampling) | 0.0901 | 25.96 |
| `probe_legacy_ep20` | old ep20 | 0.0965 | 33.11 |
| `probe_dv2_ep100` | **NEW ep100 — end of 224 bulk, NO tail** | 0.0974 | 25.95 |
| `probe_legacy_ep10` | old ep10 | 0.1453 | 47.08 |

**The 518 high-res tail is a large, clean win:** `ep104` (with tail) vs `ep100` (same run, no tail) —
**0.0747 vs 0.0974 (~23% better)**, isolating exactly the 4-epoch tail's effect. `ep104` is now the
**best frozen probe of any encoder**, ahead of `ep20`. Curiously `ep100` (100 bulk epochs, no tail) is
*worse* than `ep20` (20 bulk epochs) — more 224-only bulk training alone doesn't help past a point;
only the resolution-matched tail does.

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

## 4. Reference (NOT fold-0 comparable)

| Run | note | blend | MRE |
|---|---|---|---|
| `p2_baseline_nossl` | **holdout split** (not fold 0; 81% of its val is in fold-0 train) — do not rank against §1 | 0.0770 | 23.85 |
| `smoke_p2` | CPU smoke test (dummy) — ignore | 0.2656 | 289.81 |

---

## Key findings

1. **The old SSL hurts.** Every legacy-`multicrop` full-FT run loses to no-SSL (`abl_nossl` 0.0740). The frozen probe shows why: the legacy encoder is *worse than off-the-shelf* (probe legacy ep10 0.145 / ep20 0.097, both above no-SSL 0.089) — the weak head degraded DINOv2's features.
2. **The new SSL fixed the encoder.** Frozen probe: NEW ep20 (0.0782, MRE **24.70**) is the **best representation of all** — beats off-the-shelf (0.0890) and improves ep10→ep20 (0.0872→0.0782). Its frozen MRE even beats the best *fine-tuned* model.
3. **But full fine-tuning erases most of that edge.** With unfreeze-4 + 150 ep, NEW-ep20 in the *simple* recipe (0.0791) lands just behind no-SSL (0.0740). A better starting point ≠ better fine-tuned optimum here.
4. **Champion (fold 0, both metrics): `abl_ep20_simplehead_ml_dv2ep20` — 0.0696 / MRE 25.98.**
   NEW ep20 + **simple (ViTPose) decoder that CONCATENATES 4 DINOv2 depths** (multi-level concat).
   Best of everything on `challenge_blend` *and* MRE, and clearly beats no-SSL (0.0740 / 28.98).
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

## Recommended recipe so far (fold 0)
**NEW-SSL encoder (prefer the final `ep104` — 224 bulk + 518 tail, currently the best-probed
representation) + multilevel `input_mode` + `decoder: simple` (concat) + full fine-tune (unfreeze 4) +
upgraded knobs (heatmap 148, llrd 0.75, sample_temp 0.5, original loss, dsnt 0.1, bf16).** =
`abl_ep20_simplehead_ml` config. Best full-FT result so far used `ep20` and scored **0.0696 / 25.98**;
`ep104` was only probed frozen (0.0747) — has not yet been full-fine-tuned with the champion recipe.

## Untested / next
- **Full fine-tune the champion recipe on `ep104`** (not yet done — only frozen-probed; likely the
  single most promising next run given `ep104` beats `ep20` frozen).
- **Confirm across folds 1–4** with the champion recipe (needed before any ensemble/submission).
- **Later encoder checkpoint** (ep60 exists, SSL still improving) with the champion recipe — likely more headroom.
- Intermediate unfreeze depth (6/8), `coord_loss=wing`, `measurement_lambda>0`, with NEW SSL.
- NEW-**fullres** encoder (`phase1_dinov2_fullres`, 518-throughout) — Phase-1 still training; no Phase-2 uses it yet.
