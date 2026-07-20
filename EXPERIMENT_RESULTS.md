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
| `abl_ep20_simplehead_dv2ep20` | **NEW** ep20 | upgraded | single | **simple** | 0.5 | **0.0738*** | 27.65 | ⏳ running ep21 |
| `abl_nossl_fold0` | none | simple | single | hrnet | 0.0 | **0.0740** | 28.98 | ✅ done |
| `phase2_simple_dv2ep20` | **NEW** ep20 | simple | single | hrnet | 0.0 | 0.0791 | 28.85 | ✅ done |
| `abl_ep20_upgraded` | old ep20 | upgraded | multilevel | hrnet | 0.5 | 0.0946 | 31.41 | ✅ done |
| `phase2_baseline_fold0_ssl20` | old ep20 | simple | single | hrnet | 0.0 | 0.0973 | 33.95 | ✅ done |
| `phase2_upgraded_fold0` | old ep10 | upgraded | multilevel | hrnet | 0.5 | 0.1006 | 31.12 | ✅ done |
| `phase2_baseline_fold0_ssl10` | old ep10 | simple | single | hrnet | 0.0 | 0.1107 | 40.32 | ✅ done |
| `abl_ep20_sampletemp0` | old ep20 | upgraded | multilevel | hrnet | **0.0** | 0.1135 | 30.05 | ✅ done |
| `abl_ep20_simplehead` | old ep20 | upgraded | single | **simple** | 0.5 | 0.1214 | 47.54 | ✅ done |

\* `abl_ep20_simplehead_dv2ep20` is **still running** (ep21/150) — 0.0738 is best-so-far, provisional.

---

## 2. Fold-0, frozen-encoder probe (isolates SSL representation quality)

All fold 0, unfreeze **0**, 25 ep, upgraded-recipe knobs (multilevel/heatmap148/etc.). Differ only in the encoder.

| Probe | SSL | blend ↓ | MRE |
|---|---|---|---|
| `probe_dv2_ep20` | **NEW** ep20 | **0.0782** | **24.70** |
| `probe_dv2_ep10` | **NEW** ep10 | 0.0872 | 26.88 |
| `probe_nossl` | none | 0.0890 | 31.27 |
| `probe_legacy_ep20` | old ep20 | 0.0965 | 33.11 |
| `probe_legacy_ep10` | old ep10 | 0.1453 | 47.08 |

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
4. **Open / promising:** NEW-ep20 with the **simple decoder + upgraded knobs** (`abl_ep20_simplehead_dv2ep20`) is *provisionally* at 0.0738 (ep21, still running) — the first full-FT run to edge past `abl_nossl`. Watch it converge before concluding.
5. **`sample_temp`** is a tradeoff, not a harm: balanced (0.0) gives best MRE (30.05) but worse blend.
6. **Champion so far (fold 0):** `abl_nossl_fold0` **0.0740** — but the new-SSL runs are now competitive, unlike the old ones.

## Untested combinations (with NEW SSL)
- **Upgraded recipe + HRNet + full FT + NEW ep20** — the natural "best recipe on best encoder" run; only exists *frozen* so far.
- Intermediate unfreeze depth (6/8), `coord_loss=wing`, `measurement_lambda>0`, folds 1–4, with NEW SSL.
- `abl_ep20_simplehead_ml` (simple decoder, multi-level concat) — never run at all.
- NEW-**fullres** encoder (`phase1_dinov2_fullres`, 518-throughout) — Phase-1 still training (~ep15); no Phase-2 uses it yet.
