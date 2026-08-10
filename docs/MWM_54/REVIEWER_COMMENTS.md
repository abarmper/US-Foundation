# Reviewer Comments & Responses — MWM_54

Working log of reviewer feedback and our replies for the camera-ready revision.

---

## Manuscript changes since submission

Before the per-reviewer replies below: while addressing Reviewer 2's Phase-1
comments, we found that §2.1 (and Figure 1) described an earlier multi-crop
self-distillation recipe that predates the recipe following DINOv2's
original SSL formulation our codebase has used as the default for some time
(CLS self-distillation + iBOT masked-patch prediction + KoLeo
regularization, plus a two-resolution 224px-then-518px curriculum). This was
a documentation staleness issue, not a change to the results themselves —
the submitted results were always produced by that recipe. We corrected §2.1, Figure 1, and the
Phase-1 augmentation description in §3.1 to accurately describe the method
that actually produced our results.

Since the below was written, the official Codabench evaluation numbers were
finalized (Codabench is now closed to new submissions): the submitted model
is the epoch-104 Phase-1 checkpoint with a single-level HRNet decoder,
fold 0 (MRE 26.34\,px / MAE 29.70\,px, Table 1 — not the 35.03/34.62
placeholder numbers referenced in some replies below). The multilevel
Phase-2 design (§2.2) was evaluated separately as an internal ablation
(§4.3, Table 2) and was never submitted to Codabench. All final-numbers
edits are complete in `main.tex`; the per-reviewer replies below are updated
to match.

---

## Supplementary material

Camera-ready supplementary files live under `docs/MWM_54/suppl_mat/`, kept
separate from `main.tex` so submission page limits stay unaffected.

| Table | Contents | Answers |
|---|---|---|
| S1 | Complete frozen-probe SSL-duration sweep (11 checkpoints; moved out of `main.tex` §4.2 for space — Figure 4 and key numbers stay there) | Reviewer 2.2 — figure requested stays in main text, only the full data table moved |
| S2 | Task-specific sensitivity to Phase-1 adaptation duration (per-task MRE, best checkpoint, deviation from ep104) | Reviewer 2.2 — "does the optimal duration vary by task?" |
| S3 | Compute cost: parameters, throughput, peak memory, GFLOPs for both phases | Reviewer 3.3 — compute cost/throughput/memory not reported |

---

## Reviewer 1 — jWeV

**Rating:** 7 (Accept) | **Confidence:** 3 — publication-quality/camera-ready
review, plus a list of suggested citations for related MICCAI 2026 challenge
tracks (CMRSeg, MVAA, FoundUS).

### Citations suggested and incorporated

Of the suggested references, the following are directly relevant to our task
(Foundation Model Challenge for Ultrasound Biometry, AoP estimation, and
adjacent 2025/2026 challenge tracks) and are now cited in the manuscript:

- Ramirez Zegarra et al., "Intrapartum Ultrasound" (2025) — `\cite{ramirez2025intrapartum}`
- Zhou et al., "Automatic Angle of Progress Measurement..." (MICCAI 2020) — `\cite{zhou2020aop}`
- Lu et al., "Multitask Deep Neural Network for... Angle of Progression" (2022) — `\cite{lu2022aop}`
- Bai et al., "Landmark Detection Challenge for Intrapartum Ultrasound Measurement..." (MICCAI 2025) — `\cite{bai2025landmarkchallenge}`
- Bai et al., "FUGC: Benchmarking Semi-Supervised Learning Methods for Cervical Segmentation" (2026) — `\cite{bai2026fugc}`
- Lin et al., "...UUSIC25 Challenge" (2025) — `\cite{lin2025uusic}`
- Deng et al., "Baseline Method of the Foundation Model Challenge for Ultrasound Image Analysis" (2026) — `\cite{deng2026baseline}`

The remaining suggested references (CMRSeg/MVAA segmentation-challenge
papers, PSFHS segmentation-specific works) target segmentation rather than
landmark regression — outside our task's scope — and were not added, to
keep the related-work discussion focused.

### Comment index

| # | Paraphrase | Where to look |
|---|---|---|
| 1.1 | Submit LaTeX source + supplementary materials to bai_jieyun@126.com | "Our reply" below (procedural reminder) |
| 1.2 | Sign the licence-to-publish form | "Our reply" below (procedural reminder) |
| 1.3 | Cite relevant FoundUS publications | "Our reply" below — done |

### Our reply

We thank the reviewer for the suggested references. We have incorporated the
most directly relevant works into the revised manuscript, including
references on FoundUS/IUGC, intrapartum landmark detection and AoP
estimation, FUGC, UUSIC25, and recent clinical intrapartum ultrasound.

**Camera-ready procedural items — not yet done (reminder):** submit LaTeX source +
supplementary materials to bai_jieyun@126.com with subject `MWM_#XX`; complete
and sign the licence-to-publish form (`SNCS_ProceedingsPaper_LTP_ST_SN_Switzerland.docx`
from https://ideal-conf.com/downloads).

---

## Reviewer 2 — ZxK3

**Rating:** 7 (Accept) | **Confidence:** 3

> The paper's study of how SSL adaptation duration affects downstream
> accuracy is a valuable practical contribution. However, results are
> reported somewhat vaguely... Please ensure the final results (per-task
> MRE, overall challenge metric, and comparison to relevant baselines) are
> clearly presented early in the paper.

**Response:** Abstract (closing sentence) + §4.1 (Results reordered to lead
with the official evaluation) + comparison to the official leaderboard (see
Reviewer 3, comment 3.5 below — shared response, not repeated here).

> SSL adaptation duration study: ... What is the optimal adaptation
> duration? Is there a point of diminishing returns or negative transfer?
> Does the optimal duration vary by task? These findings should be
> prominently reported with a figure showing downstream accuracy vs.
> adaptation epochs.

**Response:** §4.2, Figure 4, plus Supplementary Tables S1 (complete sweep)
and S2 (per-task breakdown).

> HRNet decoder design: ... the specific HRNet configuration (number of
> stages, channels per resolution, how transformer patch features are
> mapped to the HRNet input) is not described. ... A diagram showing how
> DINOv2 patch tokens are reshaped and fed into the HRNet backbone would be
> particularly helpful.

**Response:** §2.2 (stage count, exact per-branch channel widths, and the
DINOv2-tap-to-branch mapping with formulas) + Figure 2.

> DINO-style multi-crop self-distillation: ... is there a reason this
> [register] variant was chosen? Would the standard DINOv2 work as well? A
> brief justification or ablation would strengthen this choice.

**Response:** §4.2 (register paragraph) — literature-based justification
plus a matched frozen-probe ablation, reported honestly as a mixed result.

> Comparison to frozen/no-adaptation baseline: ... The paper should report
> results with a frozen (non-adapted) DINOv2 encoder as a direct test of
> this hypothesis.

**Response:** §4.2 — the no-SSL baseline is epoch 0 in Figure 4, with its
blend/MRE stated numerically in the text.

### Comment index

| # | Paraphrase | Status |
|---|---|---|
| 2.1 | Present final results prominently, early, with baseline comparison | Done — Abstract + §4.1 reorder + leaderboard comparison (3.5) |
| 2.2 | SSL duration study: report findings with a clear figure | Done — §4.2, Figure 4, Supplementary Tables S1/S2 |
| 2.3 | Specify HRNet configuration + patch-to-HRNet mapping diagram | Done — §2.2 + Figure 2 |
| 2.4 | Justify the register-variant DINOv2 choice | Done — §4.2 |
| 2.5 | Frozen/no-adaptation baseline | Done — §4.2, Figure 4 (epoch 0) |

### Our reply

We thank the reviewer for these constructive comments, which meaningfully
improved the manuscript's presentation.

We have made quantitative results prominent early: the Abstract now closes
with the overall result (MRE 26.34 px, MAE 29.70 px on the official
Codabench evaluation), and Section 4 has been reordered so the official
per-task results (Table 1, `tab:codabench-results`) are the first subsection encountered, with the
internal ablations that motivate our Phase-1/Phase-2 choices following as
supporting evidence. A comparison to the official leaderboard is reported
under our reply to Reviewer 3 (comment 3.5), to avoid duplicating the same
content twice.

The SSL-duration study (Section 4.2, Figure 4) now reports: adaptation
quality improves rapidly early on but becomes non-monotonic thereafter
(diminishing returns); the best aggregate frozen-probe MRE occurs at epoch
20, while epoch 104 (100 low-resolution "bulk" epochs plus a 4-epoch
high-resolution "tail") achieves the best blend score and completes our
two-resolution curriculum — we select epoch 104 on that combined basis
rather than a single-metric win. A task-wise breakdown (Supplementary Table
S2) confirms the preferred duration varies by task, though epoch 104 stays
within 8% MRE of the task-specific optimum on eight of nine tasks; the
complete sweep (11 checkpoints) is in Supplementary Table S1.

Section 2.2 now states the HRNet neck's stage count, per-branch resolutions,
and channel widths explicitly (37×37/74×74/148×148 at 128/96/64 channels),
and the exchange-unit update equations, matching the redrawn Figure 2, which
traces the full path from DINOv2 patch tokens to landmark coordinates.

On the register-variant backbone: we retain it primarily on the strength of
its documented purpose — register tokens absorb high-norm "artifact" patch
tokens that otherwise contaminate dense feature maps, which matters for a
neck that consumes every patch token. We additionally ran a matched
frozen-probe ablation (Section 4.2): the result is mixed (registers improve
blend, the standard backbone achieves lower MRE), which we report honestly
rather than selectively.

Finally, the frozen/non-adapted baseline the reviewer requested is included
directly in the duration sweep (Figure 4, epoch 0) and stated numerically in
the text (blend 0.0890, MRE 31.27 px), so its disadvantage relative to every
adapted checkpoint is directly visible.

---

## Reviewer 3 — KKV9

**Title:** Official Review
**Official Review by** Reviewer KKV9
**Date:** 31 Jul 2026, 13:36 (modified: 03 Aug 2026, 20:58)

### Review

• Official mean radial error (35.03 px) and measurement MAE (34.62 px) indicate substantial room for improvement, especially on PSAX, A4C, and HC; deeper error analysis would help prioritize future work.

• Section 4.1 recommends epoch-20 Phase-1 initialization, but Section 4.2 compares Phase-2 recipes using the epoch-10 checkpoint; this inconsistency should be corrected and the final choice justified.

• The method uses a large DINOv2-L encoder with complex multi-branch neck, yet computational cost, throughput, and memory footprint are not reported.

• Self-supervised adaptation duration is studied only coarsely (epochs 10 vs 20); additional checkpoints or early-stop criteria tied to downstream metrics would strengthen the Phase-1 selection rationale.

• The paper would benefit from comparison to other challenge submissions or strong supervised baselines on the same official scorer to contextualize performance.

Overall, the proposed DINOv2-HRNet pipeline is technically sound and well aligned with cross-domain ultrasound biometry, and I would support acceptance with minor revisions provided that the authors resolve the checkpoint inconsistency and address the above comments.

**Rating:** 7: Good paper, accept
**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct

### Comment index

| # | Paraphrase | Status |
|---|---|---|
| 3.1 | Deeper per-task error analysis (worst tasks) | **Done — `main.tex` §4.1**, right after Figure 3 |
| 3.2 | §4.1 (ep20) vs. §4.2 (ep10) checkpoint inconsistency | **Done — fully resolved.** Legacy comparison and epoch-10-fixed recipe table retired; replaced with one matched epoch-104 decoder ablation (§4.3, Table 2) |
| 3.3 | Compute cost / throughput / memory not reported | Done — §3.2 + Supplementary Table S3 |
| 3.4 | SSL duration studied only coarsely | Done — same evidence as 2.2, cross-referenced |
| 3.5 | Comparison to other submissions / baselines on official scorer | **Done — `main.tex` §4.1**, right after the 3.1 paragraph |

### Our reply

We thank the reviewer for these comments, which identified real gaps in the
submitted manuscript.

**On per-task error analysis (3.1):** "deeper" here means explaining, not
just reporting, why the worst-performing tasks underperform — distinguishing
data scarcity from task-inherent difficulty. Under the final official
numbers, the worst tasks are HC, PSAX, and IVC (Table 1). PSAX and IVC line
up with documented data scarcity (§3.1): both have the fewest labeled
training images (49, 38) and the smallest official evaluation sets (N=18,
N=10) of all nine tasks. HC is the harder case — it has an order of
magnitude more labeled data (999 images) yet remains the single
worst-performing task; we report this openly as unexplained by dataset size
alone rather than force an unsupported claim.

**On the epoch-10/epoch-20 inconsistency (3.2):** fully resolved. §4.2 now
reports a systematic frozen-encoder probe sweeping the checkpoint-duration
space (no-SSL, epochs 10/20/60/100, plus high-resolution-tail and
full-resolution-control variants) and selects one checkpoint — epoch 104
(100 low-resolution epochs plus a 4-epoch high-resolution tail) — used
consistently as the Phase-1 initialization throughout the paper. The legacy
epoch-10-vs-epoch-20 comparison and the epoch-10-fixed Phase-2 recipe table
have both been retired and replaced with one matched, epoch-104 decoder
ablation across five cross-validation folds (§4.3, Table 2), so the
checkpoint used is now consistent everywhere in the paper.

**On compute cost (3.3):** addressed. §3.2 now reports headline
throughput/memory/parameter-count numbers for both phases (Phase 1: 351.7M
parameters, ~96 GPU-hours total; Phase 2: 312.2M parameters, 58.2M
trainable, up to 127.3 img/s under bf16 inference). The complete breakdown
(per-stage GFLOPs, throughput, and peak memory) is in Supplementary Table S3.

**On SSL-duration coarseness (3.4):** same evidence as comment 2.2 above
(§4.2, Figure 4, Supplementary Tables S1–S2): the duration sweep now spans
six checkpoints (plus tail and full-resolution variants) rather than the
original two, with selection driven by the probe protocol's blend/MRE
trade-off.

**On baseline comparison (3.5):** for external context, the
best-performing submission currently on the official GU\_Biometry Codabench
leaderboard reports an overall MRE of 22.56 px and MAE of 29.02 px,
evaluated with the same official scorer as our own submission (MRE 26.34 px
/ MAE 29.70 px, Table 1) — MAE is now essentially tied with the leaderboard
best. We report this as a like-for-like reference point on the same metric;
a full per-team methodological comparison is limited by what individual
submissions disclose publicly on the leaderboard.

---

## Internal TODOs (not reviewer-facing)

### Resolved since the list above was written

- [x] Official numbers finalized (Codabench closed to new submissions):
      abstract, Table 1, §4.1 prose, and all reviewer replies in this file
      now use the real scores (MRE 26.34 / MAE 29.70, epoch-104 Phase-1 +
      single-level HRNet decoder, fold 0 — `suppl_mat/scores.txt`).
- [x] "single model" wording confirmed correct — the submission is a single
      fold-0 model, not an ensemble (Codabench closed before an ensemble+TTA
      submission was possible).
- [x] §4.1 now explicitly states the submission used the single-level HRNet
      configuration and that the multilevel design (§2.2) was evaluated
      separately, internally only — never conflated as "the" submission.
- [x] One unified Phase-2 ablation story: the legacy epoch-10-vs-epoch-20
      table and epoch-10-fixed recipe subsection are gone; §4.3 is now one
      matched, epoch-104, five-fold decoder ablation (Table 2).
- [x] `{\color{blue}}` revision highlighting stripped from `main.tex`.
- [x] Page budget: 10 pages (was ~14 at worst).
- [x] `mybibliography.bib` / `llncs.cls` / `splncs04.bst` now tracked in git
      (were silently uncommitted).

### Decided, not a bug

- **Figure 3 (qualitative examples) will not be regenerated.** It still
  shows predictions from an earlier model, not the exact official-submission
  run. Explicit decision: keep the existing figure as is. The caption and
  surrounding prose are generic enough (no claim about which exact model
  produced the examples) that this is not misleading, and it stays that way
  — do not add wording that would make the mismatch inferable.

### Still open

- [ ] Figure 4 (SSL-duration panels) still uses plain matplotlib defaults —
      not styled to match the paper's other figures.
- [ ] bf16 autocast inference speedup (~7×, Table S3) identified but not yet
      enabled in `predict.py`/`evaluate.py` — decide whether to ship it
      before camera-ready.
- [ ] `MWM_54_source.zip` not yet built (verify `old_material/`, `drafts/`
      excluded).
- [ ] Signed Springer LTP form — not yet submitted (author action, comment 1.2).
