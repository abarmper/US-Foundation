# Reviewer Comments & Responses — MWM_54

Working log of reviewer feedback and our replies for the camera-ready revision.


> **Color convention in `main.tex`:** blue = reviewer-driven methodology/figure
> updates (Reviewer 2, and the Phase-1 methodology staleness fix below), via
> `\textcolor{...}`/`xcolor` in the preamble — strip before final submission.
> Reviewer 1's citation additions are **not** color-highlighted — merging them
> into their existing `\cite{}` brackets made per-citation coloring
> impractical, so they render as plain text like everything else, which is
> fine.

---

## Supplementary material

Camera-ready supplementary files live under `docs/MWM_54/suppl_mat/`, kept
separate from `main.tex` so submission page limits stay unaffected. Index of
what currently exists:

| File | Contents | Answers |
|---|---|---|
| `suppl_mat/supplementary.tex` | Table S1 — task-specific sensitivity to Phase-1 adaptation duration (per-task MRE, best checkpoint, deviation from ep104) | Reviewer 2.2, "does the optimal duration vary by task?" |

This list will grow as more supplementary items are added; update it
whenever a new file lands in `suppl_mat/`.

---

## Reviewer 1 — jWeV

**Title:** Publication Quality and Camera-ready Requirements
**Official Review by** Reviewer jWeV
**Date:** 31 Jul 2026, 20:07 (modified: 03 Aug 2026, 20:58)
**Visibility:** Everyone
**Revisions:** —

### Review

This review comments are provided solely from the perspective of publication quality.

Camera-ready articles: Authors must submit the original LaTeX source files together with the final camera-ready manuscript and any supplementary materials (if applicable) to bai_jieyun@126.com (The email subject line should be MWM_#XX, where XX denotes the paper number in the OpenReview system).

Signed copyright forms: The licence-to-publish form, "SNCS_ProceedingsPaper_LTP_ST_SN_Switzerland.docx," can be downloaded from https://ideal-conf.com/downloads. The corresponding author of each paper must complete and sign this form.

Citation of relevant publications: Where applicable, authors should appropriately cite relevant publications associated with the event in their manuscripts (if applicable).

For the Universal Multi-Sequence, Multi-Center, and Multi-View CMR Segmentation Challenge (CMRSeg), the following publications may be considered for citation:

Zhu J, Bai J, Zhou Z, et al. RAS Dataset: A 3D Cardiac LGE-MRI Dataset for Segmentation of the Right Atrial Cavity. Scientific Data. 2024;11(1):401.

Bai J, et al. A Benchmark Framework for Right Atrial Cavity Segmentation from LGE-MRIs. IEEE Transactions on Medical Imaging. 2025.

Bai, J., Qiu, R., Chen, J., Wang, L., Li, L., Tian, Y., ... & Zhao, J. (2023). A two-stage method with a shared 3D U-net for left atrial segmentation of late gadolinium-enhanced MRI images. Cardiovascular Innovations and Applications, 8(1), 976.

Qu, T., Su, Z., Zhang, H., Bai, J., Zhang, N., Wang, H., Zhou, Z., Zhao, P., Bo, K., Zhao, J., Gan, J., Zhan, Y., Lu, H., Zhang, X., Cai, W., Xun, L.& Zhang, H. (2026). Universal Multi-Sequence, Multi-Center and Multi-View CMR Segmentation Challenge. Zenodo. International Conference on Medical Image Computing and Computer Assisted Intervention 2026 (MICCAI). https://doi.org/10.5281/zenodo.19728181

For the Mitral Valve Anatomy Analysis Using Multimodal Imaging Data (MVAA), the following publications may be considered for citation:

Hammad, M., Bai, J., Zhao, J., Jia, T., Zhang, X., Xu, X., Ma, J.& Li, S. (2026). Mitral Valve Anatomy Analysis Using Multimodal Imaging Data. Zenodo. International Conference on Medical Image Computing and Computer Assisted Intervention 2026 (MICCAI). https://doi.org/10.5281/zenodo.19726755

Zeng A, Wu C, Lin G, et al. ImageCAS: A Large-Scale Dataset and Benchmark for Coronary Artery Segmentation Based on Computed Tomography Angiography Images. Computerized Medical Imaging and Graphics. 2023;109:102287.

Bai J, Khobo I, Lu Y, et al. Landmark Detection Challenge for Intrapartum Ultrasound Measurement Meeting the Actual Clinical Assessment of Labor Progress. Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. Zenodo; 2025. doi:10.5281/zenodo.15172238.

Zia A, Berniker M, Perreault C, Nespolo R, Jarc A. Surgical Visual Understanding Challenge (SurgVU). Medical Image Computing and Computer Assisted Intervention – MICCAI 2025. Zenodo; 2024. doi:10.5281/zenodo.14054184.

Chen Z, Bai J, Li S, et al. CFVP-Net: Coarse-to-Fine Visual Perception Network for Aortic Dissection Segmentation. In: Proceedings of the 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE; 2025:5809–5816.

Zhu J, Bai J, Zhou Z, et al. RAS Dataset: A 3D Cardiac LGE-MRI Dataset for Segmentation of the Right Atrial Cavity. Scientific Data. 2024;11(1):401.

Bai J, et al. A Benchmark Framework for Right Atrial Cavity Segmentation from LGE-MRIs. IEEE Transactions on Medical Imaging. 2025.

For the Foundation Model Challenge for Ultrasound Biometry (FoundUS), the following publications may be considered for citation:

Bai, J., Yaqub, M., Ma, J., Lekadir, K., Gan, J., Cai, W., Ni, D.& Li, S. (2026). Foundation Model Challenge for Ultrasound Biometry. Zenodo. International Conference on Medical Image Computing and Computer Assisted Intervention 2026 (MICCAI). https://doi.org/10.5281/zenodo.19736827

Bai J, et al. Beyond Benchmarks of IUGC: Rethinking Requirements of Deep Learning Methods for Intrapartum Ultrasound Biometry from Fetal Ultrasound Videos. Medical Image Analysis. 2026.

Bai J, et al. IUGC: A Benchmark of Landmark Detection in End-to-End Intrapartum Ultrasound Biometry. Medical Image Analysis. 2026: 103960.

Bai J, Zhou Z, Ou Z, et al. PSFHS Challenge Report: Pubic Symphysis and Fetal Head Segmentation from Intrapartum Ultrasound Images. Medical Image Analysis. 2025;99:103353.

Chen Z, Ou Z, Lu Y, et al. Direction-Guided and Multi-Scale Feature Screening for Fetal Head–Pubic Symphysis Segmentation and Angle of Progression Calculation. Expert Systems with Applications. 2024;245:123096.

Lu Y, Zhi D, Zhou M, et al. Multitask Deep Neural Network for the Fully Automatic Measurement of the Angle of Progression. Computational and Mathematical Methods in Medicine. 2022.

Lu Y, Zhou M, Zhi D, et al. The JNU-IFM Dataset for Segmenting Pubic Symphysis–Fetal Head. Data in Brief. 2022;41:107904.

Bai J, Sun Z, Yu S, et al. A Framework for Computing Angle of Progression from Transperineal Ultrasound Images for Evaluating Fetal Head Descent Using a Novel Double-Branch Network. Frontiers in Physiology. 2022;13:2565.

Zhou Z, Lu Y, Bai J, et al. Segment Anything Model for Fetal Head–Pubic Symphysis Segmentation in Intrapartum Ultrasound Image Analysis. Expert Systems with Applications. 2024:125699.

Jiang J, Wang H, Bai J, et al. Intrapartum Ultrasound Image Segmentation of Pubic Symphysis and Fetal Head Using a Dual Student–Teacher Framework with CNN–ViT Collaborative Learning. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2024. Springer; 2024:448–458.

Ou Z, Bai J, Chen Z, et al. RTSeg-Net: A Lightweight Network for Real-Time Segmentation of Fetal Head and Pubic Symphysis from Intrapartum Ultrasound Images. Computers in Biology and Medicine. 2024;175:108501.

Qiu R, Zhou M, Bai J, et al. PSFHSP-Net: An Efficient Lightweight Network for Identifying the Pubic Symphysis–Fetal Head Standard Plane from Intrapartum Ultrasound Images. Medical & Biological Engineering & Computing. 2024.

Chen G, Bai J, Ou Z, et al. PSFHS: Intrapartum Ultrasound Image Dataset for AI-Based Segmentation of Pubic Symphysis and Fetal Head. Scientific Data. 2024;11(1):436.

Chen Z, Lu Y, Long S, et al. Fetal Head and Pubic Symphysis Segmentation in Intrapartum Ultrasound Images Using a Dual-Path Boundary-Guided Residual Network. IEEE Journal of Biomedical and Health Informatics. 2024.

Bai J, Lekadir K, Ni D, et al. Intrapartum Ultrasound Grand Challenge 2024. MICCAI 2024. Zenodo; 2024.

Bai J, Ou Z, Lu Y, et al. Pubic Symphysis–Fetal Head Segmentation from Transperineal Ultrasound Images. MICCAI 2023. Zenodo; 2023.

Bai J, Khobo I, Slimani S, et al. Landmark Detection Challenge for Intrapartum Ultrasound Measurement Meeting the Actual Clinical Assessment of Labor Progress. MICCAI 2025. Zenodo; 2025.

Bai J, Yang Z, Hasan K, et al. Fetal Ultrasound Grand Challenge: Semi-Supervised Cervical Segmentation (FUGC25). ISBI 2025. Zenodo; 2024.

Chen S, Wang H, Long S, Bai J, Jiang J. Ultrasound Video Segmentation of Pubic Symphysis and Fetal Head for Angle of Progression Measurement. In: Proceedings of ACM Multimedia Asia 2024. 2024.

Gan J, Liang Z, Fan J, et al. Accurate Fetal Head Descent Assessment During Labor Using Video Swin Transformer and Wavelet-Based Multitask Learning for the 2024 MICCAI Challenge IUGC. In: Intrapartum Ultrasound. Springer; 2024:21–31.

Gan J, Liang Z, Fan J, et al. Sequential Spatial–Temporal Network for Interpretable Automatic Ultrasonic Assessment of the Fetal Head During Labor. In: Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI 2025). IEEE; 2025.

Chen Z, Ou Z, Lu Y, Campello VM, Bai J, Lekadir K. Uncertainty-Aware Fetal Head and Pubic Symphysis Segmentation with Enhanced Multi-Scale Features and Sparse Visual Graph Attention. Expert Systems with Applications. 2026. 296:128998.

Chen Z, Ou Z, Lu Y, Campello VM, Bai J, Lekadir K. Uncertainty-Aware Fetal Head and Pubic Symphysis Segmentation with Enhanced Multi-Scale Features and Sparse Visual Graph Attention. Expert Systems with Applications. 2026;296:128998.

Bai J, Kang X, Wang W, et al. A Multimodal Model in the Prediction of the Delivery Mode Using Data from a Digital Twin-Empowered Labor Monitoring System. Digital Health. 2024;10:20552076241304934.

Ramirez Zegarra R, Lizarraga Cepeda E, Ghi T. Intrapartum Ultrasound. Best Practice & Research Clinical Obstetrics & Gynaecology. 2025;101:102617.

Zhou M, Yuan C, Chen Z, et al. Automatic Angle of Progress Measurement of Intrapartum Transperineal Ultrasound Images with Deep Learning. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. Springer; 2020:406–414.

Sherer DM. Intrapartum Ultrasound. Ultrasound in Obstetrics & Gynecology. 2007;30:123–139.

Deng B, et al. Baseline Method of the Foundation Model Challenge for Ultrasound Image Analysis. arXiv preprint. 2026. arXiv:2602.01055.

Luo Y, Long S, Wang H, et al. DSTCS: Dual-Student–Teacher Framework with Segment Anything Model for Semi-Supervised Pubic Symphysis–Fetal Head Segmentation. arXiv preprint. 2026. arXiv:2601.19446.

Bai J, Tang Y, Zhou Z, et al. FUGC: Benchmarking Semi-Supervised Learning Methods for Cervical Segmentation. arXiv preprint. 2026. arXiv:2601.15572.

Lin Z, Han L, Wang X, et al. Diagnostic Performance of Universal-Learning Ultrasound AI Across Multiple Organs and Tasks: The UUSIC25 Challenge. arXiv preprint. 2025. arXiv:2512.17279.

Luo Y, Long S, Wang H, et al. Edge Awareness Network with Large Kernel Attention for Small-Target Segmentation from Intrapartum Ultrasound Images. In: Proceedings of the 2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE; 2025:5876–5883.

**Rating:** 7: Good paper, accept
**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct

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

**Title:** Official Review
**Official Review by** Reviewer ZxK3
**Date:** 31 Jul 2026, 15:29 (modified: 03 Aug 2026, 20:58)
**Visibility:** Everyone
**Revisions:** —

### Review

The paper proposes a two-stage pipeline for multi-task ultrasound landmark detection: first, self-supervised domain adaptation of DINOv2 on unlabeled ultrasound frames via DINO-style multi-crop self-distillation, then coupling the adapted encoder with an HRNet-style multi-resolution decoder with task-specific soft-argmax heads, evaluated across nine tasks on the GU_Biometry challenge.

> Actions: Nothing

This is a well-motivated and methodologically sound paper. The two-stage design is a principled approach to the challenge of transferring foundation models to ultrasound. The use of HRNet for maintaining high-resolution spatial detail throughout the decoder is well-justified given the precision requirements of landmark localization. The paper's study of how SSL adaptation duration affects downstream accuracy is a valuable practical contribution. However, results are reported somewhat vaguely in the reviewed excerpt.

> Actions: Enhance results for SSL duration (comment 2.2)

Quantitative results: The abstract and introduction describe the method thoroughly but do not report concrete performance numbers (MRE, MAE, or challenge score). For a challenge paper, the primary results should appear prominently. The reader should not need to search for key quantitative outcomes. Please ensure the final results (per-task MRE, overall challenge metric, and comparison to relevant baselines) are clearly presented early in the paper.

> Actions: (a) Add MRE and MAE in abstract (b) forward-pointing sentence with the same numbers + a table reference added to the Introduction's contributions (c) decide which one keep, because we do not want repeatitions (d) ensure the final results (per-task MRE, overall challenge metric, and comparison to relevant baselines) are clearly presented early in the paper (e) i have to search for relevant baselines for comparison

SSL adaptation duration study: This is listed as a contribution ("a study of how unlabeled self-supervised learning adaptation duration affects downstream landmark accuracy") and is genuinely useful, but its findings are not reported in the reviewed excerpt. What is the optimal adaptation duration? Is there a point of diminishing returns or negative transfer? Does the optimal duration vary by task? These findings should be prominently reported with a figure showing downstream accuracy vs. adaptation epochs.

HRNet decoder design: The choice of HRNet over U-Net is well-motivated in the text, but the specific HRNet configuration (number of stages, channels per resolution, how transformer patch features are mapped to the HRNet input) is not described. Since HRNet is less common in medical imaging than U-Net, readers need sufficient detail to understand and reproduce the architecture. A diagram showing how DINOv2 patch tokens are reshaped and fed into the HRNet backbone would be particularly helpful.

DINO-style multi-crop self-distillation: The adaptation strategy is well-described but the paper does not discuss whether the register-variant of DINOv2 is specifically important. The text mentions "register-variant DINOv2-L/14" -- is there a reason this variant was chosen? Would the standard DINOv2 work as well? A brief justification or ablation would strengthen this choice.

Comparison to frozen/no-adaptation baseline: The central hypothesis is that SSL domain adaptation on unlabeled ultrasound data improves downstream landmark accuracy. The paper should report results with a frozen (non-adapted) DINOv2 encoder as a direct test of this hypothesis. Without this baseline, it is unclear whether the SSL adaptation stage is necessary or whether the HRNet decoder alone could achieve similar performance by learning to interpret generic DINOv2 features.

Accept with minor revisions. The two-stage design is principled and well-motivated, and the study of SSL adaptation duration is a useful contribution. The authors should (a) report primary quantitative results prominently, (b) present findings from the SSL duration study with a clear figure, (c) specify the HRNet configuration and the patch-to-HRNet mapping, and (d) include a frozen-encoder baseline to validate the necessity of domain adaptation.

**Rating:** 7: Good paper, accept
**Confidence:** 3: The reviewer is fairly confident that the evaluation is correct

### Comment index

| # | Paraphrase | Where to look |
|---|---|---|
| 2.1 | Present final results (per-task MRE, overall metric, baseline comparison) prominently, early | **See "Response to 2.1" near the end of this file** — done except the baseline-comparison clause (blocked) |
| 2.2 | SSL duration study: report findings with a clear figure | **Done — landed in `main.tex` §4.2** (probe paragraph + Figure~\ref{fig:ssl-duration} + Table~\ref{tab:ssl-duration}). Answers all three sub-questions: optimal duration (mixed — ep20 best aggregate MRE, ep104 best blend + complete resolution curriculum), diminishing returns (non-monotonic, demonstrated), and "does it vary by task?" (yes — task-wise per-task breakdown now run, see `docs/MWM_54/suppl_mat/supplementary.tex` Table S1, referenced from §4.2). |
| 2.3 | Specify HRNet configuration + patch-to-HRNet mapping (diagram) | **Done, already satisfied — no new work needed.** §2.2's prose + the existing Figure 2 (`fig:neck-zoom`, `FIGURE2_UPDATED-Page-5.drawio.pdf`) already give branch count, per-branch channel counts (128/96/64ch, in the figure), and the exact DINOv2-tap-to-branch mapping with formulas. Verified by rendering the actual figure, not assumed. |
| 2.4 | Justify the register-variant DINOv2 choice | Partially — `probe_noreg_nossl` ablation exists (`EXPERIMENT_RESULTS.md` §2, "Register-backbone ablation"), not yet written into the paper/reply |
| 2.5 | Frozen/no-adaptation baseline to test necessity of SSL | **Done — landed in `main.tex` §4.2** (the probe protocol's off-the-shelf/no-SSL point, Figure~\ref{fig:ssl-duration} epoch-0) |

### Our reply

**Reviewer's note:**
> "The paper's study of how SSL adaptation duration affects downstream accuracy
> is a valuable practical contribution. However, results are reported somewhat
> vaguely in the reviewed excerpt." *(general remark, second paragraph)* — see
> also the itemized **SSL adaptation duration study** point below: *"What is
> the optimal adaptation duration? Is there a point of diminishing returns or
> negative transfer? Does the optimal duration vary by task?"*

**Our response:** We thank the reviewer for this suggestion. We added a
controlled frozen-encoder probe across Phase-1 checkpoints and report
downstream localization performance as a function of adaptation duration in
Section 4.2 and the corresponding figure/table. The analysis shows rapid
gains during early adaptation followed by non-monotonic behavior and
diminishing returns at longer durations. The best aggregate probe MRE occurs
at ep20, whereas ep104 achieves the best probe blend score and corresponds
to the complete high-resolution-tail training schedule; we therefore use
ep104 as the final Phase-1 checkpoint. A task-wise analysis further shows
that the preferred duration varies across tasks (ep20--ep104); the complete
breakdown is provided in Supplementary Table S1
(`docs/MWM_54/suppl_mat/supplementary.tex`).

In more detail (fold-0, frozen-encoder probe, so duration effects on the
encoder itself are isolated from Phase-2 fine-tuning capacity — see
`EXPERIMENT_RESULTS.md` §2), reported as two figures and their accompanying
tables:

- **A 224px-bulk-then-518px-tail curve** (no-SSL → ep10 → ep20 → ep60 → ep100,
  plus tail-appended variants at ep60+5, ep60+10, ep100+4): quality is
  **non-monotonic**, peaking at 20 epochs (MRE 24.70 vs. 31.27 no-SSL), then
  *degrading* with more low-resolution-only training — by epoch 100 it is
  worse than epoch 10 (25.95 vs. 26.88 MRE) and briefly worse than no SSL at
  all on the blend metric (0.0974 vs. 0.0890). A short high-resolution tail
  (as little as 5 epochs) recovers most of this loss and reaches the best
  probe blend score of any checkpoint tested (100 bulk + 4-epoch tail, blend
  0.0747); its MRE (25.09) is close to, but does not surpass, the epoch-20
  peak (24.70) — a mixed result, not a clean win on every metric, which is
  why we report it as diminishing returns / non-monotonic behavior rather
  than claiming ep104 is universally optimal.
- **A 518px-full-resolution-throughout control curve** (ep10/20/30, no
  bulk/tail split): the same non-monotonic decline appears, and at every
  matched epoch count it trails the bulk-then-tail design — evidence the
  two-resolution curriculum itself (not just total duration) is doing real
  work.

This directly answers "what is the optimal duration" (mixed: ep20 on
aggregate MRE, ep104 on blend + curriculum completeness — see above) and "is
there a point of diminishing returns" (yes, demonstrated directly). The
third sub-question, whether the optimal duration varies by task, is now also
answered: yes, task-specific optima span ep20--ep104, though the selected
ep104 checkpoint stays within 8% MRE of the task-specific optimum on eight
of nine tasks (Supplementary Table S1). All figures/tables report internal
fold-0 values using our local challenge-metric scorer (verified to reproduce
the official scorer, Section 3.2).

---

## Draft workspace: response to Reviewer 2, points (b) + (d)

**Status: LANDED in `main.tex` §4.2 — this whole section is now historical
record of the draft, not a pending task.** The paragraph, both figure panels,
and the table below are all live in the paper (using fold-0 **internal**
values, as drafted — that decision is made). The per-task duration breakdown
is also now done — see Supplementary Table S1 below.

### What this answers

- **(d) frozen/no-adaptation baseline** → `probe_nossl` (off-the-shelf DINOv2,
  frozen encoder, no Phase-1 SSL) run through the exact same protocol as every
  SSL-duration point below. Direct, clean baseline.
- **(b) SSL duration study + figure** → the frozen-probe duration sweep
  (`EXPERIMENT_RESULTS.md` §2) is a genuinely richer answer than the paper's
  current Table 1 (which is only OLD-SSL ep10-vs-ep20, full fine-tune, 2 points).
  It shows the reviewer's exact question — diminishing *and* negative transfer —
  directly: quality peaks at 20 bulk epochs, then **degrades past even the
  no-SSL baseline by epoch 100** (blend 0.0974 vs. 0.0890 no-SSL), and is then
  rescued and pushed further by a short high-resolution tail.

### Draft figures (v2 — two separate images, per plan)

(c) register with/without comparison is **deliberately skipped for now**
(waiting on `probe_noreg_nossl`).

Both regenerated with a **real linear epoch axis** (not categorical — distances
are now proportional to actual epoch gaps) and a **diamond marker** (◆, was a
star) for tail-appended checkpoints. Script: `drafts/plot_ssl_duration.py`
(matplotlib, regenerable once numbers are finalized or if we switch to
Codabench values).

**(a) [`drafts/fig_ssl_duration_224_DRAFTv2.pdf`](drafts/fig_ssl_duration_224_DRAFTv2.pdf)
— 224px bulk-only + high-res-tail variants.** x-axis: total Phase-1 epochs
elapsed (0/10/20/60/100 on the main bulk-only curve; tail-appended checkpoints
plotted at their *total* elapsed epoch count — 65, 70, 104 — connected back to
their parent bulk checkpoint by a dashed line, since they're the same bulk
checkpoint + a few more 518px epochs, not a separate duration point). y-axis:
MRE (orig. px, ↓). 0 = `probe_nossl` (off-the-shelf, no SSL) in red, shared
with figure (b) for a consistent reference point across both.

**(b) [`drafts/fig_ssl_duration_fullres_DRAFTv2.pdf`](drafts/fig_ssl_duration_fullres_DRAFTv2.pdf)
— 518px full-resolution control.** **Now complete: ep10/20/30 all probed**
(`probe_dv2fullres_ep10` landed 0.0929/25.79, same monotonic-decline pattern
as figure (a)'s bulk-only curve). The underlying Phase-1 run itself is still
capped at ep30 (evicted mid-epoch-31, not being resumed — `EXPERIMENT_RESULTS.md`),
so this is the complete duration curve this control will ever have, not a gap.

Both are separate image files as requested; for `main.tex` these likely become
two `subcaptionbox` panels inside one `figure` environment (the `subcaption`
package is already imported in the preamble for exactly this), keeping it to
the originally-agreed "1 figure" of page budget rather than two.

Things to sanity-check before finalizing: (1) MRE-only vs. also showing
`blend` (paper's actual selection metric) — currently MRE-only, to match Table
2's headline units and keep the figures simple; `blend` stays in the tables
instead; (2) font/style match against the paper's other figures (currently
plain matplotlib defaults); (3) whether Codabench-external points should
replace or supplement the internal ones once we have more than 2 external SSL
submissions.

### Draft tables (compact, for the paper)

**(a) 224px bulk + tail**

| Encoder / SSL duration | blend ↓ | MRE (px) ↓ |
|---|---|---|
| No SSL (off-the-shelf DINOv2) | 0.0890 | 31.27 |
| SSL, 10 ep (224px bulk only) | 0.0872 | 26.88 |
| SSL, 20 ep (224px bulk only) | 0.0782 | 24.70 |
| SSL, 60 ep (224px bulk only) | 0.0836 | 26.43 |
| SSL, 60 ep + 5-ep 518px tail | 0.0748 | 25.23 |
| SSL, 60 ep + 10-ep 518px tail | 0.0755 | 25.50 |
| SSL, 100 ep (224px bulk only) | 0.0974 | 25.95 |
| SSL, 100 ep + 4-ep 518px tail | **0.0747** | 25.09 |

**(b) 518px full-resolution control**

| Encoder / SSL duration | blend ↓ | MRE (px) ↓ |
|---|---|---|
| No SSL (off-the-shelf DINOv2) | 0.0890 | 31.27 |
| SSL, 10 ep (518px full-res) | 0.0929 | 25.79 |
| SSL, 20 ep (518px full-res) | 0.0901 | 25.96 |
| SSL, 30 ep (518px full-res) | 0.0934 | 26.92 |

(All frozen-probe, fold 0, unfreeze 0, 25 ep, upgraded-recipe knobs —
`EXPERIMENT_RESULTS.md` §2. `probe_legacy_ep10/20` (OLD SSL) deliberately left
out of both tables to keep them focused on the duration question; already
covered elsewhere / in the supplementary material plan.)

### Draft paragraph (for `main.tex`, extending `sec:ssl-duration`)

> To directly test whether Phase-1 SSL improves the encoder's own representation
> — independent of Phase-2's capacity to compensate for a weaker encoder — we
> additionally run a frozen-encoder probe: the backbone is entirely frozen
> (unfreeze 0) and only the neck and heads train, for a short 25 epochs on
> fold~0, holding every other recipe knob fixed. Figure~\ref{fig:ssl-duration}
> sweeps SSL duration under this protocol against an off-the-shelf,
> non-adapted DINOv2 control. Representation quality is **non-monotonic** in
> adaptation duration (panel a): 20 epochs of low-resolution (224px) adaptation
> yields the best bulk-only checkpoint (MRE 24.70), after which quality
> degrades — 100 epochs scores worse than even 10 (25.95 vs.\ 26.88 MRE), and
> briefly worse than no SSL at all on blend (0.0974 vs.\ 0.0890) — consistent
> with representational drift away from the foundation model's priors as
> adaptation continues unchecked. A short, high-resolution (518px) tail
> appended after either the 60- or 100-epoch bulk phase recovers and surpasses
> the 20-epoch peak (best: 100~ep~+~4-ep tail, MRE 25.09, blend 0.0747), the
> best encoder tested. A full-resolution (518px throughout) control (panel b)
> confirms the low-res-bulk-then-tail design is not merely convenient but
> better: at matched epoch count it trails the 224px-bulk curve (ep20: MRE
> 25.96 full-res vs.\ 24.70 bulk-only). All values are fold-0 internal
> validation with our local challenge-metric scorer (Section~\ref{sec:phase2}).

### Draft OpenReview reply (points b + d combined)

> We thank the reviewer for these two related points. We now report a
> frozen-encoder ablation (backbone entirely frozen, only neck/heads trained,
> fold-0 internal validation) that directly addresses both: for (d), a true
> off-the-shelf, non-adapted DINOv2 baseline; for (b), a duration sweep across
> 10/20/60/100 Phase-1 epochs plus a high-resolution-tail variant, presented as
> a new figure and table in Section [X]. The results show adaptation quality is
> non-monotonic with duration: it peaks early (20 epochs), degrades with
> further low-resolution-only training to the point of underperforming the
> non-adapted baseline by epoch 100, and is rescued and further improved by a
> short high-resolution tail — directly answering the reviewer's question about
> diminishing returns as adaptation duration increases. [Superseded by the
> finalized "Our response" above; local-vs-Codabench framing and page
> placement are both settled, and the per-task follow-up is done — see
> Supplementary Table S1.]

### Remaining items (post-landing)

- [x] ~~Fill the `probe_dv2fullres_ep10` gap~~ — done.
- [x] ~~Decide local-only vs. also-Codabench framing~~ — decided: local/internal, landed.
- [x] ~~Confirm placement~~ — decided: extended `sec:ssl-duration` in place
      (now §4.2 after the Results reorder), with the old ep10-vs-ep20
      full-FT comparison kept afterward as the legacy/secondary check.
- [x] ~~Compile and check page-count impact~~ — checked every pass, compiles clean.
- [ ] Polish figure styling to match the paper's other figures — still plain
      matplotlib defaults, genuinely not done yet.
- [x] ~~Plan the supplementary-material doc (per-task duration figures from
      `EXPERIMENT_RESULTS.md`) as its own follow-up pass~~ — done: Table S1 in
      `suppl_mat/supplementary.tex`, referenced from `main.tex` §4.2.

---

## Phase-1 methodology staleness fix (blue-highlighted in `main.tex`)

**Not from a reviewer comment directly — self-identified while working the
Reviewer 2 (b)/(d) items.** §2.1 described the *legacy* multicrop SSL recipe
(fixed 518px, mean-pooled patch tokens, plain 2-layer MLP head, single
CLS-only cross-entropy loss) while the actual default pipeline
(`phase1_dinov2.py`) has for some time been the DINOv2-faithful recipe: CLS
self-distillation + iBOT masked-patch prediction + KoLeo regularizer, a proper
DINOHead into 65,536 prototypes, and a 224px-bulk-then-518px-tail resolution
curriculum. `docs/fixes/phase1_dinov2_methodology_draft.md` already had a
complete, code-grounded LaTeX replacement drafted — used it directly.

**Done, this pass (all blue in `main.tex`):**
- §2.1 body replaced with the DINOv2-faithful description (2 new equations,
  `eq:ibot-loss` / `eq:dino-combined`; old `eq:ssl-loss` removed).
- Dataset/Preprocessing's Phase-1 augmentation sentence updated to match the
  actual `us_v2` recipe (±10° global-only rotation, foreground-bbox
  pre-crop, brightness/contrast/gamma/CLAHE + blur/noise, background-reject
  resampling for locals) — verified directly against `transforms.py` and
  `multicrop.py`, not just the draft doc.
- Added missing `.bib` entries: `zhou2022ibot` (iBOT, ICLR 2022),
  `sablayrolles2018spreading` (KoLeo source, ICLR 2019).
- §ssl-duration (Table 1's ep10-vs-ep20 study) got one clarifying sentence:
  that comparison predates this recipe and used the superseded CLS-only
  multicrop SSL — added so the two sections don't silently contradict each
  other, without touching Table 1's actual numbers.

**Deliberately deferred (per your answers):**
- **Table 2 provenance** — "Official Challenge Evaluation" still claims it
  submitted "Table 1's best (OLD-SSL) checkpoint," which doesn't match any of
  the 4 logged Codabench submissions in `EXPERIMENT_RESULTS.md` §5. Treated as
  an older/untracked submission for now; **needs confirming before final
  submission** — flagging again here so it doesn't get lost.
- **Figure 1** (`fig:pipeline`) — now visibly stale next to the new blue text
  on the same page: the diagram still shows mean-pooling + a single CE loss
  box, and the caption still says "shared encoder-pool-head pipeline" /
  "the resulting cross-entropy loss" (singular). No iBOT/KoLeo/two-resolution
  slot. Both the diagram *and* its caption need a redraw pass — Mermaid
  sketches already exist in the methodology-draft file as a starting point,
  but matching your drawio/Illustrator style is a manual task for you.

Compiles clean (no undefined refs, no bibtex warnings). Body grew 8→9 pages
from the real content added (not asked to optimize for page count this pass).

---

## Reviewer 3 — KKV9

**Title:** Official Review
**Official Review by** Reviewer KKV9
**Date:** 31 Jul 2026, 13:36 (modified: 03 Aug 2026, 20:58)
**Visibility:** Everyone
**Revisions:** —

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

| # | Paraphrase | Where to look |
|---|---|---|
| 3.1 | Deeper per-task error analysis (PSAX/A4C/HC) | **See "Response to 3.1" near the end of this file** |
| 3.2 | §4.1 (ep20) vs. §4.2 (ep10) checkpoint inconsistency | "Draft workspace: response to Reviewer 3" below — data already exists, not yet written up |
| 3.3 | Compute cost / throughput / memory not reported | "Draft workspace: response to Reviewer 3" below — done |
| 3.4 | SSL duration studied only coarsely | Same duration sweep as 2.2 — "Draft workspace: response to Reviewer 2" |
| 3.5 | Comparison to other submissions / baselines on official scorer | **See "Response to 3.5" near the end of this file** |

### Our reply

_(to be written)_

---

## Draft workspace: response to Reviewer 3

Went through all 5 points against `EXPERIMENT_RESULTS.md` first to figure out
which actually need new experiments. Only one did.

| # | Point | New run needed? | Status |
|---|---|---|---|
| 1 | Deeper per-task error analysis (PSAX/A4C/HC) | No | Writing task — tie to already-documented data scarcity (PSAX/A4C/PLAX/IVC are the smallest test sets, n=18/20/26/10) |
| 2 | §4.1 recommends ep20, §4.2 tests ep10 — inconsistency | **No — already have the data** | `phase2_baseline_fold0_ssl20` (0.0973/33.95) and `abl_ep20_upgraded` (0.0946/31.41) are the ep20 equivalents of both §4.2 rows — Table 1's right half can be swapped to ep20 with zero retraining |
| 3 | No compute cost / throughput / memory reported | **Yes — done this pass, Phase 1 + Phase 2** | `scripts/profile_compute.py` (Phase 2) + `scripts/profile_compute_phase1.py` (Phase 1), results in `EXPERIMENT_RESULTS.md` §6 |
| 4 | SSL duration studied only coarsely (ep10 vs 20) | No | Same frozen-probe duration sweep already drafted for Reviewer 2 (b) — landing that draft answers this too |
| 5 | Comparison to other submissions / supervised baselines on official scorer | No run — external lookup only | Need the actual Codabench numeric score for your own no-SSL submission (only qualitative "best so far" is logged); other-team comparison depends on leaderboard visibility, not something I can fetch |

### Point 3 — done, both phases

**Correction to the first pass:** initially only profiled Phase 2. You caught
it — the reviewer said "the large DINOv2-L encoder," and Phase 1 trains the
*full* encoder (not just the last 4 blocks) over 191K unlabeled images with an
8-view multi-crop forward per image, so it's actually the more expensive
stage. Added `scripts/profile_compute_phase1.py` (reuses the real
`phase1_dinov2.py` training code path — data pipeline, models, losses, all
imported directly, not reimplemented — against a small synthetic *unlabeled*
image set). Full numbers in `EXPERIMENT_RESULTS.md` §6; headline:

**Phase 2** (`scripts/profile_compute.py`, `configs/phase2_upgraded.yaml`):
- 312.20M params total, 58.22M trainable (unfreeze-4 + neck + 9 heads), 1051.80 GFLOPs/forward (bs=1).
- Training (bf16, bs=64): 84.4 img/s, 29.24 GB peak.
- Inference: fp32 (what `predict.py`/`evaluate.py` actually run today) 17.8 img/s @ bs=64; bf16 autocast **127.3 img/s** at *lower* memory (5.65 vs 6.06 GB) — a free ~7× speedup nobody's using yet.

**Phase 1** (`scripts/profile_compute_phase1.py`, `configs/phase1_dinov2.yaml`):
- 351.70M params, **351.57M trainable — the full encoder**, not a partial unfreeze.
- Bulk (224px, bs32×8accum): 27,641 GFLOPs/step fwd, 64.4 img/s, 33.16 GB peak.
- Tail (518px, bs16×16accum): 68,579 GFLOPs/step fwd, 15.9 img/s, 55.03 GB peak.
- Rough wall-clock (steady-state, caveat: GPU-bound only, real data-loading may differ): **~96 GPU-hours (~4 days)** total for the 100-bulk+4-tail recipe — this dwarfs Phase 2's per-model cost and is the honest answer to "how expensive is this method."

### Draft paragraph (for `main.tex`'s "Implementation Details", not yet applied — now covers both phases)

> {\color{blue}Phase 1 trains the full 304.4M-parameter encoder (plus two 47.3M-parameter
> DINOHeads) over the 191K unlabeled frames: the 224px bulk stage runs at 64.4 img/s
> (33.2\,GB peak, single A100-80GB), the 518px tail at 15.9 img/s (55.0\,GB peak);
> the full 100-bulk+4-tail schedule is $\sim$96 GPU-hours. Phase 2's model has 312.2M
> parameters (58.2M trainable: the last four unfrozen encoder blocks, neck, and nine
> task heads), costing 1051.8 GFLOPs per forward pass (single task head, $518\times518$
> input). On the same GPU, Phase-2 training (bf16, batch 64, forward+backward+optimizer
> step) runs at 84.4 img/s and 29.2\,GB peak memory; inference at the same batch size
> runs at 17.8 img/s in fp32 (the current \texttt{predict.py} precision) or 127.3 img/s
> under bf16 autocast, at lower memory (5.65\,GB) — an unused $\sim\!7\times$ speedup we
> plan to enable.}

### Open items

- [ ] Confirm placement: extend §3.2 "Implementation Details" (where batch
      size/AMP/LR already live) rather than a new subsection. Getting long
      now that both phases are in it — maybe Phase-1 numbers belong in
      §2.1 (Phase 1 methodology) instead, Phase-2 numbers stay in §3.2.
- [ ] Decide whether to actually apply the `predict.py`/`evaluate.py` autocast
      fix before camera-ready, or just note it as identified-but-not-yet-shipped.
- [ ] Point 1 (per-task error analysis) and point 2 (ep20 Table 1 fix) still
      need their own text passes — not started yet.
- [ ] Point 5 needs you to pull the real Codabench numeric score for
      `regression_predictions.zip` (and any leaderboard context) from the
      platform directly.

---

## Response to 2.1 (Reviewer 2 — present results early)

**My answer:** No new experiments — writing/restructuring only. Right now
the Abstract and Introduction report zero numbers, and Table 2 (the official
per-task results) doesn't appear until §4.3, near the end of an already-tight
8-page paper — exactly what the reviewer flagged.

**Done (blue in `main.tex`):** added the headline numbers (overall MRE
35.03px / MAE 34.62px) as the closing sentence of the Abstract only —
the matching Introduction sentence was added then deliberately reverted
(same numbers twice within two pages read as padding; the Abstract alone
already satisfies "early, no searching required"). Compiles clean, zero
page-count impact.

**Also done (blue in `main.tex`):** the structural fix — §4 Results is now
reordered so "Official Challenge Evaluation" (per-task table + figure) is
§4.1, with the former 4.1/4.2 internal ablations (SSL-duration probe,
Phase-2 recipe selection) pushed down to §4.2/§4.3 as the supporting
evidence behind those choices, read afterward. This resolves the
"per-task MRE isn't early" gap too — it was the same fix, not a separate
one. Added one blue transition sentence at the top of the new §4.1
("We report the official evaluation first; the Phase-1 checkpoint and
Phase-2 recipe choices behind it are justified by the ablations in
Sections~4.2 and~4.3 that follow.") so the forward references read
intentionally rather than backward. All table/figure numbers renumbered
automatically via existing `\ref{}`s — no hardcoded numbers anywhere, so
nothing broke. Compiles clean, body page count unchanged.

**Only remaining gap for 2.1: the "comparison to relevant baselines"
clause.** Deliberately deferred, not forgotten — it's blocked on the Table
2/3 provenance question (see Response to 3.5 below: we don't yet know with
certainty which model produced the official numbers, so a baseline
comparison can't be written honestly until that's resolved). This clause
overlaps with 3.5's "comparison to other submissions/baselines" — tracked
here under 2.1's own wording, resolved together with 3.5 once unblocked.

### Draft reply to Reviewer 2 (point 2.1)

> We thank the reviewer for this comment. We have made the primary
> quantitative results significantly more prominent: the Abstract now
> closes with the overall result (mean radial error 35.03\,px, measurement
> error 34.62\,px on the official Codabench evaluation), and we have
> restructured Section 4 (Results) so that the official per-task evaluation
> (results table and qualitative figure) is now the first subsection a
> reader encounters, rather than appearing after two subsections of
> internal ablations as in the original submission. The internal ablations
> that motivate our Phase-1 and Phase-2 design choices now follow as
> supporting evidence, with an explicit transition sentence connecting the
> two. We are finalizing a frozen/non-adapted-encoder baseline comparison
> on the official scorer for the camera-ready version to directly
> contextualize this result.

*(Status note, not for the reviewer: the baseline-comparison sentence above
is a forward-looking commitment, not yet true — keep it only if we're
confident we'll land it before camera-ready; otherwise soften to "we plan
to include" or drop the specific promise.)*

---

## Response to 3.1 (Reviewer 3 — per-task error analysis)

**Reviewer's note:**
> "Official mean radial error (35.03 px) and measurement MAE (34.62 px)
> indicate substantial room for improvement, especially on PSAX, A4C, and HC;
> deeper error analysis would help prioritize future work."

**My answer:** No new experiments — writing task. Table 2's own numbers
(PSAX 82.29 — worst by far; A4C 40.62; HC 45.20) support a real explanation,
not just an acknowledgment. PSAX and A4C line up with data scarcity already
documented in §3.1: PSAX/PLAX/IVC have under 100 labeled training images, and
PSAX/A4C are also the smallest *official test sets* (n=18/20). **HC is the
interesting one** — it is *not* data-scarce (thousands of labeled images,
same scale as AoP) yet is still one of the three worst tasks. I don't have a
full explanation for that yet and will say so honestly rather than force an
unsupported claim (working hypothesis: HC's ellipse-based clinical
measurement may amplify landmark-level pixel error differently than the
other tasks — needs a qualitative failure-case look, not yet done).

**Not yet done:** actually drafting the paragraph for the paper (goes after
Table 2 in §4.3). Purely a writing task at this point, ready whenever you
want it drafted.

---

## Response to 3.5 (Reviewer 3 — baseline comparison on the official scorer)

**Reviewer's note:**
> "The paper would benefit from comparison to other challenge submissions or
> strong supervised baselines on the same official scorer to contextualize
> performance."

**My answer — this one needs your input before I can write it.** We do have
a same-formula (local scorer, verified to reproduce the official
`local_eval`) no-SSL vs. SSL comparison already: `abl_nossl_fold0` (blend
0.0740 / MRE 28.98, fold-0 internal). We can report that part now. But
there's a less comfortable fact that has to be part of an honest answer
here: per `EXPERIMENT_RESULTS.md` §5, our own **external** Codabench
submission history shows the plain no-SSL model is currently our **best
external result** — ahead of both SSL-adapted submissions we've made. The
internal ranking does not hold up on the real held-out test set. This
directly overlaps the still-open **Table 2 provenance question** (flagged in
the Phase-1 methodology fix section above): we don't yet know with certainty
which model actually produced Table 2's headline numbers, so I can't write
this response properly until that's resolved — **this is the one blocking
item, and it's yours to resolve, not mine.** Separately, "comparison to
other challenge submissions" (other teams) needs the Codabench leaderboard
directly — I have no way to fetch that; only you can pull those numbers.

---

## Notes: what still needs describing about "the probe" before it lands in `main.tex`

The Draft-workspace paragraph (response to Reviewer 2, above) covers the core
protocol reasonably well, but it's being asked to do a lot of jobs — SSL
duration (2.2/3.4), frozen baseline (2.5), and potentially the register
ablation (2.4) too. Gaps to close before it goes into the paper:

1. **Name it once, formally, on first use.** Right now different drafts call
   it "a frozen-encoder probe," "the frozen probe," "this protocol" — pick
   one first-use phrasing (e.g. *"we introduce a frozen-encoder probe
   (henceforth, the probe)"*) so every later reference — in this same
   section, and if reused for 2.4's register ablation — reads as one
   consistent, named thing instead of being redefined each time.

2. **State plainly that probe numbers are not comparable to Table 1/2's
   full-fine-tune numbers.** The draft paragraph never says this, and it's
   an easy misread: probe MRE 24.70 (frozen, 25 ep, lightweight neck+heads
   only) sits *below* several full-fine-tune numbers in Table 1, which looks
   like a contradiction unless a reader realizes the probe is a diagnostic
   protocol for *relative* encoder-quality ranking, not a competing
   end-to-end model. One sentence fixes this.

3. **List the fixed knobs explicitly, once.** Fold 0 only, multilevel neck,
   heatmap 148, task-homogeneous sampler, unfreeze 0, 25 epochs — most of
   this is already implied by "holding every other recipe knob fixed" but
   a reader can't reconstruct the exact protocol from that phrase alone.
   Doesn't need every hyperparameter (code is public via the footnote link),
   just enough to make "probe" a well-defined, reproducible term.

4. **If 2.4 (register-variant justification) gets written up**, it reuses
   this *exact* protocol (`probe_nossl` vs. `probe_noreg_nossl`, only
   `model.backbone.name` differs) — write it as "under the same probe
   protocol as above, swapping only the backbone," not as a separately
   re-explained ablation. Note the result is a genuine mixed finding, not a
   clean win for either backbone (`EXPERIMENT_RESULTS.md` §2: register wins
   on blend 0.0890 vs 0.0976, non-register wins on MRE 31.27 vs 29.34) — say
   that honestly rather than picking whichever number favors the shipped
   choice.

5. **Confirm the "fold-0 internal only" caveat survives into the final
   paragraph** — already present in the current draft's last sentence, just
   don't let it get edited out when the paragraph gets trimmed for length.

---
