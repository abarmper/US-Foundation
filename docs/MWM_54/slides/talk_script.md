# FUB 2026 — 10-minute talk script

> Word-for-word script for `31_annapan_54_presentation.pptx` (13 slides).
> ~1,130 spoken words ≈ **8:30–9:00 at a calm 130–140 words/minute**, leaving
> real buffer under the 10-minute cap. Timestamps below assume that pace.
> If you are running long, the cuts that hurt least are marked ✂ inline.
> Every number matches the deck and `../main.tex`.

| # | Slide | Start | Budget |
|---|---|---|---|
| 1 | Title | 0:00 | 0:20 |
| 2 | Nine tasks, one model | 0:20 | 0:55 |
| 3 | The metric dictates the design | 1:15 | 1:00 |
| 4 | The method in one picture | 2:15 | 0:40 |
| 5 | From image to patch tokens | 2:55 | 0:35 |
| 6 | Phase 1 — domain adaptation | 3:30 | 0:30 |
| 7 | Phase 1 — the recipe in brief | 4:00 | 1:00 |
| 8 | Phase 2 — HRNet neck | 5:00 | 0:45 |
| 9 | Phase 2 — routing & decoding | 5:45 | 0:55 |
| 10 | Official evaluation | 6:40 | 1:00 |
| 11 | Qualitative results | 7:40 | 0:15 |
| 12 | How long should SSL run? | 7:55 | 0:55 |
| 13 | Conclusions | 8:50 | 0:30 |

---

## Slide 1 — Title  (0:00)

> Good morning, everyone. This is our submission to the FU-Biometry
> challenge: a self-supervised domain-adaptation pipeline — DINOv2 with an
> HRNet-style decoder — that handles all nine biometry tasks with a single
> model. It is joint work between the National Technical University of Athens
> and the Athena Research Center.

## Slide 2 — Nine tasks, one model — why that is hard  (0:20)

> Ultrasound biometry turns a few anatomical landmarks into clinical
> measurements — fetal growth, labour progression, cardiac structure. Doing
> this manually is slow and operator-dependent, and task-specific models
> transfer poorly across scanners and sites.
>
> The challenge asks for one model across nine tasks, spanning three clinical
> domains. And the data is the real difficulty. Only three-point-four percent
> of the released images are labeled — about seven thousand labeled, against
> a hundred and ninety-one thousand unlabeled.
>
> *[gesture at the chart]* And the labels span three orders of magnitude:
> angle-of-progression has four thousand labeled images; IVC has
> thirty-eight. Train naively on the pooled data, and you get an AoP model
> that ignores the heart.

## Slide 3 — The metric dictates the design  (1:15)

> Why does that matter so much? Because of the metric. The score is fifty
> percent landmark error and fifty percent clinical-measurement error,
> macro-averaged with equal weight per task.
>
> *[point at the two strips]* AoP is fifty-nine percent of the labeled data —
> but one ninth of the score. A thirty-eight-image task counts exactly as
> much as a four-thousand-image one, so the tiny cardiac tasks decide the
> ranking.
>
> We therefore treated the metric as a design constraint, in three ways.
> Task sampling is balanced by the square root of task size, so AoP is seen
> nine times as often as PSAX, not eighty-two times. The loss is computed in
> original-image pixels — the metric's own units. And checkpoints are
> selected on a metric-aligned blend of landmark and measurement error,
> never on canvas loss.

## Slide 4 — The method in one picture  (2:15)

> Here is the whole method in one picture. Phase one, on the left, adapts a
> DINOv2 encoder to ultrasound using all hundred-and-ninety-one thousand
> unlabeled frames — no labels anywhere in that phase. Phase two takes the
> adapted encoder, taps it at four depths, and trains an HRNet-style neck
> with nine task-specific heads on the labeled images. Landmarks are read
> out with a soft-argmax and become the clinical measurements. The next
> slides walk through this, left to right.

## Slide 5 — From image to patch tokens  (2:55)

> One piece of background first. Images are letterboxed to five-eighteen by
> five-eighteen, preserving aspect ratio, and the ViT-Large backbone turns
> them into a thirty-seven by thirty-seven grid of patch tokens. Two things
> to remember: this grid is coarse — fourteen pixels per token — and we tap
> the encoder at four depths, not just the last block, so coarse semantics
> and finer detail both reach the decoder.

## Slide 6 — Phase 1 — self-supervised domain adaptation  (3:30)

> Phase one. Off-the-shelf DINOv2 is trained on natural images, and
> sonography has very different texture statistics, acquisition physics, and
> artifacts. So before asking the encoder to localize anything, we continue
> DINOv2's own pretraining on the unlabeled ultrasound frames — the
> student–teacher setup you see here, where the student is trained and an
> EMA teacher provides the targets.

## Slide 7 — Phase 1 — the recipe in brief  (4:00)

> The recipe in brief. Multi-crop self-distillation: two global crops and
> six small local crops. The student sees all eight views; the teacher only
> the two globals. The loss is DINO self-distillation on the CLS token, plus
> iBOT masked patch prediction, plus a small KoLeo term against collapse.
>
> The part I want to highlight is the resolution curriculum: a hundred
> epochs at two-twenty-four pixels, then just a four-epoch tail at
> five-eighteen — the deployment resolution. Full resolution throughout
> would cost roughly five times more per epoch; the whole phase is about
> ninety-six GPU-hours on a single A100.
>
> ✂ And the augmentations are ultrasound-aware: we crop to the fan, keep
> rotations gentle because view orientation carries meaning, and resample
> background-heavy crops.

## Slide 8 — Phase 2 — HRNet-style multi-resolution neck  (5:00)

> Phase two decodes landmarks from those coarse tokens. Instead of a U-Net
> that downsamples and then climbs back up, we use an HRNet-style neck:
> three parallel branches, at resolutions thirty-seven, seventy-four, and
> one-forty-eight, with two rounds of bidirectional exchange, fused into a
> single high-resolution map.
>
> One detail that mattered a lot: GroupNorm everywhere, never BatchNorm.
> Our batches are task-homogeneous, so BatchNorm's running statistics would
> blend nine incompatible domains — we actually watched training error fall
> while validation stayed flat, until we changed this.

## Slide 9 — Phase 2 — task routing and sub-pixel decoding  (5:45)

> On top of the neck sit nine independent heads, and the batch's task ID
> selects exactly one — which works precisely because every batch contains a
> single task.
>
> Each head outputs one logit map per landmark, and we decode with
> soft-argmax: a temperature-scaled softmax, then the spatial expectation.
> That is differentiable and sub-pixel, so the network is a direct
> coordinate regressor — heatmaps are never supervised with MSE. The loss is
> a masked L1 over the annotated landmarks, and predictions are
> inverse-letterboxed back to original-image pixels — so training happens in
> the metric's units. ✂ Everything trains on a single A100, and what we
> validate and deploy is the EMA teacher.

## Slide 10 — Official challenge evaluation  (6:40)

> On the official validation phase — six hundred and nineteen images, the
> official scorer — we reach an overall mean radial error of twenty-six
> point three pixels and a measurement error of twenty-nine point seven.
> For context, the best leaderboard entry at the time of writing was at
> twenty-two point six and twenty-nine point zero.
>
> In red are our weak points: HC, PSAX, and IVC. For PSAX and IVC this is
> somewhat expected — they have the fewest training images, and evaluation
> sets of only eighteen and ten images, so those numbers carry real
> uncertainty. HC is the interesting one: it has nine hundred and
> ninety-nine labeled images — an order of magnitude more than the cardiac
> tasks — and is still the worst task. So difficulty is not explained by
> dataset size alone.

## Slide 11 — Qualitative results  (7:40)

> Qualitatively: the same network, across four different clinical domains —
> ground truth in green, predictions in red — handling very different
> anatomy and image appearance.

## Slide 12 — How long should self-supervised adaptation run?  (7:55)

> Finally, the question we studied: how long should self-supervised
> adaptation actually run? We probe frozen encoder checkpoints with a small
> twenty-five-epoch decoder, which isolates representation quality from
> fine-tuning.
>
> The answer: adaptation is not monotonic. Error bottoms out at epoch
> twenty, and by epoch one hundred the score is actually worse than the
> non-adapted encoder. But the short full-resolution tail rescues the late
> checkpoints — it gives the best score of any checkpoint we tested, so we
> selected epoch one-oh-four. And at matched epochs, training at full
> resolution from the start is worse than the two-twenty-four curriculum.
>
> So: more self-supervision is not automatically better, the failure is
> invisible unless you probe the frozen encoder — and the cheap curriculum
> is also the better one.

## Slide 13 — Conclusions  (8:50)

> To conclude. One model for nine heterogeneous biometry tasks. A hundred
> and ninety-one thousand unlabeled frames supporting seven thousand labels,
> by separating representation learning from localization. Self-supervision
> duration is non-monotonic, and a short high-resolution tail beats more
> low-resolution epochs. And the design follows the metric throughout. The
> code is on GitHub, and we thank the NVIDIA Academic Hardware Grant Program
> for the compute. Thank you — happy to take questions.

---

## Q&A appendix — likely questions

There are no backup slides, so these answers are for saying out loud.

**"Does the HRNet neck actually earn its complexity?"**
Honest answer: the evidence is mixed. Over 5 matched CV folds from the same
ep104 initialization, the HRNet neck wins our internal blend criterion in 3
of 5 folds (0.0707 ± 0.0089 vs 0.0740 ± 0.0068), but the simple decoder has
slightly lower mean MRE (25.24 vs 25.69 px). Differences are small against
fold-to-fold variance, so we don't claim a decisive architectural advantage.

**"What were the Phase-2 training details?"**
Batch 64, bf16, AdamW at base LR 2e-4 with 15-epoch warmup and 150-epoch
cosine decay, weight decay excluded from norms and biases. Last 4 backbone
blocks unfrozen at 0.1× base LR with layer-wise decay 0.75. EMA teacher
α = 0.999, early-stopping patience 40, sampler temperature 0.5. 312.2M
parameters, 58.2M trainable; 84.4 img/s training, up to 127.3 img/s
inference. Phase 2 is cheap — the cost was Phase 1's ~96 GPU-hours.

**"Phase-1 hyperparameters?"**
DINO head over K = 65,536 prototypes; iBOT masks restricted to foreground;
teacher temperature annealed 0.04 → 0.07; EMA momentum 0.994 → 1.0;
effective batch 256 via gradient accumulation.

**"Why no ensembling or TTA?"**
Deliberately left as future work — 5-fold ensembling with multi-scale and
intensity TTA is the obvious lever for the tiny tasks, but the submitted
model is a single network so the ablations stay interpretable.

**"What do the adapted features look like?"**
A PCA of the patch tokens at ep104 (figure 5 in the paper) shows the adapted
encoder separating anatomy from background and speckle.

**"Why is HC so bad despite 999 labels?"**
Open question. What we can say from the data is only the negative result:
with an order of magnitude more labels than the cardiac tasks, HC is still
the single worst task, so difficulty is not explained by dataset size. We
did not isolate the cause — if you speculate (annotation ambiguity of the
4-point ellipse, image scale), flag it explicitly as speculation.
