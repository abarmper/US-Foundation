#!/usr/bin/env python3
"""Fill the FUB 2026 "Structured Documentation of Your Approach" template.

python-docx is not available here, and a .docx is just a zip: the 16 template
placeholders (6 x "[Enter response]" in the header block, 10 x "[Enter response
here]" in the detailed-description table) each live in their own <w:r> run, so
they can be substituted directly in word/document.xml without disturbing the
organizers' styling.

The placeholder runs are styled italic + grey (#6B7280) to read as unfilled
prompts; that formatting is stripped on the runs we fill so answers render as
normal body text.

ANSWERS is the single source of truth: this script emits both the .docx and a
plain-text mirror from it, so the two cannot drift.

Usage:
    python3 fill_docx.py [path/to/template.docx]
"""
import html
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE / "template_structured_documentation.docx"
OUT_DOCX = HERE / "structured_documentation_filled.docx"
OUT_MD = HERE / "structured_documentation_filled.md"

TASKS = [
    # task, landmarks, validation N, MRE, MAE
    ("HC (fetal head circumference)", 4, 215, "49.07", "76.70"),
    ("FA (fetal abdomen)", 4, 188, "21.61", "92.16"),
    ("fetal_femur (fetal femur)", 2, 62, "25.33", "18.10"),
    ("FUGC (cervical)", 2, 20, "9.88", "11.55"),
    ("AOP (angle of progression)", 4, 60, "16.02", "9.07"),
    ("PLAX (parasternal long axis)", 22, 26, "18.94", "9.48"),
    ("A4C (apical 4-chamber)", 16, 20, "23.54", "20.37"),
    ("PSAX (parasternal short axis)", 4, 18, "39.02", "20.00"),
    ("IVC (inferior vena cava)", 2, 10, "33.65", "9.87"),
]

_TASK_LINES = "\n".join(
    f"  - {name}: {k} landmarks, N={n}, MRE {mre} px, MAE {mae}"
    for name, k, n, mre, mae in TASKS
)

# Hidden-test-phase results, final CodaBench submission 886072 (per-task N not
# provided by the platform's results table at time of writing).
TEST_TASKS = [
    # task, MRE, MAE
    ("HC", "26.49", "44.30"),
    ("FA", "30.24", "116.84"),
    ("fetal_femur", "21.99", "19.17"),
    ("FUGC", "13.79", "10.30"),
    ("AOP", "65.01", "71.92"),
    ("PLAX", "20.32", "10.54"),
    ("A4C", "27.62", "20.48"),
    ("PSAX", "41.96", "19.81"),
    ("IVC", "28.49", "17.67"),
]
_TEST_TASK_LINES = "\n".join(
    f"  - {name}: MRE {mre} px, MAE {mae}" for name, mre, mae in TEST_TASKS
)

# --------------------------------------------------------------------------
# Header block (placeholders 1-6)
# --------------------------------------------------------------------------

A1_TEAM_NAME = "apbiomed1234"

A2_ACCOUNT = "annapan"

A3_CORRESPONDING = "Anna Panagiotakopoulou"

A4_EMAIL = "annapan@biomed.ntua.gr"

A5_DOCKER = (
    "apbiomed1234/fu-biometry@sha256:"
    "d8cec8c12dfddd98cab1383f11fbecd5962c7f92c27f0f04b293814218f79806"
)

A6_SUBMISSION_ID = "886072"

# --------------------------------------------------------------------------
# Detailed description (placeholders 7-16)
# --------------------------------------------------------------------------

A7_ARCHITECTURE = f"""Overview. A single unified multi-task landmark-detection network covering all nine tasks, built in two phases: (1) self-supervised domain adaptation of a DINOv2 encoder on the challenge's unlabeled ultrasound frames, and (2) supervised training of an HRNet-style multi-resolution neck plus nine task-specific coordinate-regression heads on the labeled frames. The final submission is one such model; there is no ensemble.

Phase 1 - self-supervised domain adaptation (semi-supervised component).
The backbone is DINOv2 ViT-L/14 with registers (dinov2_vitl14_reg; 24 transformer blocks, embedding dimension 1024). It is adapted to ultrasound by continued pretraining that follows the original DINOv2 formulation: CLS-token self-distillation (DINO), masked patch-token prediction (iBOT), and a KoLeo entropic spread regularizer, all under an exponential-moving-average (EMA) teacher that receives no gradient. A local-to-global multi-crop scheme produces 2 global crops and 6 local crops; the student sees all 8 views, the teacher only the 2 global views. This is the only place unlabeled data enters the pipeline: there is no pseudo-labeling and no consistency loss on the labeled data in Phase 2.

Phase 2 - encoder.
The adapted encoder consumes a 518x518 letterboxed image and produces a 37x37 grid of 1024-dimensional patch tokens (518/14 = 37). The submitted model uses the FINAL block's patch tokens only (single-level input; a multi-level variant tapping four intermediate depths exists in the codebase but was NOT used for this submission). The encoder is frozen except for its last 4 transformer blocks and the final norm layer, which are fine-tuned.

Phase 2 - neck (TrueHRNetNeck).
A multi-stage HRNet-style neck with parallel branches at three resolutions and repeated bidirectional exchange, with branch widths (w1, w2, w3) = (128, 96, 64):
  - Stage 1: b1 = Conv3x3-GroupNorm-ReLU (1024 -> 128) at 37x37; b2 = ConvTranspose2d (1024 -> 96, kernel 4, stride 2, pad 1) + GroupNorm + ReLU at 74x74.
  - Exchange 1 (two branches, bidirectional): 74 -> 37 by a stride-2 3x3 conv; 37 -> 74 by 1x1 projection + bilinear upsample. Fused by residual addition then ReLU.
  - Stage 2: b1 = Conv3x3-GN-ReLU (128 -> 128); b2 = Conv3x3-GN-ReLU (96 -> 96); b3 = ConvTranspose2d (96 -> 64) + GN + ReLU at 148x148.
  - Exchange 2 (three branches, all pairs): six resampling paths (74->37, 148->37 via two stride-2 steps, 37->74, 148->74, 37->148, 74->148), again residual-added and ReLU'd.
  - Fusion: each branch is projected to 128 channels and resampled to 148x148 (b1 x4 up, b2 x2 up, b3 1x1 only) and the three are summed, followed by Dropout2d(p=0.3) and a final Conv3x3-GN-ReLU (128 -> 128). Output: 148x148 x 128.

Normalization. GroupNorm is used throughout the neck and heads, never BatchNorm. Training batches are task-homogeneous (see section 3), so BatchNorm would normalize with per-task statistics at training time but with a single set of running statistics blended across cardiac, fetal and vascular domains at evaluation time. That train/eval mismatch was observed to keep validation error flat while training error fell. GroupNorm computes statistics per sample and behaves identically in train and eval mode.

Phase 2 - task-specific heads.
Nine independent heads, one per task, each: Conv3x3 (128 -> 128) + GroupNorm + ReLU; Conv3x3 (128 -> 64) + GroupNorm + ReLU; Dropout2d(p=0.3); bilinear upsample to 128x128; Conv1x1 (64 -> K) producing K per-landmark logit maps. Landmark counts K per task:
{_TASK_LINES}

Task routing. Each forward pass runs exactly one head, selected by the task identity of the batch (at inference, the task_id column of test_metadata.csv). This is why training batches must be task-homogeneous.

Coordinate extraction. Coordinates are decoded by soft-argmax rather than by taking the arg-max bin: the K logit maps are flattened, scaled by a temperature of 10.0, passed through a softmax, and the spatial expectation of the resulting distribution is taken over the pixel grid. The result is rescaled by 518/128 into 518-canvas pixels. This is differentiable and gives sub-pixel resolution, so the network is trained as a direct coordinate regressor; the heatmaps are never supervised with an MSE/heatmap loss.

Model size: approximately 312M parameters total, of which approximately 58M are trainable in Phase 2."""

A8_PREPROCESSING = """Identical geometric preprocessing is used in training, validation and inference, so there is no train/test preprocessing mismatch.

Steps, in order (implemented with Albumentations in the submitted container):
  1. Read the image with OpenCV and convert BGR -> RGB.
  2. LongestMaxSize(max_size=518): aspect-ratio-preserving resize so that the longer side becomes 518 px. Aspect ratio is deliberately preserved because native image sizes differ substantially across tasks (roughly 336x544 median for FUGC up to 768x1024 for FA), and a non-uniform resize would distort the anatomy that the landmarks describe.
  3. PadIfNeeded(min_height=518, min_width=518, border_mode=BORDER_CONSTANT, fill=0): centered zero-padding onto a 518x518 canvas ("letterbox").
  4. Normalize with ImageNet statistics (mean 0.485/0.456/0.406, std 0.229/0.224/0.225).
  5. ToTensorV2 -> CHW float tensor.

518 is used because it is the DINOv2 ViT-L/14 evaluation resolution and is divisible by the patch size 14 (518 = 14 x 37); the positional embeddings are interpolated to that grid.

Not used: intensity clipping or windowing, per-image histogram equalization at inference, resampling to physical spacing, patch or sliding-window extraction, channel construction beyond RGB replication, or any modality-specific correction. Images are used at their native grey levels.

The exact letterbox geometry (scale factor and pad offsets) is recomputed at output time to invert this mapping back to original-image pixels; see section 8."""

A9_DATA = """Data used. Only data released by the challenge. The released training set comprises 197,938 images: 6,768 labeled and 191,170 unlabeled (a labeled fraction of roughly 3.4%).

Phase 1 (self-supervised) used the unlabeled pool, with no labels of any kind.

Phase 2 (supervised) used the labeled pool only, split 80% / 20% into training and internal validation. The submitted model was trained on fold 0 of that split.

Cross-validation: NO. The submitted model is a single model trained on one fixed 80/20 partition. A 5-fold cross-validation facility exists in the released code and was used for internal ablations, but the submitted model is not a fold ensemble and no cross-validation result contributed to it.

Class/task imbalance and sampling. The labeled pool is severely imbalanced across tasks, from roughly 4,000 images for AOP down to fewer than 100 for IVC (38), PSAX (49) and PLAX; HC has 999. Because each forward pass routes to exactly one task head, batches must be task-homogeneous, so a plain shuffling loader cannot be used. A deterministic task-homogeneous sampler builds each batch from a single task. For the submitted model the per-task allocation was fully balanced across the nine tasks (temperature parameter 0.0), i.e. equal batch allocation per task rather than allocation proportional to task size.

Missing landmarks. Landmarks that are not annotated for a given image are encoded as (-1, -1) and masked out of the loss, so they contribute no gradient.

Validation and model selection. An EMA teacher copy of the weights is evaluated on the held-out 20% after each epoch. Checkpoint selection and early stopping use an internal criterion referred to as "blend": a macro-averaged, per-task-normalized combination of landmark error and derived clinical-measurement error, computed in ORIGINAL-image pixel space rather than on the 518 canvas, so that selection is aligned with the challenge metric rather than with the training-canvas loss. Lower is better. The selected checkpoint is the deployed artifact.

See the note at the end of section 4 for the training-schedule values that still require confirmation."""

A10_LOSS = """PHASE 1 (self-supervised).
Total objective: L = L_dino + L_ibot + 0.1 * L_koleo.
  - L_dino: cross-entropy between the teacher's and student's CLS-token distributions over K = 65,536 prototypes, after centering and sharpening the teacher's logits. Applied across the local-to-global crop pairs.
  - L_ibot: masked patch-token prediction. A block-wise, foreground-restricted subset of the student's global-crop patch tokens is masked; the student must recover the teacher's distribution for each masked patch, through a second, separately parameterized head of identical architecture. Averaged over the masked patches.
  - L_koleo: a Kozachenko-Leonenko entropic regularizer penalizing overly similar CLS embeddings within a batch, discouraging representational collapse independently of centering. Weight 0.1.
Teacher temperature annealed 0.04 -> 0.07 over the first 30 epochs; teacher EMA momentum 0.994 -> 1.0; the final projection layer is frozen for the first epoch; gradient clipping at 3.0.
Optimizer AdamW, learning rate 1e-4 with 10 epochs of linear warmup then cosine decay, weight decay cosine 0.04 -> 0.2, effective batch size 256 (batch 32 with 8 gradient-accumulation steps).
Resolution curriculum: 100 epochs at 224x224 global crops, followed by a 4-epoch high-resolution adaptation tail at 518x518 (learning rate 5e-5, 1 warmup epoch, effective batch 256 via 16 x 16), so the encoder is finally tuned at the resolution Phase 2 deploys. The Phase-1 checkpoint used by the submission is the one at the end of that tail (epoch 104).

PHASE 2 (supervised).
Primary loss: a masked L1 loss on the soft-argmax coordinates - a direct coordinate regression. For each annotated landmark the L1 distance between predicted and ground-truth (x, y) is taken in 518-canvas pixel space; unannotated landmarks are masked out. There is no heatmap/MSE supervision anywhere.
Soft-argmax temperature: 10.0.
Auxiliary losses: none. The DSNT-style heatmap regularizer available in the codebase was disabled for this model (weight 0.0), and no clinical-measurement auxiliary loss was used. There is no consistency loss, no adversarial loss and no boundary loss; the semi-supervised contribution is entirely in Phase 1.
Optimizer: AdamW, weight decay 1e-2, with the fine-tuned encoder blocks on a 0.1x multiple of the base learning rate. Layer-wise learning-rate decay is disabled (factor 1.0). Schedule: linear warmup then cosine decay. Mixed precision: fp16. Gradient-norm clipping is applied.
An EMA teacher (decay 0.999) is maintained and is what gets validated and deployed.

Phase-2 schedule, confirmed directly against the submitted run's own record (runs/dv2ep104_hrnet_reg/config.json and its training log -- not inferred): batch size 16; base learning rate 1e-4; warmup epochs 15; early-stopping patience 30. The run's epoch cap was 500, but early stopping fired at epoch 114 off a best checkpoint saved at epoch 84 (Val Loss 10.88043, Val MRE 8.85px in 518-canvas space). Weight decay (1e-2) is applied as a single flat AdamW value across all trainable parameters -- it is NOT excluded from norm or bias parameters; no such exclusion logic exists in the training script. EMA decay is exactly 0.999. Wherever the paper's Section 3 description and this run's actual recipe disagreed, the values above and in section 3 follow what was actually shipped in the submitted container (single-level neck, 128x128 heatmaps, balanced sampling, canvas-space loss, no DSNT, no layer-wise decay, fp16)."""

A11_AUGMENTATION = """Augmentation is applied only during training. Validation and inference use the deterministic letterbox of section 2 with no augmentation of any kind.

PHASE 2 (supervised training), applied between the letterbox resize and the normalization, with keypoints transformed consistently:
  - Affine: scale 0.95-1.05, translation +/- 10% of image size, rotation +/- 30 degrees, probability 0.5.
  - RandomBrightnessContrast: brightness limit 0.2, contrast limit 0.2, probability 0.3.
  - OneOf (probability 0.5): RandomGamma with gamma limits (70, 130), or CLAHE with clip limit 4.0 and 8x8 tile grid.
  - GaussNoise: standard-deviation range 0.02-0.1, probability 0.3.

PHASE 1 (self-supervised), an ultrasound-specific variant of the DINOv2 multi-crop pipeline:
  - Foreground cropping to the ultrasound fan bounding box before cropping, so crops land on tissue rather than on the black surround.
  - 2 global crops at 224x224 (scale range from 0.32) and 6 local crops at 98x98 (scale range 0.05-0.32); the high-resolution tail uses 518x518 global crops.
  - Rotation limited to +/- 10 degrees - deliberately milder than Phase 2, because in several tasks the orientation of the probe view is semantically meaningful.
  - Local crops whose foreground fraction falls below 0.02 are rejected and resampled, to avoid near-empty background crops.
  - iBOT masking is restricted to foreground patches.
  - Standard photometric augmentation (brightness/contrast, gamma/CLAHE, blur/noise).

No MixUp, CutMix, style transfer, elastic deformation or copy-paste augmentation was used. No temporal augmentation was used; all data are treated as independent 2D frames."""

A12_EXTERNAL = """External datasets: NONE. No data other than that released by the challenge organizers was used, for either phase. No additional private annotation of any public dataset was performed.

Pretrained model: YES - exactly one.
  - Model: DINOv2 ViT-L/14 with registers (dinov2_vitl14_reg), by Meta AI.
  - Source: https://github.com/facebookresearch/dinov2 (obtained through torch.hub).
  - License: Apache License 2.0 - publicly available, permits this use.
  - How used: as the initialization for Phase-1 self-supervised domain adaptation on the challenge's unlabeled ultrasound frames. The adapted encoder is then the Phase-2 backbone. No other pretrained weights (ImageNet classifiers, medical foundation models, detection backbones) are used anywhere in the pipeline.

Note on the submitted container: because the evaluation container runs fully offline (--network none), a copy of the facebookresearch/dinov2 repository is vendored into the image and the model is constructed with pretrained=False. This builds the architecture without any network access; every weight is then loaded from our own trained checkpoint, which is baked into the image. Meta's pretrained weights are therefore not shipped in the image and are not present at inference - they enter the pipeline only as the Phase-1 starting point during training."""

A13_INFERENCE = """Inference. Full-image inference at the native 518x518 letterboxed resolution - no sliding window, no tiling, no overlap or blending, since the target is a small set of global landmarks rather than a dense map. Images are processed one at a time: the entry script offers a batch size of 8, but each forward pass routes to exactly one task head, so batching would require grouping by task first for no meaningful gain at this scale. Inference runs in fp32.

For each image the task_id from test_metadata.csv selects the head; the model emits K logit maps at 128x128, which are decoded to coordinates by soft-argmax (temperature 10.0, spatial expectation, rescaled by 518/128).

Test-time augmentation: NO. The submitted run uses a single forward pass per image with no flipping, no multi-scale views and no intensity views.

Ensemble: NO. The submission is one model, from one training run, on one data partition. Multi-scale and intensity TTA and a 5-fold ensemble are both implemented in the released codebase and were used in internal validation-phase experiments, but neither was used for the final test submission.

Postprocessing: essentially none. Because the model regresses coordinates directly, there is no thresholding, no argmax peak-picking, no connected-component analysis, no morphological operation, no CRF and no smoothing. The only operations applied after decoding are the geometric inverse-letterbox and a clip to the image bounds, both described in section 8."""

A14_OUTPUT = """Restoring original geometry. Predicted coordinates come out in 518-canvas pixels and are mapped back to original-image pixels by inverting the letterbox exactly:
    scale   = 518 / max(orig_h, orig_w)
    new_h   = round(orig_h * scale);  new_w = round(orig_w * scale)
    pad_top = (518 - new_h) // 2;     pad_left = (518 - new_w) // 2
    x_orig  = (x_canvas - pad_left) / scale
    y_orig  = (y_canvas - pad_top)  / scale
The resized dimensions are computed with round(), not truncation. Albumentations' LongestMaxSize rounds to nearest when it computes the resized target size; truncating here instead reproduces its actual output for only about half of all aspect ratios, and introduces a systematic one-pixel coordinate drift on the remainder. This was identified and fixed before submission.

Bounds. Coordinates are clipped to [0, orig_w] in x and [0, orig_h] in y.

Format. For each row of test_metadata.csv one record is emitted with image_path and task_id copied verbatim from the CSV, plus predicted_points_pixels: a flat float list [x1, y1, x2, y2, ...] of length num_classes x 2, in the task's canonical landmark order. All records are written as a single JSON list to /output/regression_predictions.json.

Checks performed before submission:
  - The checkpoint loads with strict=True: every parameter key matches, with no DataParallel prefix mismatch.
  - The containerized pipeline reproduces the reference (already-scored) prediction file bit-for-bit on the same inputs.
  - A real docker build and run with --network none, /input read-only and /output writable exits with code 0.
  - The output file is re-read and checked for record count and for the set of task_ids present.
An image that cannot be decoded is skipped with a warning rather than aborting the run; this did not occur on the evaluation data, and every expected record was produced."""

A15_REPRODUCIBILITY = """Entry points (inference, as executed by the evaluation platform).
  - /app/predict.py - the organizer-provided container entry script, used UNMODIFIED. It prepares the working directory and calls model.Model.predict().
  - /app/model.py - defines class Model with predict(data_root, output_dir, batch_size): preprocessing, per-task routing, soft-argmax decode, inverse letterbox, JSON writing.
  - /app/model_factory.py - defines the network. The submitted model is constructed as UnifiedBiometryModel(freeze_encoder=True, unfreeze_last_n_blocks=4, neck_branch_width=(128, 96, 64), backbone_name='dinov2_vitl14_reg'), with the default heatmap size of 128.
  - ENTRYPOINT ["python3", "/app/predict.py"]; output fixed at /output/regression_predictions.json.

Checkpoint. /app/best_model.pth, copied into the image at build time (the container is fully offline, so nothing is downloaded at run time). It is loaded with strict=True.

Checkpoint selection rule. The EMA-teacher weights at the epoch with the best internal "blend" score on the held-out 20% of fold 0 (see section 3).

Environment.
  - Base image: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime (torch 2.3.1, CUDA 12.1). torch and torchvision are deliberately NOT reinstalled from requirements.txt, to avoid pulling a CPU-only or CUDA-mismatched wheel over the base image's correct build.
  - Additional Python dependencies, pinned: numpy 1.26.4, opencv-python-headless 4.11.0.86, albumentations 2.0.8, albucore 0.0.24.
  - System packages: libgl1-mesa-glx, libglib2.0-0, libsm6, libxrender1, libxext6.
  - NO_ALBUMENTATIONS_UPDATE=1, so that Albumentations makes no version-check network call on import.
  - Platform linux/amd64. Verified within the evaluation profile (RTX 3080, 10 GB VRAM, 7 GB container memory, 4 CPU cores, offline).

Training hardware. A single NVIDIA A100-SXM4-80GB GPU per run. Phase 1 took roughly 96 GPU-hours in total.

Random seed. 42, for both phases.

Source code. https://github.com/abarmper/US-Foundation - the released package (gubiometry) exposes the full pipeline through one CLI: `python -m gubiometry make-splits`, then `phase1 --config ...`, `phase2 --config ...`, and `predict --config ...`, each driven by a single YAML configuration.

Important note for reviewers reproducing this exactly. The submitted checkpoint was produced by an earlier iteration of our training code (an experiment directory, v4_true_hrnet_softargmax_reg), which the released gubiometry package supersedes and cleans up. The network definition shipped in the container is a byte-for-byte copy of that earlier architecture file, with a single change: the DINOv2 backbone is built from the vendored local repository copy with pretrained=False so that construction succeeds offline. The single-level neck parameter keys are identical between the two code versions, so the submitted checkpoint loads unchanged into the released package as well; this was verified."""

A16_ADDITIONAL = f"""Per-task summary. All nine tasks are served by one network; only the head and its landmark count differ. Official validation-phase evaluation (619 images, the official scorer) for the submitted model:
{_TASK_LINES}
  - Overall (unweighted average over the nine tasks): N=619, MRE 26.34 px, MAE 29.70.
These validation-phase figures are the ones reported in our paper.

Hidden-test-phase evaluation, official CodaBench scorer, final submission ID 886072 (per-task N not shown in the platform's results table):
{_TEST_TASK_LINES}
  - Overall (unweighted average over the nine tasks): MRE 30.66 px, MAE 36.78.
This hidden-test result is the authoritative one for competition ranking; it is higher than the validation-phase figures on most tasks (most notably AOP: 65.01 vs 16.02 px), consistent with ordinary validation-to-test generalization gap rather than any known pipeline discrepancy. HC is a partial exception, improving from 49.07 to 26.49 px.

Architectural constraint worth stating explicitly. Each forward pass executes exactly one task head, chosen by the batch's task identity. Every batch must therefore be task-homogeneous. This is enforced by a dedicated sampler during training and by per-image routing at inference; a conventional shuffling data loader would silently produce wrong results.

Design choices specific to this problem.
  - GroupNorm rather than BatchNorm throughout the neck and heads, because the nine tasks span cardiac, fetal and vascular appearance statistics and the batches are task-homogeneous (see section 1).
  - Soft-argmax coordinate regression rather than heatmap regression, so that the training objective is measured in the same units as the challenge metric and is sub-pixel from the outset.
  - Aspect-ratio-preserving letterboxing rather than direct resizing, because native resolutions differ by more than a factor of two across tasks and the landmarks encode anatomical geometry.
  - Balanced per-task batch allocation, so that AOP (about 4,000 labeled images) does not dominate the tasks with fewer than 100.

Known limitations.
  - The submission is a single model on a single data partition, with no test-time augmentation and no ensembling. All three are implemented in our codebase and improve validation scores; they were not used here.
  - Per-task reliability varies widely. PSAX and IVC have both the fewest labeled training images and the smallest evaluation sets (N=18 and N=10), so their per-task figures carry considerable uncertainty.
  - HC is the weakest task despite having 999 labeled images - an order of magnitude more than the cardiac tasks - so per-task difficulty in this benchmark is not explained by training-set size alone.
  - The clinical-measurement half of the metric is derived from the predicted landmarks by the official scorer (an angle in degrees for AOP, lengths for the remaining tasks); we optimize the landmarks and the measurement error follows, rather than regressing the measurements directly."""

ANSWERS = [
    A1_TEAM_NAME, A2_ACCOUNT, A3_CORRESPONDING, A4_EMAIL, A5_DOCKER, A6_SUBMISSION_ID,
    A7_ARCHITECTURE, A8_PREPROCESSING, A9_DATA, A10_LOSS, A11_AUGMENTATION,
    A12_EXTERNAL, A13_INFERENCE, A14_OUTPUT, A15_REPRODUCIBILITY, A16_ADDITIONAL,
]

LABELS = [
    "Team name", "CodaBench account / team account", "Corresponding author",
    "Contact email", "Final Docker image name and immutable digest",
    "Final CodaBench submission ID selected for reporting",
    "1. Architecture and complete processing pipeline",
    "2. Preprocessing",
    "3. Training data, partitioning, and validation",
    "4. Loss functions and optimization",
    "5. Data augmentation",
    "6. External data and pretrained models",
    "7. Inference, postprocessing, and ensemble strategy",
    "8. Output generation and validation",
    "9. Reproducibility",
    "10. Additional information",
]

PLACEHOLDER_RUN = re.compile(
    r"<w:r>(?:(?!</w:r>).)*?\[Enter response(?: here)?\](?:(?!</w:r>).)*?</w:r>",
    re.DOTALL,
)
RPR = re.compile(r"<w:rPr>.*?</w:rPr>", re.DOTALL)


def _runs(text):
    """Escape and turn newlines into <w:br/>-separated <w:t> runs."""
    escaped = html.escape(text, quote=False)
    parts = escaped.split("\n")
    joined = '</w:t><w:br/><w:t xml:space="preserve">'.join(parts)
    return f'<w:t xml:space="preserve">{joined}</w:t>'


def _rebuild(match, answer):
    """Replace one placeholder run, dropping the italic/grey placeholder styling."""
    rpr = RPR.search(match.group(0))
    props = ""
    if rpr:
        props = rpr.group(0)
        props = props.replace("<w:i/>", "")
        props = re.sub(r'<w:color w:val="6B7280"\s*/>', "", props)
        if props == "<w:rPr></w:rPr>":
            props = ""
    return f"<w:r>{props}{_runs(answer)}</w:r>"


def main():
    template = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEMPLATE
    if not template.exists():
        sys.exit(f"template not found: {template}")

    with zipfile.ZipFile(template) as zin:
        names = zin.namelist()
        blobs = {n: zin.read(n) for n in names}

    xml = blobs["word/document.xml"].decode("utf-8")

    found = PLACEHOLDER_RUN.findall(xml)
    if len(found) != len(ANSWERS):
        sys.exit(f"expected {len(ANSWERS)} placeholder runs, found {len(found)}")

    counter = {"i": 0}

    def sub(m):
        i = counter["i"]
        counter["i"] += 1
        return _rebuild(m, ANSWERS[i])

    xml = PLACEHOLDER_RUN.sub(sub, xml)
    blobs["word/document.xml"] = xml.encode("utf-8")

    if OUT_DOCX.exists():
        OUT_DOCX.unlink()
    with zipfile.ZipFile(OUT_DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:  # preserve original entry order
            zout.writestr(n, blobs[n])

    md = ["# Structured Documentation of Your FUB 2026 Approach",
          "",
          "> Plain-text mirror of `structured_documentation_filled.docx`.",
          "> Generated by `fill_docx.py` - edit ANSWERS there and re-run, so the",
          "> two never drift apart.",
          "",
          "## Team and submission information",
          ""]
    for label, answer in zip(LABELS, ANSWERS):
        if label.startswith("1. "):
            md += ["", "## Detailed description", ""]
        heading = f"**{label}:**" if not label[0].isdigit() else f"### {label}"
        md += [heading, "", answer, ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"wrote {OUT_DOCX}  ({OUT_DOCX.stat().st_size:,} bytes)")
    print(f"wrote {OUT_MD}  ({OUT_MD.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
