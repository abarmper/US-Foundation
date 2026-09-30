#!/usr/bin/env python3
"""Build the FUB 2026 challenge presentation.

13 slides, 16:9, from the paper (docs/MWM_54/main.tex), paced for a 10-minute
talk: the front half is picture-led (three slide-only schematics from
make_extra_schematics.py plus the paper's figures), and the dense
hyperparameter material lives in the speaker notes and in talk_script.md's
Q&A appendix (there are no backup slides).

SLIDES below is the single source of truth: it drives the .pptx AND outline.md,
so the deck and its text mirror cannot drift apart. Speaker notes go into the
deck's real notes slides as well as the mirror; where a bullet was trimmed for
time, the removed facts were moved into the notes.

Bullet/caption text supports inline markup:  **bold**  ·  *italic*  ·
__bold accent color__ .  The markers pass through verbatim into outline.md,
where they are ordinary Markdown.

Usage:
    python3 make_vit_schematic.py && python3 make_extra_schematics.py
    python3 build_slides.py
"""
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"
OUT_PPTX = HERE / "31_annapan_54_presentation.pptx"
OUT_MD = HERE / "outline.md"

SLIDE_W, SLIDE_H = 13.333, 7.5

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x6B, 0x4E, 0x8E)
MUTED = RGBColor(0x66, 0x66, 0x66)
RULE = RGBColor(0xD9, 0xC7, 0xA8)
BG = RGBColor(0xFF, 0xFF, 0xFF)
BAD = RGBColor(0xB3, 0x3A, 0x30)
GOOD = RGBColor(0x2E, 0x7D, 0x32)

# (name, landmarks, N, MRE, MAE, mre_flag, mae_flag); flags: "worst" | "best"
TASKS = [
    ("HC  (head circumference)", 4, 215, "49.07", "76.70", "worst", "worst"),
    ("FA  (fetal abdomen)", 4, 188, "21.61", "92.16", "", "worst"),
    ("fetal_femur", 2, 62, "25.33", "18.10", "", ""),
    ("FUGC  (cervical)", 2, 20, "9.88", "11.55", "best", ""),
    ("AoP  (angle of progression)", 4, 60, "16.02", "9.07", "", "best"),
    ("PLAX", 22, 26, "18.94", "9.48", "", ""),
    ("A4C", 16, 20, "23.54", "20.37", "", ""),
    ("PSAX", 4, 18, "39.02", "20.00", "worst", ""),
    ("IVC", 2, 10, "33.65", "9.87", "worst", ""),
]

# ---------------------------------------------------------------------------
# Slide content.  kind: "title" | "bullets" | "image" | "image_bullets" |
#                       "two_images" | "table"
# ---------------------------------------------------------------------------

SLIDES = [
    dict(
        kind="title",
        title="Self-Supervised Domain Adaptation with DINOv2-HRNet\n"
              "for Multi-Task Ultrasound Landmark Detection",
        subtitle=[
            "A. Panagiotakopoulou   ·   A. Barmperis   ·   V. E. Katsigiannis   ·   G. K. Matsopoulos",
            "Biomedical Engineering Laboratory, School of Electrical and Computer Engineering,",
            "National Technical University of Athens   ·   Archimedes Research Unit, Athena Research Center",
            "",
            "FU_Biometry Challenge  ·  MICCAI 2026",
        ],
        notes="One unified model for all nine biometry tasks. Two phases: "
              "self-supervised domain adaptation of DINOv2 on unlabeled "
              "ultrasound, then a supervised HRNet-style decoder with nine "
              "task-specific heads.",
    ),
    dict(
        kind="image_bullets",
        title="Nine tasks, one model — why that is hard",
        image="task_imbalance.png",
        image_frac=0.50,
        bullets=[
            ("A few anatomical landmarks become clinical measurements — fetal growth, "
             "labour progression, cardiac structure. *Manual measurement is slow and "
             "operator-dependent.*", 0),
            ("**9 tasks, 3 clinical domains**, 20+ parameters in one benchmark — and "
             "task-specific models transfer poorly across scanners and sites.", 0),
            ("Labels are scarce: __only 3.4% labeled__ — 6,768 labeled vs 191,170 "
             "unlabeled images.", 0),
            ("**Three orders of magnitude** between the largest and smallest task: "
             "naive pooled training gives an AoP model that ignores the heart.", 0),
        ],
        notes="The chart is the headline: three orders of magnitude between the "
              "largest and smallest task (AoP 4,000 labeled images vs IVC 38), "
              "colored by clinical domain — blue prenatal/gyn, orange "
              "intrapartum, red cardiac/vascular. Anything that trains naively "
              "on the pooled data will be an AoP model that ignores the "
              "cardiac tasks. Also note 191,170 unlabeled frames exist — "
              "they are used only in Phase 1.",
    ),
    dict(
        kind="image_bullets",
        title="The metric dictates the design",
        image="metric_equal_weight.png",
        image_frac=0.50,
        bullets=[
            ("Score = **50% landmark error (MRE)** + **50% clinical-measurement "
             "error (MAE)**, macro-averaged with __equal weight per task__.", 0),
            ("So a **38-image task counts exactly as much as a 4,000-image one** — "
             "the tiny cardiac tasks decide the ranking.", 0),
            ("Three consequences that run through the whole system:", 0),
            ("**√-balanced task sampling** — AoP is seen ~9× as often as PSAX, "
             "not 82×", 1),
            ("loss in **original-image pixels**, not on the 518 training canvas", 1),
            ("checkpoint selection on a **metric-aligned blend** — never on "
             "canvas loss", 1),
        ],
        notes="Worth dwelling on: we treat the metric as a design constraint "
              "rather than something measured at the end. The two strips say it "
              "visually: AoP is 59% of the labeled data but 1/9 of the score. "
              "The per-task macro average is what makes the tiny cardiac tasks "
              "decisive.",
    ),
    dict(
        kind="image",
        title="The method in one picture",
        image="method_overview.png",
        caption="**Phase 1** adapts DINOv2 to ultrasound on the 191k unlabeled frames — "
                "**Phase 2** trains an HRNet-style neck and 9 task heads on the 6.7k "
                "labeled images. *The next slides walk left to right.*",
        notes="Left, Phase 1: continued DINOv2 self-supervision (DINO + iBOT + "
              "KoLeo, student/EMA-teacher) on all 191,170 unlabeled frames — no "
              "labels anywhere. The adapted encoder checkpoint seeds Phase 2. "
              "Right, Phase 2: four encoder depths feed an HRNet-style "
              "multi-resolution neck, a task router picks one of nine heads, and "
              "soft-argmax reads out sub-pixel landmark coordinates that become "
              "the clinical measurements. Every batch is task-homogeneous, "
              "because each forward pass runs exactly one head.",
    ),
    dict(
        kind="image_bullets",
        title="From image to patch tokens",
        image="vit_schematic.png",
        bullets=[
            ("**518 × 518 letterboxed** input, aspect ratio preserved (native sizes "
             "range from ~336 × 544 to 768 × 1024).", 0),
            ("DINOv2 ViT-L/14 with registers → **37 × 37 grid** of 1024-d patch "
             "tokens.", 0),
            ("We tap **four depths** (blocks 5, 11, 17, 23), not only the last — "
             "*coarse semantics and finer detail both reach the decoder*.", 0),
        ],
        image_frac=0.62,
        notes="This is the piece the paper assumes you know. The g05/g11/g17/g23 "
              "notation on the next architecture figure refers to these taps.",
    ),
    dict(
        kind="image",
        title="Phase 1 — self-supervised domain adaptation",
        image="fig1_phase1_pipeline.png",
        caption="Continued DINOv2 pretraining on **191,170 unlabeled ultrasound frames** — "
                "*no labels are used anywhere in this phase*.",
        notes="Off-the-shelf DINOv2 is trained on natural images; sonography has "
              "very different texture statistics, acquisition physics and "
              "artifacts. We adapt the encoder to the domain before asking it "
              "to localize anything.",
    ),
    dict(
        kind="bullets",
        title="Phase 1 — the recipe in brief",
        bullets=[
            ("**Multi-crop self-distillation**: 2 global crops + 6 local crops (98 px); "
             "the student sees all 8 views, the *EMA teacher* only the 2 globals.", 0),
            ("**L  =  L_DINO  +  L_iBOT  +  0.1 · L_KoLeo** — CLS self-distillation, "
             "masked patch-token prediction, and an anti-collapse spread term.", 0),
            ("**Two-resolution curriculum**: 100 epochs at 224 px, then a **4-epoch tail "
             "at 518 px** — the deployment resolution, ~5× cheaper than full-res "
             "throughout.  *~96 GPU-hours on one A100.*", 0),
            ("**Ultrasound-specific**: fan bounding-box crop, gentler ±10° rotation "
             "(*view orientation is semantically meaningful*), background-heavy crops "
             "resampled.", 0),
        ],
        notes="Details kept for questions: DINO loss over K=65,536 prototypes; "
              "iBOT masks restricted to foreground; teacher temperature annealed "
              "0.04 to 0.07; EMA momentum 0.994 to 1.0; effective batch 256 via "
              "gradient accumulation. The 224-then-518 curriculum is the cost "
              "trick: 518 px throughout is roughly 5x more expensive per epoch, "
              "and as the ablation shows, it is not better.",
    ),
    dict(
        kind="image",
        title="Phase 2 — HRNet-style multi-resolution neck",
        image="fig2_left_neck.png",
        caption="Three parallel branches at **37 / 74 / 148** with widths 128 / 96 / 64, "
                "two rounds of bidirectional exchange, fused to **148 × 148 × 128**.  "
                "*GroupNorm throughout — never BatchNorm.*",
        notes="ViT tokenization is coarse — 37×37 for a 518 image. Rather than a "
              "U-Net decoder that downsamples then climbs back, HRNet keeps a "
              "high-resolution branch alive throughout and exchanges between "
              "scales. GroupNorm everywhere, never BatchNorm: batches are "
              "task-homogeneous, so BatchNorm's eval-time running statistics "
              "would blend nine incompatible domains — training error fell "
              "while validation error stayed flat until we changed it.",
    ),
    dict(
        kind="image_bullets",
        title="Phase 2 — task routing and sub-pixel decoding",
        image="fig2_right_heads.png",
        bullets=[
            ("**Nine independent heads**; the batch's task id selects __exactly one__. "
             "Landmarks per task: 2 (FUGC, IVC, femur) up to 22 (PLAX).", 0),
            ("**Soft-argmax** (τ = 10): temperature-scaled softmax over the logits, "
             "then the spatial expectation — *differentiable and sub-pixel*, so the "
             "network is a direct coordinate regressor. Heatmaps are **never "
             "MSE-supervised**.", 0),
            ("**Masked L1** over annotated landmarks; missing ones are (−1, −1) and "
             "contribute no gradient.", 0),
            ("Predictions **inverse-letterboxed to original-image pixels** — training "
             "in the metric's own units.", 0),
            ("*Training: single A100, AdamW + layer-wise LR decay; the EMA teacher is "
             "what gets validated and deployed.*", 0),
        ],
        image_frac=0.28,
        notes="Soft-argmax is the method, not a detail. It is what lets us train "
              "in the metric's own coordinate space. Training facts behind the "
              "last bullet: batch 64, bf16, AdamW base LR 2e-4 with 15-epoch "
              "warmup and 150-epoch cosine decay, last 4 backbone blocks "
              "unfrozen at 0.1x base LR with LLRD 0.75, EMA alpha 0.999, "
              "early-stopping patience 40, sampler temperature 0.5 (allocation "
              "proportional to sqrt of task size), affine + intensity "
              "augmentation. 312.2M parameters, 58.2M trainable; 84.4 img/s "
              "training, up to 127.3 img/s inference. Phase 2 is comparatively "
              "cheap — the expensive part was Phase 1's 96 GPU-hours.",
    ),
    dict(
        kind="table",
        title="Official challenge evaluation",
        subtitle="Validation phase · 619 images · official scorer      "
                 "(red = worst per column, green = best)",
        notes="HC, PSAX and IVC are the weak points. PSAX and IVC have both the "
              "least training data and the smallest evaluation sets (N=18, N=10), "
              "so those numbers carry real uncertainty. HC is the interesting "
              "one: 999 labeled images, an order of magnitude more than the "
              "cardiac tasks, and still the worst task — difficulty here is not "
              "explained by dataset size.",
        footnote="For context, the best leaderboard entry at the time of writing reported "
                 "**MRE 22.56 px / MAE 29.02** on the same scorer.",
    ),
    dict(
        kind="image",
        title="Qualitative results",
        image="fig3_best_per_task.png",
        caption="Representative examples across four tasks spanning different clinical "
                "domains. Ground truth in **green**, predictions in **red**.",
        notes="Four tasks from different domains, showing the same network "
              "handling very different anatomy and image appearance.",
    ),
    dict(
        kind="two_images",
        title="How long should self-supervised adaptation run?",
        images=["fig4a_ssl_duration_224.png", "fig4b_ssl_duration_fullres.png"],
        sub_captions=["(a)  224 px bulk + high-resolution tail",
                      "(b)  518 px full-resolution control"],
        bullets=[
            ("**Frozen-encoder probe** (25 epochs, fold 0) isolates representation "
             "quality from fine-tuning.", 0),
            ("Adaptation is __not monotonic__: bulk-only MRE bottoms out at **epoch 20** "
             "(24.70 px); by epoch 100 the blend is *worse than the non-adapted "
             "encoder* (0.0974 vs 0.0890).", 0),
            ("A **4-epoch 518 px tail rescues** the later checkpoints — best blend of "
             "any checkpoint tested (**0.0747**).  We select **ep104**.", 0),
            ("At matched 20 epochs, full-resolution adaptation is **worse** than 224 px "
             "(25.96 vs 24.70 px) — *the cheap curriculum is also the better one*.", 0),
        ],
        notes="The honest finding of the paper: more self-supervision is not "
              "better, and the failure is invisible unless you probe the frozen "
              "encoder. Decoder ablation over 5 folds was inconclusive — HRNet "
              "0.0707±0.0089 vs simple decoder 0.0740±0.0068, with MRE favouring "
              "the simple one; differences are small against fold variance.",
    ),
    dict(
        kind="bullets",
        title="Conclusions",
        bullets=[
            ("**One model, nine tasks**: a two-stage DINOv2-HRNet pipeline for "
             "heterogeneous ultrasound biometry.", 0),
            ("**191k unlabeled frames support 6.7k labeled ones** by separating "
             "domain-level representation learning from task-specific localization.", 0),
            ("**SSL duration is non-monotonic** — a short high-resolution tail matters "
             "more than more low-resolution epochs.", 0),
            ("**Design follows the metric**: balanced sampling, original-pixel loss, "
             "metric-aligned checkpoint selection.", 0),
            ("*Future work*: 5-fold ensembling with multi-scale and intensity TTA; "
             "ablating the neck's multi-level fusion and fine-tuning depth.", 0),
            ("Code: **github.com/abarmper/US-Foundation**   ·   *A100 80GB instances "
             "provided by the NVIDIA Academic Hardware Grant Program (MedViLA).*", 0),
        ],
        notes="Thank you — happy to take questions.",
    ),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_MARKUP = re.compile(r"(\*\*.+?\*\*|__.+?__|\*[^*]+?\*)")


def parse_markup(txt):
    """Split '**b** *i* __a__' into (segment, bold, italic, accent) pieces."""
    out = []
    for part in _MARKUP.split(txt):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append((part[2:-2], True, False, False))
        elif part.startswith("__") and part.endswith("__") and len(part) > 4:
            out.append((part[2:-2], True, False, True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append((part[1:-1], False, True, False))
        else:
            out.append((part, False, False, False))
    return out


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """runs: list of (text, size, bold, color, space_before, indent_level).

    Text may contain inline **bold** / *italic* / __accent__ markup.
    """
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, (txt, size, bold, color, space_before, level) in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.level = level
        if space_before:
            p.space_before = Pt(space_before)
        for seg, b, it, acc in parse_markup(txt):
            r = p.add_run()
            r.text = seg
            r.font.size = Pt(size)
            r.font.bold = bold or b
            if it:
                r.font.italic = True
            r.font.color.rgb = ACCENT if acc else color
            r.font.name = "Calibri"
    return tb


def add_title(slide, title):
    add_text(slide, 0.6, 0.34, SLIDE_W - 1.2, 0.9,
             [(title, 30, True, INK, 0, 0)])
    ln = slide.shapes.add_connector(1, Inches(0.62), Inches(1.24),
                                    Inches(SLIDE_W - 0.62), Inches(1.24))
    ln.line.color.rgb = RULE
    ln.line.width = Pt(2.25)


def fit(img_path, max_w, max_h):
    """Contain-fit an image, returning (w, h) in inches."""
    from PIL import Image
    with Image.open(img_path) as im:
        iw, ih = im.size
    s = min(max_w / iw, max_h / ih)
    return iw * s, ih * s


def place_image(slide, name, cx, cy, max_w, max_h):
    path = IMG / name
    w, h = fit(path, max_w, max_h)
    slide.shapes.add_picture(str(path), Inches(cx - w / 2), Inches(cy - h / 2),
                             Inches(w), Inches(h))
    return w, h


def bullet_runs(bullets, size=15.5):
    out = []
    for txt, level in bullets:
        out.append((("• " if level == 0 else "– ") + txt,
                    size if level == 0 else size - 1.5,
                    False,
                    INK if level == 0 else MUTED,
                    9 if level == 0 else 4,
                    level))
    return out


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------------------
# slide renderers
# ---------------------------------------------------------------------------

def render_title(slide, s):
    add_text(slide, 0.9, 2.05, SLIDE_W - 1.8, 1.9,
             [(s["title"], 34, True, INK, 0, 0)], align=PP_ALIGN.CENTER)
    ln = slide.shapes.add_connector(1, Inches(4.4), Inches(4.02),
                                    Inches(SLIDE_W - 4.4), Inches(4.02))
    ln.line.color.rgb = ACCENT
    ln.line.width = Pt(2.5)
    runs = []
    for i, line in enumerate(s["subtitle"]):
        bold = i == 0
        size = 16 if i == 0 else (14.5 if i < 3 else 15)
        color = INK if i == 0 else MUTED
        if line.startswith("FU_Biometry"):
            bold, color, size = True, ACCENT, 16
        runs.append((line, size, bold, color, 7 if i else 0, 0))
    add_text(slide, 0.9, 4.30, SLIDE_W - 1.8, 2.4, runs, align=PP_ALIGN.CENTER)


def render_bullets(slide, s):
    add_title(slide, s["title"])
    add_text(slide, 0.75, 1.62, SLIDE_W - 1.5, 5.4, bullet_runs(s["bullets"]))


def render_image(slide, s):
    add_title(slide, s["title"])
    cap = s.get("caption")
    top, bottom = 1.55, (6.45 if cap else 7.05)
    place_image(slide, s["image"], SLIDE_W / 2, (top + bottom) / 2,
                SLIDE_W - 1.1, bottom - top)
    if cap:
        add_text(slide, 0.9, 6.52, SLIDE_W - 1.8, 0.85,
                 [(cap, 13, False, MUTED, 0, 0)], align=PP_ALIGN.CENTER)


def render_image_bullets(slide, s):
    add_title(slide, s["title"])
    frac = s.get("image_frac", 0.5)
    gap = 0.35
    avail = SLIDE_W - 1.5 - gap
    iw_max, bw = avail * frac, avail * (1 - frac)
    place_image(slide, s["image"], 0.75 + iw_max / 2, 4.25, iw_max, 5.0)
    add_text(slide, 0.75 + iw_max + gap, 1.75, bw, 5.2,
             bullet_runs(s["bullets"], size=14.5))


def render_two_images(slide, s):
    add_title(slide, s["title"])
    half = (SLIDE_W - 1.5 - 0.4) / 2
    for i, (name, cap) in enumerate(zip(s["images"], s["sub_captions"])):
        cx = 0.75 + half / 2 + i * (half + 0.4)
        place_image(slide, name, cx, 2.85, half, 2.15)
        add_text(slide, cx - half / 2, 4.02, half, 0.4,
                 [(cap, 12, True, MUTED, 0, 0)], align=PP_ALIGN.CENTER)
    add_text(slide, 0.75, 4.52, SLIDE_W - 1.5, 2.6,
             bullet_runs(s["bullets"], size=13.5))


def render_table(slide, s):
    add_title(slide, s["title"])
    add_text(slide, 0.75, 1.30, SLIDE_W - 1.5, 0.4,
             [(s["subtitle"], 13.5, False, MUTED, 0, 0)])
    rows, cols = len(TASKS) + 2, 4
    tw, th = 8.6, 4.75
    left = (SLIDE_W - tw) / 2
    shape = slide.shapes.add_table(rows, cols, Inches(left), Inches(1.78),
                                   Inches(tw), Inches(th))
    tbl = shape.table
    for c, w in zip(range(cols), (3.7, 1.5, 1.7, 1.7)):
        tbl.columns[c].width = Inches(w)

    def cell(r, c, txt, bold=False, size=13, align=PP_ALIGN.LEFT, color=None):
        cl = tbl.cell(r, c)
        cl.text = txt
        p = cl.text_frame.paragraphs[0]
        p.alignment = align
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.name = "Calibri"
            if color is not None:
                run.font.color.rgb = color
        cl.vertical_anchor = MSO_ANCHOR.MIDDLE

    flag_color = {"worst": BAD, "best": GOOD}
    for c, h in enumerate(["Task", "Landmarks", "MRE (px) ↓", "MAE ↓"]):
        cell(0, c, h, bold=True, size=13.5,
             align=PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
    for r, (name, k, n, mre, mae, f_mre, f_mae) in enumerate(TASKS, start=1):
        cell(r, 0, f"{name}   (N={n})")
        cell(r, 1, str(k), align=PP_ALIGN.CENTER)
        cell(r, 2, mre, align=PP_ALIGN.CENTER,
             bold=bool(f_mre), color=flag_color.get(f_mre))
        cell(r, 3, mae, align=PP_ALIGN.CENTER,
             bold=bool(f_mae), color=flag_color.get(f_mae))
    last = len(TASKS) + 1
    cell(last, 0, "Overall  (unweighted mean, N=619)", bold=True, size=13.5)
    cell(last, 1, "—", bold=True, align=PP_ALIGN.CENTER)
    cell(last, 2, "26.34", bold=True, size=13.5, align=PP_ALIGN.CENTER)
    cell(last, 3, "29.70", bold=True, size=13.5, align=PP_ALIGN.CENTER)

    add_text(slide, 0.75, 6.72, SLIDE_W - 1.5, 0.5,
             [(s["footnote"], 12, False, MUTED, 0, 0)], align=PP_ALIGN.CENTER)


RENDERERS = {
    "title": render_title,
    "bullets": render_bullets,
    "image": render_image,
    "image_bullets": render_image_bullets,
    "two_images": render_two_images,
    "table": render_table,
}


# ---------------------------------------------------------------------------

def images_of(s):
    return ([s["image"]] if "image" in s else []) + list(s.get("images", []))


def main():
    missing = [n for s in SLIDES for n in images_of(s) if not (IMG / n).exists()]
    if missing:
        raise SystemExit("missing images: " + ", ".join(missing))

    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    for s in SLIDES:
        slide = blank(prs)
        RENDERERS[s["kind"]](slide, s)
        set_notes(slide, s["notes"])

    prs.save(OUT_PPTX)

    md = ["# FUB 2026 presentation — outline", "",
          "> Text mirror of `31_annapan_54_presentation.pptx`.",
          "> Generated by `build_slides.py`; edit `SLIDES` there and re-run.", ""]
    for i, s in enumerate(SLIDES, start=1):
        tag = "  *(backup)*" if s["title"].startswith("Backup") else ""
        md += [f"## Slide {i} — {s['title'].splitlines()[0]}{tag}", ""]
        for name in images_of(s):
            md.append(f"![{name}](img/{name})")
        if images_of(s):
            md.append("")
        subs = s.get("subtitle")
        for line in ([subs] if isinstance(subs, str) else subs or []):
            if line:
                md.append(f"*{line}*")
        if subs:
            md.append("")
        if s["kind"] == "table":
            md += ["| Task | Landmarks | MRE (px) | MAE |", "|---|---|---|---|"]
            md += [f"| {n} (N={nn}) | {k} | {a} | {b} |"
                   for n, k, nn, a, b, _, _ in TASKS]
            md += ["| **Overall (N=619)** | — | **26.34** | **29.70** |", "",
                   s["footnote"], ""]
        for txt, level in s.get("bullets", []):
            md.append(("- " if level == 0 else "    - ") + txt)
        if s.get("bullets"):
            md.append("")
        if s.get("caption"):
            md += [f"*{s['caption']}*", ""]
        md += [f"**Notes:** {s['notes']}", ""]
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    size_mb = OUT_PPTX.stat().st_size / 1e6
    print(f"wrote {OUT_PPTX.name}  ({len(SLIDES)} slides, {size_mb:.1f} MB)")
    print(f"wrote {OUT_MD.name}")


if __name__ == "__main__":
    main()
