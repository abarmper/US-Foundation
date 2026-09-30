#!/usr/bin/env python3
"""Draw the three slide-only schematics the paper does not have.

    img/task_imbalance.png      labeled images per task, log scale (slide 2)
    img/metric_equal_weight.png data share vs score share (slide 3)
    img/method_overview.png     the two-phase pipeline in one strip (slide 4)

Per-task labeled counts come from data/splits/train_val_split_keys.json
(train+val pooled); they match the paper's claims (AoP 4,000 · IVC 38 ·
PSAX 49 · HC 999 · "under 100" for IVC/PLAX/PSAX).

Usage:  python3 make_extra_schematics.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"

INK, ACCENT, MUTED = "#1A1A1A", "#6B4E8E", "#666666"

# one color per clinical domain — reused on both charts so the domains are
# introduced visually before slide 2's bullets name them
PRENATAL, INTRAPARTUM, CARDIAC = "#4E79A7", "#E8A33D", "#C0504D"
DOMAINS = [("prenatal / gyn", PRENATAL), ("intrapartum", INTRAPARTUM),
           ("cardiac / vascular", CARDIAC)]

TASKS = [  # (name, labeled images, domain color), descending
    ("AoP", 4000, INTRAPARTUM),
    ("HC", 999, PRENATAL),
    ("femur", 702, PRENATAL),
    ("FA", 500, PRENATAL),
    ("FUGC", 260, PRENATAL),
    ("A4C", 108, CARDIAC),
    ("PLAX", 87, CARDIAC),
    ("PSAX", 49, CARDIAC),
    ("IVC", 38, CARDIAC),
]


def task_imbalance():
    fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=200)
    names = [t[0] for t in TASKS][::-1]
    vals = [t[1] for t in TASKS][::-1]
    cols = [t[2] for t in TASKS][::-1]

    bars = ax.barh(names, vals, color=cols, edgecolor="white", height=0.66, log=True)
    for b, v in zip(bars, vals):
        ax.text(v * 1.18, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=11.5, color=INK, fontweight="bold")

    ax.set_xlim(25, 30000)
    ax.set_xlabel("labeled images  (log scale)", fontsize=11, color=MUTED)
    ax.tick_params(axis="y", labelsize=12, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=9.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)

    ax.legend(handles=[Patch(fc=c, label=l) for l, c in DOMAINS],
              loc="lower right", frameon=False, fontsize=10.5)
    ax.text(0.985, 0.30, "+ 191,170 unlabeled frames\n(used only in Phase 1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            style="italic", color=ACCENT)

    fig.tight_layout()
    fig.savefig(IMG / "task_imbalance.png", facecolor="white")
    plt.close(fig)


def metric_equal_weight():
    fig = plt.figure(figsize=(7.2, 4.5), dpi=200)
    ax = fig.add_axes([0.03, 0.02, 0.94, 0.96])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.955, "Score  =  ½ · MRE  +  ½ · MAE",
            ha="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.875, "macro-averaged: every task carries the same weight",
            ha="center", fontsize=11.5, style="italic", color=MUTED)

    total = sum(v for _, v, _ in TASKS)
    y1, y2, h = 0.565, 0.13, 0.155

    # top strip — segments proportional to labeled data
    ax.text(0, y1 + h + 0.03, "share of the labeled data",
            fontsize= 12, fontweight="bold", color=INK)
    x = 0.0
    for name, v, c in TASKS:
        w = v / total
        ax.add_patch(Rectangle((x, y1), w, h, fc=c, ec="white", lw=1.2))
        if w > 0.055:
            ax.text(x + w / 2, y1 + h / 2, name, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="white")
        x += w
    ax.annotate("FUGC · A4C · PLAX · PSAX · IVC — 8% together",
                xy=(0.96, y1), xytext=(0.99, y1 - 0.085),
                ha="right", fontsize=9.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))

    # bottom strip — nine equal segments
    ax.text(0, y2 + h + 0.03, "share of the score  —  1/9 each",
            fontsize=12, fontweight="bold", color=INK)
    for i, (name, _, c) in enumerate(TASKS):
        ax.add_patch(Rectangle((i / 9, y2), 1 / 9, h, fc=c, ec="white", lw=1.2))
        ax.text(i / 9 + 1 / 18, y2 + h / 2, name, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")

    ax.annotate("", xy=(0.5, y2 + h + 0.075), xytext=(0.5, y1 - 0.12),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2))
    ax.text(0.515, (y1 - 0.12 + y2 + h + 0.075) / 2, "equal weight per task",
            fontsize=11.5, style="italic", color=ACCENT, va="center")

    fig.savefig(IMG / "metric_equal_weight.png", facecolor="white")
    plt.close(fig)


def method_overview():
    W, H = 10.6, 3.1
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.axis("off")

    def panel(x, w, title, fc, ec):
        ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.88,
                                    boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=fc, ec=ec, lw=1.8, mutation_aspect=W / H))
        ax.text(x + w / 2, 0.845, title, ha="center", fontsize=11.5,
                fontweight="bold", color=ec)

    def box(cx, w, text, fc="white", fs=9.3, cy=0.47, h=0.34):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc=fc, ec=INK, lw=1.2, mutation_aspect=W / H))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK)

    def arrow(x0, x1, y=0.47, color=INK, lw=1.6):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    shrinkA=0, shrinkB=0))

    BLUE, PURPLE, YELLOW, PINK, RED = ("#D6E6F6", "#E3D7F0", "#FDF3CF",
                                       "#F3D9E7", "#FBDCD2")
    panel(0.012, 0.39, "PHASE 1 — self-supervised domain adaptation",
          "#F1EAF8", ACCENT)
    panel(0.475, 0.515, "PHASE 2 — supervised landmark decoding",
          "#E9F3E6", "#4E7B4A")

    # phase 1
    box(0.075, 0.114, "191,170\nunlabeled frames", fc=RED, fs=8.8)
    arrow(0.134, 0.15)
    box(0.226, 0.145, "DINOv2 ViT-L/14\nstudent ⇆ EMA teacher", fc=BLUE, fs=8.8)
    arrow(0.3, 0.316)
    box(0.353, 0.072, "adapted\nencoder", fc=BLUE, fs=8.8)
    ax.text(0.2, 0.155, "DINO + iBOT + KoLeo  ·  multi-crop 224 / 98 px",
            ha="center", fontsize=9, style="italic", color=MUTED)

    # bridge between the phases
    arrow(0.391, 0.472, color=ACCENT, lw=2.2)
    ax.text(0.4315, 0.585, "ckpt", ha="center", fontsize=9, color=ACCENT)

    # phase 2
    box(0.528, 0.086, "encoder\n4 depth taps", fc=BLUE, fs=8.8)
    arrow(0.573, 0.585)
    box(0.63, 0.083, "HRNet neck\n37 · 74 · 148", fc=PURPLE, fs=8.8)
    arrow(0.673, 0.685)
    box(0.727, 0.077, "task router\n9 heads", fc=YELLOW, fs=8.8)
    arrow(0.767, 0.779)
    box(0.833, 0.1, "soft-argmax\nsub-pixel coords", fc=YELLOW, fs=8.8)
    arrow(0.885, 0.897)
    box(0.9425, 0.084, "landmarks →\nmeasurements", fc=PINK, fs=8.5)
    ax.text(0.7325, 0.155, "6,768 labeled images  ·  loss in original-image pixels",
            ha="center", fontsize=9, style="italic", color=MUTED)

    fig.savefig(IMG / "method_overview.png", facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


if __name__ == "__main__":
    task_imbalance()
    metric_equal_weight()
    method_overview()
    for n in ("task_imbalance", "metric_equal_weight", "method_overview"):
        p = IMG / f"{n}.png"
        print(f"wrote {p.name}  ({p.stat().st_size / 1e3:.0f} kB)")
