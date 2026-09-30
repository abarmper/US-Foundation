#!/usr/bin/env python3
"""Draw the ViT / patch-tokenization schematic the paper does not have.

The paper's fig1 and fig2 both assume the reader already knows how a 518x518
image becomes a 37x37 grid of 1024-d tokens, and fig2's g05/g11/g17/g23 taps
are unexplained without it. This slide-4 diagram fills that gap:

    518x518 letterbox  ->  14x14 patchify  ->  37x37 tokens
                       ->  24 transformer blocks, four of them tapped
                       ->  the multi-level features the HRNet neck consumes

Palette matches the drawio figures so it does not look imported.

Writes img/vit_schematic.png (and .pdf).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon

OUT = Path(__file__).resolve().parent / "img"

# palette lifted from the paper's figures
PEACH = "#FCE4CE"
PEACH_E = "#E8B98A"
GREEN = "#CFE7CF"
GREEN_E = "#6E9E6E"
PURPLE = "#DDD0EA"
PURPLE_E = "#8B72AE"
YELLOW = "#FDF2CC"
YELLOW_E = "#C9A227"
GREY = "#F2F2F2"
GREY_E = "#9A9A9A"
INK = "#1A1A1A"
MUTED = "#6B6B6B"

W, H = 13.0, 5.6
MID = 2.60  # vertical centreline of the flow


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.06):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))


def text(ax, x, y, s, size=11, weight="normal", color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, zorder=5)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, ms=12):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, zorder=4, shrinkA=0, shrinkB=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # no dead margin
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.add_patch(FancyBboxPatch(
        (0.12, 0.12), W - 0.24, H - 0.24,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=PEACH, edgecolor=PEACH_E, linewidth=1.6, zorder=0))

    text(ax, 0.50, 5.05, "From image to patch tokens", size=15,
         weight="bold", ha="left")

    # ------------------------------------------------------------- input
    ix, iy, isz = 0.80, 1.55, 2.10
    ax.add_patch(Rectangle((ix, iy), isz, isz, facecolor="#111111",
                           edgecolor="#7A7A7A", linewidth=1.3, zorder=2))
    # ultrasound fan, so the black region reads as letterbox padding
    apex = (ix + isz / 2, iy + isz * 0.90)
    ax.add_patch(Polygon(
        [apex, (ix + isz * 0.10, iy + isz * 0.16), (ix + isz * 0.90, iy + isz * 0.16)],
        closed=True, facecolor="#5C5C5C", edgecolor="none", zorder=3))
    ax.add_patch(Polygon(
        [apex, (ix + isz * 0.28, iy + isz * 0.34), (ix + isz * 0.72, iy + isz * 0.34)],
        closed=True, facecolor="#8A8A8A", edgecolor="none", zorder=3))
    n = 7
    for i in range(1, n):
        f = i / n
        ax.plot([ix + f * isz] * 2, [iy, iy + isz], color="#FFFFFF",
                linewidth=0.45, alpha=0.30, zorder=4)
        ax.plot([ix, ix + isz], [iy + f * isz] * 2, color="#FFFFFF",
                linewidth=0.45, alpha=0.30, zorder=4)
    text(ax, ix + isz / 2, iy - 0.30, "letterboxed input", size=10.5, weight="bold")
    text(ax, ix + isz / 2, iy - 0.60, "518 x 518 x 3", size=10)

    # ---------------------------------------------------------- patchify
    arrow(ax, 2.98, MID, 3.72, MID)
    text(ax, 3.35, MID + 0.42, "patchify", size=9.5, weight="bold")
    text(ax, 3.35, MID + 0.17, "14 x 14", size=9)

    # -------------------------------------------------------- token grid
    gx, gy, gsz = 3.85, 1.55, 2.10
    box(ax, gx, gy, gsz, gsz, GREEN, GREEN_E, r=0.05)
    m = 9
    for i in range(1, m):
        f = i / m
        ax.plot([gx + f * gsz] * 2, [gy, gy + gsz], color=GREEN_E,
                linewidth=0.5, alpha=0.55, zorder=3)
        ax.plot([gx, gx + gsz], [gy + f * gsz] * 2, color=GREEN_E,
                linewidth=0.5, alpha=0.55, zorder=3)
    text(ax, gx + gsz / 2, gy + gsz + 0.26, "518 / 14 = 37", size=9.5, color=MUTED)
    text(ax, gx + gsz / 2, gy - 0.30, "patch tokens", size=10.5, weight="bold")
    text(ax, gx + gsz / 2, gy - 0.60, "37 x 37,  1024-d", size=10)

    arrow(ax, 6.05, MID, 6.85, MID)

    # ------------------------------------------------ transformer stack
    sx, sw = 6.95, 2.00
    bh, gap, n_blocks = 0.105, 0.028, 24
    total = n_blocks * bh + (n_blocks - 1) * gap
    sy = MID - total / 2

    taps = {4: "g05", 10: "g11", 16: "g17", 22: "g23"}
    bus_x = 9.62
    tap_ys = []
    for i in range(n_blocks):
        y = sy + i * (bh + gap)
        tap = i in taps
        box(ax, sx, y, sw, bh,
            PURPLE if tap else GREY,
            PURPLE_E if tap else GREY_E,
            lw=1.3 if tap else 0.7, r=0.02)
        if tap:
            cy = y + bh / 2
            tap_ys.append(cy)
            ax.plot([sx + sw + 0.04, bus_x], [cy, cy], color=PURPLE_E,
                    linewidth=1.6, zorder=4, solid_capstyle="round")
            text(ax, 9.28, cy + 0.155, taps[i], size=9, weight="bold",
                 color=PURPLE_E)

    # collector bus -> one arrow into the output box
    ax.plot([bus_x, bus_x], [min(tap_ys), max(tap_ys)], color=PURPLE_E,
            linewidth=1.8, zorder=4, solid_capstyle="round")
    arrow(ax, bus_x, MID, 10.02, MID, color=PURPLE_E, lw=1.8)

    text(ax, sx + sw / 2, sy + total + 0.30, "DINOv2 ViT-L/14 (register variant)",
         size=11, weight="bold")
    text(ax, sx + sw / 2, sy - 0.30, "24 transformer blocks", size=10.5, weight="bold")
    text(ax, sx + sw / 2, sy - 0.60, "+ CLS token and 4 registers", size=9.5, color=MUTED)

    # ------------------------------------------------------------ output
    ox, ow, oh = 10.10, 2.50, 2.00
    box(ax, ox, MID - oh / 2, ow, oh, YELLOW, YELLOW_E)
    cx = ox + ow / 2
    text(ax, cx, MID + 0.60, "four depths tapped,", size=11, weight="bold")
    text(ax, cx, MID + 0.30, "not just the last", size=11, weight="bold")
    text(ax, cx, MID - 0.04, "block", size=11, weight="bold")
    text(ax, cx, MID - 0.44, "-> HRNet neck", size=10.5, color="#7A6414")
    text(ax, cx, MID - 0.70, "(Phase 2)", size=10.5, color="#7A6414")

    fig.savefig(OUT / "vit_schematic.png", dpi=200, facecolor="white")
    fig.savefig(OUT / "vit_schematic.pdf", facecolor="white")
    print(f"wrote {OUT / 'vit_schematic.png'}")


if __name__ == "__main__":
    main()
