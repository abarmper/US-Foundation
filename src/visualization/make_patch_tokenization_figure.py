"""
"ViT-to-HRNet interface" figure: shows how the 518x518 processed ultrasound
frame is divided into a 37x37 grid of 14x14px patches, and what the actual
DINOv2 encoder (as fine-tuned in the v3 checkpoint) produces from it -- a
37x37 grid of 1024-channel patch tokens, visualized via PCA-to-RGB (the
standard DINO/DINOv2 way of showing what the encoder "sees": semantically
similar patches get similar colors).

Panel (a): the actual model input (518x518, letterboxed) with the 37x37
patch grid drawn on top and one patch highlighted (nearest patch to a GT
keypoint, so the highlight is anatomically meaningful).
Panel (b): the real patch-token PCA map (37x37, nearest-neighbor upscaled so
each token renders as a sharp, distinct cell) with the same patch index
highlighted, connected to panel (a) by an arrow through the encoder.

Usage:
  CUDA_VISIBLE_DEVICES=3 python make_patch_tokenization_figure.py \
      --checkpoint ../runs/v3_true_hrnet_softargmax/checkpoints/best_teacher_model.pth \
      --manifest figures/qualitative_best/manifest.json \
      --task_id HC \
      --out_stem figures/fig_patch_tokenization
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for "common"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from common import load_image_rgb, INFERENCE_TRANSFORM, GT_JSON_PATH
from gubiometry.models import UnifiedBiometryModel

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch_gu_biometry"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANVAS = 518
GRID = 37
CELL = CANVAS / GRID  # ~14.0


def pca_feature_map(patch_tokens, grid_h, grid_w):
    pca = PCA(n_components=3)
    proj = pca.fit_transform(patch_tokens)
    proj -= proj.min(axis=0, keepdims=True)
    denom = proj.max(axis=0, keepdims=True) - proj.min(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    proj = proj / denom
    return proj.reshape(grid_h, grid_w, 3)


def draw_patch_grid(ax, color="white", alpha=0.35, lw=0.5):
    for i in range(1, GRID):
        ax.axhline(i * CELL, color=color, alpha=alpha, linewidth=lw)
        ax.axvline(i * CELL, color=color, alpha=alpha, linewidth=lw)


def main(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(GT_JSON_PATH) as f:
        gt_records = json.load(f)
    gt_by_key = {}
    for rec in gt_records:
        fname = os.path.basename(rec["image_path"])
        gt_by_key[(rec["task_id"], fname)] = rec["ground_truth_points_pixels"]

    best = manifest[args.task_id]["selected"][0]
    abs_path = best["image_path"]
    fname = os.path.basename(abs_path)
    gt_pts_flat = gt_by_key[(args.task_id, fname)]
    gt_pts = np.asarray(gt_pts_flat, dtype=float).reshape(-1, 2)

    print("Loading model...")
    model = UnifiedBiometryModel(
        freeze_encoder=True,
        dropout_p=0.3,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        neck_branch_width=tuple(args.neck_branch_width),
        shared_head=args.shared_head,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=True)
    print("Load result:", load_result)
    model.eval()

    img_rgb = load_image_rgb(abs_path)
    orig_h, orig_w = img_rgb.shape[:2]
    transformed = INFERENCE_TRANSFORM(image=img_rgb)
    tensor_img = transformed["image"].unsqueeze(0).to(device)

    # reconstruct the 518x518 *displayable* (unnormalized) canvas identically
    # to what the encoder actually sees, for panel (a)
    import albumentations as A
    import cv2
    display_transform = A.Compose([
        A.LongestMaxSize(max_size=CANVAS),
        A.PadIfNeeded(min_height=CANVAS, min_width=CANVAS, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ])
    canvas_rgb = display_transform(image=img_rgb)["image"]

    # map the first visible GT keypoint into the same 518-canvas coordinate
    # system, to choose an anatomically meaningful patch to highlight
    scale = CANVAS / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_top = (CANVAS - new_h) // 2
    pad_left = (CANVAS - new_w) // 2
    kp_x, kp_y = gt_pts[0]
    kp_x_518 = kp_x * scale + pad_left
    kp_y_518 = kp_y * scale + pad_top
    patch_col = int(np.clip(kp_x_518 // CELL, 0, GRID - 1))
    patch_row = int(np.clip(kp_y_518 // CELL, 0, GRID - 1))

    print("Running encoder...")
    with torch.no_grad():
        patch_tokens = model.encoder.forward_features(tensor_img)["x_norm_patchtokens"]
    patch_tokens = patch_tokens[0].cpu().numpy()  # (1369, 1024)
    pca_map = pca_feature_map(patch_tokens, GRID, GRID)  # (37, 37, 3) in [0,1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    ax_a, ax_b = axes

    # ---- Panel (a): processed image + patch grid ----
    ax_a.imshow(canvas_rgb)
    draw_patch_grid(ax_a, color="white", alpha=0.35, lw=0.5)
    ax_a.add_patch(Rectangle((patch_col * CELL, patch_row * CELL), CELL, CELL,
                              fill=False, edgecolor="#FF7A00", linewidth=2.5, zorder=5))
    ax_a.set_xlim(0, CANVAS); ax_a.set_ylim(CANVAS, 0)
    ax_a.set_title(f"(a) Model input: {CANVAS}×{CANVAS}\ndivided into a {GRID}×{GRID} grid of "
                   f"{CELL:.0f}×{CELL:.0f}px patches", fontsize=11)
    ax_a.axis("off")

    # ---- Panel (b): PCA token grid, nearest-upscaled for sharp cells ----
    upscale = 8
    pca_disp = np.repeat(np.repeat(pca_map, upscale, axis=0), upscale, axis=1)
    ax_b.imshow(pca_disp, interpolation="nearest")
    draw_patch_grid_scaled = lambda ax, cell, n: [
        (ax.axhline(i * cell, color="white", alpha=0.25, linewidth=0.4),
         ax.axvline(i * cell, color="white", alpha=0.25, linewidth=0.4))
        for i in range(1, n)
    ]
    draw_patch_grid_scaled(ax_b, upscale, GRID)
    ax_b.add_patch(Rectangle((patch_col * upscale, patch_row * upscale), upscale, upscale,
                              fill=False, edgecolor="#FF7A00", linewidth=2.5, zorder=5))
    ax_b.set_title(f"(b) DINOv2-L/14 patch tokens\n{GRID}×{GRID} grid, 1024 channels/token "
                   f"(top-3 PCA → RGB)", fontsize=11)
    ax_b.axis("off")

    # ---- connecting arrow ----
    arrow = FancyArrowPatch((1.03, 0.5), (1.2, 0.5),
                             transform=ax_a.transAxes, mutation_scale=25,
                             color="black", arrowstyle="-|>", linewidth=2, clip_on=False)
    fig.add_artist(arrow) if False else ax_a.annotate(
        "", xy=(1.22, 0.5), xytext=(1.0, 0.5), xycoords=ax_a.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=2))
    ax_a.text(1.11, 0.56, "DINOv2-L/14\nencoder", transform=ax_a.transAxes,
              ha="center", va="bottom", fontsize=9.5, style="italic")

    orange_patch = mpatches.Patch(edgecolor="#FF7A00", facecolor="none", linewidth=2.5,
                                   label="Same patch in both panels (nearest to a GT keypoint)")
    fig.legend(handles=[orange_patch], loc="lower center", fontsize=9.5, frameon=False)

    fig.suptitle(f"Patch tokenization of the DINOv2 encoder ({args.task_id} example)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    out_dir = os.path.dirname(args.out_stem) or "."
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png", "pdf", "eps"]:
        out_path = f"{args.out_stem}.{ext}"
        kwargs = {"dpi": 300} if ext == "png" else {}
        plt.savefig(out_path, format=ext, bbox_inches="tight", **kwargs)
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--task_id", type=str, default="HC")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=4)
    parser.add_argument("--neck_branch_width", type=int, nargs=3, default=[128, 96, 64])
    parser.add_argument("--shared_head", action="store_true")
    parser.add_argument("--out_stem", type=str, default="figures/fig_patch_tokenization")
    args = parser.parse_args()
    main(args)
