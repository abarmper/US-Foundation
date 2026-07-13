"""
Soft-argmax probability heatmap figure: for the best validation case of a
task, shows softmax(temperature * logits) for each keypoint -- literally the
distribution the soft-argmax expectation (Eq. eq:softargmax in the paper) is
computed over -- overlaid on the model input, next to the GT/predicted point.
A tight, high-contrast blob means the model is confident and well-localized;
a diffuse blob means the model is uncertain even though soft-argmax still
outputs a single expected coordinate.

Usage:
  CUDA_VISIBLE_DEVICES=3 python make_softargmax_heatmap_figure.py \
      --checkpoint ../runs/v3_true_hrnet_softargmax/checkpoints/best_teacher_model.pth \
      --manifest figures/qualitative_best/manifest.json \
      --task_id HC \
      --out_dir figures/softargmax_heatmaps
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import albumentations as A

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for "common"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from common import load_image_rgb, INFERENCE_TRANSFORM, GT_JSON_PATH, remove_padding_and_scale
from gubiometry.models import UnifiedBiometryModel

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch_gu_biometry"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANVAS = 518


def get_soft_argmax_coords_and_maps(logits, temperature=10.0):
    B, K, H, W = logits.shape
    probs = F.softmax(logits.view(B, K, -1) * temperature, dim=-1).view(B, K, H, W)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=logits.device),
        torch.arange(W, device=logits.device),
        indexing="ij"
    )
    grid_x = grid_x.flatten().float()
    grid_y = grid_y.flatten().float()
    flat = probs.view(B, K, -1)
    pred_x = torch.sum(flat * grid_x, dim=-1) * (CANVAS / W)
    pred_y = torch.sum(flat * grid_y, dim=-1) * (CANVAS / H)
    return pred_x, pred_y, probs


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
    gt_pts = np.asarray(gt_by_key[(args.task_id, fname)], dtype=float).reshape(-1, 2)

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
    tensor_img = INFERENCE_TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(device)

    display_transform = A.Compose([
        A.LongestMaxSize(max_size=CANVAS),
        A.PadIfNeeded(min_height=CANVAS, min_width=CANVAS, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ])
    canvas_rgb = display_transform(image=img_rgb)["image"]

    print("Running inference...")
    with torch.no_grad():
        logits = model.forward_phase2(tensor_img, args.task_id)
        pred_x_518, pred_y_518, probs = get_soft_argmax_coords_and_maps(logits, temperature=args.softargmax_temp)
    probs = probs[0].cpu().numpy()  # (K, Hh, Wh)
    n_kp = min(logits.shape[1], gt_pts.shape[0])

    scale = CANVAS / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_top = (CANVAS - new_h) // 2
    pad_left = (CANVAS - new_w) // 2

    os.makedirs(args.out_dir, exist_ok=True)

    n_cols = min(4, n_kp)
    n_rows = int(np.ceil(n_kp / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows))
    axes = np.atleast_2d(axes)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    ZOOM_HALF_WIN = 22  # pixels either side, in 518-canvas coords
    DISPLAY_GAMMA = 0.3  # <1 stretches the visible falloff of an inherently
                          # sharp softmax peak for legibility; does not move
                          # the peak location, only its perceived radius

    for k in range(n_kp):
        ax = axes[k // n_cols, k % n_cols]
        heat = probs[k]
        heat_up = cv2.resize(heat, (CANVAS, CANVAS), interpolation=cv2.INTER_CUBIC)
        heat_up = np.clip(heat_up, 0, None)
        heat_up = heat_up / (heat_up.max() + 1e-8)
        heat_up = heat_up ** DISPLAY_GAMMA

        gt_x_518 = gt_pts[k, 0] * scale + pad_left
        gt_y_518 = gt_pts[k, 1] * scale + pad_top
        pred_x, pred_y = pred_x_518[0, k].item(), pred_y_518[0, k].item()

        ax.imshow(canvas_rgb)
        ax.imshow(heat_up, cmap="inferno", alpha=0.55, extent=(0, CANVAS, CANVAS, 0))
        ax.scatter([gt_x_518], [gt_y_518], c="lime", marker="o", s=70,
                   edgecolors="black", linewidths=0.8, zorder=4, label="Ground truth")
        ax.scatter([pred_x], [pred_y], c="cyan", marker="x", s=70,
                   linewidths=2.2, zorder=4, label="Soft-argmax prediction")
        peak_conf = heat.max()
        ax.set_title(f"Keypoint {k}  (peak prob {peak_conf:.3f})", fontsize=10)
        ax.set_xlim(0, CANVAS); ax.set_ylim(CANVAS, 0)
        ax.axis("off")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

        # zoomed inset around the peak so the heatmap blob is actually
        # legible (it's only a few px wide at full-canvas scale)
        cx, cy = pred_x, pred_y
        x0, x1 = cx - ZOOM_HALF_WIN, cx + ZOOM_HALF_WIN
        y0, y1 = cy - ZOOM_HALF_WIN, cy + ZOOM_HALF_WIN
        axins = inset_axes(ax, width="42%", height="42%", loc="lower left",
                            borderpad=0.6)
        axins.imshow(canvas_rgb)
        axins.imshow(heat_up, cmap="inferno", alpha=0.6, extent=(0, CANVAS, CANVAS, 0))
        axins.scatter([gt_x_518], [gt_y_518], c="lime", marker="o", s=140,
                      edgecolors="black", linewidths=1.0, zorder=4)
        axins.scatter([pred_x], [pred_y], c="cyan", marker="x", s=140,
                      linewidths=2.6, zorder=4)
        axins.set_xlim(x0, x1); axins.set_ylim(y1, y0)
        axins.set_xticks([]); axins.set_yticks([])
        for spine in axins.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(1.5)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="white", linewidth=0.8, alpha=0.7)

    for j in range(n_kp, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle(f"Soft-argmax probability heatmaps ({args.task_id}, MRE {best['mre_px']:.2f}px)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["png", "pdf", "eps"]:
        out_path = os.path.join(args.out_dir, f"fig_softargmax_{args.task_id}.{ext}")
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
    parser.add_argument("--softargmax_temp", type=float, default=10.0)
    parser.add_argument("--out_dir", type=str, default="figures/softargmax_heatmaps")
    args = parser.parse_args()
    main(args)
