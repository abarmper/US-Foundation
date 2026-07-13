"""
Compact 1xN row version of make_best_per_task_summary_figure.py: shows the
best (lowest-MRE) validation case for a small, representative subset of
tasks (default: one per clinical domain -- AOP/intrapartum, HC/prenatal,
A4C/cardiac, IVC/vascular) instead of all 9 tasks, to save space in the
manuscript. Reuses the same ranking (figures/qualitative_best/manifest.json)
and re-runs inference only on the selected tasks' already-chosen best images.

Usage:
  CUDA_VISIBLE_DEVICES=3 python make_best_per_task_row_figure.py \
      --checkpoint ../runs/v3_true_hrnet_softargmax/checkpoints/best_teacher_model.pth \
      --manifest figures/qualitative_best/manifest.json \
      --tasks AOP HC A4C IVC \
      --out_stem figures/fig_best_per_task_row
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # for "common"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from common import build_path_index, load_image_rgb, INFERENCE_TRANSFORM, remove_padding_and_scale, GT_JSON_PATH
from gubiometry.models import UnifiedBiometryModel

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch_gu_biometry"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_soft_argmax_coords(logits, temperature=10.0):
    B, K, H, W = logits.shape
    heatmaps = F.softmax(logits.view(B, K, -1) * temperature, dim=-1)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=logits.device),
        torch.arange(W, device=logits.device),
        indexing="ij"
    )
    grid_x = grid_x.flatten().float()
    grid_y = grid_y.flatten().float()
    pred_x = torch.sum(heatmaps * grid_x, dim=-1) * (518.0 / W)
    pred_y = torch.sum(heatmaps * grid_y, dim=-1) * (518.0 / H)
    return pred_x, pred_y


@torch.no_grad()
def infer_one(model, task_id, abs_path, softargmax_temp, gt_pts_flat):
    img_rgb = load_image_rgb(abs_path)
    orig_h, orig_w = img_rgb.shape[:2]
    tensor_img = INFERENCE_TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(device)

    logits = model.forward_phase2(tensor_img, task_id)
    pred_x_518, pred_y_518 = get_soft_argmax_coords(logits, temperature=softargmax_temp)

    gt_pts = np.asarray(gt_pts_flat, dtype=float).reshape(-1, 2)
    n_kp = min(logits.shape[1], gt_pts.shape[0])
    pred_pts = []
    for k in range(n_kp):
        px, py = remove_padding_and_scale(pred_x_518[0, k].item(), pred_y_518[0, k].item(), orig_h, orig_w)
        pred_pts.append([px, py])
    pred_pts = np.asarray(pred_pts)
    visible = np.all(gt_pts[:n_kp] >= 0, axis=1)
    return img_rgb, gt_pts[:n_kp], pred_pts, visible


def main(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(GT_JSON_PATH) as f:
        gt_records = json.load(f)
    gt_by_key = {}
    for rec in gt_records:
        fname = os.path.basename(rec["image_path"])
        gt_by_key[(rec["task_id"], fname)] = rec["ground_truth_points_pixels"]

    print("Loading model...")
    model = UnifiedBiometryModel(
        freeze_encoder=True,
        dropout_p=0.3,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        neck_branch_width=tuple(args.neck_branch_width),
        shared_head=args.shared_head,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    print("Load result:", model.load_state_dict(state_dict, strict=True))
    model.eval()

    build_path_index()  # not strictly needed here (paths come from manifest), kept for parity

    n_cols = len(args.tasks)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.3))
    axes = np.atleast_1d(axes)

    for ax, task_id in zip(axes, args.tasks):
        if task_id not in manifest or not manifest[task_id]["selected"]:
            ax.axis("off")
            continue

        best = manifest[task_id]["selected"][0]  # rank 1 == lowest MRE
        abs_path = best["image_path"]
        fname = os.path.basename(abs_path)
        gt_pts_flat = gt_by_key.get((task_id, fname))
        if gt_pts_flat is None:
            ax.axis("off")
            continue

        img_rgb, gt_pts, pred_pts, visible = infer_one(
            model, task_id, abs_path, args.softargmax_temp, gt_pts_flat
        )

        ax.imshow(img_rgb)
        ax.scatter(gt_pts[visible, 0], gt_pts[visible, 1], c="lime", marker="o", s=55,
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.scatter(pred_pts[visible, 0], pred_pts[visible, 1], c="red", marker="x", s=55,
                   linewidths=1.8, zorder=3)
        for k in range(len(gt_pts)):
            if visible[k]:
                ax.plot([gt_pts[k, 0], pred_pts[k, 0]], [gt_pts[k, 1], pred_pts[k, 1]],
                        c="yellow", linewidth=0.9, alpha=0.9, zorder=2)
        ax.set_title(f"{task_id}  (MRE {best['mre_px']:.2f}px)", fontsize=12, fontweight="bold")
        ax.axis("off")

    gt_patch = mpatches.Patch(color="lime", label="Ground truth")
    pred_patch = mpatches.Patch(color="red", label="Prediction")
    line_patch = mpatches.Patch(color="yellow", label="GT–Pred error vector")
    fig.legend(handles=[gt_patch, pred_patch, line_patch], loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

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
    parser.add_argument("--tasks", type=str, nargs="+", default=["AOP", "HC", "A4C", "IVC"])
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=4)
    parser.add_argument("--neck_branch_width", type=int, nargs=3, default=[128, 96, 64])
    parser.add_argument("--shared_head", action="store_true")
    parser.add_argument("--softargmax_temp", type=float, default=10.0)
    parser.add_argument("--out_stem", type=str, default="figures/fig_best_per_task_row")
    args = parser.parse_args()
    main(args)
