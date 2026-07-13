"""
For every task, run inference with the v3 True-HRNet teacher checkpoint on
ALL of that task's validation images (same split as training -- verified
against splits/train_val_split_keys.json), rank samples by per-image mean
radial error (MRE, px, original image scale), and save the best N as
individually-overlaid PNGs (GT vs. predicted keypoints) plus one paper-ready
grid montage per task.

Usage:
  CUDA_VISIBLE_DEVICES=3 python select_best_qualitative.py \
      --checkpoint ../runs/v3_true_hrnet_softargmax/checkpoints/best_teacher_model.pth \
      --out_dir figures/qualitative_best \
      --n_per_task 10
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
from common import (build_path_index, load_image_rgb, INFERENCE_TRANSFORM, TASK_ORDER,
                     remove_padding_and_scale, GT_JSON_PATH)
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


def load_all_val_records(gt_json_path=GT_JSON_PATH):
    with open(gt_json_path) as f:
        gt_records = json.load(f)
    by_task = {}
    for rec in gt_records:
        by_task.setdefault(rec["task_id"], []).append(rec)
    return by_task


@torch.no_grad()
def run_inference_for_task(model, task_id, records, path_index, softargmax_temp):
    """Returns a list of dicts: {rec, abs_path, img_rgb, gt_pts, pred_pts, mre}."""
    results = []
    for rec in records:
        fname = os.path.basename(rec["image_path"])
        abs_path = path_index.get((task_id, fname))
        if abs_path is None:
            continue
        img_rgb = load_image_rgb(abs_path)
        orig_h, orig_w = img_rgb.shape[:2]
        tensor_img = INFERENCE_TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(device)

        logits = model.forward_phase2(tensor_img, task_id)
        pred_x_518, pred_y_518 = get_soft_argmax_coords(logits, temperature=softargmax_temp)

        gt_pts = np.asarray(rec["ground_truth_points_pixels"], dtype=float).reshape(-1, 2)
        n_kp = min(logits.shape[1], gt_pts.shape[0])
        pred_pts = []
        for k in range(n_kp):
            px, py = remove_padding_and_scale(pred_x_518[0, k].item(), pred_y_518[0, k].item(), orig_h, orig_w)
            pred_pts.append([px, py])
        pred_pts = np.asarray(pred_pts)

        # visible-keypoint mask: GT of (-1,-1) (or any negative) marks an absent point
        visible = np.all(gt_pts[:n_kp] >= 0, axis=1)
        if visible.sum() == 0:
            continue
        dists = np.linalg.norm(pred_pts[visible] - gt_pts[:n_kp][visible], axis=1)
        mre = float(dists.mean())

        results.append({
            "rec": rec,
            "abs_path": abs_path,
            "img_rgb": img_rgb,
            "gt_pts": gt_pts[:n_kp],
            "pred_pts": pred_pts,
            "visible": visible,
            "mre": mre,
        })
    return results


def save_single_overlay(sample, task_id, rank, out_dir):
    img_rgb = sample["img_rgb"]
    gt_pts = sample["gt_pts"]
    pred_pts = sample["pred_pts"]
    visible = sample["visible"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_rgb)
    ax.scatter(gt_pts[visible, 0], gt_pts[visible, 1], c="lime", marker="o", s=60,
               edgecolors="black", linewidths=0.8, label="Ground truth", zorder=3)
    ax.scatter(pred_pts[visible, 0], pred_pts[visible, 1], c="red", marker="x", s=60,
               linewidths=2.0, label="Prediction", zorder=3)
    for k in range(len(gt_pts)):
        if visible[k]:
            ax.plot([gt_pts[k, 0], pred_pts[k, 0]], [gt_pts[k, 1], pred_pts[k, 1]],
                    c="yellow", linewidth=1.0, alpha=0.8, zorder=2)
    ax.set_title(f"{task_id} -- MRE {sample['mre']:.2f}px", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    fname = f"{task_id}_rank{rank:02d}_mre{sample['mre']:.2f}_{os.path.basename(sample['abs_path'])}"
    out_path = os.path.join(out_dir, task_id, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_task_grid(selected, task_id, out_dir, n_cols=5):
    n = len(selected)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes)

    for i, sample in enumerate(selected):
        ax = axes[i // n_cols, i % n_cols]
        img_rgb = sample["img_rgb"]
        gt_pts = sample["gt_pts"]
        pred_pts = sample["pred_pts"]
        visible = sample["visible"]

        ax.imshow(img_rgb)
        ax.scatter(gt_pts[visible, 0], gt_pts[visible, 1], c="lime", marker="o", s=30,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(pred_pts[visible, 0], pred_pts[visible, 1], c="red", marker="x", s=30,
                   linewidths=1.5, zorder=3)
        for k in range(len(gt_pts)):
            if visible[k]:
                ax.plot([gt_pts[k, 0], pred_pts[k, 0]], [gt_pts[k, 1], pred_pts[k, 1]],
                        c="yellow", linewidth=0.7, alpha=0.8, zorder=2)
        ax.set_title(f"{sample['mre']:.2f}px", fontsize=9)
        ax.axis("off")

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    gt_patch = mpatches.Patch(color="lime", label="Ground truth")
    pred_patch = mpatches.Patch(color="red", label="Prediction")
    line_patch = mpatches.Patch(color="yellow", label="GT–Pred error vector")
    fig.legend(handles=[gt_patch, pred_patch, line_patch], loc="lower center", ncol=3, fontsize=9)

    fig.suptitle(f"{task_id}: {n} best validation cases by MRE", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_path = os.path.join(out_dir, f"grid_{task_id}.png")
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main(args):
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

    path_index = build_path_index()
    by_task = load_all_val_records()

    manifest = {}
    os.makedirs(args.out_dir, exist_ok=True)

    for task_id in TASK_ORDER:
        records = by_task.get(task_id, [])
        if not records:
            print(f"[{task_id}] no validation records found, skipping")
            continue
        print(f"[{task_id}] running inference on {len(records)} validation images...")
        results = run_inference_for_task(model, task_id, records, path_index, args.softargmax_temp)
        if not results:
            print(f"[{task_id}] no valid predictions, skipping")
            continue

        results.sort(key=lambda r: r["mre"])
        n_take = min(args.n_per_task, len(results))
        selected = results[:n_take]

        print(f"[{task_id}] {len(results)} scored, selecting best {n_take} "
              f"(MRE range {selected[0]['mre']:.2f}-{selected[-1]['mre']:.2f}px; "
              f"full-set median {np.median([r['mre'] for r in results]):.2f}px)")

        entry = []
        for rank, sample in enumerate(selected, start=1):
            out_path = save_single_overlay(sample, task_id, rank, args.out_dir)
            entry.append({
                "rank": rank,
                "mre_px": sample["mre"],
                "image_path": sample["abs_path"],
                "saved_overlay": out_path,
            })
        manifest[task_id] = {
            "n_validation_total": len(results),
            "n_selected": n_take,
            "full_set_median_mre_px": float(np.median([r["mre"] for r in results])),
            "selected": entry,
        }

        grid_path = save_task_grid(selected, task_id, args.out_dir, n_cols=min(5, n_take))
        print(f"[{task_id}] saved grid -> {grid_path}")

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest -> {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=4)
    parser.add_argument("--neck_branch_width", type=int, nargs=3, default=[128, 96, 64])
    parser.add_argument("--shared_head", action="store_true")
    parser.add_argument("--softargmax_temp", type=float, default=10.0)
    parser.add_argument("--out_dir", type=str, default="figures/qualitative_best")
    parser.add_argument("--n_per_task", type=int, default=10)
    args = parser.parse_args()
    main(args)
