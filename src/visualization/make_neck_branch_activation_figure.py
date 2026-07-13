"""
Per-branch neck activation figure: replays TrueHRNetNeck.forward() step by
step (same submodules, same order, just called manually so we can keep the
intermediate b1/b2/b3 tensors instead of only the final fused output), takes
the post-Exchange-Unit-2 branch activations (37x37, 74x74, 148x148 -- the
fully cross-informed, pre-fusion representations) plus the final fused map,
mean-pools each over channels, upsamples to the 518-canvas, and overlays as a
heatmap on the model input. This is meant to visualize, with real
activations, the coarse-to-fine / fine-to-coarse exchange story described in
the Phase 2 section (Section "Top-down multi-scale exchange").

No architecture changes -- this only calls the already-loaded submodules of
`model.shared_upsampler` (TrueHRNetNeck) in the same order as its own
forward(), so it works with the existing best_teacher_model.pth unchanged.

Usage:
  CUDA_VISIBLE_DEVICES=3 python make_neck_branch_activation_figure.py \
      --checkpoint ../runs/v3_true_hrnet_softargmax/checkpoints/best_teacher_model.pth \
      --manifest figures/qualitative_best/manifest.json \
      --task_id HC \
      --out_dir figures/neck_branch_activations
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
from common import load_image_rgb, INFERENCE_TRANSFORM
from gubiometry.models import UnifiedBiometryModel

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch_gu_biometry"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANVAS = 518


@torch.no_grad()
def replay_neck_forward(neck, x):
    """Same steps as TrueHRNetNeck.forward(), but returns every intermediate
    branch tensor instead of only the final fused map."""
    intermediates = {}

    b1 = neck.stage1_b1(x)
    b2 = neck.stage1_b2(x)
    intermediates["stage1_b1"] = b1
    intermediates["stage1_b2"] = b2

    b1, b2 = neck.exchange1(b1, b2)
    intermediates["exchange1_b1"] = b1
    intermediates["exchange1_b2"] = b2

    b1 = neck.stage2_b1(b1)
    b2 = neck.stage2_b2(b2)
    b3 = neck.stage2_b3(b2)
    intermediates["stage2_b1"] = b1
    intermediates["stage2_b2"] = b2
    intermediates["stage2_b3"] = b3

    b1, b2, b3 = neck.exchange2(b1, b2, b3)
    intermediates["exchange2_b1"] = b1
    intermediates["exchange2_b2"] = b2
    intermediates["exchange2_b3"] = b3

    out = neck.fuse_b1(b1) + neck.fuse_b2(b2) + neck.fuse_b3(b3)
    out = neck.dropout(out)
    out = neck.final_layer(out)
    intermediates["fused_output"] = out

    return intermediates


def channel_mean_heatmap(tensor, canvas=CANVAS):
    """(1, C, H, W) -> (canvas, canvas) normalized [0,1] heatmap."""
    m = tensor.mean(dim=1, keepdim=True)  # (1, 1, H, W)
    m = F.interpolate(m, size=(canvas, canvas), mode="bilinear", align_corners=False)
    m = m[0, 0].cpu().numpy()
    m -= m.min()
    denom = m.max() - m.min() if m.max() > m.min() else 1.0
    return m / denom if isinstance(denom, float) and denom != 0 else m / (m.max() + 1e-8)


def save_clean_panel(base_img, heatmap, valid_mask, out_stem, canvas=CANVAS):
    """Title-free, axis-free, tightly-cropped single-panel export -- meant
    to be dropped directly into a diagram (e.g. as a node image in the
    pipeline drawio) rather than read as a standalone figure."""
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(canvas / 100, canvas / 100), dpi=100)
    ax.imshow(base_img)
    rgba = cm.jet(heatmap)
    rgba[..., 3] = np.where(valid_mask, 0.45, 0.0) if valid_mask is not None else 0.45
    ax.imshow(rgba, extent=(0, canvas, canvas, 0))
    ax.set_xlim(0, canvas); ax.set_ylim(canvas, 0)
    ax.axis("off")
    for ext in ["png", "pdf", "eps"]:
        out_path = f"{out_stem}.{ext}"
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_path, format=ext, bbox_inches="tight", pad_inches=0, **kwargs)
        print(f"Saved {out_path}")
    plt.close(fig)


def overlay_heatmap(ax, base_img, heatmap, title, valid_mask=None, alpha=0.45):
    import matplotlib.cm as cm
    ax.imshow(base_img)
    rgba = cm.jet(heatmap)
    if valid_mask is not None:
        rgba[..., 3] = np.where(valid_mask, alpha, 0.0)
    else:
        rgba[..., 3] = alpha
    ax.imshow(rgba, extent=(0, CANVAS, CANVAS, 0))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def main(args):
    with open(args.manifest) as f:
        manifest = json.load(f)
    best = manifest[args.task_id]["selected"][0]
    abs_path = best["image_path"]

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
    tensor_img = INFERENCE_TRANSFORM(image=img_rgb)["image"].unsqueeze(0).to(device)

    display_transform = A.Compose([
        A.LongestMaxSize(max_size=CANVAS),
        A.PadIfNeeded(min_height=CANVAS, min_width=CANVAS, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ])
    canvas_rgb = display_transform(image=img_rgb)["image"]

    # mask out the letterbox padding so its uninformative activations don't
    # show up as spurious "hot corners" in the overlay
    orig_h, orig_w = img_rgb.shape[:2]
    scale = CANVAS / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_top = (CANVAS - new_h) // 2
    pad_left = (CANVAS - new_w) // 2
    valid_mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    valid_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = True

    print("Running encoder + neck...")
    with torch.no_grad():
        patch_tokens = model.encoder.forward_features(tensor_img)["x_norm_patchtokens"]
        B, N, C = patch_tokens.shape
        patch_h = patch_w = int(N ** 0.5)
        feat_grid = patch_tokens.permute(0, 2, 1).view(B, C, patch_h, patch_w)
        intermediates = replay_neck_forward(model.shared_upsampler, feat_grid)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Main 2x2 figure: post-Exchange-2 B1/B2/B3 (pre-fusion) + final fused ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    panels = [
        ("exchange2_b1", f"B1 (37×37) after Exchange 2\n(coarse, semantically anchored)"),
        ("exchange2_b2", f"B2 (74×74) after Exchange 2\n(intermediate resolution)"),
        ("exchange2_b3", f"B3 (148×148) after Exchange 2\n(fine, boundary-sensitive)"),
        ("fused_output", f"Final fused output (148×148, 128ch)\nfed to task-specific heads"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        heat = channel_mean_heatmap(intermediates[key])
        overlay_heatmap(ax, canvas_rgb, heat, title, valid_mask=valid_mask)

    fig.suptitle(f"Multi-Stage HRNet Neck: per-branch activations ({args.task_id} example)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ["png", "pdf", "eps"]:
        out_path = os.path.join(args.out_dir, f"fig_neck_branches.{ext}")
        kwargs = {"dpi": 300} if ext == "png" else {}
        plt.savefig(out_path, format=ext, bbox_inches="tight", **kwargs)
        print(f"Saved {out_path}")
    plt.close(fig)

    # ---- Clean standalone panels (for embedding in the pipeline diagram) ----
    panel_dir = os.path.join(args.out_dir, "individual_panels")
    os.makedirs(panel_dir, exist_ok=True)
    panel_names = {
        "exchange2_b1": "branch_b1_37x37",
        "exchange2_b2": "branch_b2_74x74",
        "exchange2_b3": "branch_b3_148x148",
        "fused_output": "branch_fused_148x148",
    }
    for key, name in panel_names.items():
        heat = channel_mean_heatmap(intermediates[key])
        save_clean_panel(canvas_rgb, heat, valid_mask, os.path.join(panel_dir, name))
    # plain (no-overlay) input image too, useful as the diagram's "input" node
    fig0, ax0 = plt.subplots(figsize=(CANVAS / 100, CANVAS / 100), dpi=100)
    ax0.imshow(canvas_rgb); ax0.axis("off")
    for ext in ["png", "pdf", "eps"]:
        out_path = os.path.join(panel_dir, f"input_image.{ext}")
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig0.savefig(out_path, format=ext, bbox_inches="tight", pad_inches=0, **kwargs)
        print(f"Saved {out_path}")
    plt.close(fig0)

    # ---- Bonus: full stage-by-stage progression (all 10 intermediates) ----
    order = ["stage1_b1", "stage1_b2", "exchange1_b1", "exchange1_b2",
             "stage2_b1", "stage2_b2", "stage2_b3",
             "exchange2_b1", "exchange2_b2", "exchange2_b3", "fused_output"]
    labels = ["Stage1 B1 init (37)", "Stage1 B2 init (74)",
              "Exch1 B1 (37)", "Exch1 B2 (74)",
              "Stage2 B1 refine (37)", "Stage2 B2 refine (74)", "Stage2 B3 spawn (148)",
              "Exch2 B1 (37)", "Exch2 B2 (74)", "Exch2 B3 (148)",
              "Final fused (148)"]
    n = len(order)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    axes2 = np.atleast_2d(axes2)
    for i, (key, label) in enumerate(zip(order, labels)):
        ax = axes2[i // n_cols, i % n_cols]
        heat = channel_mean_heatmap(intermediates[key])
        overlay_heatmap(ax, canvas_rgb, heat, label, valid_mask=valid_mask)
    for j in range(n, n_rows * n_cols):
        axes2[j // n_cols, j % n_cols].axis("off")
    fig2.suptitle(f"Full stage-by-stage neck progression ({args.task_id} example)",
                  fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["png", "pdf", "eps"]:
        out_path = os.path.join(args.out_dir, f"fig_neck_full_progression.{ext}")
        kwargs = {"dpi": 300} if ext == "png" else {}
        plt.savefig(out_path, format=ext, bbox_inches="tight", **kwargs)
        print(f"Saved {out_path}")
    plt.close(fig2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--task_id", type=str, default="HC")
    parser.add_argument("--unfreeze_last_n_blocks", type=int, default=4)
    parser.add_argument("--neck_branch_width", type=int, nargs=3, default=[128, 96, 64])
    parser.add_argument("--shared_head", action="store_true")
    parser.add_argument("--out_dir", type=str, default="figures/neck_branch_activations")
    args = parser.parse_args()
    main(args)
