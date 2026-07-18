"""MultiCropBiometryDataset -- wraps the base dataset to yield 2 global + N local
crops per image for DINO-style Phase-1 self-distillation.

`objective="multicrop"` (legacy) uses this dataset with the default flags (byte-identical
to the original). `objective="dinov2"` opts into foreground/fan-aware cropping and pairs the
dataset with `make_dinov2_collate` (crop-major stacking + iBOT mask tensors).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import read_rgb


def fan_bbox(image, thresh=10):
    """Bounding box of the non-black ultrasound fan (numpy, cv2-free). Returns
    (top, bottom, left, right); falls back to the full frame on a degenerate/black image."""
    gray = image.max(axis=2) if image.ndim == 3 else image
    rows = np.where(gray.max(axis=1) > thresh)[0]
    cols = np.where(gray.max(axis=0) > thresh)[0]
    if rows.size < 2 or cols.size < 2:
        return 0, image.shape[0], 0, image.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


class MultiCropBiometryDataset(Dataset):
    def __init__(self, base_dataset, transforms_dict, n_local_crops=6,
                 foreground_crop=False, min_local_fg_frac=0.0, fg_thresh=10):
        self.base_dataset = base_dataset
        self.global_transform = transforms_dict["global"]
        self.local_transform = transforms_dict["local"]
        self.n_local_crops = n_local_crops
        self.foreground_crop = foreground_crop
        self.min_local_fg_frac = min_local_fg_frac      # >0 -> reject mostly-black locals (best-of-N)
        self.fg_thresh = fg_thresh

    def __len__(self):
        return len(self.base_dataset)

    def _local_crop(self, image):
        """A local crop; if min_local_fg_frac>0, draw up to 4 and keep the most-textured
        (highest spatial variance) so the crop isn't pure black surround/speckle."""
        crop = self.local_transform(image=image)["image"]
        if self.min_local_fg_frac <= 0:
            return crop
        best, best_var = crop, float(crop.float().var())
        for _ in range(3):
            if best_var >= self.min_local_fg_frac:
                break
            c = self.local_transform(image=image)["image"]
            v = float(c.float().var())
            if v > best_var:
                best, best_var = c, v
        return best

    def __getitem__(self, idx):
        info = self.base_dataset.samples[idx]
        image = read_rgb(info["abs_path"])
        if image is None:
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())

        if self.foreground_crop:
            t, b, l, r = fan_bbox(image, self.fg_thresh)
            image = image[t:b, l:r]

        crops = [self.global_transform(image=image)["image"] for _ in range(2)]
        crops += [self._local_crop(image) for _ in range(self.n_local_crops)]
        return {"crops": crops, "task_id": info["task_id"]}


# --------------------------------------------------------------------------- #
# DINOv2 collate: crop-major stacking + iBOT mask tensors
# --------------------------------------------------------------------------- #
def _foreground_patch_map(global_crops, patch_size, std_thresh=0.05):
    """(B2, P*P) bool: patches whose per-patch std exceeds a small threshold (flat/black
    background regions have ~0 std). Adaptive-free, normalization-agnostic best-effort."""
    b2, c, h, w = global_crops.shape
    p = h // patch_size
    gp = global_crops.reshape(b2, c, p, patch_size, p, patch_size)
    gp = gp.permute(0, 2, 4, 1, 3, 5).reshape(b2, p * p, c * patch_size * patch_size)
    return gp.float().std(dim=-1) > std_thresh


def make_dinov2_collate(patch_size, mask_ratio_min, mask_ratio_max, mask_sample_probability,
                        mask_generator, mask_foreground=True, generator=None):
    """Build a DataLoader collate_fn for the dinov2 objective.

    Stacks global crops CROP-MAJOR into (2B, C, Hg, Wg) (all crop-0 then all crop-1, so a
    later `.chunk(2)` splits per-crop) and locals into (n_local*B, C, Hl, Wl). Generates
    per-global-crop iBOT block masks (optionally restricted to foreground patches) and the
    (mask_indices_list, masks_weight, n_masked_patches) tensors iBOTPatchLossV2 consumes.
    """
    rng = np.random.default_rng() if generator is None else generator

    def collate(batch):
        n_local = len(batch[0]["crops"]) - 2
        globals_ = [item["crops"][0] for item in batch] + [item["crops"][1] for item in batch]
        collated_global = torch.stack(globals_, dim=0)                       # (2B, C, Hg, Wg)
        locals_ = [item["crops"][2 + j] for j in range(n_local) for item in batch]
        collated_local = torch.stack(locals_, dim=0) if n_local else None    # (n_local*B, C, Hl, Wl)

        b2, _, hg, _ = collated_global.shape
        p = hg // patch_size
        n_tokens = p * p
        fg = _foreground_patch_map(collated_global, patch_size) if mask_foreground else None

        masks = torch.zeros(b2, n_tokens, dtype=torch.bool)
        for j in range(b2):
            if rng.random() >= mask_sample_probability:
                continue
            ratio = rng.uniform(mask_ratio_min, mask_ratio_max)
            m = torch.from_numpy(mask_generator(int(n_tokens * ratio)).reshape(-1))
            if fg is not None:
                m = m & fg[j]
            masks[j] = m

        flat = masks.reshape(-1)
        mask_indices_list = torch.nonzero(flat, as_tuple=False).squeeze(1)    # (M,)
        counts = masks.sum(dim=1).clamp(min=1).float()                       # (2B,)
        masks_weight = (1.0 / counts).unsqueeze(1).expand_as(masks)[masks]    # (M,)
        return {
            "collated_global_crops": collated_global,
            "collated_local_crops": collated_local,
            "collated_masks": masks,
            "mask_indices_list": mask_indices_list,
            "masks_weight": masks_weight,
            "n_masked_patches": int(mask_indices_list.numel()),
            "n_global_crops": b2,
            "n_local_crops": n_local,
            "task_id": batch[0]["task_id"],
        }

    return collate
