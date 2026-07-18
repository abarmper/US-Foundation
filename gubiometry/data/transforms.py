"""Image transforms.

The inference/validation letterbox is implemented in pure numpy/PIL (no
albumentations, no cv2) so prediction/evaluation -- and the Docker test container --
have no heavy augmentation dependency. Albumentations is used only for train-time
augmentation and is imported lazily.

All transforms follow the albumentations call signature
`transform(image=rgb_uint8, keypoints=list_of_xy) -> {"image": chw_tensor, "keypoints": list}`
so the dataset is transform-agnostic.
"""

import numpy as np
import torch

from ..geometry import letterbox_params, forward_letterbox_kps

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_to_tensor(img_rgb, canvas=518):
    """RGB uint8 HxWx3 -> normalized CHW float tensor via centered letterbox (fill=0)."""
    from PIL import Image

    h, w = img_rgb.shape[:2]
    _, pad_left, pad_top, new_h, new_w = letterbox_params(h, w, canvas)
    im = Image.fromarray(img_rgb).resize((new_w, new_h), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    out = np.zeros((canvas, canvas, 3), dtype=np.float32)
    out[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = arr
    out = (out - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


class LetterboxTransform:
    """Deterministic val/inference transform (no augmentation)."""

    def __init__(self, canvas=518):
        self.canvas = canvas

    def __call__(self, image, keypoints=None):
        tensor = letterbox_to_tensor(image, self.canvas)
        out = {"image": tensor}
        if keypoints is not None:
            h, w = image.shape[:2]
            kps = forward_letterbox_kps(np.asarray(keypoints, dtype=np.float64), h, w, self.canvas)
            out["keypoints"] = [tuple(p) for p in kps]
        return out


AUG_STRENGTHS = ("none", "light", "medium", "strong")


def _train_aug_ops(strength):
    """Albumentations op list between resize and normalize, by strength.
    `medium` is the original fixed pipeline (default)."""
    import albumentations as A
    if strength == "none":
        return []
    if strength == "light":
        return [
            A.Affine(scale=(0.98, 1.02), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
    if strength == "medium":
        return [
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.10, 0.10), rotate=(-30, 30), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.OneOf([
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.5),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        ]
    if strength == "strong":
        return [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.15, 0.15), rotate=(-45, 45), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.OneOf([
                A.RandomGamma(gamma_limit=(50, 150), p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
            ], p=0.5),
        ]
    raise ValueError(f"Unknown aug_strength: {strength!r} (expected one of {AUG_STRENGTHS})")


def get_train_transforms(canvas=518, strength="medium"):
    """Albumentations train-augmentation pipeline at the given strength
    (none|light|medium|strong; `medium` = the original fixed pipeline).

    Falls back to the no-augmentation LetterboxTransform if albumentations is not
    installed, so training still runs (e.g. CPU smoke tests). The training
    environment (requirements.txt) has albumentations.
    """
    if strength not in AUG_STRENGTHS:
        raise ValueError(f"Unknown aug_strength: {strength!r} (expected one of {AUG_STRENGTHS})")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2
    except ImportError:
        import warnings
        warnings.warn("albumentations/cv2 not installed; training WITHOUT augmentation.")
        return LetterboxTransform(canvas)

    ops = [
        A.LongestMaxSize(max_size=canvas),
        A.PadIfNeeded(min_height=canvas, min_width=canvas, border_mode=cv2.BORDER_CONSTANT, fill=0),
    ] + _train_aug_ops(strength) + [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(ops, keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


# --------------------------------------------------------------------------- #
# Phase-1 SSL transforms (albumentations, lazy) -- ported from transforms_final.py
# --------------------------------------------------------------------------- #
def get_unlabeled_transforms(canvas=518):
    """Strong augmentation for same-view Phase-1 SSL (train_unlabeled)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2

    return A.Compose([
        A.LongestMaxSize(max_size=canvas),
        A.PadIfNeeded(min_height=canvas, min_width=canvas, border_mode=cv2.BORDER_CONSTANT, fill=0),
        A.Affine(scale=(0.7, 1.3), translate_percent=(-0.15, 0.15), rotate=(-45, 45), p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.RandomGamma(gamma_limit=(50, 150), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
        ], p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_multicrop_transforms(global_size=518, local_size=98):
    """DINO multi-crop: 2 global (>=50%) + N local (<40%) crops. Lazy albumentations."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    normalization = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    color_noise = [
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.RandomGamma(gamma_limit=(50, 150), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
        ], p=0.6),
    ]
    global_t = A.Compose([
        A.RandomResizedCrop(size=(global_size, global_size), scale=(0.5, 1.0), p=1.0),
        A.Affine(rotate=(-45, 45), translate_percent=(-0.1, 0.1), p=0.5),
        *color_noise, *normalization,
    ])
    local_t = A.Compose([
        A.RandomResizedCrop(size=(local_size, local_size), scale=(0.1, 0.4), p=1.0),
        A.Affine(rotate=(-45, 45), p=0.5),
        *color_noise, *normalization,
    ])
    return {"global": global_t, "local": local_t}


class _NumpyMultiCrop:
    """Albumentations-free fallback (CPU smoke venv): random-resized-crop + normalize + CHW tensor,
    exposing the same `t(image=arr)["image"]` interface. No rotation/color aug (identity-ish)."""
    def __init__(self, size, scale, mean, std):
        self.size, self.scale = size, scale
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image):
        import torch
        from PIL import Image
        h, w = image.shape[:2]
        area = h * w
        s = np.random.uniform(*self.scale)
        ch = max(1, min(h, int(round((area * s) ** 0.5))))
        cw = max(1, min(w, int(round((area * s) ** 0.5))))
        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)
        crop = image[top:top + ch, left:left + cw]
        arr = np.asarray(Image.fromarray(crop).resize((self.size, self.size))).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)                      # HWC -> CHW
        arr = (arr - self.mean) / self.std
        return {"image": torch.from_numpy(arr)}


def get_multicrop_transforms_v2(global_size=224, local_size=98, global_scale=(0.32, 1.0),
                                local_scale=(0.05, 0.32), rotate_limit=10.0, normalization="imagenet"):
    """Ultrasound-aware DINOv2 multi-crop (2 global + N local). Same interface as
    get_multicrop_transforms; differences: DINOv2 crop scales, gentler rotation on globals only,
    aspect-ratio clamp, horizontal flip. Foreground/fan handling lives in the dataset
    (MultiCropBiometryDataset) since it needs the raw image. Falls back to a numpy pipeline
    when albumentations is unavailable (CPU smoke venv)."""
    if normalization == "grayscale":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        import warnings
        warnings.warn("albumentations not installed; using numpy multi-crop fallback (no aug).")
        return {"global": _NumpyMultiCrop(global_size, global_scale, mean, std),
                "local": _NumpyMultiCrop(local_size, local_scale, mean, std)}

    norm = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    color_noise = [
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.RandomGamma(gamma_limit=(50, 150), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
        ], p=0.5),
    ]
    ratio = (0.85, 1.18)
    global_t = A.Compose([
        A.RandomResizedCrop(size=(global_size, global_size), scale=tuple(global_scale), ratio=ratio, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-rotate_limit, rotate_limit), p=0.5),
        *color_noise, *norm,
    ])
    local_t = A.Compose([
        A.RandomResizedCrop(size=(local_size, local_size), scale=tuple(local_scale), ratio=ratio, p=1.0),
        A.HorizontalFlip(p=0.5),
        *color_noise, *norm,
    ])
    return {"global": global_t, "local": local_t}


# --------------------------------------------------------------------------- #
# Test-time augmentation (safe views only -- NO naive flips; keypoints are semantic)
# --------------------------------------------------------------------------- #
def _gamma(img_rgb, g):
    lut = (np.linspace(0, 1, 256, dtype=np.float32) ** g * 255.0).astype(np.uint8)
    return lut[img_rgb]


def make_tta_views(img_rgb, canvases=(518,), intensity=False):
    """Return a list of {"tensor": CHW, "canvas": int} views.

    Multi-scale (each canvas must be a multiple of 14) and optional intensity
    (gamma) variants. Each view carries its own canvas so soft-argmax + inverse
    letterbox use the correct scale -- never hardcode 518.
    """
    views = []
    for c in canvases:
        views.append({"tensor": letterbox_to_tensor(img_rgb, c), "canvas": c})
        if intensity:
            for g in (0.8, 1.2):
                views.append({"tensor": letterbox_to_tensor(_gamma(img_rgb, g), c), "canvas": c})
    return views
