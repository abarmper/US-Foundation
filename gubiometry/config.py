"""Typed, YAML-backed configuration.

Replaces the scattered argparse across the three original training scripts with a
single nested-dataclass config that round-trips to `runs/<run>/config.json`, so
inference/evaluation can rebuild the exact model with `build_model_from_config`
(no more re-specifying --neck_branch_width/--unfreeze_last_n_blocks by hand under
strict=True).

Defaults reproduce the original pipeline's behavior:
  * neck.input_mode="single", optim.llrd_decay=1.0, optim.dsnt_lambda=0.0
    -> identical graph, param groups and loss as before the upgrades.
"""

import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Optional

import yaml


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class BackboneConfig:
    name: str = "dinov2_vitl14_reg"      # register variant (cleaner dense features); "dinov2_vitl14" | "dummy"
    embed_dim: int = 1024
    unfreeze_last_n_blocks: int = 4


@dataclass
class NeckConfig:
    branch_width: tuple = (128, 96, 64)
    dropout_p: float = 0.3
    input_mode: str = "multilevel"       # multi-level DINOv2 features into the neck; "single" = legacy
    feature_layers: tuple = (5, 11, 17, 23)
    shared_head: bool = False
    decoder: str = "hrnet"               # post-backbone decoder: "hrnet" (multi-branch fusion) | "simple"
                                         # ("simple" = ViTPose-style deconv upsampler; uses the last-layer
                                         #  grid, so input_mode/feature_layers are ignored under it)


@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    heatmap_size: int = 148   # match the neck's 148x148 output (finer soft-argmax)


@dataclass
class DataConfig:
    data_root: str = "data/data/train_data"   # this checkout's layout (images/ csv/ splits/ live here)
    split_file: str = "splits/train_val_split_keys.json"
    fold: Optional[int] = None           # if set -> splits/kfold_v1/fold_{fold}.json
    kfold_dir: str = "splits/kfold_v1"
    batch_size: int = 16
    num_workers: int = 8
    canvas: int = 518
    aug_strength: str = "medium"         # "none" | "light" | "medium" | "strong"
    sample_temp: float = 0.5             # sqrt task sampling (small tasks not replayed ~100x); 0=fully balanced


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    encoder_lr_mult: float = 0.1
    llrd_decay: float = 0.75            # layer-wise LR decay (canonical ViT fine-tune); 1.0 = legacy
    amp_dtype: str = "bf16"            # "bf16" (A100+) | "fp16" | "fp32"
    warmup_epochs: int = 15
    epochs: int = 500
    softargmax_temp: float = 10.0
    dsnt_lambda: float = 0.1            # DSNT spread regularizer (keeps heatmaps tight); 0.0 = off
    dsnt_type: str = "js"               # "js" | "var"
    dsnt_sigma: float = 1.0
    grad_clip: float = 1.0
    ema_alpha: float = 0.999
    early_stop_patience: int = 30
    select_metric: str = "challenge_blend"   # challenge_blend | average_mre | average_avg_mae
    # --- experiment knobs (defaults reproduce current behavior) ---
    loss_space: str = "original"        # "original" aligns loss with the original-px metric; "canvas" = legacy
    coord_loss: str = "l1"              # "l1" | "smooth_l1" | "wing"
    wing_omega: float = 10.0            # Wing loss params (used only when coord_loss="wing")
    wing_epsilon: float = 2.0
    measurement_lambda: float = 0.0     # >0 -> add clinical-measurement auxiliary loss
    scheduler: str = "warmup_cosine"    # "warmup_cosine" | "cosine" | "constant"
    max_train_batches: int = 0
    max_val_batches: int = 0


@dataclass
class Phase1Config:
    mode: str = "multicrop"             # "sameview" | "multicrop" | "dinov2"
    epochs: int = 100
    batch_size: int = 8
    lr: float = 5e-5
    weight_decay: float = 1e-4
    out_dim: int = 256                  # legacy DINO head width (multicrop mode)
    n_local_crops: int = 6
    num_workers: int = 8
    ema_alpha: float = 0.999
    # --- "dinov2" mode: DINOv2-faithful recipe (all read only when mode == "dinov2";
    #     defaults leave the legacy multicrop/sameview paths untouched) ---
    grad_accum_steps: int = 1           # effective batch = batch_size * grad_accum_steps
    # heads (vendored DINOHead)
    dino_out_dim: int = 65536           # DINO prototypes; drop to 16384/32768 if occupancy collapses
    ibot_out_dim: int = 0               # 0 -> same as dino_out_dim
    ibot_separate_head: bool = True     # official vitl14 uses a separate iBOT head
    head_hidden_dim: int = 2048
    head_bottleneck_dim: int = 256
    head_nlayers: int = 3
    # loss weights
    dino_loss_weight: float = 1.0
    ibot_loss_weight: float = 1.0
    koleo_weight: float = 0.1
    # iBOT masking
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.5
    mask_sample_probability: float = 0.5
    mask_foreground: bool = True        # US: restrict masks to foreground (fan) patches
    # temperatures / centering
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30
    student_temp: float = 0.1
    center_momentum: float = 0.9
    # EMA teacher momentum (cosine over effective steps)
    momentum_base: float = 0.994
    momentum_final: float = 1.0
    # optim schedules
    warmup_epochs: int = 10             # linear LR warmup (in epochs)
    min_lr: float = 1e-6
    weight_decay_end: float = 0.0       # 0 -> constant wd (= weight_decay); official uses 0.2
    clip_grad: float = 3.0
    freeze_last_layer_epochs: int = 1
    # high-resolution adaptation tail (DINOv2 §5: short high-res phase after the bulk)
    highres_epochs: int = 0             # 0 -> no tail; >0 -> that many extra epochs at highres_crop_size
    highres_crop_size: int = 518
    highres_batch_size: int = 0         # 0 -> batch_size (usually smaller: 518 is heavy)
    highres_grad_accum_steps: int = 0   # 0 -> grad_accum_steps (usually larger to keep the effective batch)
    highres_lr: float = 0.0             # 0 -> lr; the tail runs a compressed warmup+cosine of its own
    highres_warmup_epochs: int = 1
    init_encoder: str = ""              # dinov2 mode: init the student/teacher encoder from this
                                        # bare-encoder checkpoint (heads start fresh). Lets a tail run
                                        # start from a mid-bulk checkpoint. Set epochs=0 for a pure tail.
    # crops
    global_crop_size: int = 0           # 0 -> data.canvas (DINOv2-faithful pretrain uses 224)
    local_crop_size: int = 98
    global_scale_min: float = 0.32
    global_scale_max: float = 1.0
    local_scale_min: float = 0.05
    local_scale_max: float = 0.32
    save_encoder_from: str = "teacher"  # "teacher" | "student"
    # ultrasound augmentation
    aug: str = "legacy"                 # "legacy" | "us_v2"
    foreground_crop: bool = False       # pre-crop to the ultrasound fan bbox
    rotate_limit: float = 45.0          # us_v2 recipe uses ~10
    min_local_fg_frac: float = 0.0      # us_v2: resample local crops that are mostly black


@dataclass
class PredictConfig:
    member_run_dirs: list = field(default_factory=list)
    checkpoint_name: str = "best_teacher_model.pth"
    image_dir: str = ""                 # predict every image in a <task>/<file> folder tree (challenge val/test set)
    tta_canvases: tuple = (518,)        # multi-scale views; all must be % 14 == 0
    tta_intensity: bool = False
    average_space: str = "coord"        # "coord" | "heatmap"
    reduce: str = "mean"                # "mean" | "median" (across members x TTA views)
    oof: bool = False                   # out-of-fold: each fold member predicts only its held-out val
    out_json: str = "regression_predictions.json"   # zip always contains regression_predictions.json (Codabench)
    zip_output: bool = True
    gt_json: Optional[str] = None       # for evaluate


@dataclass
class RunConfig:
    run_name: str = "run"
    seed: int = 42
    phase: str = "phase2"
    phase1_weights: str = ""
    resume: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    predict: PredictConfig = field(default_factory=PredictConfig)


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def _wants_tuple(f):
    default = f.default if f.default is not None and not callable(getattr(f, "default_factory", None)) else None
    return isinstance(default, tuple)


def from_dict(cls, data):
    """Build a dataclass from a (possibly partial) dict; missing keys keep defaults."""
    if data is None:
        return cls()
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if is_dataclass(f.type) and isinstance(v, dict):
            kwargs[f.name] = from_dict(f.type, v)
        elif _wants_tuple(f) and isinstance(v, list):
            kwargs[f.name] = tuple(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _yaml_safe(obj):
    """Recursively turn tuples into lists so yaml.safe_dump is happy."""
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    return obj


def to_dict(cfg):
    return asdict(cfg)


def load_config(path):
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return from_dict(RunConfig, data or {})


def save_config(cfg, path):
    with open(path, "w") as fh:
        yaml.safe_dump(_yaml_safe(asdict(cfg)), fh, sort_keys=False)


def save_config_json(cfg, path, extra=None):
    """Write runs/<run>/config.json (the inference-time reconstruction source)."""
    payload = asdict(cfg)
    if extra:
        payload["_meta"] = extra
    with open(path, "w") as fh:
        json.dump(_yaml_safe(payload), fh, indent=2)


# --------------------------------------------------------------------------- #
# Overrides ("optim.epochs=1", "model.backbone.name=dummy", "neck.branch_width=96,64,48")
# --------------------------------------------------------------------------- #
def _coerce(current, raw):
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        parts = [p for p in raw.split(",") if p != ""]
        elem = current[0] if current else 0
        cast = type(elem) if current else str
        return tuple(cast(p) for p in parts)
    if isinstance(current, list):
        return [p for p in raw.split(",") if p != ""]
    if current is None:
        # best-effort: int if it parses, else string
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def apply_overrides(cfg, overrides):
    """Apply ['dotted.key=value', ...] in place, coercing to the existing type."""
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override (expected key=value): {ov!r}")
        key, raw = ov.split("=", 1)
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        leaf = parts[-1]
        current = getattr(obj, leaf)
        setattr(obj, leaf, _coerce(current, raw))
    return cfg


# --------------------------------------------------------------------------- #
# Legacy adapter: old runs/<run>/config.json stored vars(args)
# --------------------------------------------------------------------------- #
def legacy_args_to_config(args):
    """Map an old argparse `args` dict (or {'args': {...}}) to a RunConfig.

    Lets checkpoints trained by the original train_phase2_hrnet.py be rebuilt
    by build_model_from_config without hand-matching hyperparameters.
    """
    a = args.get("args", args) if isinstance(args, dict) else {}
    cfg = RunConfig(run_name=a.get("run_name", "legacy"))
    bw = a.get("neck_branch_width", [128, 96, 64])
    cfg.model.neck.branch_width = tuple(bw)
    cfg.model.neck.dropout_p = a.get("dropout_p", 0.3)
    cfg.model.neck.shared_head = bool(a.get("shared_head", False))
    cfg.model.backbone.unfreeze_last_n_blocks = a.get("unfreeze_last_n_blocks", 4)
    cfg.optim.lr = a.get("lr", 1e-4)
    cfg.optim.encoder_lr_mult = a.get("encoder_lr_mult", 0.1)
    cfg.optim.weight_decay = a.get("weight_decay", 1e-2)
    cfg.optim.softargmax_temp = a.get("softargmax_temp", 10.0)
    cfg.optim.warmup_epochs = a.get("warmup_epochs", 15)
    cfg.optim.epochs = a.get("epochs", 500)
    cfg.optim.early_stop_patience = a.get("early_stop_patience", 30)
    cfg.data.data_root = a.get("data_root", "./data")
    cfg.data.batch_size = a.get("batch_size", 16)
    cfg.data.num_workers = a.get("num_workers", 8)
    cfg.phase1_weights = a.get("phase1_weights", "")
    return cfg


def config_from_run_dir(run_dir):
    """Load a RunConfig from runs/<run>/config.json, handling both new and legacy layouts."""
    import os
    path = os.path.join(run_dir, "config.json")
    with open(path) as fh:
        data = json.load(fh)
    # New layout: full RunConfig asdict (has 'model'); legacy: {'args': {...}}
    if "model" in data:
        return from_dict(RunConfig, data)
    return legacy_args_to_config(data)
