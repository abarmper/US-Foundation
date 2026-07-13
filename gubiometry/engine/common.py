"""Shared engine helpers: run dirs, logging, optional TensorBoard, seeding,
device, checkpoint IO, and the robust Phase-1 encoder-weight loader.

Consolidates boilerplate that was triplicated across the three original training
scripts.
"""

import os
import sys
import json
import random
import logging
from datetime import datetime

import numpy as np
import torch

os.environ.setdefault("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch_gu_biometry"))

# Repo root: gubiometry/engine/common.py -> repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def runs_dir():
    return os.path.join(PROJECT_ROOT, "runs")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(amp_dtype="bf16"):
    """Return (enabled, torch_dtype, use_scaler) for autocast.

    bf16 (default) on Ampere+ -- wider range, no gradient scaler needed. Falls back to
    fp16 (+ scaler) if bf16 is unsupported, and disables autocast for "fp32"/CPU.
    """
    if not torch.cuda.is_available() or amp_dtype == "fp32":
        return False, torch.float32, False
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _install_excepthook(logger):
    """Record uncaught exceptions in the run log (so a crash on a long unattended run
    is captured in the file, not just lost in a scrolled-away terminal)."""
    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.error("Uncaught exception -- run aborted", exc_info=(exc_type, exc, tb))
    sys.excepthook = _hook


def create_logger(log_dir, tag="train"):
    """Log to stdout AND a per-run file under `log_dir`, handled entirely in-code (no
    shell redirect needed -- run it in a tmux pane and it still persists the log).

    Writes `<tag>_<timestamp>.log` and points a stable `<tag>_latest.log` symlink at it
    for `tail -f` from another pane. Line-buffered, so the tail updates live.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger(log_file)   # unique name per run so handlers don't leak
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    latest = os.path.join(log_dir, f"{tag}_latest.log")
    try:                                    # best-effort stable path for tailing
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(log_file), latest)
    except OSError:
        pass

    logger.info(f"Logging to {log_file}  (live tail: tail -f {latest})")
    _install_excepthook(logger)
    return logger


class _NullWriter:
    """No-op stand-in when TensorBoard is unavailable."""
    def add_scalar(self, *a, **k):
        pass

    def close(self):
        pass


def get_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        return _NullWriter()


def md5_of_file(path, chunk_size=8 * 1024 * 1024):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_phase1_encoder_weights(encoder, ckpt_path, device, logger=None):
    """Load a Phase-1 checkpoint into `encoder`, tolerating every shape Phase-1
    saves: bare encoder dict, full training checkpoint ({'student_state_dict':...}),
    or the multi-crop dict with `encoder.*`-prefixed keys."""
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and "student_state_dict" in raw:
        raw = raw["student_state_dict"]
    elif isinstance(raw, dict) and "teacher_state_dict" in raw:
        raw = raw["teacher_state_dict"]
    if any(k.startswith("encoder.") for k in raw):
        raw = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(raw, strict=False)
    if logger:
        logger.info(f"--> Loaded Phase-1 encoder weights ({len(raw)} tensors) | "
                    f"missing={len(missing)}, unexpected={len(unexpected)}")
        if unexpected:
            logger.warning(f"Phase-1 checkpoint had {len(unexpected)} unexpected keys "
                           f"(e.g. {list(unexpected)[:3]}) -- wrong file for this encoder?")
    return missing, unexpected


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
