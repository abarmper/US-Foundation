"""gubiometry -- DINOv2-HRNet multi-task ultrasound landmark detection.

Two-phase pipeline (SSL domain adaptation -> HRNet neck + per-task soft-argmax
heads) for the GU/FU Biometry MICCAI 2026 challenge. See METHOD_CHANGES.md for the
architectural upgrades layered on top of the original method.

Light-weight package init: heavy submodules (torch models, albumentations
transforms) are imported lazily via __getattr__ so that e.g. `import gubiometry`
does not pull torch until something actually needs it.
"""

__version__ = "0.1.0"

_LAZY = {
    "build_model_from_config": ("gubiometry.models.model", "build_model_from_config"),
    "load_config": ("gubiometry.config", "load_config"),
    "RunConfig": ("gubiometry.config", "RunConfig"),
    "challenge_score": ("gubiometry.metrics", "challenge_score"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        return getattr(importlib.import_module(module_name), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
