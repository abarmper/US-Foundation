# Pretrained / Phase-1 checkpoints

This folder is where you place the Phase 1 (DINOv2 SSL domain-adapted
encoder) checkpoint before running Phase 2, e.g.:

```
pretrained/
└── dinov2_adapted_ep20.pth
```

Checkpoints are produced by `src/training/train_phase1.py` (same-view
alignment) or `src/training/train_phase1_multicrop.py` (asymmetric
multi-crop, DINO-style), and consumed via `--phase1_weights` in
`src/training/train_phase2_hrnet.py`.

**Checkpoint shape (handled automatically):** `train_phase2_hrnet.py` accepts
any of the shapes the Phase 1 scripts save and extracts the encoder weights for
you (`load_phase1_encoder_weights`):

- a *bare encoder* state dict (`cls_token`, `blocks.23...`, no prefix) —
  what `train_phase1.py` saves as `dinov2_adapted_epN.pth`;
- a full training checkpoint (`{'student_state_dict': ...}` /
  `latest_*checkpoint.pth`);
- the multi-crop checkpoint from `train_phase1_multicrop.py`, whose keys are
  prefixed (`encoder.*` alongside `shared_upsampler.*` / `heads.*`) — the
  `encoder.*` keys are selected and the prefix stripped for you.

So you can point `--phase1_weights` straight at any of these; the loader logs
how many tensors matched and warns if the file looks wrong for this encoder.
(Manual pre-stripping like the snippet below is no longer required.)

```python
# Legacy manual extraction — no longer needed, kept only for reference.
import torch
sd = torch.load("dinov2_multicrop_adapted_epX.pth", map_location="cpu")
encoder_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
torch.save(encoder_sd, "dinov2_multicrop_adapted_epX_encoder_only.pth")
```

Checkpoints are not tracked in git (see `.gitignore`) — they're large binary
artifacts, fully reproducible by re-running Phase 1.
