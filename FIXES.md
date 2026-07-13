# Fixes & usability improvements

> Note: the `src/{models,data,training}` modules referenced below were
> subsequently refactored into the installable `gubiometry/` package (all fixes
> carried over). See [METHOD_CHANGES.md](METHOD_CHANGES.md) for the full refactor
> and the challenge-specific method upgrades.


Small, clearly-accidental problems fixed. **No method/architecture change** —
same SSL objective, same HRNet neck, same soft-argmax loss.

## Correctness (things that were plainly wrong)

- **`make_splits.py` produced the wrong file.** It wrote unused
  `train_metadata.csv` / `val_metadata.csv`, but the dataset loader, viz scripts,
  and README all consume `splits/train_val_split_keys.json`. Rewrote it to emit
  that exact JSON (`{"train": [[task_id, filename], …], "val": […]}`, 80/20
  stratified by task, seed 42), and to skip `pseudo` CSVs like the loader does.
  The documented "regenerate the split" step now actually works.

- **Multi-crop Phase 1 crashed for short runs.** `DINOLoss` built
  `torch.ones(nepochs - 30)` → negative size when `epochs < 30` (e.g. smoke
  tests). Clamped the warmup length to `nepochs`. Verified for `nepochs ∈ {1,5,30,100}`.

- **Wrong script name in an error message.** Dataset's "split not found" error
  pointed at `make_split_index.py` (doesn't exist) → now `src/data/make_splits.py`.

## Ease of use

- **Phase 2 now accepts any Phase 1 checkpoint shape.** `--phase1_weights`
  previously required a hand-stripped bare encoder dict; the multi-crop
  checkpoint had to be manually edited first. New `load_phase1_encoder_weights`
  auto-handles bare dicts, full `student_state_dict` checkpoints, and the
  multi-crop `encoder.*`-prefixed dict, logging matched/missing/unexpected
  counts. Verified against all three shapes. (`pretrained/README.md` updated.)

- **`--num_workers` flag** added to all three training scripts (default 8; was
  hardcoded), so the pipeline runs on machines with fewer cores.

- **`train_phase1.py` output path fixed.** It wrote `runs/` relative to the
  current directory while the other two scripts used the repo root; now all
  three use `PROJECT_ROOT` consistently.

- **Doc command fixed.** `docs/ARCHITECTURE.md` "Running" block used a hardcoded
  absolute interpreter path and the wrong filename (`train_hrnet.py`) → now
  `python src/training/train_phase2_hrnet.py` with a repo-relative checkpoint.

## Efficiency

- **`generate_heatmaps` hoisted the coordinate grid out of the per-keypoint
  loop** (was re-allocating two 128×128 grids per keypoint per image in the
  dataloader hot path). Identical output, less work. Note: these heatmaps are
  currently unused by the soft-argmax Phase 2 loop — left in place to avoid a
  behavioral change.
