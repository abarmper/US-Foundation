#!/usr/bin/env bash
# Phase 1: DINOv2 SSL domain adaptation (same-view cosine alignment).
# The multi-crop variant (run_phase1_multicrop.sh) is recommended -- this one is
# collapse-prone under long training.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase1.sh [-o key=val ...]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m gubiometry phase1 --config configs/phase1_multicrop.yaml \
  -o run_name=phase1_sameview -o phase1.mode=sameview -o phase1.epochs=60 "$@"
