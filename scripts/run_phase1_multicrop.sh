#!/usr/bin/env bash
# Phase 1: DINOv2 SSL domain adaptation (DINO multi-crop).
# Set model.backbone.name to match your intended Phase 2 backbone.
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase1_multicrop.sh [-o key=val ...]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m gubiometry phase1 --config configs/phase1_multicrop.yaml "$@"
