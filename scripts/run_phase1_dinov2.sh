#!/usr/bin/env bash
# Phase 1 -- DINOv2-faithful continued pretraining. Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> scripts/run_phase1_dinov2.sh [extra -o overrides...]
# Logs to runs/phase1_dinov2/phase1_latest.log (in-code). Resume with
#   -o resume=runs/phase1_dinov2/checkpoints/latest_checkpoint.pth
set -euo pipefail
cd "$(dirname "$0")/.."
python -m gubiometry phase1 --config configs/phase1_dinov2.yaml "$@"
