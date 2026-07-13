#!/usr/bin/env bash
# Phase 2: HRNet neck + soft-argmax heads, initialized from a Phase 1 checkpoint.
# Baseline config reproduces the original method; use configs/phase2_upgraded.yaml
# for the challenge upgrades (multi-level features, LLRD, DSNT, register backbone).
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_phase2_hrnet.sh \
#          -o phase1_weights=pretrained/dinov2_adapted_ep20.pth [-o key=val ...]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m gubiometry phase2 --config configs/phase2_baseline.yaml "$@"
