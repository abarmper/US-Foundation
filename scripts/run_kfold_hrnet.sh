#!/usr/bin/env bash
# 5-fold cross-validation training (one Phase 2 model per fold -> ensemble members).
# First generate fold splits:  python -m gubiometry make-splits --kfold --n-splits 5
#
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_kfold_hrnet.sh \
#          -o phase1_weights=pretrained/dinov2_reg_adapted_ep20.pth [-o key=val ...]
set -euo pipefail
cd "$(dirname "$0")/.."

python -m gubiometry kfold --config configs/phase2_upgraded.yaml "$@"
