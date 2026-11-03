#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || exit 2
: "${WORLD_POLICY_VENV:?Set the isolated environment path}"
: "${TRAIN_DATASET:?Set verified train_dataset directory}"
: "${RUN_DIR:?Set unique persistent run directory}"
mkdir -p "$RUN_DIR"
"$WORLD_POLICY_VENV/bin/python" scripts/cloud_preflight.py \
  --train-dir "$TRAIN_DATASET" --output "$RUN_DIR/gpu_preflight.json" \
  --device cuda --batch-size 32 --updates 5 --target-updates 1500
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$RUN_DIR/gpu_identity.csv"
