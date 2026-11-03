#!/usr/bin/env bash
set -euo pipefail
: "${WORLD_POLICY_VENV:?Set isolated environment path}"
: "${RUN_DIR:?Set completed or interrupted run directory}"
: "${BACKUP_DIR:?Set an approved persistent destination; never a public bucket}"
[[ ! -e "$BACKUP_DIR" ]] || { echo 'Choose a new, empty backup directory' >&2; exit 2; }
mkdir -p "$BACKUP_DIR"
rsync -a --checksum "$RUN_DIR/" "$BACKUP_DIR/"
"$WORLD_POLICY_VENV/bin/python" scripts/cloud_verify.py create \
  --root "$RUN_DIR" --manifest "${RUN_DIR}_sha256.json"
"$WORLD_POLICY_VENV/bin/python" scripts/cloud_verify.py verify \
  --root "$BACKUP_DIR" --manifest "${RUN_DIR}_sha256.json"
echo 'Verified persistent backup. Check provider rules: storage may remain billable after GPU stop.'
