#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Linux required' >&2; exit 2; }
: "${WORLD_POLICY_VENV:?Set the isolated environment path}"
: "${RUN_DIR:?Set a unique persistent run directory}"
: "${GPU_HOURLY_USD:?Set the current quoted all-in hourly price}"
: "${MAX_SPEND_USD:?Set the user-authorized spending cap}"
: "${CLOUD_RUN_AUTHORIZED:?Set yes only after instance, platform and cap approval}"
: "${TRAIN_DATASET:?Set the exact dataset used for GPU preflight}"
CONFIG="${CONFIG:-configs/v13/visual_fetch_push_task_visible_cloud.json}"
[[ "$CLOUD_RUN_AUTHORIZED" == yes ]] || { echo 'Paid run not authorized' >&2; exit 2; }
[[ -f "$RUN_DIR/gpu_preflight.json" ]] || { echo 'Missing measured GPU preflight' >&2; exit 2; }
[[ "$(realpath "$TRAIN_DATASET")" == "$(realpath "$RUN_DIR/train_dataset")" ]] || {
  echo 'Preflight must use the actual formal training dataset in RUN_DIR' >&2; exit 2;
}
"$WORLD_POLICY_VENV/bin/python" - "$RUN_DIR/gpu_preflight.json" "$GPU_HOURLY_USD" "$MAX_SPEND_USD" "$TRAIN_DATASET" "$CONFIG" <<'PY'
import hashlib, json, pathlib, sys
report = json.load(open(sys.argv[1], encoding='utf-8'))
rate, cap = map(float, sys.argv[2:4])
dataset = pathlib.Path(sys.argv[4])
config = json.load(open(sys.argv[5], encoding='utf-8'))
digest = hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()
if digest != report['dataset_manifest_sha256']:
    raise SystemExit('Training dataset changed after GPU preflight')
if (report['target_updates_per_policy'] != config['updates'] or
        report['batch_size'] != config['batch_size'] or
        report['camera_setup'] != config['camera_setup']):
    raise SystemExit('GPU preflight does not match formal training config')
if report['status'] != 'GPU_PROBE_MEASURED' or rate <= 0 or cap <= 0:
    raise SystemExit('GPU measurement or budget invalid')
seconds = sum(row['estimated_target_training_seconds'] for row in report['policies'].values())
# Budget two times measured training for rendering, evaluation, startup and backup.
estimated = 2 * seconds / 3600 * rate
print(f'Estimated total ${estimated:.2f}, authorized maximum ${cap:.2f}')
if estimated > cap:
    raise SystemExit('Estimated run exceeds authorized budget')
PY
MAX_SECONDS=$("$WORLD_POLICY_VENV/bin/python" -c 'import sys; print(int(3600*float(sys.argv[2])/float(sys.argv[1])))' "$GPU_HOURLY_USD" "$MAX_SPEND_USD")
mkdir -p "$RUN_DIR/logs"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 5 > "$RUN_DIR/logs/gpu_usage.csv" &
MONITOR_PID=$!
"$WORLD_POLICY_VENV/bin/python" scripts/cloud_cost_monitor.py \
  --output "$RUN_DIR/logs/cost.jsonl" --hourly-usd "$GPU_HOURLY_USD" \
  --max-usd "$MAX_SPEND_USD" &
COST_PID=$!
trap 'kill "$MONITOR_PID" "$COST_PID" 2>/dev/null || true' EXIT
ARGS=(--config "$CONFIG" --output "$RUN_DIR" --device cuda --resume)
timeout --signal=TERM --kill-after=30s "${MAX_SECONDS}s" \
  "$WORLD_POLICY_VENV/bin/python" -m worldpolicy.visual_experiment "${ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/logs/train.log"
"$WORLD_POLICY_VENV/bin/python" scripts/cloud_verify.py create \
  --root "$RUN_DIR" --manifest "${RUN_DIR}_sha256.json"
