#!/usr/bin/env bash
# Official LeRobot LIBERO evaluation. Run only inside a separately prepared Linux checkout.
set -euo pipefail

[[ "$(uname -s)" == Linux ]] || { echo 'BLOCKED: LIBERO requires Linux' >&2; exit 2; }
: "${LEROBOT_COMMIT:?Pin the official LeRobot source commit}"
: "${LEROBOT_DIR:?Set official LeRobot source checkout}"
: "${LEROBOT_VENV:?Set official LeRobot isolated venv}"
: "${POLICY_PATH:?Set an official checkpoint path or Hub ID with pinned revision}"
: "${POLICY_REVISION:?Pin the checkpoint revision / commit}"
: "${POLICY_WEIGHTS_FILE:?Point to locally cached official weights file}"
: "${POLICY_WEIGHTS_SHA256:?Pin the weights SHA256}"
: "${DATASET_REVISION:?Record the compatible training dataset revision}"
: "${CONTROL_MODE:?Set relative or absolute after checking checkpoint metadata}"
: "${SCHEMA_VERIFIED:?Set to yes only after checking both camera names, 8D state and 7D action}"
: "${SUITE:?Choose libero_spatial, libero_object, libero_goal or libero_10}"

[[ "$SCHEMA_VERIFIED" == yes ]] || { echo 'BLOCKED: policy schema not verified' >&2; exit 2; }
[[ "$CONTROL_MODE" == relative || "$CONTROL_MODE" == absolute ]] || exit 2
case "$SUITE" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) exit 2 ;; esac
[[ "$(git -C "$LEROBOT_DIR" rev-parse HEAD)" == "$LEROBOT_COMMIT" ]] || {
  echo 'BLOCKED: LeRobot commit mismatch' >&2; exit 2;
}
[[ -f "$POLICY_WEIGHTS_FILE" ]] || { echo 'BLOCKED: cached official weights missing' >&2; exit 2; }
echo "$POLICY_WEIGHTS_SHA256  $POLICY_WEIGHTS_FILE" | sha256sum --check --status || {
  echo 'BLOCKED: official weights checksum mismatch' >&2; exit 2;
}
[[ -x "$LEROBOT_VENV/bin/lerobot-eval" ]] || { echo 'BLOCKED: lerobot-eval missing' >&2; exit 2; }

RUN_DIR="${RUN_DIR:-./worldpolicy_libero_${SUITE}_${POLICY_REVISION:0:12}}"
mkdir -p "$RUN_DIR"
printf 'lerobot_commit=%s\npolicy_path=%s\npolicy_revision=%s\npolicy_weights_sha256=%s\ndataset_revision=%s\ncontrol_mode=%s\nsuite=%s\n' \
  "$LEROBOT_COMMIT" "$POLICY_PATH" "$POLICY_REVISION" "$POLICY_WEIGHTS_SHA256" "$DATASET_REVISION" "$CONTROL_MODE" "$SUITE" \
  > "$RUN_DIR/protocol.txt"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$LEROBOT_VENV/bin/lerobot-eval" \
  --output_dir="$RUN_DIR/eval" \
  --policy.path="$POLICY_PATH" \
  --env.type=libero \
  --env.task="$SUITE" \
  --env.control_mode="$CONTROL_MODE" \
  --env.init_states=true \
  --env.hard_reset=true \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed="${EVAL_SEED:-31}"
