#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || exit 2
: "${WORLD_POLICY_VENV:?Set the isolated environment path}"
: "${RUN_DIR:?Set completed visual run directory}"
: "${EVAL_OUTPUT:?Set unique persistent evaluation JSON path}"
"$WORLD_POLICY_VENV/bin/python" -m worldpolicy.visual_evaluate \
  --run-dir "$RUN_DIR" --output "$EVAL_OUTPUT" \
  --seed-start "${EVAL_SEED_START:-104000}" --episodes "${EVAL_EPISODES:-10}"
