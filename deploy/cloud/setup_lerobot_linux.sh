#!/usr/bin/env bash
# Separate official LeRobot environment; do not reuse the WorldPolicy virtualenv.
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || exit 2
: "${LEROBOT_COMMIT:?Pin the official upstream commit}"
: "${LEROBOT_DIR:?Set a unique clone directory}"
: "${LEROBOT_VENV:?Set a separate venv path}"
[[ ! -e "$LEROBOT_DIR" && ! -e "$LEROBOT_VENV" ]] || {
  echo 'Choose fresh LeRobot clone and venv paths' >&2; exit 2;
}
git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
git -C "$LEROBOT_DIR" checkout --detach "$LEROBOT_COMMIT"
[[ "$(git -C "$LEROBOT_DIR" rev-parse HEAD)" == "$LEROBOT_COMMIT" ]] || exit 2
python3.12 -m venv "$LEROBOT_VENV"
"$LEROBOT_VENV/bin/python" -m pip install --upgrade pip
"$LEROBOT_VENV/bin/python" -m pip install -e "$LEROBOT_DIR[libero]"
"$LEROBOT_VENV/bin/python" -m pip check
"$LEROBOT_VENV/bin/python" -m pip freeze > "$LEROBOT_DIR/installed_requirements.lock"
"$LEROBOT_VENV/bin/python" -c 'import torch, mujoco, lerobot; print(torch.__version__, torch.cuda.is_available(), mujoco.__version__)'
