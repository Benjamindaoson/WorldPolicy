#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Linux required' >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
: "${WORLD_POLICY_VENV:?Set a project-exclusive absolute virtualenv path}"
python3.12 -m venv "$WORLD_POLICY_VENV"
"$WORLD_POLICY_VENV/bin/python" -m pip install --upgrade pip
"$WORLD_POLICY_VENV/bin/python" -m pip install --require-hashes -r deploy/cloud/requirements-linux.lock
"$WORLD_POLICY_VENV/bin/python" -m pip install --no-deps -e .
"$WORLD_POLICY_VENV/bin/python" -m pip check
"$WORLD_POLICY_VENV/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
