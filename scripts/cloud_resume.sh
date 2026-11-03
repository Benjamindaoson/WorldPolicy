#!/usr/bin/env bash
set -euo pipefail
export RESUME=1
exec "$(dirname "$0")/cloud_train.sh"
