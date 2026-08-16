#!/usr/bin/env bash
set -euo pipefail

ROVER_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROVER_DEMO_ROOT"

ROVER_DEMO_PYTHON="${ROVER_DEMO_PYTHON:-.venv-gemma4/bin/python}"
ROVER_DEMO_CHECK=false
ROVER_DEMO_HELP=false

for ROVER_DEMO_ARGUMENT in "$@"; do
  case "$ROVER_DEMO_ARGUMENT" in
    --check) ROVER_DEMO_CHECK=true ;;
    --help|-h) ROVER_DEMO_HELP=true ;;
  esac
done

if [[ ! -x "$ROVER_DEMO_PYTHON" ]]; then
  echo "Local Gemma environment is unavailable: $ROVER_DEMO_PYTHON" >&2
  echo "Install the pinned environments with: make setup" >&2
  exit 2
fi

if [[ "$ROVER_DEMO_HELP" == true ]]; then
  exec env PYTHONPATH=src "$ROVER_DEMO_PYTHON" \
    -m semantic_3d_chat.robot.rover_demo "$@"
fi

# Check mode is strictly read-only. A real launch may materialize only the
# sanitized two-file runtime copy from its already authenticated local source.
if [[ "$ROVER_DEMO_CHECK" == true ]]; then
  PYTHONPATH=src "$ROVER_DEMO_PYTHON" scripts/prepare_demo_runtime.py --check >/dev/null || {
    echo "The sanitized local runtime release is unavailable." >&2
    echo "Prepare it with: make prepare-demo-runtime" >&2
    exit 2
  }
else
  PYTHONPATH=src "$ROVER_DEMO_PYTHON" scripts/prepare_demo_runtime.py >/dev/null
fi

# Authenticate the exact prepared room map, human-only scan figures, adapters,
# and pinned local Gemma snapshot. --fast still hashes every project artifact;
# it avoids rehashing only the already size/revision-bound 10.25 GB base model.
PYTHONPATH=src "$ROVER_DEMO_PYTHON" scripts/check_demo_artifacts.py --fast >/dev/null || {
  echo "The prepared room/scan/model assets are missing or no longer authentic." >&2
  echo "Inspect the exact problem with: make demo-artifacts-check-fast" >&2
  exit 2
}

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec env PYTHONPATH=src "$ROVER_DEMO_PYTHON" \
  -m semantic_3d_chat.robot.rover_demo "$@"
