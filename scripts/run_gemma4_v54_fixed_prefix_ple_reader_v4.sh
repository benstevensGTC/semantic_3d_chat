#!/usr/bin/env bash
set -euo pipefail

PLE_V54_V4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PLE_V54_V4_ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 preregister|preflight|smoke|train|authenticate" >&2
  exit 2
fi

PLE_V54_V4_MODE="$1"
PLE_V54_V4_PYTHON="${PLE_V54_V4_PYTHON:-.venv-gemma4/bin/python}"
PLE_V54_V4_PREREG="reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v4_preregistration.json"

if [[ ! -x "$PLE_V54_V4_PYTHON" ]]; then
  echo "Gemma-4 Python environment is missing: $PLE_V54_V4_PYTHON" >&2
  exit 2
fi

case "$PLE_V54_V4_MODE" in
  preregister)
    PYTHONPATH=src "$PLE_V54_V4_PYTHON" \
      -m semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v4_preregistration \
      --output "$PLE_V54_V4_PREREG"
    ;;
  preflight|smoke|train|authenticate)
    if [[ ! -f "$PLE_V54_V4_PREREG" ]]; then
      echo "Immutable PLE-V54 V4 preregistration is missing: $PLE_V54_V4_PREREG" >&2
      exit 2
    fi
    PYTHONPATH=src "$PLE_V54_V4_PYTHON" \
      -m semantic_3d_chat.training.train_fixed_prefix_ple_v54_v4 \
      "$PLE_V54_V4_MODE"
    ;;
  *)
    echo "Unknown PLE-V54 V4 mode: $PLE_V54_V4_MODE" >&2
    exit 2
    ;;
esac
