#!/usr/bin/env bash
set -euo pipefail

PLE_V54_V5_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PLE_V54_V5_ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 preregister|preflight|smoke|train|authenticate" >&2
  exit 2
fi

PLE_V54_V5_MODE="$1"
PLE_V54_V5_PYTHON="${PLE_V54_V5_PYTHON:-.venv-gemma4/bin/python}"
PLE_V54_V5_PREREG="reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v5_preregistration.json"

if [[ ! -x "$PLE_V54_V5_PYTHON" ]]; then
  echo "Gemma-4 Python environment is missing: $PLE_V54_V5_PYTHON" >&2
  exit 2
fi

case "$PLE_V54_V5_MODE" in
  preregister)
    PYTHONPATH=src "$PLE_V54_V5_PYTHON" \
      -m semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v5_preregistration \
      --output "$PLE_V54_V5_PREREG"
    ;;
  preflight|smoke|train|authenticate)
    if [[ ! -f "$PLE_V54_V5_PREREG" ]]; then
      echo "Immutable PLE-V54 V5 preregistration is missing: $PLE_V54_V5_PREREG" >&2
      exit 2
    fi
    PYTHONPATH=src "$PLE_V54_V5_PYTHON" \
      -m semantic_3d_chat.training.train_fixed_prefix_ple_v54_v5 \
      "$PLE_V54_V5_MODE"
    ;;
  *)
    echo "Unknown PLE-V54 V5 mode: $PLE_V54_V5_MODE" >&2
    exit 2
    ;;
esac
