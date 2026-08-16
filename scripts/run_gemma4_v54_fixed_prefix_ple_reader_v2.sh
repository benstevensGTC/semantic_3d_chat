#!/usr/bin/env bash
set -euo pipefail

PLE_V54_V2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PLE_V54_V2_ROOT"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 preregister|preflight|smoke|train|authenticate" >&2
  exit 2
fi

PLE_V54_V2_MODE="$1"
PLE_V54_V2_PYTHON="${PLE_V54_V2_PYTHON:-.venv-gemma4/bin/python}"
PLE_V54_V2_PREREG="reports/gemma4/metrics/gemma4_v54_fixed_prefix_ple_reader_v2_preregistration.json"

if [[ ! -x "$PLE_V54_V2_PYTHON" ]]; then
  echo "Gemma-4 Python environment is missing: $PLE_V54_V2_PYTHON" >&2
  exit 2
fi

case "$PLE_V54_V2_MODE" in
  preregister)
    PYTHONPATH=src "$PLE_V54_V2_PYTHON" \
      -m semantic_3d_chat.evaluation.fixed_prefix_ple_v54_v2_preregistration \
      --output "$PLE_V54_V2_PREREG"
    ;;
  preflight|smoke|train|authenticate)
    if [[ ! -f "$PLE_V54_V2_PREREG" ]]; then
      echo "Immutable PLE-V54 V2 preregistration is missing: $PLE_V54_V2_PREREG" >&2
      exit 2
    fi
    PYTHONPATH=src "$PLE_V54_V2_PYTHON" \
      -m semantic_3d_chat.training.train_fixed_prefix_ple_v54_v2 \
      "$PLE_V54_V2_MODE"
    ;;
  *)
    echo "Unknown PLE-V54 V2 mode: $PLE_V54_V2_MODE" >&2
    exit 2
    ;;
esac
