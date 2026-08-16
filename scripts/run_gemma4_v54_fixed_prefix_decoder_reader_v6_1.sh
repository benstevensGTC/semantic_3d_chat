#!/usr/bin/env bash
set -euo pipefail

V61_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V61_ROOT"

V61_PYTHON="${V61_PYTHON:-.venv-gemma4/bin/python}"
if [[ ! -x "$V61_PYTHON" ]]; then
  echo "Gemma-4 Python environment is missing: $V61_PYTHON" >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 release-smoke|authenticate-release-smoke|smoke|authenticate-smoke|preflight|release-training|authenticate-release-training|train|authenticate" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

case "$1" in
  release-smoke)
    PYTHONPATH=src "$V61_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_1_release import write_v6_1_mps_smoke_release; print(write_v6_1_mps_smoke_release())'
    ;;
  authenticate-release-smoke)
    PYTHONPATH=src "$V61_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_1_release import authenticate_v6_1_mps_smoke_release; print(authenticate_v6_1_mps_smoke_release()[1])'
    ;;
  smoke)
    PYTHONPATH=src "$V61_PYTHON" -c \
      'import json; from semantic_3d_chat.training.smoke_fixed_prefix_decoder_reader_v6_1 import run_released_full_model_mps_smoke_v6_1; print(json.dumps(run_released_full_model_mps_smoke_v6_1(), sort_keys=True, allow_nan=False))'
    ;;
  authenticate-smoke)
    PYTHONPATH=src "$V61_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_1_release import authenticate_v6_1_passing_smoke; print(authenticate_v6_1_passing_smoke()[1])'
    ;;
  preflight)
    PYTHONPATH=src "$V61_PYTHON" -m \
      semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_1 preflight
    ;;
  release-training)
    PYTHONPATH=src "$V61_PYTHON" -m \
      semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_1 release
    ;;
  authenticate-release-training)
    PYTHONPATH=src "$V61_PYTHON" -m \
      semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_1 authenticate-release
    ;;
  train|authenticate)
    PYTHONPATH=src "$V61_PYTHON" -m \
      semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_1 "$1"
    ;;
  *)
    echo "Unknown V6.1 mode: $1" >&2
    exit 2
    ;;
esac
