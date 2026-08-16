#!/usr/bin/env bash
set -euo pipefail

V6_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V6_ROOT"

V6_PYTHON="${V6_PYTHON:-.venv-gemma4/bin/python}"
if [[ ! -x "$V6_PYTHON" ]]; then
  echo "Gemma-4 Python environment is missing: $V6_PYTHON" >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 preflight|seal|authenticate-seal|release-smoke|smoke|release-training|train|authenticate" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

case "$1" in
  preflight)
    PYTHONPATH=src "$V6_PYTHON" -m \
      semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_preregistration preflight
    ;;
  seal)
    PYTHONPATH=src "$V6_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_release import write_sealed_preregistration; print(write_sealed_preregistration())'
    ;;
  authenticate-seal)
    PYTHONPATH=src "$V6_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_release import authenticate_sealed_preregistration; print(authenticate_sealed_preregistration()[1])'
    ;;
  release-smoke)
    PYTHONPATH=src "$V6_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_release import write_mps_smoke_release; print(write_mps_smoke_release())'
    ;;
  smoke)
    PYTHONPATH=src "$V6_PYTHON" -c \
      'import json; from semantic_3d_chat.training.smoke_fixed_prefix_decoder_reader_v6 import run_released_full_model_mps_smoke; print(json.dumps(run_released_full_model_mps_smoke(), sort_keys=True, allow_nan=False))'
    ;;
  release-training)
    PYTHONPATH=src "$V6_PYTHON" -c \
      'from semantic_3d_chat.evaluation.fixed_prefix_decoder_reader_v6_release import write_training_release; print(write_training_release())'
    ;;
  train|authenticate)
    PYTHONPATH=src "$V6_PYTHON" -m \
      semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6 "$1"
    ;;
  *)
    echo "Unknown V6 mode: $1" >&2
    exit 2
    ;;
esac
