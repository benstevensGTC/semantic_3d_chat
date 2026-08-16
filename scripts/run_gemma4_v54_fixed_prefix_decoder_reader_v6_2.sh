#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv-gemma4/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Gemma 4 environment: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_DIR}"

case "${1:-}" in
  preflight|release|authenticate-release|attempt-status|train|authenticate)
    exec "${PYTHON_BIN}" -m \
      semantic_3d_chat.training.train_fixed_prefix_decoder_reader_v6_2 "$1"
    ;;
  *)
    echo "usage: $0 {preflight|release|authenticate-release|attempt-status|train|authenticate}" >&2
    exit 2
    ;;
esac
