#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:src"

exec .venv-gemma4/bin/python \
  -m semantic_3d_chat.training.train_fixed_prefix_attention_reader_v6_4 "$@"
