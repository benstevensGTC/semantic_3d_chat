#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHONPATH=src .venv-gemma4/bin/python -m \
  semantic_3d_chat.training.train_question_control_v73 "$@"
