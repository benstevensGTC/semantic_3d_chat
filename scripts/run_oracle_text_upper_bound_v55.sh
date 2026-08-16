#!/usr/bin/env bash
set -euo pipefail

# Evaluation control only. This deliberately supplies textual oracle facts to
# local Gemma and is categorically prohibited as a primary runtime input.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

ORACLE_TEXT_PYTHON="${ORACLE_TEXT_PYTHON:-.venv-gemma4/bin/python}"
if [[ ! -x "$ORACLE_TEXT_PYTHON" ]]; then
  echo "Missing Gemma environment: $ORACLE_TEXT_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

CONTROL_ROOT="reports/gemma4/evaluation_only/oracle_text_upper_bound"
SCENE_TEXT="$CONTROL_ROOT/v55_scene_descriptions.json"
PREDICTIONS="$CONTROL_ROOT/v55_predictions.jsonl"

"$ORACLE_TEXT_PYTHON" -m semantic_3d_chat.evaluation.oracle_text_prepare \
  --config configs/experiments/gemma4_oracle_text_v55.yaml \
  --questions reports/gemma4/questions/v55_development_validation.json \
  --oracle-root data/oracle \
  --output "$SCENE_TEXT"

"$ORACLE_TEXT_PYTHON" -m semantic_3d_chat.evaluation.oracle_text_predict \
  --config configs/experiments/gemma4_oracle_text_v55.yaml \
  --questions reports/gemma4/questions/v55_development_validation.json \
  --scene-text "$SCENE_TEXT" \
  --predictions "$PREDICTIONS"

"$ORACLE_TEXT_PYTHON" -m semantic_3d_chat.evaluation.oracle_text_score \
  --references data_diverse28/qa/validation.jsonl \
  --predictions "$PREDICTIONS" \
  --output reports/gemma4/metrics/oracle_text_upper_bound_v55_development.json
