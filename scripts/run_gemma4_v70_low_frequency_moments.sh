#!/usr/bin/env bash
set -euo pipefail

V70_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V70_PROJECT_ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--print-command" ) ]]; then
  echo "Usage: $0 [--print-command]" >&2
  exit 2
fi

V70_PYTHON_BIN="${V70_PYTHON_BIN:-.venv-gemma4/bin/python}"
V70_BASELINE_LOCK="${V70_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V70_PREREGISTRATION="${V70_PREREGISTRATION:-reports/gemma4/metrics/v70_low_frequency_moments_preregistration.json}"
V70_TRAINING_BASELINE_LOCK="${V70_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V70_TRAIN_QA="${V70_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V70_TEACHER_CACHE="${V70_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V70_SUPPLEMENTAL_TEACHER_CACHE="${V70_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V70_PREFIX_CACHE="${V70_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V70_RUNTIME_CONFIG="${V70_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V70_BASE_CHECKPOINT="${V70_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V70_SOURCE_V60="${V70_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V70_WORK_DIRECTORY="${V70_WORK_DIRECTORY:-data_gemma4/training/v70_low_frequency_moments_screen_work}"
V70_OUTPUT_CHECKPOINT="${V70_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v70_low_frequency_moments_control}"
V70_SCREEN_REPORT="${V70_SCREEN_REPORT:-reports/gemma4/metrics/v70_low_frequency_moments_numeric_screen.json}"
V70_DEVICE="${V70_DEVICE:-auto}"

V70_COMMAND=(
  "$V70_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v70
  --baseline-lock "$V70_BASELINE_LOCK"
  --preregistration "$V70_PREREGISTRATION"
  --training-baseline-lock "$V70_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V70_TRAIN_QA"
  --teacher-cache "$V70_TEACHER_CACHE"
  --supplemental-teacher-cache "$V70_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V70_PREFIX_CACHE"
  --base-runtime-config "$V70_RUNTIME_CONFIG"
  --base-checkpoint "$V70_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V70_SOURCE_V60"
  --work-directory "$V70_WORK_DIRECTORY"
  --output-checkpoint "$V70_OUTPUT_CHECKPOINT"
  --training-report "$V70_SCREEN_REPORT"
  --device "$V70_DEVICE"
)

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V70_COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$V70_PYTHON_BIN" ]]; then
  echo "V70 Python environment is missing: $V70_PYTHON_BIN" >&2
  exit 2
fi
for V70_REQUIRED_INPUT in \
  "$V70_BASELINE_LOCK" \
  "$V70_PREREGISTRATION" \
  "$V70_TRAINING_BASELINE_LOCK" \
  "$V70_TRAIN_QA" \
  "$V70_TEACHER_CACHE" \
  "$V70_SUPPLEMENTAL_TEACHER_CACHE" \
  "$V70_PREFIX_CACHE" \
  "$V70_RUNTIME_CONFIG" \
  "$V70_BASE_CHECKPOINT" \
  "$V70_SOURCE_V60"; do
  if [[ ! -e "$V70_REQUIRED_INPUT" ]]; then
    echo "Required V70 training input is missing: $V70_REQUIRED_INPUT" >&2
    exit 2
  fi
done
if [[ -e "$V70_OUTPUT_CHECKPOINT" ]]; then
  echo "V70 checkpoint target must remain absent: $V70_OUTPUT_CHECKPOINT" >&2
  exit 2
fi

echo "V70 mode: numeric screen only"
echo "V70 controlled change: first 8 -> first 32 fixed DCT moments"
echo "V70 wall-time budget: 1200 seconds"
echo "V70 report: $V70_SCREEN_REPORT"
PYTHONPATH=src "${V70_COMMAND[@]}"
