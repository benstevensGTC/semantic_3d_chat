#!/usr/bin/env bash
set -euo pipefail

V69_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V69_PROJECT_ROOT"

V69_MODE="${1:-screen}"
if [[ "$V69_MODE" != "screen" && "$V69_MODE" != "full" ]]; then
  echo "Usage: $0 {screen|full} [--print-command]" >&2
  exit 2
fi
shift || true

V69_PYTHON_BIN="${V69_PYTHON_BIN:-.venv-gemma4/bin/python}"
V69_BASELINE_LOCK="${V69_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V69_PREREGISTRATION="${V69_PREREGISTRATION:-reports/gemma4/metrics/v69_pair_augmentation_preregistration.json}"
V69_TRAINING_BASELINE_LOCK="${V69_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V69_TRAIN_QA="${V69_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V69_TEACHER_CACHE="${V69_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V69_SUPPLEMENTAL_TEACHER_CACHE="${V69_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V69_PREFIX_CACHE="${V69_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V69_RUNTIME_CONFIG="${V69_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V69_BASE_CHECKPOINT="${V69_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V69_SOURCE_V60="${V69_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V69_OUTPUT_CHECKPOINT="${V69_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v69_pair_augmentation_control}"
V69_SCREEN_REPORT="${V69_SCREEN_REPORT:-reports/gemma4/metrics/v69_pair_augmentation_numeric_grid.json}"
V69_FULL_REPORT="${V69_FULL_REPORT:-reports/gemma4/metrics/v69_pair_augmentation_training.json}"
V69_DEVICE="${V69_DEVICE:-auto}"

if [[ "$V69_MODE" == "screen" ]]; then
  V69_WORK_DIRECTORY="${V69_WORK_DIRECTORY:-data_gemma4/training/v69_pair_augmentation_screen_work}"
  V69_TRAINING_REPORT="$V69_SCREEN_REPORT"
else
  V69_WORK_DIRECTORY="${V69_WORK_DIRECTORY:-data_gemma4/training/v69_pair_augmentation_full_work}"
  V69_TRAINING_REPORT="$V69_FULL_REPORT"
fi

V69_COMMAND=(
  "$V69_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v69
  --mode "$V69_MODE"
  --baseline-lock "$V69_BASELINE_LOCK"
  --preregistration "$V69_PREREGISTRATION"
  --training-baseline-lock "$V69_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V69_TRAIN_QA"
  --teacher-cache "$V69_TEACHER_CACHE"
  --supplemental-teacher-cache "$V69_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V69_PREFIX_CACHE"
  --base-runtime-config "$V69_RUNTIME_CONFIG"
  --base-checkpoint "$V69_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V69_SOURCE_V60"
  --work-directory "$V69_WORK_DIRECTORY"
  --output-checkpoint "$V69_OUTPUT_CHECKPOINT"
  --training-report "$V69_TRAINING_REPORT"
  --device "$V69_DEVICE"
)
if [[ "$V69_MODE" == "full" ]]; then
  V69_COMMAND+=(--screen-authorization "$V69_SCREEN_REPORT")
fi

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V69_COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 {screen|full} [--print-command]" >&2
  exit 2
fi

if [[ ! -x "$V69_PYTHON_BIN" ]]; then
  echo "V69 Python environment is missing: $V69_PYTHON_BIN" >&2
  exit 2
fi
for V69_REQUIRED_INPUT in \
  "$V69_BASELINE_LOCK" \
  "$V69_PREREGISTRATION" \
  "$V69_TRAINING_BASELINE_LOCK" \
  "$V69_TRAIN_QA" \
  "$V69_TEACHER_CACHE" \
  "$V69_SUPPLEMENTAL_TEACHER_CACHE" \
  "$V69_PREFIX_CACHE" \
  "$V69_RUNTIME_CONFIG" \
  "$V69_BASE_CHECKPOINT" \
  "$V69_SOURCE_V60"; do
  if [[ ! -e "$V69_REQUIRED_INPUT" ]]; then
    echo "Required V69 training input is missing: $V69_REQUIRED_INPUT" >&2
    exit 2
  fi
done
if [[ "$V69_MODE" == "full" && ! -f "$V69_SCREEN_REPORT" ]]; then
  echo "V69 full mode requires the passed numeric screen: $V69_SCREEN_REPORT" >&2
  exit 2
fi

echo "V69 mode: $V69_MODE"
echo "V69 selection: fixed-priority first all-gate-passing arm"
echo "V69 work: $V69_WORK_DIRECTORY"
echo "V69 gated checkpoint: $V69_OUTPUT_CHECKPOINT"
echo "V69 report: $V69_TRAINING_REPORT"
PYTHONPATH=src "${V69_COMMAND[@]}"
