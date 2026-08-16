#!/usr/bin/env bash
set -euo pipefail

V68_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V68_PROJECT_ROOT"

V68_MODE="${1:-screen}"
if [[ "$V68_MODE" != "screen" && "$V68_MODE" != "full" ]]; then
  echo "Usage: $0 {screen|full} [--print-command]" >&2
  exit 2
fi
shift || true

V68_PYTHON_BIN="${V68_PYTHON_BIN:-.venv-gemma4/bin/python}"
V68_BASELINE_LOCK="${V68_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V68_PREREGISTRATION="${V68_PREREGISTRATION:-reports/gemma4/metrics/v68_regularized_pair_preregistration.json}"
V68_TRAINING_BASELINE_LOCK="${V68_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V68_TRAIN_QA="${V68_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V68_TEACHER_CACHE="${V68_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V68_SUPPLEMENTAL_TEACHER_CACHE="${V68_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V68_PREFIX_CACHE="${V68_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V68_RUNTIME_CONFIG="${V68_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V68_BASE_CHECKPOINT="${V68_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V68_SOURCE_V60="${V68_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V68_OUTPUT_CHECKPOINT="${V68_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v68_regularized_pair_control}"
V68_SCREEN_REPORT="${V68_SCREEN_REPORT:-reports/gemma4/metrics/v68_regularized_pair_numeric_grid.json}"
V68_FULL_REPORT="${V68_FULL_REPORT:-reports/gemma4/metrics/v68_regularized_pair_training.json}"

if [[ "$V68_MODE" == "screen" ]]; then
  V68_WORK_DIRECTORY="${V68_WORK_DIRECTORY:-data_gemma4/training/v68_regularized_pair_screen_work}"
  V68_TRAINING_REPORT="$V68_SCREEN_REPORT"
else
  V68_WORK_DIRECTORY="${V68_WORK_DIRECTORY:-data_gemma4/training/v68_regularized_pair_full_work}"
  V68_TRAINING_REPORT="$V68_FULL_REPORT"
fi

V68_COMMAND=(
  "$V68_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v68
  --mode "$V68_MODE"
  --baseline-lock "$V68_BASELINE_LOCK"
  --preregistration "$V68_PREREGISTRATION"
  --training-baseline-lock "$V68_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V68_TRAIN_QA"
  --teacher-cache "$V68_TEACHER_CACHE"
  --supplemental-teacher-cache "$V68_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V68_PREFIX_CACHE"
  --base-runtime-config "$V68_RUNTIME_CONFIG"
  --base-checkpoint "$V68_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V68_SOURCE_V60"
  --work-directory "$V68_WORK_DIRECTORY"
  --output-checkpoint "$V68_OUTPUT_CHECKPOINT"
  --training-report "$V68_TRAINING_REPORT"
  --device auto
)
if [[ "$V68_MODE" == "full" ]]; then
  V68_COMMAND+=(--screen-authorization "$V68_SCREEN_REPORT")
fi

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V68_COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 {screen|full} [--print-command]" >&2
  exit 2
fi

if [[ ! -x "$V68_PYTHON_BIN" ]]; then
  echo "V68 Python environment is missing: $V68_PYTHON_BIN" >&2
  exit 2
fi
for V68_REQUIRED_INPUT in \
  "$V68_BASELINE_LOCK" \
  "$V68_PREREGISTRATION" \
  "$V68_TRAINING_BASELINE_LOCK" \
  "$V68_TRAIN_QA" \
  "$V68_TEACHER_CACHE" \
  "$V68_SUPPLEMENTAL_TEACHER_CACHE" \
  "$V68_PREFIX_CACHE" \
  "$V68_RUNTIME_CONFIG" \
  "$V68_BASE_CHECKPOINT" \
  "$V68_SOURCE_V60"; do
  if [[ ! -e "$V68_REQUIRED_INPUT" ]]; then
    echo "Required V68 training input is missing: $V68_REQUIRED_INPUT" >&2
    exit 2
  fi
done
if [[ "$V68_MODE" == "full" && ! -f "$V68_SCREEN_REPORT" ]]; then
  echo "V68 full mode requires the passed numeric screen: $V68_SCREEN_REPORT" >&2
  exit 2
fi

echo "V68 mode: $V68_MODE"
echo "V68 selection: fixed-priority first all-gate-passing arm"
echo "V68 work: $V68_WORK_DIRECTORY"
echo "V68 gated checkpoint: $V68_OUTPUT_CHECKPOINT"
echo "V68 report: $V68_TRAINING_REPORT"
PYTHONPATH=src "${V68_COMMAND[@]}"
