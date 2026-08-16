#!/usr/bin/env bash
set -euo pipefail

V67_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V67_PROJECT_ROOT"

V67_MODE="${1:-screen}"
if [[ "$V67_MODE" != "screen" && "$V67_MODE" != "full" ]]; then
  echo "Usage: $0 {screen|full} [--print-command]" >&2
  exit 2
fi
shift || true

V67_PYTHON_BIN="${V67_PYTHON_BIN:-.venv-gemma4/bin/python}"
V67_BASELINE_LOCK="${V67_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V67_PREREGISTRATION="${V67_PREREGISTRATION:-reports/gemma4/metrics/v67_pair_objective_preregistration.json}"
V67_TRAINING_BASELINE_LOCK="${V67_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V67_TRAIN_QA="${V67_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V67_TEACHER_CACHE="${V67_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V67_SUPPLEMENTAL_TEACHER_CACHE="${V67_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V67_PREFIX_CACHE="${V67_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V67_RUNTIME_CONFIG="${V67_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V67_BASE_CHECKPOINT="${V67_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V67_SOURCE_V60="${V67_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V67_OUTPUT_CHECKPOINT="${V67_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v67_pair_objective_control}"
V67_SCREEN_REPORT="${V67_SCREEN_REPORT:-reports/gemma4/metrics/v67_pair_objective_numeric_screen.json}"
V67_FULL_REPORT="${V67_FULL_REPORT:-reports/gemma4/metrics/v67_pair_objective_training.json}"

if [[ "$V67_MODE" == "screen" ]]; then
  V67_WORK_DIRECTORY="${V67_WORK_DIRECTORY:-data_gemma4/training/v67_pair_objective_screen_work}"
  V67_TRAINING_REPORT="$V67_SCREEN_REPORT"
else
  V67_WORK_DIRECTORY="${V67_WORK_DIRECTORY:-data_gemma4/training/v67_pair_objective_full_work}"
  V67_TRAINING_REPORT="$V67_FULL_REPORT"
fi

V67_COMMAND=(
  "$V67_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v67
  --mode "$V67_MODE"
  --baseline-lock "$V67_BASELINE_LOCK"
  --preregistration "$V67_PREREGISTRATION"
  --training-baseline-lock "$V67_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V67_TRAIN_QA"
  --teacher-cache "$V67_TEACHER_CACHE"
  --supplemental-teacher-cache "$V67_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V67_PREFIX_CACHE"
  --base-runtime-config "$V67_RUNTIME_CONFIG"
  --base-checkpoint "$V67_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V67_SOURCE_V60"
  --work-directory "$V67_WORK_DIRECTORY"
  --output-checkpoint "$V67_OUTPUT_CHECKPOINT"
  --training-report "$V67_TRAINING_REPORT"
  --device auto
)
if [[ "$V67_MODE" == "full" ]]; then
  V67_COMMAND+=(--screen-authorization "$V67_SCREEN_REPORT")
fi

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V67_COMMAND[@]}"
  printf '\n'
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: $0 {screen|full} [--print-command]" >&2
  exit 2
fi

if [[ ! -x "$V67_PYTHON_BIN" ]]; then
  echo "V67 Python environment is missing: $V67_PYTHON_BIN" >&2
  exit 2
fi
for V67_REQUIRED_INPUT in \
  "$V67_BASELINE_LOCK" \
  "$V67_PREREGISTRATION" \
  "$V67_TRAINING_BASELINE_LOCK" \
  "$V67_TRAIN_QA" \
  "$V67_TEACHER_CACHE" \
  "$V67_SUPPLEMENTAL_TEACHER_CACHE" \
  "$V67_PREFIX_CACHE" \
  "$V67_RUNTIME_CONFIG" \
  "$V67_BASE_CHECKPOINT" \
  "$V67_SOURCE_V60"; do
  if [[ ! -e "$V67_REQUIRED_INPUT" ]]; then
    echo "Required V67 training input is missing: $V67_REQUIRED_INPUT" >&2
    exit 2
  fi
done
if [[ "$V67_MODE" == "full" && ! -f "$V67_SCREEN_REPORT" ]]; then
  echo "V67 full mode requires the passed numeric screen: $V67_SCREEN_REPORT" >&2
  exit 2
fi

echo "V67 mode: $V67_MODE"
echo "V67 work: $V67_WORK_DIRECTORY"
echo "V67 gated checkpoint: $V67_OUTPUT_CHECKPOINT"
echo "V67 report: $V67_TRAINING_REPORT"
PYTHONPATH=src "${V67_COMMAND[@]}"
