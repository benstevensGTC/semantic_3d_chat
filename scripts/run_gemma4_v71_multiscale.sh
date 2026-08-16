#!/usr/bin/env bash
set -euo pipefail

V71_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V71_PROJECT_ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--print-command" ) ]]; then
  echo "Usage: $0 [--print-command]" >&2
  exit 2
fi

V71_PYTHON_BIN="${V71_PYTHON_BIN:-.venv-gemma4/bin/python}"
V71_BASELINE_LOCK="${V71_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V71_PREREGISTRATION="${V71_PREREGISTRATION:-reports/gemma4/metrics/v71_multiscale_preregistration.json}"
V71_TRAINING_BASELINE_LOCK="${V71_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V71_TRAIN_QA="${V71_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V71_TEACHER_CACHE="${V71_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V71_SUPPLEMENTAL_TEACHER_CACHE="${V71_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V71_PREFIX_CACHE="${V71_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V71_RUNTIME_CONFIG="${V71_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V71_BASE_CHECKPOINT="${V71_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V71_SOURCE_V60="${V71_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V71_WORK_DIRECTORY="${V71_WORK_DIRECTORY:-data_gemma4/training/v71_multiscale_screen_work}"
V71_OUTPUT_CHECKPOINT="${V71_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v71_multiscale_control}"
V71_SCREEN_REPORT="${V71_SCREEN_REPORT:-reports/gemma4/metrics/v71_multiscale_numeric_screen.json}"
V71_DEVICE="${V71_DEVICE:-auto}"

V71_COMMAND=(
  "$V71_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v71
  --baseline-lock "$V71_BASELINE_LOCK"
  --preregistration "$V71_PREREGISTRATION"
  --training-baseline-lock "$V71_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V71_TRAIN_QA"
  --teacher-cache "$V71_TEACHER_CACHE"
  --supplemental-teacher-cache "$V71_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V71_PREFIX_CACHE"
  --base-runtime-config "$V71_RUNTIME_CONFIG"
  --base-checkpoint "$V71_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V71_SOURCE_V60"
  --work-directory "$V71_WORK_DIRECTORY"
  --output-checkpoint "$V71_OUTPUT_CHECKPOINT"
  --training-report "$V71_SCREEN_REPORT"
  --device "$V71_DEVICE"
)

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V71_COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$V71_PYTHON_BIN" ]]; then
  echo "V71 Python environment is missing: $V71_PYTHON_BIN" >&2
  exit 2
fi
for V71_REQUIRED_INPUT in \
  "$V71_BASELINE_LOCK" \
  "$V71_PREREGISTRATION" \
  "$V71_TRAINING_BASELINE_LOCK" \
  "$V71_TRAIN_QA" \
  "$V71_TEACHER_CACHE" \
  "$V71_SUPPLEMENTAL_TEACHER_CACHE" \
  "$V71_PREFIX_CACHE" \
  "$V71_RUNTIME_CONFIG" \
  "$V71_BASE_CHECKPOINT" \
  "$V71_SOURCE_V60"; do
  if [[ ! -e "$V71_REQUIRED_INPUT" ]]; then
    echo "Required V71 input is missing: $V71_REQUIRED_INPUT" >&2
    exit 2
  fi
done
if [[ -e "$V71_OUTPUT_CHECKPOINT" ]]; then
  echo "V71 checkpoint target must remain absent: $V71_OUTPUT_CHECKPOINT" >&2
  exit 2
fi

echo "V71 mode: numeric screen only"
echo "V71 architecture: independent all-latent DCT[0:8] + DCT[0:32] branches"
echo "V71 training: exact V69 balanced_extrapolation_010 arm"
echo "V71 wall-time budget: 1200 seconds"
echo "V71 report: $V71_SCREEN_REPORT"
PYTHONPATH=src "${V71_COMMAND[@]}"
