#!/usr/bin/env bash
set -euo pipefail

V66_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V66_PROJECT_ROOT"

V66_PYTHON_BIN="${V66_PYTHON_BIN:-.venv-gemma4/bin/python}"
V66_BASELINE_LOCK="${V66_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V66_PREREGISTRATION="${V66_PREREGISTRATION:-reports/gemma4/metrics/v66b_paired_opposite_preregistration.json}"
V66_TRAINING_BASELINE_LOCK="${V66_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V66_TRAIN_QA="${V66_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V66_TEACHER_CACHE="${V66_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V66_SUPPLEMENTAL_TEACHER_CACHE="${V66_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V66_PREFIX_CACHE="${V66_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V66_RUNTIME_CONFIG="${V66_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V66_BASE_CHECKPOINT="${V66_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V66_SOURCE_V60="${V66_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V66_WORK_DIRECTORY="${V66_WORK_DIRECTORY:-data_gemma4/training/v66b_allrow_pair_cv_work}"
V66_OUTPUT_CHECKPOINT="${V66_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control}"
V66_TRAINING_REPORT="${V66_TRAINING_REPORT:-reports/gemma4/metrics/v66b_allrow_always_on_distillation.json}"

V66_COMMAND=(
  "$V66_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v66
  --baseline-lock "$V66_BASELINE_LOCK"
  --preregistration "$V66_PREREGISTRATION"
  --training-baseline-lock "$V66_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V66_TRAIN_QA"
  --teacher-cache "$V66_TEACHER_CACHE"
  --supplemental-teacher-cache "$V66_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V66_PREFIX_CACHE"
  --base-runtime-config "$V66_RUNTIME_CONFIG"
  --base-checkpoint "$V66_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V66_SOURCE_V60"
  --work-directory "$V66_WORK_DIRECTORY"
  --output-checkpoint "$V66_OUTPUT_CHECKPOINT"
  --training-report "$V66_TRAINING_REPORT"
  --device auto
  --epochs 160
  --prototype-classification-epochs 40
)

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V66_COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$V66_PYTHON_BIN" ]]; then
  echo "V66b Python environment is missing: $V66_PYTHON_BIN" >&2
  exit 2
fi
for V66_REQUIRED_INPUT in \
  "$V66_BASELINE_LOCK" \
  "$V66_PREREGISTRATION" \
  "$V66_TRAINING_BASELINE_LOCK" \
  "$V66_TRAIN_QA" \
  "$V66_TEACHER_CACHE" \
  "$V66_SUPPLEMENTAL_TEACHER_CACHE" \
  "$V66_PREFIX_CACHE" \
  "$V66_RUNTIME_CONFIG" \
  "$V66_BASE_CHECKPOINT" \
  "$V66_SOURCE_V60"; do
  if [[ ! -e "$V66_REQUIRED_INPUT" ]]; then
    echo "Required V66b training input is missing: $V66_REQUIRED_INPUT" >&2
    exit 2
  fi
done

echo "V66b resumable folds: $V66_WORK_DIRECTORY"
echo "V66b gated checkpoint: $V66_OUTPUT_CHECKPOINT"
echo "V66b training report: $V66_TRAINING_REPORT"
PYTHONPATH=src "${V66_COMMAND[@]}" "$@"
