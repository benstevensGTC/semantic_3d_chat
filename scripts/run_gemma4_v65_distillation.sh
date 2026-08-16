#!/usr/bin/env bash
set -euo pipefail

V65_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V65_PROJECT_ROOT"

V65_PYTHON_BIN="${V65_PYTHON_BIN:-.venv-gemma4/bin/python}"
V65_BASELINE_LOCK="${V65_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V65_TRAINING_BASELINE_LOCK="${V65_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V65_TRAIN_QA="${V65_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V65_TEACHER_CACHE="${V65_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V65_PREFIX_CACHE="${V65_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V65_RUNTIME_CONFIG="${V65_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V65_BASE_CHECKPOINT="${V65_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V65_SOURCE_V60="${V65_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V65_WORK_DIRECTORY="${V65_WORK_DIRECTORY:-data_gemma4/training/v65_magnitude_gated_pair_cv_work}"
V65_OUTPUT_CHECKPOINT="${V65_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v65_magnitude_gated_control}"
V65_TRAINING_REPORT="${V65_TRAINING_REPORT:-reports/gemma4/metrics/v65_magnitude_gated_distillation.json}"

v65_usage() {
  cat <<'EOF'
Usage: ./scripts/run_gemma4_v65_distillation.sh [--print-command] [TRAINER OPTIONS]

Runs the training-only, resumable V65 canonical-answer prototype experiment.
An interrupted run resumes completed pair folds from:

  data_gemma4/training/v65_magnitude_gated_pair_cv_work

Create-once successful outputs are:

  data_gemma4/checkpoints/gemma4_v65_magnitude_gated_control
  reports/gemma4/metrics/v65_magnitude_gated_distillation.json

No checkpoint is written unless the held-fold and final greedy-Gemma gates pass.
Use --print-command to inspect the exact command without loading Gemma or training.
All paths can be overridden with the V65_* environment variables defined near the
top of this script. Additional trainer options are appended and override defaults.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  v65_usage
  exit 0
fi

V65_PRINT_COMMAND=0
if [[ "${1:-}" == "--print-command" ]]; then
  V65_PRINT_COMMAND=1
  shift
fi

V65_COMMAND=(
  "$V65_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v65
  --baseline-lock "$V65_BASELINE_LOCK"
  --training-baseline-lock "$V65_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V65_TRAIN_QA"
  --teacher-cache "$V65_TEACHER_CACHE"
  --prefix-cache "$V65_PREFIX_CACHE"
  --base-runtime-config "$V65_RUNTIME_CONFIG"
  --base-checkpoint "$V65_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V65_SOURCE_V60"
  --work-directory "$V65_WORK_DIRECTORY"
  --output-checkpoint "$V65_OUTPUT_CHECKPOINT"
  --training-report "$V65_TRAINING_REPORT"
  --device auto
  --epochs 160
)

if [[ "$V65_PRINT_COMMAND" -eq 1 ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V65_COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi

if [[ ! -x "$V65_PYTHON_BIN" ]]; then
  echo "V65 Python environment is missing: $V65_PYTHON_BIN" >&2
  exit 2
fi
for V65_REQUIRED_INPUT in \
  "$V65_BASELINE_LOCK" \
  "$V65_TRAINING_BASELINE_LOCK" \
  "$V65_TRAIN_QA" \
  "$V65_TEACHER_CACHE" \
  "$V65_PREFIX_CACHE" \
  "$V65_RUNTIME_CONFIG" \
  "$V65_BASE_CHECKPOINT" \
  "$V65_SOURCE_V60"; do
  if [[ ! -e "$V65_REQUIRED_INPUT" ]]; then
    echo "Required V65 training input is missing: $V65_REQUIRED_INPUT" >&2
    exit 2
  fi
done

echo "V65 resumable folds: $V65_WORK_DIRECTORY"
echo "V65 gated checkpoint: $V65_OUTPUT_CHECKPOINT"
echo "V65 training report: $V65_TRAINING_REPORT"
PYTHONPATH=src "${V65_COMMAND[@]}" "$@"
