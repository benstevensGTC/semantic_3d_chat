#!/usr/bin/env bash
set -euo pipefail

V72_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V72_PROJECT_ROOT"

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--print-command" ) ]]; then
  echo "Usage: $0 [--print-command]" >&2
  exit 2
fi

V72_PYTHON_BIN="${V72_PYTHON_BIN:-.venv-gemma4/bin/python}"
V72_BASELINE_LOCK="${V72_BASELINE_LOCK:-reports/gemma4/metrics/v62_v54_no_control_baseline_lock.json}"
V72_TRAINING_BASELINE_LOCK="${V72_TRAINING_BASELINE_LOCK:-reports/gemma4/metrics/v65_v54_training_baseline_lock.json}"
V72_TRAIN_QA="${V72_TRAIN_QA:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
V72_TEACHER_CACHE="${V72_TEACHER_CACHE:-data_gemma4/training/v62_changed_teachers}"
V72_SUPPLEMENTAL_TEACHER_CACHE="${V72_SUPPLEMENTAL_TEACHER_CACHE:-data_gemma4/training/v66_answer_class_teachers}"
V72_PREFIX_CACHE="${V72_PREFIX_CACHE:-data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes}"
V72_RUNTIME_CONFIG="${V72_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
V72_BASE_CHECKPOINT="${V72_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
V72_SOURCE_V60="${V72_SOURCE_V60:-data_gemma4/checkpoints/gemma4_v60_teacher_basis_control}"
V72_HELD_PAIRS="${V72_HELD_PAIRS:-pair_000011 pair_000016}"
V72_OUTPUT_CHECKPOINT="${V72_OUTPUT_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v72_adaptive_fusion_dev_forbidden}"
V72_REPORT="${V72_REPORT:-reports/gemma4/metrics/v72_adaptive_fusion_development.json}"
V72_DEVICE="${V72_DEVICE:-auto}"

read -r -a V72_PAIR_ARRAY <<< "$V72_HELD_PAIRS"
V72_COMMAND=(
  "$V72_PYTHON_BIN"
  -m semantic_3d_chat.training.train_question_control_v72
  --baseline-lock "$V72_BASELINE_LOCK"
  --training-baseline-lock "$V72_TRAINING_BASELINE_LOCK"
  --filtered-train-qa "$V72_TRAIN_QA"
  --teacher-cache "$V72_TEACHER_CACHE"
  --supplemental-teacher-cache "$V72_SUPPLEMENTAL_TEACHER_CACHE"
  --prefix-cache "$V72_PREFIX_CACHE"
  --base-runtime-config "$V72_RUNTIME_CONFIG"
  --base-checkpoint "$V72_BASE_CHECKPOINT"
  --source-v60-checkpoint "$V72_SOURCE_V60"
  --held-pairs "${V72_PAIR_ARRAY[@]}"
  --output-checkpoint "$V72_OUTPUT_CHECKPOINT"
  --training-report "$V72_REPORT"
  --device "$V72_DEVICE"
)

if [[ "${1:-}" == "--print-command" ]]; then
  printf 'PYTHONPATH=src '
  printf '%q ' "${V72_COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$V72_PYTHON_BIN" ]]; then
  echo "V72 Python environment is missing: $V72_PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$V72_OUTPUT_CHECKPOINT" ]]; then
  echo "V72 development checkpoint target must remain absent" >&2
  exit 2
fi
if [[ -e "$V72_REPORT" ]]; then
  echo "V72 create-once development report already exists: $V72_REPORT" >&2
  exit 2
fi

echo "V72 mode: two pair-disjoint train-only development folds"
echo "V72 held pairs: $V72_HELD_PAIRS"
echo "V72 scene policy: both 8/32 branches process all 256 latents; no retrieval"
echo "V72 publication: no Gemma generation and no checkpoint"
PYTHONPATH=src "${V72_COMMAND[@]}"
