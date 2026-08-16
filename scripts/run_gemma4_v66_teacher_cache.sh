#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec env PYTHONPATH=src .venv-gemma4/bin/python -m \
  semantic_3d_chat.training.soft_prompt_teacher_v66 \
  --training-baseline-lock reports/gemma4/metrics/v65_v54_training_baseline_lock.json \
  --filtered-train-qa data_gemma4/training/v62_pair_disjoint/train.jsonl \
  --v62-teacher-cache data_gemma4/training/v62_changed_teachers \
  --prefix-cache data_gemma4/scene_tokens/v62_pair_disjoint_train_prefixes \
  --base-runtime-config configs/runtime/gemma4_v54.yaml \
  --base-checkpoint data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000 \
  --source-control-checkpoint data_gemma4/checkpoints/gemma4_v60_teacher_basis_control \
  --work-directory data_gemma4/training/v66_answer_class_teacher_work_v2 \
  --output-artifact data_gemma4/training/v66_answer_class_teachers \
  --device auto \
  "$@"
