#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${GEMMA4_PYTHON:-.venv-gemma4/bin/python}"
preregistration="${NAVIGATION_V3_2_PREREGISTRATION:-reports/gemma4/metrics/navigation_policy_v3_2_runtime_preregistration.json}"
journal="reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_2.json"
audit="reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_2.json"
score="reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_2.json"
context="reports/gemma4/metrics/navigation_continuous_context_v3_2.json"
result="reports/gemma4/metrics/navigation_policy_v3_2_runtime_acceptance.json"
persistent_map="data_gemma4/robot_benchmark_learned_v3_2/scene_000001/semantic_map.npz"

if [[ $# -ne 0 ]]; then
    echo "Usage: ./scripts/run_learned_navigation_benchmark_v3_2.sh" >&2
    exit 2
fi

PYTHONPATH=src "$python_bin" \
    -m semantic_3d_chat.evaluation.navigation_policy_v3_2_preregistration \
    authenticate --preregistration "$preregistration"

for output in "$journal" "$audit" "$score" "$context" "$result" "$persistent_map"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite V3.2 evidence: $output" >&2
        exit 2
    fi
done

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
    "$python_bin" scripts/run_llm_navigation_inference_v3_2.py \
    --config configs/runtime/embodied_navigation_v2.yaml \
    --tasks configs/benchmarks/llm_navigation_v2_scene_000001.json \
    --navigation-policy-checkpoint data_gemma4/checkpoints/navigation_policy_v3 \
    --navigation-policy-version 3 \
    --journal "$journal" \
    --audit-report "$audit" \
    --persistent-map "$persistent_map"

PYTHONPATH=src "$python_bin" scripts/score_llm_navigation.py \
    --journal "$journal" \
    --scoring-spec configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json \
    --output "$score"

PYTHONPATH=src "$python_bin" scripts/audit_navigation_continuous_context.py \
    --journal "$journal" \
    --output "$context"

PYTHONPATH=src "$python_bin" \
    -m semantic_3d_chat.evaluation.navigation_policy_v3_2_preregistration \
    result --preregistration "$preregistration" --output "$result"
