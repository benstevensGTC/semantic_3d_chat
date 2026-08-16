#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${GEMMA4_PYTHON:-.venv-gemma4/bin/python}"
preregistration="${NAVIGATION_V3_3_PREREGISTRATION:-reports/gemma4/metrics/navigation_policy_v3_3_runtime_preregistration.json}"
journal="reports/gemma4/predictions/llm_navigation_scene_000001_learned_v3_3.json"
audit="reports/gemma4/metrics/llm_navigation_inference_access_learned_v3_3.json"
score="reports/gemma4/metrics/llm_navigation_scene_000001_learned_v3_3.json"
context="reports/gemma4/metrics/navigation_continuous_context_v3_3.json"
result="reports/gemma4/metrics/navigation_policy_v3_3_runtime_acceptance.json"
run_directory="data_gemma4/robot_benchmark_learned_v3_3"
persistent_map="$run_directory/scene_000001/semantic_map.npz"

if [[ $# -ne 0 ]]; then
    echo "Usage: ./scripts/run_learned_navigation_benchmark_v3_3.sh" >&2
    exit 2
fi

if [[ ! -x "$python_bin" ]]; then
    echo "V3.3 Python environment is unavailable: $python_bin" >&2
    exit 2
fi

# This verifies the frozen implementation and input hashes, including the
# exact CPU routing integration test that was run before any full Gemma load.
PYTHONPATH=src "$python_bin" \
    -m semantic_3d_chat.evaluation.navigation_policy_v3_3_preregistration \
    authenticate --preregistration "$preregistration"

for output in "$journal" "$audit" "$score" "$context" "$result"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite sealed V3.3 evidence: $output" >&2
        exit 2
    fi
done
if [[ -e "$run_directory" ]]; then
    echo "Refusing to reuse the V3.3 runtime output tree: $run_directory" >&2
    exit 2
fi

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
    "$python_bin" scripts/run_llm_navigation_inference_v3_3.py \
    --config configs/runtime/embodied_navigation_v2.yaml \
    --tasks configs/benchmarks/llm_navigation_v2_scene_000001.json \
    --navigation-policy-checkpoint data_gemma4/checkpoints/navigation_policy_v3 \
    --navigation-policy-version 3 \
    --journal "$journal" \
    --audit-report "$audit" \
    --persistent-map "$persistent_map"

# Oracle access starts only after local inference exits. It is confined to the
# scorer and never enters the policy process or continuous action context.
PYTHONPATH=src "$python_bin" scripts/score_llm_navigation.py \
    --journal "$journal" \
    --scoring-spec configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json \
    --output "$score"

context_status=0
PYTHONPATH=src "$python_bin" scripts/audit_navigation_continuous_context.py \
    --journal "$journal" \
    --output "$context" || context_status=$?

result_status=0
PYTHONPATH=src "$python_bin" \
    -m semantic_3d_chat.evaluation.navigation_policy_v3_3_preregistration \
    result --preregistration "$preregistration" --output "$result" \
    || result_status=$?

# Authenticate the stored verdict through a fresh evidence recomputation even
# when an acceptance gate rejected the sole live run.
auth_status=0
PYTHONPATH=src "$python_bin" \
    -m semantic_3d_chat.evaluation.navigation_policy_v3_3_preregistration \
    authenticate-result --preregistration "$preregistration" --result "$result" \
    || auth_status=$?

if [[ "$context_status" -ne 0 || "$result_status" -ne 0 || "$auth_status" -ne 0 ]]; then
    exit 1
fi
