#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${GEMMA4_PYTHON:-.venv-gemma4/bin/python}"
run_id="${NAVIGATION_RUN_ID:-learned_v2_demo}"
policy_checkpoint="${NAVIGATION_POLICY_CHECKPOINT:-data_gemma4/checkpoints/navigation_policy_v2}"
policy_version="${NAVIGATION_POLICY_VERSION:-1}"
embodied_config="${NAVIGATION_EMBODIED_CONFIG:-configs/runtime/embodied_navigation_v2.yaml}"
tasks="${NAVIGATION_TASKS:-configs/benchmarks/llm_navigation_v2_scene_000001.json}"
scoring_spec="${NAVIGATION_SCORING_SPEC:-configs/benchmarks/oracle/llm_navigation_v2_scene_000001.json}"
journal="reports/gemma4/predictions/llm_navigation_scene_000001_${run_id}.json"
audit="reports/gemma4/metrics/llm_navigation_inference_access_${run_id}.json"
score="reports/gemma4/metrics/llm_navigation_scene_000001_${run_id}.json"
persistent_map="data_gemma4/robot_benchmark_${run_id}/scene_000001/semantic_map.npz"

mode="run"
if [[ "${1:-}" == "--check" ]]; then
    mode="check"
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: ./scripts/run_learned_navigation_benchmark.sh [--check]" >&2
    exit 2
fi

if [[ "$policy_version" != "1" && "$policy_version" != "3" && "$policy_version" != "4" ]]; then
    echo "NAVIGATION_POLICY_VERSION must be 1, 3, or 4; got: $policy_version" >&2
    exit 2
fi

if [[ ! -x "$python_bin" ]]; then
    echo "Gemma Python environment is unavailable: $python_bin" >&2
    exit 2
fi

if [[ "$mode" == "check" ]]; then
    PYTHONPATH=src "$python_bin" scripts/check_navigation_policy_demo.py \
        --checkpoint "$policy_checkpoint" \
        --runtime-config "$embodied_config" \
        --tasks "$tasks" \
        --journal "$journal" \
        --audit "$audit" \
        --score "$score" \
        --expected-checkpoint-tree-sha256 "${NAVIGATION_EXPECTED_CHECKPOINT_TREE_SHA256:-8ca5f5278680a5cc2c9c48b170c3f78f6adb973696ad9107779c66a7806b4735}" \
        --expected-runtime-config-sha256 "${NAVIGATION_EXPECTED_RUNTIME_CONFIG_SHA256:-5ee4610104f4d8058a5fd739678f0e400c2181d6875e0e223727bbbef40ccb13}" \
        --expected-tasks-sha256 "${NAVIGATION_EXPECTED_TASKS_SHA256:-29bd12966f28b0b9ecc4ba444af25bde712b98512d7c74e322cbc7019e4f5e07}" \
        --expected-journal-file-sha256 "${NAVIGATION_EXPECTED_JOURNAL_FILE_SHA256:-3fa33bc463dce73b14527056d598c64db13b49caf042d16347754d76246f5474}" \
        --expected-journal-sha256 "${NAVIGATION_EXPECTED_JOURNAL_SHA256:-12a34ca7dd5cbe4f971ee210feaa4806be72bad400fd06aa5643b02e1e092c1f}" \
        --expected-audit-sha256 "${NAVIGATION_EXPECTED_AUDIT_SHA256:-9cb6dbdb72f7c9f7d7043ec3ee42cb0a615692ab287cd235a944f16049a3eac0}" \
        --expected-score-sha256 "${NAVIGATION_EXPECTED_SCORE_SHA256:-f4e290613912a3020f80feeb7355f05149a0c4db29bbbdbcf11d5e15506c8ae2}"
    exit 0
fi

for output in "$journal" "$audit" "$score" "$persistent_map"; do
    if [[ -e "$output" ]]; then
        echo "Refusing to overwrite benchmark evidence: $output" >&2
        echo "Set NAVIGATION_RUN_ID to a new opaque run identifier." >&2
        exit 2
    fi
done

if [[ ! -f "$policy_checkpoint/policy.safetensors" || ! -f "$policy_checkpoint/runtime_metadata.json" ]]; then
    echo "Missing accepted navigation policy checkpoint: $policy_checkpoint" >&2
    echo "Run: make navigation-policy-train" >&2
    exit 2
fi

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
    "$python_bin" scripts/run_llm_navigation_inference.py \
    --config "$embodied_config" \
    --tasks "$tasks" \
    --navigation-policy-checkpoint "$policy_checkpoint" \
    --navigation-policy-version "$policy_version" \
    --journal "$journal" \
    --audit-report "$audit" \
    --persistent-map "$persistent_map"

PYTHONPATH=src "$python_bin" scripts/score_llm_navigation.py \
    --journal "$journal" \
    --scoring-spec "$scoring_spec" \
    --output "$score"

echo "Sealed learned-policy journal: $journal"
echo "Oracle-isolation audit: $audit"
echo "Post-inference score: $score"
