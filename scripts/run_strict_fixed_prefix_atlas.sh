#!/usr/bin/env bash
set -euo pipefail

STRICT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STRICT_PROJECT_ROOT"

STRICT_PYTHON="${STRICT_PYTHON:-.venv-gemma4/bin/python}"
STRICT_RUNTIME_CONFIG="${STRICT_RUNTIME_CONFIG:-configs/runtime/gemma4_v54.yaml}"
STRICT_BASE_CHECKPOINT="${STRICT_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
STRICT_CONTROL_CHECKPOINT="${STRICT_CONTROL_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control}"
STRICT_TRAINING_SOURCE="${STRICT_TRAINING_SOURCE:-data_gemma4/training/v62_pair_disjoint/train.jsonl}"
STRICT_ATLAS_CHECKPOINT="${STRICT_ATLAS_CHECKPOINT:-data_gemma4/checkpoints/gemma4_strict_fixed_prefix_atlas_v1}"
STRICT_QUESTIONS_MANIFEST="${STRICT_QUESTIONS_MANIFEST:-reports/gemma4/questions/v62_internal_validation.json}"
STRICT_PREDICTIONS="${STRICT_PREDICTIONS:-reports/gemma4/predictions/strict_fixed_prefix_atlas_v1.jsonl}"
STRICT_SCORER_REFERENCES="${STRICT_SCORER_REFERENCES:-}"
STRICT_SCORER_REFERENCES_SHA256="${STRICT_SCORER_REFERENCES_SHA256:-}"
STRICT_MIN_EXACT_ACCURACY="${STRICT_MIN_EXACT_ACCURACY:-}"
STRICT_LAUNCH_CLAIM="${STRICT_LAUNCH_CLAIM:-reports/gemma4/metrics/strict_fixed_prefix_atlas_claim.json}"
STRICT_TERMINAL_REPORT="${STRICT_TERMINAL_REPORT:-reports/gemma4/metrics/strict_fixed_prefix_atlas_terminal.json}"
STRICT_SCENE="${STRICT_SCENE:-scene_000001}"
STRICT_PROBE_COUNT="${STRICT_PROBE_COUNT:-96}"
STRICT_CLUSTER_ITERATIONS="${STRICT_CLUSTER_ITERATIONS:-12}"
STRICT_MODE="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi

strict_usage() {
  cat <<'EOF'
Usage: ./scripts/run_strict_fixed_prefix_atlas.sh MODE [ARGS]

Modes:
  check       Verify that prerequisites and the converted atlas exist
  build       Convert an accepted sealed scene controller to a fixed atlas
  chat        Run audited local chat; remaining args are forwarded to the CLI
  evaluate    Run questions-only held evaluation; remaining args are forwarded
  score       Seal one attempt, authenticate it, then open and score references
  leakage     Hide oracle/training data and verify loaded files + prefix invariance

The atlas is compiled before any user question. Every question for an unchanged
scene receives the exact same complete environmental prefix.
EOF
}

if [[ ! -x "$STRICT_PYTHON" ]]; then
  echo "Gemma Python environment is unavailable: $STRICT_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$STRICT_RUNTIME_CONFIG" || ! -d "$STRICT_BASE_CHECKPOINT" ]]; then
  echo "Strict fixed-prefix base runtime is unavailable" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

case "$STRICT_MODE" in
  check)
    [[ -d "$STRICT_CONTROL_CHECKPOINT" ]] || {
      echo "No accepted sealed scene controller is available: $STRICT_CONTROL_CHECKPOINT" >&2
      exit 2
    }
    [[ -d "$STRICT_ATLAS_CHECKPOINT" ]] || {
      echo "Fixed-prefix atlas is not built yet: $STRICT_ATLAS_CHECKPOINT" >&2
      exit 2
    }
    echo "Strict fixed-prefix prerequisites are present"
    ;;
  build)
    [[ -d "$STRICT_CONTROL_CHECKPOINT" ]] || {
      echo "No accepted sealed scene controller is available: $STRICT_CONTROL_CHECKPOINT" >&2
      exit 2
    }
    PYTHONPATH=src "$STRICT_PYTHON" scripts/build_fixed_prefix_atlas_checkpoint.py \
      --config "$STRICT_RUNTIME_CONFIG" \
      --base-checkpoint "$STRICT_BASE_CHECKPOINT" \
      --control-checkpoint "$STRICT_CONTROL_CHECKPOINT" \
      --training-source "$STRICT_TRAINING_SOURCE" \
      --probe-count "$STRICT_PROBE_COUNT" \
      --cluster-iterations "$STRICT_CLUSTER_ITERATIONS" \
      --output "$STRICT_ATLAS_CHECKPOINT"
    ;;
  chat)
    [[ -d "$STRICT_ATLAS_CHECKPOINT" ]] || {
      echo "Fixed-prefix atlas is unavailable; run build first" >&2
      exit 2
    }
    PYTHONPATH=src "$STRICT_PYTHON" \
      -m semantic_3d_chat.chat.fixed_prefix_cli \
      --config "$STRICT_RUNTIME_CONFIG" \
      --scene "$STRICT_SCENE" \
      --base-checkpoint "$STRICT_BASE_CHECKPOINT" \
      --atlas-checkpoint "$STRICT_ATLAS_CHECKPOINT" \
      "$@"
    ;;
  evaluate)
    [[ -d "$STRICT_ATLAS_CHECKPOINT" ]] || {
      echo "Fixed-prefix atlas is unavailable; run build first" >&2
      exit 2
    }
    PYTHONPATH=src "$STRICT_PYTHON" \
      -m semantic_3d_chat.evaluation.predict_fixed_prefix_atlas \
      --config "$STRICT_RUNTIME_CONFIG" \
      --questions-manifest "$STRICT_QUESTIONS_MANIFEST" \
      --base-checkpoint "$STRICT_BASE_CHECKPOINT" \
      --atlas-checkpoint "$STRICT_ATLAS_CHECKPOINT" \
      --output "$STRICT_PREDICTIONS" \
      "$@"
    ;;
  score)
    [[ -d "$STRICT_ATLAS_CHECKPOINT" ]] || {
      echo "Fixed-prefix atlas is unavailable; run build first" >&2
      exit 2
    }
    [[ -n "$STRICT_SCORER_REFERENCES" ]] || {
      echo "Set STRICT_SCORER_REFERENCES to the scorer-only reference file" >&2
      exit 2
    }
    [[ -n "$STRICT_SCORER_REFERENCES_SHA256" ]] || {
      echo "Set STRICT_SCORER_REFERENCES_SHA256 to its predeclared SHA-256" >&2
      exit 2
    }
    [[ -n "$STRICT_MIN_EXACT_ACCURACY" ]] || {
      echo "Set STRICT_MIN_EXACT_ACCURACY to the preregistered terminal threshold" >&2
      exit 2
    }
    PYTHONPATH=src "$STRICT_PYTHON" \
      -m semantic_3d_chat.evaluation.fixed_prefix_atlas_gate \
      --config "$STRICT_RUNTIME_CONFIG" \
      --questions-manifest "$STRICT_QUESTIONS_MANIFEST" \
      --base-checkpoint "$STRICT_BASE_CHECKPOINT" \
      --atlas-checkpoint "$STRICT_ATLAS_CHECKPOINT" \
      --predictions "$STRICT_PREDICTIONS" \
      --references "$STRICT_SCORER_REFERENCES" \
      --expected-references-sha256 "$STRICT_SCORER_REFERENCES_SHA256" \
      --split validation \
      --minimum-normalized-exact-accuracy "$STRICT_MIN_EXACT_ACCURACY" \
      --launch-claim "$STRICT_LAUNCH_CLAIM" \
      --output "$STRICT_TERMINAL_REPORT" \
      "$@"
    ;;
  leakage)
    [[ -d "$STRICT_ATLAS_CHECKPOINT" ]] || {
      echo "Fixed-prefix atlas is unavailable; run build first" >&2
      exit 2
    }
    PYTHONPATH=src "$STRICT_PYTHON" \
      -m semantic_3d_chat.evaluation.fixed_prefix_atlas_leakage \
      --config "$STRICT_RUNTIME_CONFIG" \
      --scene "$STRICT_SCENE" \
      --base-checkpoint "$STRICT_BASE_CHECKPOINT" \
      --atlas-checkpoint "$STRICT_ATLAS_CHECKPOINT" \
      --training-directory data_gemma4/training \
      --output reports/gemma4/metrics/strict_fixed_prefix_atlas_leakage.json \
      "$@"
    ;;
  -h|--help|help)
    strict_usage
    ;;
  *)
    echo "Unknown strict fixed-prefix mode: $STRICT_MODE" >&2
    strict_usage >&2
    exit 2
    ;;
esac
