#!/usr/bin/env bash
set -euo pipefail

SCHEMA7_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCHEMA7_PROJECT_ROOT"

SCHEMA7_PYTHON="${SCHEMA7_PYTHON:-.venv-gemma4/bin/python}"
SCHEMA7_CONFIG="${SCHEMA7_CONFIG:-configs/runtime/gemma4_v54.yaml}"
SCHEMA7_SCENE="${SCHEMA7_SCENE:-scene_000001}"
SCHEMA7_BASE_CHECKPOINT="${SCHEMA7_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
SCHEMA7_CONTROL_CHECKPOINT="${SCHEMA7_CONTROL_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control}"
SCHEMA7_TRAINING_ARTIFACT="${SCHEMA7_TRAINING_ARTIFACT:-data_gemma4/training/v66_answer_class_teachers}"
SCHEMA7_AUDIT_LOG="${SCHEMA7_AUDIT_LOG:-reports/gemma4/metrics/schema7_demo_access.json}"
SCHEMA7_CHAT_LOG="${SCHEMA7_CHAT_LOG:-reports/gemma4/examples/schema7_demo_chat.jsonl}"
SCHEMA7_LEAKAGE_REPORT="${SCHEMA7_LEAKAGE_REPORT:-reports/gemma4/metrics/schema7_demo_leakage.json}"
SCHEMA7_MODE="finite"
SCHEMA7_QUESTIONS=()

schema7_usage() {
  cat <<'EOF'
Usage: ./scripts/run_schema7_question_control_demo.sh [OPTIONS]

Runs local Gemma 4 with the immutable full-scene prefix plus a sealed schema-7
continuous question controller. The default is a finite three-question demo.

Options:
  --config PATH               Standalone sanitized runtime YAML
  --scene ID                  Opaque scene ID
  --base-checkpoint PATH      Full-scene base checkpoint
  --control-checkpoint PATH   Sealed two-file schema-7 checkpoint
  --question TEXT             Ask a finite question (repeatable)
  --interactive               Start the audited interactive CLI
  --leakage                   Hide oracle/training artifacts and run finite inference
  --training-artifact PATH    Exact training-only directory to hide in leakage mode
  --audit-log PATH            Runtime file-access audit output
  --chat-log PATH             Finite/interactive answer JSONL output
  --leakage-report PATH       Leakage report output
  --python PATH               Gemma-capable Python executable
  -h, --help                  Show this help

The same options can be supplied through the SCHEMA7_* environment variables.
Inference is forced offline; no cloud inference API is used.
EOF
}

schema7_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) schema7_require_value "$@"; SCHEMA7_CONFIG="$2"; shift 2 ;;
    --scene) schema7_require_value "$@"; SCHEMA7_SCENE="$2"; shift 2 ;;
    --base-checkpoint)
      schema7_require_value "$@"; SCHEMA7_BASE_CHECKPOINT="$2"; shift 2 ;;
    --control-checkpoint)
      schema7_require_value "$@"; SCHEMA7_CONTROL_CHECKPOINT="$2"; shift 2 ;;
    --question) schema7_require_value "$@"; SCHEMA7_QUESTIONS+=("$2"); shift 2 ;;
    --interactive) SCHEMA7_MODE="interactive"; shift ;;
    --leakage) SCHEMA7_MODE="leakage"; shift ;;
    --training-artifact)
      schema7_require_value "$@"; SCHEMA7_TRAINING_ARTIFACT="$2"; shift 2 ;;
    --audit-log) schema7_require_value "$@"; SCHEMA7_AUDIT_LOG="$2"; shift 2 ;;
    --chat-log) schema7_require_value "$@"; SCHEMA7_CHAT_LOG="$2"; shift 2 ;;
    --leakage-report)
      schema7_require_value "$@"; SCHEMA7_LEAKAGE_REPORT="$2"; shift 2 ;;
    --python) schema7_require_value "$@"; SCHEMA7_PYTHON="$2"; shift 2 ;;
    -h|--help) schema7_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; schema7_usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$SCHEMA7_PYTHON" ]]; then
  echo "Gemma Python environment is unavailable: $SCHEMA7_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$SCHEMA7_CONFIG" ]]; then
  echo "Runtime config is unavailable: $SCHEMA7_CONFIG" >&2
  exit 2
fi
if [[ ! -d "$SCHEMA7_BASE_CHECKPOINT" ]]; then
  echo "Base checkpoint is unavailable: $SCHEMA7_BASE_CHECKPOINT" >&2
  exit 2
fi
if [[ ! -d "$SCHEMA7_CONTROL_CHECKPOINT" ]]; then
  echo "Sealed schema-7 checkpoint is unavailable: $SCHEMA7_CONTROL_CHECKPOINT" >&2
  exit 2
fi

if [[ ${#SCHEMA7_QUESTIONS[@]} -eq 0 ]]; then
  SCHEMA7_QUESTIONS=(
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
  )
fi

SCHEMA7_QUESTION_ARGS=()
for SCHEMA7_QUESTION in "${SCHEMA7_QUESTIONS[@]}"; do
  SCHEMA7_QUESTION_ARGS+=(--question "$SCHEMA7_QUESTION")
done

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Schema-7 local continuous-scene demo"
echo "  scene: $SCHEMA7_SCENE"
echo "  base: $SCHEMA7_BASE_CHECKPOINT"
echo "  control: $SCHEMA7_CONTROL_CHECKPOINT"
echo "  mode: $SCHEMA7_MODE"

if [[ "$SCHEMA7_MODE" == "leakage" ]]; then
  PYTHONPATH=src "$SCHEMA7_PYTHON" \
    -m semantic_3d_chat.evaluation.question_control_leakage \
    --config "$SCHEMA7_CONFIG" \
    --scene "$SCHEMA7_SCENE" \
    --base-checkpoint "$SCHEMA7_BASE_CHECKPOINT" \
    --control-checkpoint "$SCHEMA7_CONTROL_CHECKPOINT" \
    --training-artifact "$SCHEMA7_TRAINING_ARTIFACT" \
    --output "$SCHEMA7_LEAKAGE_REPORT" \
    "${SCHEMA7_QUESTION_ARGS[@]}"
  exit 0
fi

SCHEMA7_CLI_ARGS=(
  --config "$SCHEMA7_CONFIG"
  --scene "$SCHEMA7_SCENE"
  --base-checkpoint "$SCHEMA7_BASE_CHECKPOINT"
  --control-checkpoint "$SCHEMA7_CONTROL_CHECKPOINT"
  --audit-log "$SCHEMA7_AUDIT_LOG"
  --chat-log "$SCHEMA7_CHAT_LOG"
)
if [[ "$SCHEMA7_MODE" == "finite" ]]; then
  SCHEMA7_CLI_ARGS+=("${SCHEMA7_QUESTION_ARGS[@]}")
fi

PYTHONPATH=src "$SCHEMA7_PYTHON" \
  -m semantic_3d_chat.chat.question_control_cli \
  "${SCHEMA7_CLI_ARGS[@]}"
