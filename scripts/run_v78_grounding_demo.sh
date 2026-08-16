#!/usr/bin/env bash
set -euo pipefail

V78_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V78_DEMO_ROOT"

V78_DEMO_PYTHON="${V78_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V78_DEMO_CONFIG="${V78_DEMO_CONFIG:-configs/runtime/gemma4_v56_question_control.yaml}"
V78_DEMO_SCENE="${V78_DEMO_SCENE:-scene_000001}"
V78_DEMO_BASE_CHECKPOINT="${V78_DEMO_BASE_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v54_release_v1}"
V78_DEMO_CONTROL_CHECKPOINT="${V78_DEMO_CONTROL_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1}"
V78_DEMO_GROUNDING_CHECKPOINT="${V78_DEMO_GROUNDING_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v78_grounding_diagnostic_release_v1}"
V78_DEMO_AUDIT_LOG="${V78_DEMO_AUDIT_LOG:-reports/gemma4/metrics/v78_demo_access.json}"
V78_DEMO_CHAT_LOG="${V78_DEMO_CHAT_LOG:-reports/gemma4/examples/v78_demo_chat.jsonl}"
V78_DEMO_LEAKAGE_REPORT="${V78_DEMO_LEAKAGE_REPORT:-reports/gemma4/metrics/v78_demo_leakage.json}"
V78_DEMO_MODE="finite"
V78_DEMO_QUESTIONS=()

v78_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v78_grounding_demo.sh [OPTIONS]

Runs the unchanged V75 answer generator with the explicitly optional V78
numeric-grounding diagnostic. V78 is not official-validation evidence and is
not a promoted replacement for V75.

Options:
  --check                     Authenticate all inputs without loading Gemma
  --interactive               Start audited interactive chat
  --leakage                   Rename oracle data and prove runtime isolation
  --question TEXT             Ask a finite question (repeatable)
  --scene ID                  Opaque scene identifier
  --config PATH               Sanitized runtime YAML
  --base-checkpoint PATH      Exact two-file V54 release
  --control-checkpoint PATH   Exact two-file V75 release
  --grounding-checkpoint PATH Exact two-file V78 diagnostic runtime release
  --audit-log PATH            File-access audit output
  --chat-log PATH             Answer JSONL output
  --leakage-report PATH       Oracle-unavailable/prefix-invariance report
  --python PATH               Gemma-capable local Python
EOF
}

v78_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) V78_DEMO_MODE="check"; shift ;;
    --interactive) V78_DEMO_MODE="interactive"; shift ;;
    --leakage) V78_DEMO_MODE="leakage"; shift ;;
    --question) v78_value "$@"; V78_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene) v78_value "$@"; V78_DEMO_SCENE="$2"; shift 2 ;;
    --config) v78_value "$@"; V78_DEMO_CONFIG="$2"; shift 2 ;;
    --base-checkpoint) v78_value "$@"; V78_DEMO_BASE_CHECKPOINT="$2"; shift 2 ;;
    --control-checkpoint) v78_value "$@"; V78_DEMO_CONTROL_CHECKPOINT="$2"; shift 2 ;;
    --grounding-checkpoint) v78_value "$@"; V78_DEMO_GROUNDING_CHECKPOINT="$2"; shift 2 ;;
    --audit-log) v78_value "$@"; V78_DEMO_AUDIT_LOG="$2"; shift 2 ;;
    --chat-log) v78_value "$@"; V78_DEMO_CHAT_LOG="$2"; shift 2 ;;
    --leakage-report) v78_value "$@"; V78_DEMO_LEAKAGE_REPORT="$2"; shift 2 ;;
    --python) v78_value "$@"; V78_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v78_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v78_usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$V78_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V78_DEMO_PYTHON" >&2
  exit 2
fi

PYTHONPATH=src "$V78_DEMO_PYTHON" scripts/check_v78_grounding_demo.py \
  --config "$V78_DEMO_CONFIG" \
  --scene "$V78_DEMO_SCENE" \
  --base-checkpoint "$V78_DEMO_BASE_CHECKPOINT" \
  --control-checkpoint "$V78_DEMO_CONTROL_CHECKPOINT" \
  --grounding-checkpoint "$V78_DEMO_GROUNDING_CHECKPOINT"

if [[ "$V78_DEMO_MODE" == "check" ]]; then
  exit 0
fi

if [[ ${#V78_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  V78_DEMO_QUESTIONS=(
    "Where is the chair?"
    "Where is the bowl?"
    "Which object is closest to the camera?"
  )
fi
V78_QUESTION_ARGS=()
for question in "${V78_DEMO_QUESTIONS[@]}"; do
  V78_QUESTION_ARGS+=(--question "$question")
done

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ "$V78_DEMO_MODE" == "leakage" ]]; then
  exec env PYTHONPATH=src "$V78_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.question_control_leakage \
    --config "$V78_DEMO_CONFIG" \
    --scene "$V78_DEMO_SCENE" \
    --base-checkpoint "$V78_DEMO_BASE_CHECKPOINT" \
    --control-checkpoint "$V78_DEMO_CONTROL_CHECKPOINT" \
    --grounding-checkpoint "$V78_DEMO_GROUNDING_CHECKPOINT" \
    --output "$V78_DEMO_LEAKAGE_REPORT" \
    "${V78_QUESTION_ARGS[@]}"
fi

V78_CLI_ARGS=(
  --config "$V78_DEMO_CONFIG"
  --scene "$V78_DEMO_SCENE"
  --base-checkpoint "$V78_DEMO_BASE_CHECKPOINT"
  --control-checkpoint "$V78_DEMO_CONTROL_CHECKPOINT"
  --grounding-checkpoint "$V78_DEMO_GROUNDING_CHECKPOINT"
  --audit-log "$V78_DEMO_AUDIT_LOG"
  --chat-log "$V78_DEMO_CHAT_LOG"
)
if [[ "$V78_DEMO_MODE" == "finite" ]]; then
  V78_CLI_ARGS+=("${V78_QUESTION_ARGS[@]}")
fi

exec env PYTHONPATH=src "$V78_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.question_control_cli "${V78_CLI_ARGS[@]}"
