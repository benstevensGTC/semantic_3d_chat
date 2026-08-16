#!/usr/bin/env bash
set -euo pipefail

V75_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V75_DEMO_ROOT"

V75_DEMO_PYTHON="${V75_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V75_DEMO_CONFIG="${V75_DEMO_CONFIG:-configs/runtime/gemma4_v56_question_control.yaml}"
V75_DEMO_SCENE="${V75_DEMO_SCENE:-scene_000001}"
V75_DEMO_BASE_CHECKPOINT="${V75_DEMO_BASE_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v54_release_v1}"
V75_DEMO_CONTROL_CHECKPOINT="${V75_DEMO_CONTROL_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1}"
V75_DEMO_AUDIT_LOG="${V75_DEMO_AUDIT_LOG:-reports/gemma4/metrics/v75_demo_access.json}"
V75_DEMO_CHAT_LOG="${V75_DEMO_CHAT_LOG:-reports/gemma4/examples/v75_demo_chat.jsonl}"
V75_DEMO_LEAKAGE_REPORT="${V75_DEMO_LEAKAGE_REPORT:-reports/gemma4/metrics/v75_demo_leakage.json}"
V75_DEMO_MODE="finite"
V75_DEMO_QUESTIONS=()

v75_demo_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v75_question_control_demo.sh [OPTIONS]

Runs local Gemma 4 over a complete, question-independent continuous 3D scene
prefix plus the promoted sealed V75 full-scene controller. The default is a
finite three-question demonstration; no environmental description is supplied.

Options:
  --check                     Authenticate inputs without loading Gemma
  --interactive               Start audited interactive chat
  --leakage                   Rename oracle data and prove runtime isolation
  --question TEXT             Ask a finite question (repeatable)
  --scene ID                  Opaque scene identifier
  --config PATH               Standalone sanitized runtime YAML
  --base-checkpoint PATH      Exact two-file V54 scene-prefix release
  --control-checkpoint PATH   Exact two-file schema-75 controller release
  --audit-log PATH            File-access audit output for chat
  --chat-log PATH             Answer JSONL output for chat
  --leakage-report PATH       Oracle-deletion/prefix-invariance report
  --python PATH               Gemma-capable local Python executable
  -h, --help                  Show this help

The same values can be supplied through V75_DEMO_* environment variables.
Inference is forced offline; no cloud inference API is used.
EOF
}

v75_demo_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) V75_DEMO_MODE="check"; shift ;;
    --interactive) V75_DEMO_MODE="interactive"; shift ;;
    --leakage) V75_DEMO_MODE="leakage"; shift ;;
    --question)
      v75_demo_require_value "$@"; V75_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene) v75_demo_require_value "$@"; V75_DEMO_SCENE="$2"; shift 2 ;;
    --config) v75_demo_require_value "$@"; V75_DEMO_CONFIG="$2"; shift 2 ;;
    --base-checkpoint)
      v75_demo_require_value "$@"; V75_DEMO_BASE_CHECKPOINT="$2"; shift 2 ;;
    --control-checkpoint)
      v75_demo_require_value "$@"; V75_DEMO_CONTROL_CHECKPOINT="$2"; shift 2 ;;
    --audit-log) v75_demo_require_value "$@"; V75_DEMO_AUDIT_LOG="$2"; shift 2 ;;
    --chat-log) v75_demo_require_value "$@"; V75_DEMO_CHAT_LOG="$2"; shift 2 ;;
    --leakage-report)
      v75_demo_require_value "$@"; V75_DEMO_LEAKAGE_REPORT="$2"; shift 2 ;;
    --python) v75_demo_require_value "$@"; V75_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v75_demo_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v75_demo_usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$V75_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V75_DEMO_PYTHON" >&2
  echo "Run: make setup-gemma4-probe" >&2
  exit 2
fi

# The base release can be recreated byte-for-byte from the local promoted V54
# source. The V75 control release is gate-produced and therefore only checked,
# never reconstructed from training inputs by an inference launcher.
if [[ "$V75_DEMO_BASE_CHECKPOINT" == "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" ]]; then
  PYTHONPATH=src "$V75_DEMO_PYTHON" scripts/prepare_demo_runtime.py >/dev/null
fi

if [[
  "$V75_DEMO_CONFIG" == "configs/runtime/gemma4_v56_question_control.yaml" &&
  "$V75_DEMO_SCENE" == "scene_000001" &&
  "$V75_DEMO_BASE_CHECKPOINT" == "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" &&
  "$V75_DEMO_CONTROL_CHECKPOINT" == "data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1"
]]; then
  PYTHONPATH=src "$V75_DEMO_PYTHON" scripts/check_demo_artifacts.py --fast >/dev/null || {
    echo "Promoted V75 demo artifacts are incomplete; run: make demo-artifacts-check-fast" >&2
    echo "Model weights can be fetched with: make download-gemma4-weights" >&2
    exit 2
  }
fi

PYTHONPATH=src "$V75_DEMO_PYTHON" scripts/check_v75_demo.py \
  --config "$V75_DEMO_CONFIG" \
  --scene "$V75_DEMO_SCENE" \
  --base-checkpoint "$V75_DEMO_BASE_CHECKPOINT" \
  --control-checkpoint "$V75_DEMO_CONTROL_CHECKPOINT"

if [[ "$V75_DEMO_MODE" == "check" ]]; then
  exit 0
fi

if [[ ${#V75_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  V75_DEMO_QUESTIONS=(
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
  )
fi

V75_DEMO_QUESTION_ARGS=()
for V75_DEMO_QUESTION in "${V75_DEMO_QUESTIONS[@]}"; do
  V75_DEMO_QUESTION_ARGS+=(--question "$V75_DEMO_QUESTION")
done

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Promoted V75 local continuous-scene demo"
echo "  scene: $V75_DEMO_SCENE"
echo "  base: $V75_DEMO_BASE_CHECKPOINT"
echo "  control: $V75_DEMO_CONTROL_CHECKPOINT"
echo "  mode: $V75_DEMO_MODE"

if [[ "$V75_DEMO_MODE" == "leakage" ]]; then
  exec env PYTHONPATH=src "$V75_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.question_control_leakage \
    --config "$V75_DEMO_CONFIG" \
    --scene "$V75_DEMO_SCENE" \
    --base-checkpoint "$V75_DEMO_BASE_CHECKPOINT" \
    --control-checkpoint "$V75_DEMO_CONTROL_CHECKPOINT" \
    --output "$V75_DEMO_LEAKAGE_REPORT" \
    "${V75_DEMO_QUESTION_ARGS[@]}"
fi

V75_DEMO_CLI_ARGS=(
  --config "$V75_DEMO_CONFIG"
  --scene "$V75_DEMO_SCENE"
  --base-checkpoint "$V75_DEMO_BASE_CHECKPOINT"
  --control-checkpoint "$V75_DEMO_CONTROL_CHECKPOINT"
  --audit-log "$V75_DEMO_AUDIT_LOG"
  --chat-log "$V75_DEMO_CHAT_LOG"
)
if [[ "$V75_DEMO_MODE" == "finite" ]]; then
  V75_DEMO_CLI_ARGS+=("${V75_DEMO_QUESTION_ARGS[@]}")
fi

exec env PYTHONPATH=src "$V75_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.question_control_cli \
  "${V75_DEMO_CLI_ARGS[@]}"
