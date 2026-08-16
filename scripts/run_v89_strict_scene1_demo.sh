#!/usr/bin/env bash
set -euo pipefail

V89_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V89_DEMO_ROOT"

V89_DEMO_PYTHON="${V89_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V89_DEMO_SCENE="${V89_DEMO_SCENE:-scene_000001}"
V89_DEMO_CONFIG="${V89_DEMO_CONFIG:-configs/runtime/gemma4_v89_strict_scene1.yaml}"
V89_DEMO_CHECKPOINT="${V89_DEMO_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v89_strict_scene1_release_v1}"
V89_DEMO_MEMORY="${V89_DEMO_MEMORY:-data_gemma4/runtime/scene_memories/v89/scene_000001}"
V89_DEMO_AUDIT="${V89_DEMO_AUDIT:-reports/gemma4/metrics/v89_demo_access.json}"
V89_DEMO_CHAT="${V89_DEMO_CHAT:-reports/gemma4/examples/v89_demo_chat.jsonl}"
V89_DEMO_MODE="finite"
V89_DEMO_QUESTIONS=()

v89_demo_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v89_strict_scene1_demo.sh [OPTIONS]

Runs the promoted local Gemma 4 strict scene-one proof. Environmental input is
the immutable 738-token continuous scene memory; no caption, object list,
scene graph, simulator labels, or question-dependent retrieval is used.

Options:
  --check             Authenticate the promoted release without loading Gemma
  --interactive       Start audited interactive local chat
  --leakage           Authenticate the oracle-unavailable external smoke
  --question TEXT     Ask a finite question (repeatable)
  --scene ID          Opaque scene ID; V89 accepts only scene_000001
  --audit-log PATH    Runtime file-access audit output
  --chat-log PATH     Runtime answer JSONL output
  --python PATH       Gemma-capable local Python executable
  -h, --help          Show this help

With no mode flag, three finite questions are asked. Inference is forced
offline and uses only locally cached weights.
EOF
}

v89_demo_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) V89_DEMO_MODE="check"; shift ;;
    --interactive) V89_DEMO_MODE="interactive"; shift ;;
    --leakage) V89_DEMO_MODE="leakage"; shift ;;
    --question)
      v89_demo_require_value "$@"; V89_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene) v89_demo_require_value "$@"; V89_DEMO_SCENE="$2"; shift 2 ;;
    --audit-log) v89_demo_require_value "$@"; V89_DEMO_AUDIT="$2"; shift 2 ;;
    --chat-log) v89_demo_require_value "$@"; V89_DEMO_CHAT="$2"; shift 2 ;;
    --python) v89_demo_require_value "$@"; V89_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v89_demo_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v89_demo_usage >&2; exit 2 ;;
  esac
done

if [[ "$V89_DEMO_SCENE" != "scene_000001" ]]; then
  echo "V89 strict demo accepts only scene_000001" >&2
  exit 2
fi
if [[ ! -x "$V89_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V89_DEMO_PYTHON" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

v89_release() {
  env PYTHONPATH=src "$V89_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.v89_strict_runtime_release "$1"
}

case "$V89_DEMO_MODE" in
  check)
    v89_release authenticate
    v89_release verify
    exit 0
    ;;
  leakage)
    # The create-once smoke was executed with data/oracle physically renamed.
    # Re-entry authenticates its exact bytes and all 15 gates without mutating it.
    v89_release smoke
    v89_release verify
    exit 0
    ;;
esac

v89_release verify >/dev/null

if [[ ${#V89_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  V89_DEMO_QUESTIONS=(
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
  )
fi

V89_DEMO_ARGS=(
  --config "$V89_DEMO_CONFIG"
  --scene "$V89_DEMO_SCENE"
  --base-checkpoint "$V89_DEMO_CHECKPOINT"
  --scene-memory "$V89_DEMO_MEMORY"
  --audit-log "$V89_DEMO_AUDIT"
  --chat-log "$V89_DEMO_CHAT"
)
if [[ "$V89_DEMO_MODE" == "finite" ]]; then
  for question in "${V89_DEMO_QUESTIONS[@]}"; do
    V89_DEMO_ARGS+=(--question "$question")
  done
fi

echo "Promoted V89 strict local continuous-memory demo"
echo "  scene: $V89_DEMO_SCENE"
echo "  memory: $V89_DEMO_MEMORY"
echo "  checkpoint: $V89_DEMO_CHECKPOINT"
echo "  mode: $V89_DEMO_MODE"

exec env PYTHONPATH=src "$V89_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.v89_strict_scene1_cli \
  "${V89_DEMO_ARGS[@]}"
