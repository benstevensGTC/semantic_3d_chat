#!/usr/bin/env bash
set -euo pipefail

V96_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V96_DEMO_ROOT"

V96_DEMO_PYTHON="${V96_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V96_DEMO_SCENE="${V96_DEMO_SCENE:-scene_000025}"
V96_DEMO_CONFIG="${V96_DEMO_CONFIG:-configs/runtime/gemma4_v96_strict_multiscene.yaml}"
V96_DEMO_CHECKPOINT="${V96_DEMO_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v96_strict_multiscene_release_v1}"
V96_DEMO_MEMORY_ROOT="${V96_DEMO_MEMORY_ROOT:-data_gemma4/runtime/scene_memories/v96}"
V96_DEMO_MAP_ROOT="${V96_DEMO_MAP_ROOT:-data_gemma4/runtime/maps/v96}"
V96_DEMO_AUDIT="${V96_DEMO_AUDIT:-reports/gemma4/metrics/v96_demo_access.json}"
V96_DEMO_CHAT="${V96_DEMO_CHAT:-reports/gemma4/examples/v96_demo_chat.jsonl}"
V96_DEMO_MODE="finite"
V96_DEMO_QUESTIONS=()

v96_demo_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v96_strict_multiscene_demo.sh [OPTIONS]

Runs the promoted local Gemma 4 V96 held-out-scene proof. The runtime package
contains one sanitized numeric voxel map and one immutable 738-token continuous
scene memory. No caption, object list, scene graph, simulator label, or
question-dependent retrieval is supplied to Gemma.

Options:
  --check             Authenticate the promoted release without loading Gemma
  --interactive       Start audited interactive local chat
  --leakage           Authenticate the oracle-unavailable external smoke
  --question TEXT     Ask a finite question (repeatable)
  --scene ID          One released opaque scene ID, scene_000025--scene_000030
  --audit-log PATH    Runtime file-access audit output
  --chat-log PATH     Runtime answer JSONL output
  --python PATH       Gemma-capable local Python executable
  -h, --help          Show this help

With no mode flag, two finite questions are asked. Inference is offline and
uses only locally cached weights.
EOF
}

v96_demo_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) V96_DEMO_MODE="check"; shift ;;
    --interactive) V96_DEMO_MODE="interactive"; shift ;;
    --leakage) V96_DEMO_MODE="leakage"; shift ;;
    --question)
      v96_demo_require_value "$@"; V96_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene) v96_demo_require_value "$@"; V96_DEMO_SCENE="$2"; shift 2 ;;
    --audit-log) v96_demo_require_value "$@"; V96_DEMO_AUDIT="$2"; shift 2 ;;
    --chat-log) v96_demo_require_value "$@"; V96_DEMO_CHAT="$2"; shift 2 ;;
    --python) v96_demo_require_value "$@"; V96_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v96_demo_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v96_demo_usage >&2; exit 2 ;;
  esac
done

case "$V96_DEMO_SCENE" in
  scene_000025|scene_000026|scene_000027|scene_000028|scene_000029|scene_000030) ;;
  *) echo "V96 strict demo accepts scene_000025 through scene_000030" >&2; exit 2 ;;
esac
if [[ ! -x "$V96_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V96_DEMO_PYTHON" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

v96_release() {
  env PYTHONPATH=src "$V96_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.v96_strict_runtime_release "$1"
}

case "$V96_DEMO_MODE" in
  check)
    v96_release authenticate
    v96_release verify
    exit 0
    ;;
  leakage)
    v96_release smoke
    v96_release verify
    exit 0
    ;;
esac

v96_release verify >/dev/null

if [[ ${#V96_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  V96_DEMO_QUESTIONS=(
    "What is in the room?"
    "What is closest to the camera?"
  )
fi

V96_DEMO_MEMORY="$V96_DEMO_MEMORY_ROOT/$V96_DEMO_SCENE"
V96_DEMO_ARGS=(
  --config "$V96_DEMO_CONFIG"
  --scene "$V96_DEMO_SCENE"
  --base-checkpoint "$V96_DEMO_CHECKPOINT"
  --scene-memory "$V96_DEMO_MEMORY"
  --audit-log "$V96_DEMO_AUDIT"
  --chat-log "$V96_DEMO_CHAT"
)
if [[ "$V96_DEMO_MODE" == "finite" ]]; then
  for question in "${V96_DEMO_QUESTIONS[@]}"; do
    V96_DEMO_ARGS+=(--question "$question")
  done
fi

echo "Promoted V96 strict local continuous-memory demo"
echo "  scene: $V96_DEMO_SCENE"
echo "  memory: $V96_DEMO_MEMORY"
echo "  numeric map: $V96_DEMO_MAP_ROOT/$V96_DEMO_SCENE/voxel_map.npz"
echo "  checkpoint: $V96_DEMO_CHECKPOINT"
echo "  mode: $V96_DEMO_MODE"

exec env PYTHONPATH=src "$V96_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.v96_strict_multiscene_cli \
  "${V96_DEMO_ARGS[@]}"
