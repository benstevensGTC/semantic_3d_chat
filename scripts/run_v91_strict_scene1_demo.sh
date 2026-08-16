#!/usr/bin/env bash
set -euo pipefail

V91_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V91_DEMO_ROOT"

V91_DEMO_PYTHON="${V91_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V91_DEMO_SCENE="scene_000001"
V91_DEMO_CONFIG="configs/runtime/gemma4_v91_strict_scene1.yaml"
V91_DEMO_CHECKPOINT="data_gemma4/runtime/checkpoints/gemma4_v91_strict_scene1_release_v1"
V91_DEMO_MEMORY="data_gemma4/runtime/scene_memories/v91/scene_000001"
V91_DEMO_AUDIT="reports/gemma4/metrics/v91_demo_access.json"
V91_DEMO_CHAT="reports/gemma4/examples/v91_demo_chat.jsonl"
V91_DEMO_MODE="finite"
V91_DEMO_MODE_EXPLICIT=0
V91_DEMO_ALL_PRIMARY=0
V91_DEMO_QUESTIONS=()

v91_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v91_strict_scene1_demo.sh [OPTIONS]

Runs only an authenticated, promoted V91 strict scene-one release. The script
fails closed while V91 is merely an offline candidate. Environmental input is
the immutable pre-question [1,738,1536] continuous scene memory; no caption,
object list, oracle metadata, or question-dependent retrieval enters chat.

Options:
  --check             Authenticate model gates and verify the promoted release
  --interactive       Start audited interactive local chat
  --leakage           Re-authenticate the isolated oracle-absent smoke
  --all-primary       Ask all 13 primary conversational questions
  --question TEXT     Ask a finite question (repeatable)
  --scene ID          Opaque scene ID; only scene_000001 is accepted
  --audit-log PATH    Runtime file-access audit output
  --chat-log PATH     Runtime answer JSONL output
  --python PATH       Gemma-capable local Python executable
  -h, --help          Show this help
EOF
}

v91_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

v91_set_mode() {
  if [[ "$V91_DEMO_MODE_EXPLICIT" -ne 0 ]]; then
    echo "Choose only one of --check, --interactive, or --leakage" >&2
    exit 2
  fi
  V91_DEMO_MODE="$1"
  V91_DEMO_MODE_EXPLICIT=1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) v91_set_mode "check"; shift ;;
    --interactive) v91_set_mode "interactive"; shift ;;
    --leakage) v91_set_mode "leakage"; shift ;;
    --all-primary) V91_DEMO_ALL_PRIMARY=1; shift ;;
    --question) v91_require_value "$@"; V91_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene) v91_require_value "$@"; V91_DEMO_SCENE="$2"; shift 2 ;;
    --audit-log) v91_require_value "$@"; V91_DEMO_AUDIT="$2"; shift 2 ;;
    --chat-log) v91_require_value "$@"; V91_DEMO_CHAT="$2"; shift 2 ;;
    --python) v91_require_value "$@"; V91_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v91_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v91_usage >&2; exit 2 ;;
  esac
done

if [[ "$V91_DEMO_SCENE" != "scene_000001" ]]; then
  echo "V91 strict demo accepts only scene_000001" >&2
  exit 2
fi
if [[ ! -x "$V91_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V91_DEMO_PYTHON" >&2
  exit 2
fi
if [[ "$V91_DEMO_ALL_PRIMARY" -eq 1 && ${#V91_DEMO_QUESTIONS[@]} -ne 0 ]]; then
  echo "--all-primary cannot be combined with --question" >&2
  exit 2
fi
if [[ "$V91_DEMO_MODE" != "finite" && "$V91_DEMO_ALL_PRIMARY" -eq 1 ]]; then
  echo "--all-primary is valid only in finite mode" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

v91_release() {
  env PYTHONPATH=src "$V91_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.v91_strict_runtime_release "$1"
}

case "$V91_DEMO_MODE" in
  check)
    v91_release authenticate
    v91_release verify
    exit 0
    ;;
  leakage)
    v91_release smoke
    v91_release verify
    exit 0
    ;;
esac

v91_release verify >/dev/null

if [[ "$V91_DEMO_ALL_PRIMARY" -eq 1 ]]; then
  V91_DEMO_QUESTIONS=(
    "What objects are around you?"
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
    "What is on the table?"
    "What is underneath the table?"
    "Which object is closest to you?"
    "What is hanging on the wall?"
    "Where is the red cube?"
    "Which direction would you turn to face the lamp?"
    "Is the picture frame on the wall or on the floor?"
    "What object could someone sit on?"
    "Is anything inside the bowl?"
  )
elif [[ ${#V91_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  V91_DEMO_QUESTIONS=(
    "What is on the table?"
    "What is underneath the table?"
    "What is hanging on the wall?"
    "Where is the red cube?"
    "What object could someone sit on?"
    "Is anything inside the bowl?"
  )
fi

V91_DEMO_ARGS=(
  --config "$V91_DEMO_CONFIG"
  --scene "$V91_DEMO_SCENE"
  --base-checkpoint "$V91_DEMO_CHECKPOINT"
  --scene-memory "$V91_DEMO_MEMORY"
  --audit-log "$V91_DEMO_AUDIT"
  --chat-log "$V91_DEMO_CHAT"
)
if [[ "$V91_DEMO_MODE" == "finite" ]]; then
  for question in "${V91_DEMO_QUESTIONS[@]}"; do
    V91_DEMO_ARGS+=(--question "$question")
  done
fi

exec env PYTHONPATH=src "$V91_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.v91_strict_scene1_cli \
  "${V91_DEMO_ARGS[@]}"
