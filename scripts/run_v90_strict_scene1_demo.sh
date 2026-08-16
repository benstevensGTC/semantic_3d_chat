#!/usr/bin/env bash
set -euo pipefail

V90_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V90_DEMO_ROOT"

V90_DEMO_PYTHON="${V90_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V90_DEMO_SCENE="scene_000001"
V90_DEMO_CONFIG="configs/runtime/gemma4_v90_strict_scene1.yaml"
V90_DEMO_CHECKPOINT="data_gemma4/runtime/checkpoints/gemma4_v90_strict_scene1_release_v1"
V90_DEMO_MEMORY="data_gemma4/runtime/scene_memories/v90/scene_000001"
V90_DEMO_RELEASE_REPORT="reports/gemma4/metrics/gemma4_v90_strict_runtime_release.json"
V90_DEMO_AUDIT="reports/gemma4/metrics/v90_demo_access.json"
V90_DEMO_CHAT="reports/gemma4/examples/v90_demo_chat.jsonl"
V90_DEMO_MODE="finite"
V90_DEMO_MODE_EXPLICIT=0
V90_DEMO_ALL_PRIMARY=0
V90_DEMO_QUESTIONS=()

v90_demo_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v90_strict_scene1_demo.sh [OPTIONS]

Runs the promoted local Gemma 4 V90 strict scene-one proof. Environmental
input is one immutable 738-token continuous scene memory compiled before any
question. No caption, object list, scene graph, simulator label, oracle data,
or question-dependent retrieval is supplied to chat.

Options:
  --check             Model-free preflight, authenticate, and verify release
  --authenticate      Alias for the full authenticated release check
  --interactive       Start audited interactive local chat
  --leakage           Authenticate the oracle-unavailable runtime smoke
  --all-primary       Ask all 13 preregistered primary questions
  --question TEXT     Ask a finite question (repeatable)
  --scene ID          Opaque scene ID; V90 accepts only scene_000001
  --audit-log PATH    Runtime file-access audit output
  --chat-log PATH     Runtime answer JSONL output
  --python PATH       Gemma-capable local Python executable
  -h, --help          Show this help

With no mode flag, the finite demo asks the six core actionable questions plus
the three inherited smoke questions, then exits. Inference is forced offline.
EOF
}

v90_demo_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

v90_demo_set_mode() {
  if [[ "$V90_DEMO_MODE_EXPLICIT" -ne 0 ]]; then
    echo "Choose exactly one of --check, --authenticate, --interactive, or --leakage" >&2
    exit 2
  fi
  V90_DEMO_MODE="$1"
  V90_DEMO_MODE_EXPLICIT=1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) v90_demo_set_mode "check"; shift ;;
    --authenticate) v90_demo_set_mode "authenticate"; shift ;;
    --interactive) v90_demo_set_mode "interactive"; shift ;;
    --leakage) v90_demo_set_mode "leakage"; shift ;;
    --all-primary) V90_DEMO_ALL_PRIMARY=1; shift ;;
    --question)
      v90_demo_require_value "$@"; V90_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene) v90_demo_require_value "$@"; V90_DEMO_SCENE="$2"; shift 2 ;;
    --audit-log) v90_demo_require_value "$@"; V90_DEMO_AUDIT="$2"; shift 2 ;;
    --chat-log) v90_demo_require_value "$@"; V90_DEMO_CHAT="$2"; shift 2 ;;
    --python) v90_demo_require_value "$@"; V90_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v90_demo_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v90_demo_usage >&2; exit 2 ;;
  esac
done

if [[ "$V90_DEMO_SCENE" != "scene_000001" ]]; then
  echo "V90 strict demo accepts only scene_000001" >&2
  exit 2
fi
if [[ ! -x "$V90_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V90_DEMO_PYTHON" >&2
  exit 2
fi
if [[ "$V90_DEMO_ALL_PRIMARY" -eq 1 && ${#V90_DEMO_QUESTIONS[@]} -ne 0 ]]; then
  echo "--all-primary cannot be combined with --question" >&2
  exit 2
fi
if [[ "$V90_DEMO_MODE" != "finite" && "$V90_DEMO_ALL_PRIMARY" -eq 1 ]]; then
  echo "--all-primary is valid only in finite mode" >&2
  exit 2
fi
if [[ "$V90_DEMO_MODE" != "finite" && ${#V90_DEMO_QUESTIONS[@]} -ne 0 ]]; then
  echo "--question is valid only in finite mode" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

v90_preflight() {
  local output
  if ! output="$(env PYTHONPATH=src "$V90_DEMO_PYTHON" \
    scripts/check_v90_release.py \
    --scene "$V90_DEMO_SCENE" \
    --config "$V90_DEMO_CONFIG" \
    --checkpoint "$V90_DEMO_CHECKPOINT" \
    --scene-memory "$V90_DEMO_MEMORY" \
    --release-report "$V90_DEMO_RELEASE_REPORT")"; then
    return 2
  fi
  if [[ "$output" != *'"artifact": "gemma4_v90_strict_demo_preflight_v1"'* || \
        "$output" != *'"passed": true'* || \
        "$output" != *'"loads_model": false'* ]]; then
    echo "V90 model-free preflight returned an unauthenticated result" >&2
    return 2
  fi
  printf '%s\n' "$output"
}

v90_release() {
  env PYTHONPATH=src "$V90_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.v90_strict_runtime_release "$1"
}

v90_preflight

case "$V90_DEMO_MODE" in
  check|authenticate)
    v90_release authenticate
    v90_release verify
    exit 0
    ;;
  leakage)
    # A create-once smoke must run with the oracle directory physically absent.
    # Re-entry authenticates its exact bytes and must never silently regenerate it.
    v90_release smoke
    v90_release verify
    exit 0
    ;;
esac

v90_release verify >/dev/null

if [[ "$V90_DEMO_ALL_PRIMARY" -eq 1 ]]; then
  V90_DEMO_QUESTIONS=(
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
elif [[ ${#V90_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  # The exact six core actionable intents plus V89's three inherited smoke cases.
  V90_DEMO_QUESTIONS=(
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
    "What is on the table?"
    "What is underneath the table?"
    "What is hanging on the wall?"
    "Where is the red cube?"
    "What object could someone sit on?"
    "Is anything inside the bowl?"
  )
fi

V90_DEMO_ARGS=(
  --config "$V90_DEMO_CONFIG"
  --scene "$V90_DEMO_SCENE"
  --base-checkpoint "$V90_DEMO_CHECKPOINT"
  --scene-memory "$V90_DEMO_MEMORY"
  --audit-log "$V90_DEMO_AUDIT"
  --chat-log "$V90_DEMO_CHAT"
)
if [[ "$V90_DEMO_MODE" == "finite" ]]; then
  for question in "${V90_DEMO_QUESTIONS[@]}"; do
    V90_DEMO_ARGS+=(--question "$question")
  done
fi

echo "Promoted V90 strict local continuous-memory demo"
echo "  scene: $V90_DEMO_SCENE"
echo "  memory: $V90_DEMO_MEMORY"
echo "  checkpoint: $V90_DEMO_CHECKPOINT"
echo "  mode: $V90_DEMO_MODE"

exec env PYTHONPATH=src "$V90_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.v90_strict_scene1_cli \
  "${V90_DEMO_ARGS[@]}"
