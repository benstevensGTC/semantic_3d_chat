#!/usr/bin/env bash
set -euo pipefail

V81_DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V81_DEMO_ROOT"

V81_DEMO_PYTHON="${V81_DEMO_PYTHON:-.venv-gemma4/bin/python}"
V81_DEMO_CONFIG="${V81_DEMO_CONFIG:-configs/runtime/gemma4_v56_question_control.yaml}"
V81_DEMO_SCENE="${V81_DEMO_SCENE:-scene_000001}"
V81_DEMO_BASE_CHECKPOINT="${V81_DEMO_BASE_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v54_release_v1}"
V81_DEMO_CONTROL_CHECKPOINT="${V81_DEMO_CONTROL_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1}"
V81_DEMO_PROBE_BANK="${V81_DEMO_PROBE_BANK:-reports/gemma4/artifacts/v75_fixed_atlas_historical_internal_v1/probe_bank}"
V81_DEMO_GROUNDING_CHECKPOINT="${V81_DEMO_GROUNDING_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v78_grounding_diagnostic_release_v1}"
V81_DEMO_SCENE_MEMORY="${V81_DEMO_SCENE_MEMORY:-data_gemma4/runtime/scene_memories/v81/$V81_DEMO_SCENE}"
V81_DEMO_AUDIT_LOG="${V81_DEMO_AUDIT_LOG:-}"
V81_DEMO_CHAT_LOG="${V81_DEMO_CHAT_LOG:-}"
V81_DEMO_LEAKAGE_REPORT="${V81_DEMO_LEAKAGE_REPORT:-}"
V81_DEMO_COMPILE_AUDIT="${V81_DEMO_COMPILE_AUDIT:-}"
V81_DEMO_MODE="finite"
V81_DEMO_QUESTIONS=()

v81_demo_usage() {
  cat <<'EOF'
Usage: ./scripts/run_v81_scene_memory_demo.sh [OPTIONS]

Runs local Gemma 4 over a sealed 738-token continuous room memory. The memory
is compiled before questions; chat loads no compiler, probe bank, oracle, QA,
caption, labels, or scene graph.

Options:
  --check                     Authenticate prepared inputs without loading Gemma
  --compile                   Rebuild a missing sealed memory, then authenticate it
  --verify-compile            Recompile and compare without replacing sealed memory
  --interactive               Start audited interactive chat
  --leakage                   Hide oracle and prove runtime/file isolation
  --question TEXT             Ask a finite question (repeatable)
  --scene ID                  Opaque scene identifier
  --scene-memory PATH         Two-file sealed numeric scene memory
  --grounding-checkpoint PATH Optional two-file V78 numeric grounding sidecar
  --audit-log PATH            Chat file-access audit output
  --chat-log PATH             Chat JSONL output
  --leakage-report PATH       Oracle-deletion report
  --python PATH               Gemma-capable local Python
  -h, --help                  Show this help

Inference is forced offline. The default finite demo asks three questions.
EOF
}

v81_demo_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) V81_DEMO_MODE="check"; shift ;;
    --compile) V81_DEMO_MODE="compile"; shift ;;
    --verify-compile) V81_DEMO_MODE="verify-compile"; shift ;;
    --interactive) V81_DEMO_MODE="interactive"; shift ;;
    --leakage) V81_DEMO_MODE="leakage"; shift ;;
    --question)
      v81_demo_require_value "$@"; V81_DEMO_QUESTIONS+=("$2"); shift 2 ;;
    --scene)
      v81_demo_require_value "$@"; V81_DEMO_SCENE="$2"; shift 2
      V81_DEMO_SCENE_MEMORY="data_gemma4/runtime/scene_memories/v81/$V81_DEMO_SCENE"
      ;;
    --scene-memory)
      v81_demo_require_value "$@"; V81_DEMO_SCENE_MEMORY="$2"; shift 2 ;;
    --grounding-checkpoint)
      v81_demo_require_value "$@"; V81_DEMO_GROUNDING_CHECKPOINT="$2"; shift 2 ;;
    --audit-log)
      v81_demo_require_value "$@"; V81_DEMO_AUDIT_LOG="$2"; shift 2 ;;
    --chat-log)
      v81_demo_require_value "$@"; V81_DEMO_CHAT_LOG="$2"; shift 2 ;;
    --leakage-report)
      v81_demo_require_value "$@"; V81_DEMO_LEAKAGE_REPORT="$2"; shift 2 ;;
    --python)
      v81_demo_require_value "$@"; V81_DEMO_PYTHON="$2"; shift 2 ;;
    -h|--help) v81_demo_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; v81_demo_usage >&2; exit 2 ;;
  esac
done

# Derive scene-specific report paths only after command-line scene selection.
# Explicit command-line or environment paths remain unchanged.
V81_DEMO_AUDIT_LOG="${V81_DEMO_AUDIT_LOG:-reports/gemma4/metrics/v81_chat_access_$V81_DEMO_SCENE.json}"
V81_DEMO_CHAT_LOG="${V81_DEMO_CHAT_LOG:-reports/gemma4/examples/v81_chat_$V81_DEMO_SCENE.jsonl}"
V81_DEMO_LEAKAGE_REPORT="${V81_DEMO_LEAKAGE_REPORT:-reports/gemma4/metrics/v81_scene_memory_leakage_$V81_DEMO_SCENE.json}"
V81_DEMO_COMPILE_AUDIT="${V81_DEMO_COMPILE_AUDIT:-reports/gemma4/metrics/v81_compile_access_$V81_DEMO_SCENE.json}"

if [[ ! -x "$V81_DEMO_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $V81_DEMO_PYTHON" >&2
  echo "Run: make setup-gemma4-probe" >&2
  exit 2
fi

if [[ "$V81_DEMO_MODE" == "compile" && ! -e "$V81_DEMO_SCENE_MEMORY" ]]; then
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src "$V81_DEMO_PYTHON" \
    scripts/prepare_v81_scene_memory.py \
    --config "$V81_DEMO_CONFIG" \
    --scene "$V81_DEMO_SCENE" \
    --base-checkpoint "$V81_DEMO_BASE_CHECKPOINT" \
    --control-checkpoint "$V81_DEMO_CONTROL_CHECKPOINT" \
    --probe-bank "$V81_DEMO_PROBE_BANK" \
    --output "$V81_DEMO_SCENE_MEMORY" \
    --audit-report "$V81_DEMO_COMPILE_AUDIT"
fi

if [[ "$V81_DEMO_MODE" == "verify-compile" ]]; then
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src "$V81_DEMO_PYTHON" \
    scripts/prepare_v81_scene_memory.py \
    --config "$V81_DEMO_CONFIG" \
    --scene "$V81_DEMO_SCENE" \
    --base-checkpoint "$V81_DEMO_BASE_CHECKPOINT" \
    --control-checkpoint "$V81_DEMO_CONTROL_CHECKPOINT" \
    --probe-bank "$V81_DEMO_PROBE_BANK" \
    --output "$V81_DEMO_SCENE_MEMORY" \
    --audit-report "$V81_DEMO_COMPILE_AUDIT" \
    --verify-existing
fi

PYTHONPATH=src "$V81_DEMO_PYTHON" scripts/check_v81_scene_memory_demo.py \
  --config "$V81_DEMO_CONFIG" \
  --scene "$V81_DEMO_SCENE" \
  --base-checkpoint "$V81_DEMO_BASE_CHECKPOINT" \
  --scene-memory "$V81_DEMO_SCENE_MEMORY" \
  --grounding-checkpoint "$V81_DEMO_GROUNDING_CHECKPOINT"

if [[ "$V81_DEMO_MODE" == "check" || "$V81_DEMO_MODE" == "compile" || "$V81_DEMO_MODE" == "verify-compile" ]]; then
  exit 0
fi

if [[ ${#V81_DEMO_QUESTIONS[@]} -eq 0 ]]; then
  V81_DEMO_QUESTIONS=(
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
  )
fi

V81_DEMO_QUESTION_ARGS=()
for V81_DEMO_QUESTION in "${V81_DEMO_QUESTIONS[@]}"; do
  V81_DEMO_QUESTION_ARGS+=(--question "$V81_DEMO_QUESTION")
done

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ "$V81_DEMO_MODE" == "leakage" ]]; then
  exec env PYTHONPATH=src "$V81_DEMO_PYTHON" \
    -m semantic_3d_chat.evaluation.v81_scene_memory_leakage \
    --config "$V81_DEMO_CONFIG" \
    --scene "$V81_DEMO_SCENE" \
    --base-checkpoint "$V81_DEMO_BASE_CHECKPOINT" \
    --scene-memory "$V81_DEMO_SCENE_MEMORY" \
    --compiler-checkpoint "$V81_DEMO_CONTROL_CHECKPOINT" \
    --probe-bank "$V81_DEMO_PROBE_BANK" \
    --output "$V81_DEMO_LEAKAGE_REPORT" \
    "${V81_DEMO_QUESTION_ARGS[@]}"
fi

V81_DEMO_CLI_ARGS=(
  --config "$V81_DEMO_CONFIG"
  --scene "$V81_DEMO_SCENE"
  --base-checkpoint "$V81_DEMO_BASE_CHECKPOINT"
  --scene-memory "$V81_DEMO_SCENE_MEMORY"
  --grounding-checkpoint "$V81_DEMO_GROUNDING_CHECKPOINT"
  --audit-log "$V81_DEMO_AUDIT_LOG"
  --chat-log "$V81_DEMO_CHAT_LOG"
)
if [[ "$V81_DEMO_MODE" == "finite" ]]; then
  V81_DEMO_CLI_ARGS+=("${V81_DEMO_QUESTION_ARGS[@]}")
fi

exec env PYTHONPATH=src "$V81_DEMO_PYTHON" \
  -m semantic_3d_chat.chat.v81_scene_memory_cli \
  "${V81_DEMO_CLI_ARGS[@]}"
