#!/usr/bin/env bash
set -euo pipefail

EMBODIED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EMBODIED_ROOT"

EMBODIED_PYTHON="${EMBODIED_PYTHON:-.venv-gemma4/bin/python}"
EMBODIED_CONFIG="${EMBODIED_CONFIG:-configs/runtime/embodied_live.yaml}"
EMBODIED_CONTROL_CONFIG="${EMBODIED_CONTROL_CONFIG:-configs/runtime/gemma4_v56_question_control.yaml}"
EMBODIED_SCENE="${EMBODIED_SCENE:-scene_000001}"
EMBODIED_BASE_CHECKPOINT="${EMBODIED_BASE_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v54_release_v1}"
# The promoted controller consumes the same complete, question-independent
# environmental prefix and is refreshed transactionally after every scan.
# Callers can still opt out by explicitly exporting an empty value.
EMBODIED_CONTROL_CHECKPOINT="${EMBODIED_CONTROL_CHECKPOINT-data_gemma4/runtime/checkpoints/gemma4_v75_nll_control_release_v1}"
EMBODIED_GROUNDING_CHECKPOINT="${EMBODIED_GROUNDING_CHECKPOINT-}"
EMBODIED_RUNTIME_ASSET="${EMBODIED_RUNTIME_ASSET:-data/runtime_assets/${EMBODIED_SCENE}/${EMBODIED_SCENE/scene_/s_}.blend}"
EMBODIED_ROBOT_STATE_CHECKPOINT="${EMBODIED_ROBOT_STATE_CHECKPOINT:-data_gemma4/checkpoints/robot_state_numeric_v1}"
EMBODIED_NAVIGATION_POLICY_CHECKPOINT="${EMBODIED_NAVIGATION_POLICY_CHECKPOINT-}"
EMBODIED_NAVIGATION_POLICY_VERSION="${EMBODIED_NAVIGATION_POLICY_VERSION:-3}"
EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT="${EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT-}"

if [[ ! -x "$EMBODIED_PYTHON" ]]; then
  echo "Gemma Python environment is unavailable: $EMBODIED_PYTHON" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  PYTHONPATH=src "$EMBODIED_PYTHON" \
    -m semantic_3d_chat.robot.conversation_cli --help
  exit 0
fi
for EMBODIED_INPUT in \
  "$EMBODIED_CONFIG" \
  "$EMBODIED_CONTROL_CONFIG" \
  "$EMBODIED_BASE_CHECKPOINT" \
  "$EMBODIED_RUNTIME_ASSET" \
  "$EMBODIED_ROBOT_STATE_CHECKPOINT"; do
  if [[ ! -e "$EMBODIED_INPUT" ]]; then
    echo "Required embodied-chat input is unavailable: $EMBODIED_INPUT" >&2
    exit 2
  fi
done
if [[ -n "$EMBODIED_CONTROL_CHECKPOINT" && ! -e "$EMBODIED_CONTROL_CHECKPOINT" ]]; then
  echo "Optional enhanced-readout checkpoint is unavailable: $EMBODIED_CONTROL_CHECKPOINT" >&2
  exit 2
fi
if [[ -n "$EMBODIED_GROUNDING_CHECKPOINT" && ! -e "$EMBODIED_GROUNDING_CHECKPOINT" ]]; then
  echo "Optional V78 grounding checkpoint is unavailable: $EMBODIED_GROUNDING_CHECKPOINT" >&2
  exit 2
fi
if [[ -n "$EMBODIED_GROUNDING_CHECKPOINT" && -z "$EMBODIED_CONTROL_CHECKPOINT" ]]; then
  echo "Optional V78 grounding currently requires the V75 controlled-chat wrapper" >&2
  exit 2
fi
if [[ -n "$EMBODIED_NAVIGATION_POLICY_CHECKPOINT" && ! -e "$EMBODIED_NAVIGATION_POLICY_CHECKPOINT" ]]; then
  echo "Learned navigation-policy checkpoint is unavailable: $EMBODIED_NAVIGATION_POLICY_CHECKPOINT" >&2
  exit 2
fi
if [[ -n "$EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT" && ! -e "$EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT" ]]; then
  echo "Promoted Gemma tool-decoder checkpoint is unavailable: $EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT" >&2
  exit 2
fi
if [[ -n "$EMBODIED_NAVIGATION_POLICY_CHECKPOINT" && -n "$EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT" ]]; then
  echo "Select either a learned navigation policy or the promoted Gemma tool decoder, not both" >&2
  exit 2
fi
if [[ "$EMBODIED_NAVIGATION_POLICY_VERSION" != "1" && "$EMBODIED_NAVIGATION_POLICY_VERSION" != "3" && "$EMBODIED_NAVIGATION_POLICY_VERSION" != "4" ]]; then
  echo "Learned navigation-policy version must be 1, 3, or 4: $EMBODIED_NAVIGATION_POLICY_VERSION" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

EMBODIED_ARGS=(
  --config "$EMBODIED_CONFIG"
  --control-runtime-config "$EMBODIED_CONTROL_CONFIG"
  --scene "$EMBODIED_SCENE"
  --base-checkpoint "$EMBODIED_BASE_CHECKPOINT"
  --runtime-asset "$EMBODIED_RUNTIME_ASSET"
  --robot-state-checkpoint "$EMBODIED_ROBOT_STATE_CHECKPOINT"
)
if [[ -n "$EMBODIED_CONTROL_CHECKPOINT" ]]; then
  EMBODIED_ARGS+=(--control-checkpoint "$EMBODIED_CONTROL_CHECKPOINT")
fi
if [[ -n "$EMBODIED_GROUNDING_CHECKPOINT" ]]; then
  EMBODIED_ARGS+=(--grounding-checkpoint "$EMBODIED_GROUNDING_CHECKPOINT")
fi
if [[ -n "$EMBODIED_NAVIGATION_POLICY_CHECKPOINT" ]]; then
  EMBODIED_ARGS+=(
    --navigation-policy-checkpoint "$EMBODIED_NAVIGATION_POLICY_CHECKPOINT"
    --navigation-policy-version "$EMBODIED_NAVIGATION_POLICY_VERSION"
  )
fi
if [[ -n "$EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT" ]]; then
  EMBODIED_ARGS+=(
    --gemma-tool-decoder-checkpoint "$EMBODIED_GEMMA_TOOL_DECODER_CHECKPOINT"
  )
fi

PYTHONPATH=src "$EMBODIED_PYTHON" \
  -m semantic_3d_chat.robot.conversation_cli "${EMBODIED_ARGS[@]}" "$@"
