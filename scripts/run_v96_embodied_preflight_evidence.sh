#!/usr/bin/env bash
set -euo pipefail

V96_ROBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V96_ROBOT_ROOT"

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    cat <<'EOF'
Usage: ./scripts/run_v96_embodied_preflight_evidence.sh

Authenticates the promoted V96 release, runs the model-free embodied MCP
preflight, and creates a release-bound forbidden-read evidence receipt. Runtime
paths can be overridden with the documented V96_ROBOT_* environment variables.
This does not execute robot actions or measure navigation success.
EOF
    exit 0
  fi
  echo "This wrapper accepts no positional arguments; use --help for details." >&2
  exit 2
fi

V96_ROBOT_PYTHON="${V96_ROBOT_PYTHON:-.venv-gemma4/bin/python}"
V96_ROBOT_SCENE="${V96_ROBOT_SCENE:-scene_000001}"
V96_ROBOT_CONFIG="${V96_ROBOT_CONFIG:-configs/runtime/embodied_live.yaml}"
V96_ROBOT_CHECKPOINT="${V96_ROBOT_CHECKPOINT:-reports/gemma4/artifacts/v85_strict_runtime_candidate}"
V96_ROBOT_ASSET="${V96_ROBOT_ASSET:-data/runtime_assets/${V96_ROBOT_SCENE}/${V96_ROBOT_SCENE/scene_/s_}.blend}"
V96_ROBOT_STATE="${V96_ROBOT_STATE:-data_gemma4/checkpoints/robot_state_numeric_v1}"
V96_ROBOT_HOOK="${V96_ROBOT_HOOK:-configs/runtime/gemma4_v96_explicit_candidate_mcp_bridge.yaml}"
V96_ROBOT_MEMORY="${V96_ROBOT_MEMORY:-reports/gemma4/artifacts/v85_strict_runtime_candidate_memory/${V96_ROBOT_SCENE}}"
V96_ROBOT_MAP="${V96_ROBOT_MAP:-data_gemma4/robot/v96_explicit_candidate/${V96_ROBOT_SCENE}/semantic_map.npz}"
V96_ROBOT_SCANS="${V96_ROBOT_SCANS:-data_gemma4/robot/v96_explicit_candidate/${V96_ROBOT_SCENE}/scans}"
V96_ROBOT_AUDIT="${V96_ROBOT_AUDIT:-reports/gemma4/metrics/v96_explicit_candidate_embodied_mcp_check_${V96_ROBOT_SCENE}.json}"
V96_ROBOT_EVIDENCE="${V96_ROBOT_EVIDENCE:-reports/gemma4/metrics/gemma4_v96_embodied_mcp_preflight_evidence.json}"

if [[ ! -x "$V96_ROBOT_PYTHON" ]]; then
  echo "V96 Gemma Python is unavailable: $V96_ROBOT_PYTHON" >&2
  exit 2
fi
if [[ -e "$V96_ROBOT_EVIDENCE" || -L "$V96_ROBOT_EVIDENCE" ]]; then
  echo "Authenticating existing create-once V96 robot receipt: $V96_ROBOT_EVIDENCE" >&2
  exec env PYTHONPATH=src TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$V96_ROBOT_PYTHON" -m semantic_3d_chat.evaluation.v96_final_reporting \
      robot-evidence-check \
      --python "$V96_ROBOT_PYTHON" \
      --robot-evidence "$V96_ROBOT_EVIDENCE"
fi
if [[ ! -f "$V96_ROBOT_ASSET" ]]; then
  echo "V96 sanitized runtime asset is unavailable: $V96_ROBOT_ASSET" >&2
  exit 2
fi

export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

"$V96_ROBOT_PYTHON" -m semantic_3d_chat.mcp_server.server \
  --config "$V96_ROBOT_CONFIG" \
  --scene "$V96_ROBOT_SCENE" \
  --checkpoint "$V96_ROBOT_CHECKPOINT" \
  --runtime-asset "$V96_ROBOT_ASSET" \
  --robot-state-checkpoint "$V96_ROBOT_STATE" \
  --v96-candidate-bridge-hook "$V96_ROBOT_HOOK" \
  --v96-scene-memory "$V96_ROBOT_MEMORY" \
  --allow-explicit-v96-candidate \
  --persistent-map "$V96_ROBOT_MAP" \
  --scan-output-directory "$V96_ROBOT_SCANS" \
  --audit-report "$V96_ROBOT_AUDIT" \
  --check \
| "$V96_ROBOT_PYTHON" -m semantic_3d_chat.evaluation.v96_final_reporting \
    robot-evidence \
    --python "$V96_ROBOT_PYTHON" \
    --access-audit "$V96_ROBOT_AUDIT" \
    --robot-evidence "$V96_ROBOT_EVIDENCE"
