#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/default.yaml"
CONFIG_EXPLICIT=0
SCENE="scene_000001"
CHECKPOINT=""
MODE="interactive"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_full_demo.sh [OPTIONS]

Options:
  --config PATH          YAML configuration (default: legacy configs/default.yaml)
  --scene ID             Opaque scene ID (default: scene_000001)
  --checkpoint PATH      Adapter checkpoint; required explicitly for Gemma
  --check                Offline static preflight only; no model inference or device tensor
  --non-interactive      Run the finite oracle-isolation/prefix-invariance inference check
  -h, --help             Show this help
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) require_value "$@"; CONFIG="$2"; CONFIG_EXPLICIT=1; shift 2 ;;
    --scene) require_value "$@"; SCENE="$2"; shift 2 ;;
    --checkpoint) require_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --check) MODE="check"; shift ;;
    --non-interactive) MODE="leakage"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

LANGUAGE_BACKEND="$(PYTHONPATH=src .venv/bin/python -c 'import sys; from semantic_3d_chat.config import load_config; print(load_config(sys.argv[1]).get("language", {}).get("backend", "auto"))' "$CONFIG")"
DEMO_PYTHON=".venv/bin/python"
PROMOTION_ARG=""
if [[ "$LANGUAGE_BACKEND" == "gemma4" ]]; then
  DEMO_PYTHON=".venv-gemma4/bin/python"
  PROMOTION_ARG="--require-promotion"
  if [[ "$CONFIG_EXPLICIT" -ne 1 || -z "$CHECKPOINT" ]]; then
    echo "No accepted primary Gemma demo is configured." >&2
    echo "Supply both --config and --checkpoint after that exact pair has an accepted promotion.json record." >&2
    exit 3
  fi
fi
if [[ ! -x "$DEMO_PYTHON" ]]; then
  echo "Required Python environment is missing: $DEMO_PYTHON" >&2
  exit 2
fi

RENDER_MANIFEST="$(PYTHONPATH=src "$DEMO_PYTHON" -c 'import sys; from semantic_3d_chat.config import load_config, project_path; print(project_path(load_config(sys.argv[1]), "rendered", sys.argv[2], "manifest.json"))' "$CONFIG" "$SCENE")"
MAP_PATH="$(PYTHONPATH=src "$DEMO_PYTHON" -c 'import sys; from semantic_3d_chat.config import load_config, project_path; print(project_path(load_config(sys.argv[1]), "maps", sys.argv[2], "voxel_map.npz"))' "$CONFIG" "$SCENE")"
REPORTS_ROOT="$(PYTHONPATH=src "$DEMO_PYTHON" -c 'import sys; from semantic_3d_chat.config import load_config, reports_root; print(reports_root(load_config(sys.argv[1])))' "$CONFIG")"
RENDER_ROOT="$(dirname "$RENDER_MANIFEST")"

if [[ "$MODE" == "check" ]]; then
  ./scripts/doctor.sh \
    --skip-mps-smoke \
    --output reports/metrics/demo_doctor.json
else
  ./scripts/doctor.sh
fi

if [[ "$MODE" == "check" ]]; then
  if [[ -n "$CHECKPOINT" ]]; then
    PYTHONPATH=src "$DEMO_PYTHON" scripts/demo_check.py \
      --config "$CONFIG" \
      --scene "$SCENE" \
      --checkpoint "$CHECKPOINT" \
      ${PROMOTION_ARG:+$PROMOTION_ARG}
  else
    PYTHONPATH=src "$DEMO_PYTHON" scripts/demo_check.py \
      --config "$CONFIG" \
      --scene "$SCENE" \
      ${PROMOTION_ARG:+$PROMOTION_ARG}
  fi
  exit 0
fi

if [[ ! -f "$RENDER_MANIFEST" ]]; then
  make CONFIG="$CONFIG" SCENE="$SCENE" render-smoke-scan
fi
if [[ ! -f "$MAP_PATH" ]]; then
  if [[ "$LANGUAGE_BACKEND" == "gemma4" ]]; then
    make GEMMA4_CONFIG="$CONFIG" SCENE="$SCENE" build-gemma4-map
  else
    make CONFIG="$CONFIG" SCENE="$SCENE" build-smoke-map
  fi
fi
if [[ -z "$CHECKPOINT" ]]; then
  if ! CHECKPOINT="$(PYTHONPATH=src "$DEMO_PYTHON" scripts/demo_check.py \
    --config "$CONFIG" --scene "$SCENE" --resolve-checkpoint)"; then
    echo "A compatible trained checkpoint is required. Run: make generate-dataset train" >&2
    exit 3
  fi
fi

PYTHONPATH=src "$DEMO_PYTHON" scripts/demo_check.py \
  --config "$CONFIG" \
  --scene "$SCENE" \
  --checkpoint "$CHECKPOINT" \
  ${PROMOTION_ARG:+$PROMOTION_ARG}

echo "Overview render: $RENDER_ROOT/p_000000.png"
echo "Map preview: $REPORTS_ROOT/figures/$SCENE/map_rgb.png"
echo "Point cloud: $REPORTS_ROOT/figures/$SCENE/map_rgb.ply"
echo "Open the visual previews: open $RENDER_ROOT/p_000000.png $REPORTS_ROOT/figures/$SCENE/map_rgb.png"
echo "Checkpoint: $CHECKPOINT"
if [[ "$LANGUAGE_BACKEND" == "gemma4" ]]; then
  echo "Gemma checkpoint is explicitly promoted for this config; starting the audited CLI."
else
  echo "Legacy local visual chat UI: make CONFIG=$CONFIG SCENE=$SCENE web"
fi
echo "Robot MCP server: make CONFIG=$CONFIG SCENE=$SCENE mcp"
if [[ "$MODE" == "leakage" ]]; then
  PYTHONPATH=src "$DEMO_PYTHON" -m semantic_3d_chat.evaluation.leakage \
    --config "$CONFIG" \
    --scene "$SCENE" \
    --checkpoint "$CHECKPOINT"
else
  PYTHONPATH=src "$DEMO_PYTHON" -m semantic_3d_chat.chat.cli \
    --config "$CONFIG" \
    --scene "$SCENE" \
    --checkpoint "$CHECKPOINT"
fi
