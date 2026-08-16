#!/usr/bin/env bash
set -euo pipefail

STRICT_WEB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STRICT_WEB_ROOT"

STRICT_WEB_PYTHON="${STRICT_WEB_PYTHON:-.venv-gemma4/bin/python}"
STRICT_WEB_CONFIG="${STRICT_WEB_CONFIG:-configs/runtime/gemma4_v54.yaml}"
STRICT_WEB_SCENE="${STRICT_WEB_SCENE:-scene_000001}"
STRICT_WEB_CHECKPOINT="${STRICT_WEB_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v54_release_v1}"
STRICT_WEB_HOST="${STRICT_WEB_HOST:-127.0.0.1}"
STRICT_WEB_PORT="${STRICT_WEB_PORT:-8766}"
STRICT_WEB_AUDIT="${STRICT_WEB_AUDIT:-}"
STRICT_WEB_DEFAULT_AUDIT="reports/gemma4/metrics/strict_prefix_web_access.json"
STRICT_WEB_CHECK=""

strict_web_usage() {
  cat <<'EOF'
Usage: ./scripts/run_strict_fixed_prefix_web.sh [OPTIONS]

Starts a loopback-only browser UI for the explicit V54 fixed-prefix baseline.
The complete continuous environment prefix is built before HTTP starts and is
reused exactly for every question. Only the fused-map raster is shown to the
human; it is never supplied to Gemma.

Options:
  --check             Validate local artifacts without loading Gemma
  --scene ID          Opaque scene identifier
  --config PATH       Explicit V54 sanitized runtime config
  --checkpoint PATH   Explicit V54 fixed-prefix checkpoint
  --host HOST         Loopback host only (default 127.0.0.1)
  --port PORT         Local port (default 8766)
  --audit-log PATH    File-access audit destination (serve mode has a default)
  --python PATH       Gemma-capable local Python
  -h, --help          Show this help
EOF
}

strict_web_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) STRICT_WEB_CHECK="--check"; shift ;;
    --scene) strict_web_value "$@"; STRICT_WEB_SCENE="$2"; shift 2 ;;
    --config) strict_web_value "$@"; STRICT_WEB_CONFIG="$2"; shift 2 ;;
    --checkpoint) strict_web_value "$@"; STRICT_WEB_CHECKPOINT="$2"; shift 2 ;;
    --host) strict_web_value "$@"; STRICT_WEB_HOST="$2"; shift 2 ;;
    --port) strict_web_value "$@"; STRICT_WEB_PORT="$2"; shift 2 ;;
    --audit-log) strict_web_value "$@"; STRICT_WEB_AUDIT="$2"; shift 2 ;;
    --python) strict_web_value "$@"; STRICT_WEB_PYTHON="$2"; shift 2 ;;
    -h|--help) strict_web_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; strict_web_usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$STRICT_WEB_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $STRICT_WEB_PYTHON" >&2
  echo "Run: make setup-gemma4-probe" >&2
  exit 2
fi

if [[ "$STRICT_WEB_CHECKPOINT" == "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" ]]; then
  PYTHONPATH=src "$STRICT_WEB_PYTHON" scripts/prepare_demo_runtime.py >/dev/null
  PYTHONPATH=src "$STRICT_WEB_PYTHON" scripts/check_demo_artifacts.py --fast >/dev/null || {
    echo "Strict web artifacts are incomplete; run: make demo-artifacts-check-fast" >&2
    echo "Model weights can be fetched with: make download-gemma4-weights" >&2
    exit 2
  }
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

STRICT_WEB_ARGS=(
  --config "$STRICT_WEB_CONFIG"
  --scene "$STRICT_WEB_SCENE"
  --checkpoint "$STRICT_WEB_CHECKPOINT"
  --host "$STRICT_WEB_HOST"
  --port "$STRICT_WEB_PORT"
)
if [[ -n "$STRICT_WEB_AUDIT" ]]; then
  STRICT_WEB_ARGS+=(--audit-log "$STRICT_WEB_AUDIT")
elif [[ -z "$STRICT_WEB_CHECK" ]]; then
  STRICT_WEB_ARGS+=(--audit-log "$STRICT_WEB_DEFAULT_AUDIT")
fi
if [[ -n "$STRICT_WEB_CHECK" ]]; then
  STRICT_WEB_ARGS+=("$STRICT_WEB_CHECK")
fi

exec env PYTHONPATH=src "$STRICT_WEB_PYTHON" \
  -m semantic_3d_chat.chat.strict_prefix_web "${STRICT_WEB_ARGS[@]}"
