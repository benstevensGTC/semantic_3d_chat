#!/usr/bin/env bash
set -euo pipefail

# Launch the local Gemma rover backend and attach Blender's real 3D operator UI.
# The backend is a child of this launcher when we start it; closing Blender then
# shuts that child down cleanly.  A matching backend that was already running is
# reused and deliberately left alone.

BLENDER_ROVER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BLENDER_ROVER_ROOT"

BLENDER_ROVER_SCENE="scene_000001"
BLENDER_ROVER_HOST="127.0.0.1"
BLENDER_ROVER_PORT="8770"
BLENDER_ROVER_TIMEOUT="${BLENDER_ROVER_BACKEND_TIMEOUT:-900}"
BLENDER_ROVER_CHECK=false
BLENDER_ROVER_NAVIGATION_CHECKPOINT=""
BLENDER_ROVER_NAVIGATION_CHECKPOINT_SET=false

usage() {
  cat <<'EOF'
Usage: ./scripts/run_blender_rover_demo.sh [options]

Start the local Gemma rover backend, then open the furnished room and rover in
Blender's real 3D viewport. Closing Blender also stops a backend started by this
launcher.

Options:
  --check                 Model-free readiness check; starts no process or GUI
  --scene SCENE_ID        Opaque scene ID (default: scene_000001)
  --host 127.0.0.1        Loopback backend host (only 127.0.0.1 is accepted)
  --port PORT             Loopback backend port (default: 8770)
  --navigation-checkpoint PATH
                          Explicit task-trained Gemma waypoint checkpoint.
                          Omit this flag to preserve the backend's release default.
  --backend-timeout SEC   Maximum local Gemma startup wait (default: 900)
  -h, --help              Show this help
EOF
}

while (($#)); do
  case "$1" in
    --check)
      BLENDER_ROVER_CHECK=true
      shift
      ;;
    --scene)
      [[ $# -ge 2 ]] || { echo "--scene requires a value" >&2; exit 2; }
      BLENDER_ROVER_SCENE="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || { echo "--host requires a value" >&2; exit 2; }
      BLENDER_ROVER_HOST="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a value" >&2; exit 2; }
      BLENDER_ROVER_PORT="$2"
      shift 2
      ;;
    --navigation-checkpoint)
      [[ $# -ge 2 && -n "$2" ]] || {
        echo "--navigation-checkpoint requires a nonempty path" >&2
        exit 2
      }
      [[ "$BLENDER_ROVER_NAVIGATION_CHECKPOINT_SET" == false ]] || {
        echo "--navigation-checkpoint may be supplied only once" >&2
        exit 2
      }
      [[ "$2" != *$'\n'* && "$2" != *$'\r'* ]] || {
        echo "--navigation-checkpoint may not contain line breaks" >&2
        exit 2
      }
      BLENDER_ROVER_NAVIGATION_CHECKPOINT="$2"
      BLENDER_ROVER_NAVIGATION_CHECKPOINT_SET=true
      shift 2
      ;;
    --backend-timeout)
      [[ $# -ge 2 ]] || { echo "--backend-timeout requires a value" >&2; exit 2; }
      BLENDER_ROVER_TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "$BLENDER_ROVER_SCENE" =~ ^scene_[0-9]{6}$ ]] || {
  echo "--scene must match scene_ followed by six digits" >&2
  exit 2
}
[[ "$BLENDER_ROVER_HOST" == "127.0.0.1" ]] || {
  echo "The Blender rover launcher binds only to 127.0.0.1" >&2
  exit 2
}
[[ "$BLENDER_ROVER_PORT" =~ ^[0-9]+$ ]] \
  && ((BLENDER_ROVER_PORT >= 1 && BLENDER_ROVER_PORT <= 65535)) || {
  echo "--port must be an integer between 1 and 65535" >&2
  exit 2
}
[[ "$BLENDER_ROVER_TIMEOUT" =~ ^[0-9]+$ ]] && ((BLENDER_ROVER_TIMEOUT >= 1)) || {
  echo "--backend-timeout must be a positive integer" >&2
  exit 2
}

if [[ "$BLENDER_ROVER_NAVIGATION_CHECKPOINT_SET" == true ]]; then
  BLENDER_ROVER_NAVIGATION_ROOT="$BLENDER_ROVER_NAVIGATION_CHECKPOINT"
  if [[ "$BLENDER_ROVER_NAVIGATION_ROOT" != /* ]]; then
    BLENDER_ROVER_NAVIGATION_ROOT="$BLENDER_ROVER_ROOT/$BLENDER_ROVER_NAVIGATION_ROOT"
  fi
  [[ -d "$BLENDER_ROVER_NAVIGATION_ROOT" && ! -L "$BLENDER_ROVER_NAVIGATION_ROOT" ]] || {
    echo "Explicit navigation checkpoint is not a regular directory: $BLENDER_ROVER_NAVIGATION_CHECKPOINT" >&2
    exit 2
  }
  for BLENDER_ROVER_NAVIGATION_MEMBER in policy.safetensors runtime_metadata.json; do
    [[ -f "$BLENDER_ROVER_NAVIGATION_ROOT/$BLENDER_ROVER_NAVIGATION_MEMBER" \
      && ! -L "$BLENDER_ROVER_NAVIGATION_ROOT/$BLENDER_ROVER_NAVIGATION_MEMBER" ]] || {
      echo "Explicit navigation checkpoint lacks a regular $BLENDER_ROVER_NAVIGATION_MEMBER file" >&2
      exit 2
    }
  done
fi

# Keep this array nonempty: macOS Bash 3.2 with `set -u` rejects expansion of
# an empty array. Appending the optional pair preserves exact argument
# boundaries for checkpoint paths containing whitespace or shell metacharacters.
BLENDER_ROVER_BACKEND_ARGS=(
  --scene "$BLENDER_ROVER_SCENE"
  --host "$BLENDER_ROVER_HOST"
  --port "$BLENDER_ROVER_PORT"
)
if [[ "$BLENDER_ROVER_NAVIGATION_CHECKPOINT_SET" == true ]]; then
  BLENDER_ROVER_BACKEND_ARGS+=(
    --navigation-checkpoint "$BLENDER_ROVER_NAVIGATION_CHECKPOINT"
  )
fi
BLENDER_ROVER_BACKEND_ARGS+=(--no-open)

BLENDER_ROVER_PYTHON="${ROVER_DEMO_PYTHON:-.venv-gemma4/bin/python}"
BLENDER_ROVER_ADDON="$BLENDER_ROVER_ROOT/blender/rover_control_ui.py"
BLENDER_ROVER_COMPACT_ID="${BLENDER_ROVER_SCENE#scene_}"
BLENDER_ROVER_ASSET="$BLENDER_ROVER_ROOT/data/runtime_assets/$BLENDER_ROVER_SCENE/s_${BLENDER_ROVER_COMPACT_ID}.blend"
BLENDER_ROVER_BACKEND_LAUNCHER="$BLENDER_ROVER_ROOT/scripts/run_local_rover_demo.sh"
BLENDER_ROVER_URL="http://$BLENDER_ROVER_HOST:$BLENDER_ROVER_PORT"

resolve_blender() {
  local requested="${BLENDER:-blender}"
  if [[ "$requested" == */* ]]; then
    [[ -x "$requested" ]] || return 1
    printf '%s\n' "$requested"
  else
    command -v "$requested"
  fi
}

BLENDER_ROVER_BLENDER="$(resolve_blender || true)"
[[ -n "$BLENDER_ROVER_BLENDER" ]] || {
  echo "Blender is unavailable; install it or set BLENDER to its executable" >&2
  exit 2
}
[[ -x "$BLENDER_ROVER_PYTHON" ]] || {
  echo "Local Gemma environment is unavailable: $BLENDER_ROVER_PYTHON" >&2
  echo "Install the pinned environments with: make setup" >&2
  exit 2
}
[[ -x "$BLENDER_ROVER_BACKEND_LAUNCHER" ]] || {
  echo "Backend launcher is unavailable: $BLENDER_ROVER_BACKEND_LAUNCHER" >&2
  exit 2
}
command -v curl >/dev/null 2>&1 || {
  echo "curl is required for the loopback rover readiness probe" >&2
  exit 2
}
[[ -f "$BLENDER_ROVER_ADDON" && ! -L "$BLENDER_ROVER_ADDON" ]] || {
  echo "Blender rover UI is unavailable: $BLENDER_ROVER_ADDON" >&2
  exit 2
}
[[ -f "$BLENDER_ROVER_ASSET" && ! -L "$BLENDER_ROVER_ASSET" ]] || {
  echo "Sanitized Blender room is unavailable: $BLENDER_ROVER_ASSET" >&2
  echo "Prepare the local runtime with: make prepare-demo-runtime" >&2
  exit 2
}

# Syntax-check the startup script without importing bpy or writing bytecode.
"$BLENDER_ROVER_PYTHON" - "$BLENDER_ROVER_ADDON" <<'PY'
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
PY

authenticated_preflight_identity() {
  local preflight_json
  if ! preflight_json="$(
    "$BLENDER_ROVER_BACKEND_LAUNCHER" \
      --check \
      "${BLENDER_ROVER_BACKEND_ARGS[@]}"
  )"; then
    echo "The authenticated local rover backend preflight failed." >&2
    return 1
  fi
  printf '%s\n' "$preflight_json" | "$BLENDER_ROVER_PYTHON" -c '
import json
import pathlib
import re
import sys

payload = json.load(sys.stdin)
scene_id = sys.argv[1]
requested_checkpoint = sys.argv[2]
backend = payload.get("backend_preflight")
if (
    payload.get("artifact") != "semantic_3d_chat_local_rover_demo_preflight_v1"
    or payload.get("passed") is not True
    or payload.get("scene_id") != scene_id
    or not isinstance(backend, dict)
    or backend.get("ready") is not True
):
    raise SystemExit("Local rover preflight contract differs")
checkpoint = payload.get("navigation_checkpoint_sha256")
binding = payload.get("gemma_runtime_binding_sha256")
if (
    re.fullmatch(r"[0-9a-f]{64}", str(checkpoint)) is None
    or re.fullmatch(r"[0-9a-f]{64}", str(binding)) is None
    or backend.get("navigation_checkpoint_sha256") != checkpoint
    or backend.get("gemma_runtime_binding_sha256") != binding
):
    raise SystemExit("Local rover preflight navigation identity differs")
if requested_checkpoint:
    observed_checkpoint = backend.get("navigation_checkpoint")
    if not isinstance(observed_checkpoint, str):
        raise SystemExit("Local rover preflight omitted the explicit checkpoint path")
    requested_path = pathlib.Path(requested_checkpoint).expanduser()
    if not requested_path.is_absolute():
        requested_path = pathlib.Path.cwd() / requested_path
    try:
        requested_path = requested_path.resolve(strict=True)
        observed_path = pathlib.Path(observed_checkpoint).expanduser().resolve(strict=True)
    except OSError as error:
        raise SystemExit("Local rover preflight checkpoint path is unavailable") from error
    if observed_path != requested_path:
        raise SystemExit("Local rover preflight used a different navigation checkpoint")
print(checkpoint, binding, sep="\t")
' "$BLENDER_ROVER_SCENE" "$BLENDER_ROVER_NAVIGATION_CHECKPOINT"
}

BLENDER_ROVER_PREFLIGHT_IDENTITY="$(authenticated_preflight_identity)"
IFS=$'\t' read -r BLENDER_ROVER_EXPECTED_NAVIGATION_SHA256 \
  BLENDER_ROVER_EXPECTED_RUNTIME_BINDING_SHA256 <<<"$BLENDER_ROVER_PREFLIGHT_IDENTITY"
[[ "$BLENDER_ROVER_EXPECTED_NAVIGATION_SHA256" =~ ^[0-9a-f]{64}$ \
  && "$BLENDER_ROVER_EXPECTED_RUNTIME_BINDING_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "Authenticated local rover preflight returned an invalid identity" >&2
  exit 2
}

if [[ "$BLENDER_ROVER_CHECK" == true ]]; then
  printf '{"artifact":"semantic_3d_chat_blender_rover_launcher_v1","passed":true,"loads_model":false,"starts_backend":false,"starts_blender":false,"scene_id":"%s","backend_url":"%s","navigation_checkpoint_sha256":"%s","gemma_runtime_binding_sha256":"%s"}\n' \
    "$BLENDER_ROVER_SCENE" "$BLENDER_ROVER_URL" \
    "$BLENDER_ROVER_EXPECTED_NAVIGATION_SHA256" \
    "$BLENDER_ROVER_EXPECTED_RUNTIME_BINDING_SHA256"
  exit 0
fi

BLENDER_ROVER_RUNTIME_DIR="${TMPDIR:-/tmp}/semantic_3d_chat_blender_rover_${UID}_${BLENDER_ROVER_PORT}"
mkdir -p "$BLENDER_ROVER_RUNTIME_DIR"
chmod 700 "$BLENDER_ROVER_RUNTIME_DIR"
BLENDER_ROVER_STATE_FILE="$BLENDER_ROVER_RUNTIME_DIR/state.json"
BLENDER_ROVER_LOG="$BLENDER_ROVER_RUNTIME_DIR/backend.log"
BLENDER_ROVER_PID=""
BLENDER_ROVER_OWNS_BACKEND=false

backend_matches() {
  if ! curl --fail --silent --show-error \
    --connect-timeout 1 --max-time 3 \
    "$BLENDER_ROVER_URL/api/state" \
    --output "$BLENDER_ROVER_STATE_FILE" 2>/dev/null; then
    return 1
  fi
  "$BLENDER_ROVER_PYTHON" - "$BLENDER_ROVER_STATE_FILE" "$BLENDER_ROVER_SCENE" \
    "$BLENDER_ROVER_EXPECTED_NAVIGATION_SHA256" \
    "$BLENDER_ROVER_EXPECTED_RUNTIME_BINDING_SHA256" <<'PY'
import json
import pathlib
import re
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
state = payload.get("state")
control = payload.get("control")
memory = payload.get("scene_memory")
if not isinstance(state, dict) or not isinstance(control, dict) or not isinstance(memory, dict):
    raise SystemExit(1)
if state.get("scene_id") != sys.argv[2]:
    raise SystemExit(1)
if re.fullmatch(r"[0-9a-f]{64}", str(state.get("scene_prefix_hash", ""))) is None:
    raise SystemExit(1)
if state.get("map_version") != 0 or state.get("scan_count") != 0 or state.get("action_count") != 0:
    raise SystemExit(1)
required_control = {
    "local_inference": True,
    "cloud_model_used": False,
    "high_level_natural_language_only": True,
    "task_trained_navigation": True,
    "model_selects_every_waypoint_and_heading": True,
    "model_selects_stop": True,
    "deterministic_route_planner_used": False,
    "fallback_used": False,
    "substitution_applied": False,
    "synthetic_stop_applied": False,
    "untrained_json_backend_enabled": False,
    "static_precomputed_scene_memory": True,
    "camera_control_input": False,
}
if any(control.get(name) is not expected for name, expected in required_control.items()):
    raise SystemExit(1)
if control.get("control_mode") != "actual_local_gemma_model_only_waypoint_policy":
    raise SystemExit(1)
if control.get("navigation_control_mode") != "actual_local_gemma_model_only_waypoint_policy":
    raise SystemExit(1)
if control.get("navigation_checkpoint_sha256") != sys.argv[3]:
    raise SystemExit(1)
if control.get("gemma_runtime_binding_sha256") != sys.argv[4]:
    raise SystemExit(1)
if memory.get("tensor_shape") != [1, 258, 1536]:
    raise SystemExit(1)
if memory.get("map_version") != 0 or memory.get("question_dependent_scene_retrieval") is not False:
    raise SystemExit(1)
if memory.get("all_runtime_voxels_encoded") is not True:
    raise SystemExit(1)
PY
}

port_is_listening() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$BLENDER_ROVER_PORT" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v nc >/dev/null 2>&1; then
    nc -z "$BLENDER_ROVER_HOST" "$BLENDER_ROVER_PORT" >/dev/null 2>&1
  else
    curl --silent --connect-timeout 1 --max-time 1 "$BLENDER_ROVER_URL/" >/dev/null 2>&1
  fi
}

cleanup_backend() {
  local exit_status=$?
  trap - EXIT INT TERM HUP
  if [[ "$BLENDER_ROVER_OWNS_BACKEND" == true && -n "$BLENDER_ROVER_PID" ]] \
    && kill -0 "$BLENDER_ROVER_PID" 2>/dev/null; then
    echo "Stopping the launcher-owned local Gemma rover backend (PID $BLENDER_ROVER_PID)…"
    kill -TERM "$BLENDER_ROVER_PID" 2>/dev/null || true
    local attempt
    for attempt in {1..50}; do
      kill -0 "$BLENDER_ROVER_PID" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "$BLENDER_ROVER_PID" 2>/dev/null; then
      kill -KILL "$BLENDER_ROVER_PID" 2>/dev/null || true
    fi
    wait "$BLENDER_ROVER_PID" 2>/dev/null || true
  fi
  exit "$exit_status"
}
trap cleanup_backend EXIT
trap 'exit 130' INT TERM HUP

if backend_matches; then
  echo "Reusing the matching local Gemma rover backend at $BLENDER_ROVER_URL."
elif port_is_listening; then
  echo "Port $BLENDER_ROVER_PORT is occupied by a different or unready service." >&2
  echo "Choose another port, for example: make rover-3d ROVER_DEMO_PORT=8771" >&2
  exit 2
else
  : >"$BLENDER_ROVER_LOG"
  echo "Starting local Gemma and the continuous 3D scene memory; first load can take several minutes…"
  "$BLENDER_ROVER_BACKEND_LAUNCHER" \
    "${BLENDER_ROVER_BACKEND_ARGS[@]}" >"$BLENDER_ROVER_LOG" 2>&1 &
  BLENDER_ROVER_PID=$!
  BLENDER_ROVER_OWNS_BACKEND=true

  BLENDER_ROVER_DEADLINE=$((SECONDS + BLENDER_ROVER_TIMEOUT))
  until backend_matches; do
    if ! kill -0 "$BLENDER_ROVER_PID" 2>/dev/null; then
      echo "The local Gemma rover backend exited before it became ready." >&2
      echo "Backend log: $BLENDER_ROVER_LOG" >&2
      tail -n 30 "$BLENDER_ROVER_LOG" >&2 || true
      exit 1
    fi
    if ((SECONDS >= BLENDER_ROVER_DEADLINE)); then
      echo "Timed out after ${BLENDER_ROVER_TIMEOUT}s waiting for the local rover backend." >&2
      echo "Backend log: $BLENDER_ROVER_LOG" >&2
      tail -n 30 "$BLENDER_ROVER_LOG" >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "Local Gemma rover backend is ready at $BLENDER_ROVER_URL."
fi

echo "Opening the furnished room in Blender's real 3D viewport."
echo "Use the Gemma Rover panel in the 3D Viewport sidebar (press N if hidden)."
"$BLENDER_ROVER_BLENDER" \
  --disable-autoexec \
  "$BLENDER_ROVER_ASSET" \
  --python "$BLENDER_ROVER_ADDON" \
  -- \
  --backend-url "$BLENDER_ROVER_URL" \
  --project-root "$BLENDER_ROVER_ROOT"
