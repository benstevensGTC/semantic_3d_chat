#!/usr/bin/env bash
set -euo pipefail

STRICT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STRICT_ROOT"

STRICT_PYTHON="${STRICT_PYTHON:-.venv-gemma4/bin/python}"
STRICT_CONFIG="${STRICT_CONFIG:-configs/runtime/gemma4_v54.yaml}"
STRICT_SCENE="${STRICT_SCENE:-scene_000001}"
STRICT_CHECKPOINT="${STRICT_CHECKPOINT:-data_gemma4/runtime/checkpoints/gemma4_v54_release_v1}"
STRICT_MODE="finite"
STRICT_OUTPUT="human"
STRICT_QUESTIONS=()

strict_usage() {
  cat <<'EOF'
Usage: ./scripts/run_strict_fixed_prefix_demo.sh [OPTIONS]

Runs the local fixed-prefix proof: one complete continuous 3D scene prefix is
built before user text and reused byte-identically for every question. The
default checkpoint is runnable but explicitly below the project's acceptance
accuracy gate; it is not silently presented as a successful final model.

Options:
  --check             Validate exact local inputs without loading Gemma
  --interactive       Start audited interactive chat
  --leakage           Hide the oracle directory and prove prefix invariance
  --question TEXT     Ask a finite question (repeatable)
  --human             Print concise human-readable answers (default)
  --json              Print JSON lines for automation
  --scene ID          Opaque scene identifier
  --config PATH       Sanitized runtime config
  --checkpoint PATH   Explicit fixed-prefix checkpoint
  --python PATH       Gemma-capable local Python
  -h, --help          Show this help
EOF
}

strict_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) STRICT_MODE="check"; shift ;;
    --interactive) STRICT_MODE="interactive"; shift ;;
    --leakage) STRICT_MODE="leakage"; shift ;;
    --question) strict_value "$@"; STRICT_QUESTIONS+=("$2"); shift 2 ;;
    --human) STRICT_OUTPUT="human"; shift ;;
    --json) STRICT_OUTPUT="json"; shift ;;
    --scene) strict_value "$@"; STRICT_SCENE="$2"; shift 2 ;;
    --config) strict_value "$@"; STRICT_CONFIG="$2"; shift 2 ;;
    --checkpoint) strict_value "$@"; STRICT_CHECKPOINT="$2"; shift 2 ;;
    --python) strict_value "$@"; STRICT_PYTHON="$2"; shift 2 ;;
    -h|--help) strict_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; strict_usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$STRICT_PYTHON" ]]; then
  echo "Gemma Python is unavailable: $STRICT_PYTHON" >&2
  echo "Run: make setup-gemma4-probe" >&2
  exit 2
fi

if [[ "$STRICT_CHECKPOINT" == "data_gemma4/runtime/checkpoints/gemma4_v54_release_v1" ]]; then
  PYTHONPATH=src "$STRICT_PYTHON" scripts/prepare_demo_runtime.py >/dev/null
  PYTHONPATH=src "$STRICT_PYTHON" scripts/check_demo_artifacts.py --fast >/dev/null || {
    echo "Strict demo artifacts are incomplete; run: make demo-artifacts-check-fast" >&2
    echo "Model weights can be fetched with: make download-gemma4-weights" >&2
    exit 2
  }
fi

"$STRICT_PYTHON" - "$STRICT_CONFIG" "$STRICT_SCENE" "$STRICT_CHECKPOINT" <<'PY'
from pathlib import Path
import re, sys

config, scene, checkpoint = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if not config.is_file() or config.is_symlink():
    raise SystemExit(f"Strict demo config is unavailable or unsafe: {config}")
if re.fullmatch(r"scene_[0-9]{6}", scene) is None:
    raise SystemExit("Strict demo scene ID must be opaque")
if not checkpoint.is_dir() or checkpoint.is_symlink():
    raise SystemExit(f"Strict demo checkpoint is unavailable or unsafe: {checkpoint}")
expected = {"adapter.safetensors", "runtime_metadata.json"}
observed = {item.name for item in checkpoint.iterdir()}
if observed != expected or any((checkpoint / name).is_symlink() for name in expected):
    raise SystemExit(f"Strict demo checkpoint inventory changed: {sorted(observed)}")
map_path = Path("data_gemma4/maps") / scene / "voxel_map.npz"
if not map_path.is_file() or map_path.is_symlink():
    raise SystemExit(f"Strict demo map is unavailable or unsafe: {map_path}")
preview_path = Path("reports/gemma4/figures") / scene / "map_rgb.png"
point_cloud_path = Path("reports/gemma4/figures") / scene / "map_rgb.ply"
for artifact in (preview_path, point_cloud_path):
    if not artifact.is_file() or artifact.is_symlink():
        raise SystemExit(f"Strict demo visual artifact is unavailable or unsafe: {artifact}")
print("Strict fixed-prefix demo preflight: PASS")
print(f"  scene: {scene}")
print(f"  checkpoint: {checkpoint}")
print("  behavioral status: runnable development checkpoint; acceptance gate failed")
print(f"  RGB map preview: {preview_path}")
print(f"  point cloud: {point_cloud_path}")
print(f"  macOS viewer command: open {preview_path} {point_cloud_path}")
print("  finite MCP verification: make mcp-stdio-smoke")
print("  semantic embodied MCP preflight: make gemma4-embodied-mcp-check SCENE=" + scene)
print("  embodied MCP server: make gemma4-embodied-mcp SCENE=" + scene)
PY

if [[ "$STRICT_MODE" == "check" ]]; then
  exit 0
fi

if [[ ${#STRICT_QUESTIONS[@]} -eq 0 ]]; then
  STRICT_QUESTIONS=(
    "Is there a chair?"
    "What color is the bowl?"
    "Is the bowl left or right of the chair?"
  )
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [[ "$STRICT_MODE" == "leakage" ]]; then
  STRICT_ARGS=()
  for question in "${STRICT_QUESTIONS[@]}"; do STRICT_ARGS+=(--question "$question"); done
  exec env PYTHONPATH=src "$STRICT_PYTHON" \
    -m semantic_3d_chat.evaluation.leakage \
    --config "$STRICT_CONFIG" \
    --scene "$STRICT_SCENE" \
    --checkpoint "$STRICT_CHECKPOINT" \
    --output reports/gemma4/metrics/strict_prefix_leakage.json \
    "${STRICT_ARGS[@]}"
fi

STRICT_ARGS=(
  --config "$STRICT_CONFIG"
  --scene "$STRICT_SCENE"
  --checkpoint "$STRICT_CHECKPOINT"
)
if [[ "$STRICT_OUTPUT" == "human" ]]; then
  STRICT_ARGS+=(--human)
else
  STRICT_ARGS+=(--json)
fi
if [[ "$STRICT_MODE" == "finite" ]]; then
  STRICT_ARGS+=(--replace-chat-log)
  for question in "${STRICT_QUESTIONS[@]}"; do STRICT_ARGS+=(--question "$question"); done
fi

exec env PYTHONPATH=src "$STRICT_PYTHON" \
  -m semantic_3d_chat.chat.strict_prefix_cli "${STRICT_ARGS[@]}"
