#!/usr/bin/env bash
set -euo pipefail

RESEARCH_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RESEARCH_PROJECT_ROOT"

RESEARCH_PYTHON="${RESEARCH_PYTHON:-.venv-gemma4/bin/python}"
RESEARCH_CONFIG="${RESEARCH_CONFIG:-configs/runtime/gemma4_v54.yaml}"
RESEARCH_SCENE="${RESEARCH_SCENE:-scene_000001}"
RESEARCH_BASE_CHECKPOINT="${RESEARCH_BASE_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v54_semantic_greedy_gate/update_000}"
RESEARCH_CONTROL_CHECKPOINT="${RESEARCH_CONTROL_CHECKPOINT:-data_gemma4/checkpoints/gemma4_v66b_allrow_always_on_control}"
RESEARCH_MODE="finite"
RESEARCH_QUESTIONS=()

research_usage() {
  cat <<'EOF'
Usage: ./scripts/run_research_demo.sh [OPTIONS]

Runs the gated local static vertical slice: prepared complete-image Gemma 4
features -> fused 3D map -> immutable 256-token scene prefix -> sealed schema-7
continuous controller -> local Gemma decoder. The command fails closed until the
schema-7 training/runtime gate has published its exact two-file checkpoint.

Options:
  --check                     Validate prepared artifacts without model inference
  --interactive               Start an interactive local chat
  --leakage                   Run oracle/training-removal and prefix-invariance checks
  --question TEXT             Ask a finite question (repeatable)
  --scene ID                  Opaque scene identifier
  --config PATH               Sanitized runtime configuration
  --base-checkpoint PATH      Frozen full-scene checkpoint
  --control-checkpoint PATH   Sealed schema-7 checkpoint
  --python PATH               Gemma-capable local Python
  -h, --help                  Show this help

Environment equivalents are RESEARCH_PYTHON, RESEARCH_CONFIG, RESEARCH_SCENE,
RESEARCH_BASE_CHECKPOINT, and RESEARCH_CONTROL_CHECKPOINT. Inference is offline.
EOF
}

research_require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) RESEARCH_MODE="check"; shift ;;
    --interactive) RESEARCH_MODE="interactive"; shift ;;
    --leakage) RESEARCH_MODE="leakage"; shift ;;
    --question)
      research_require_value "$@"
      RESEARCH_QUESTIONS+=("$2")
      shift 2
      ;;
    --scene)
      research_require_value "$@"
      RESEARCH_SCENE="$2"
      shift 2
      ;;
    --config)
      research_require_value "$@"
      RESEARCH_CONFIG="$2"
      shift 2
      ;;
    --base-checkpoint)
      research_require_value "$@"
      RESEARCH_BASE_CHECKPOINT="$2"
      shift 2
      ;;
    --control-checkpoint)
      research_require_value "$@"
      RESEARCH_CONTROL_CHECKPOINT="$2"
      shift 2
      ;;
    --python)
      research_require_value "$@"
      RESEARCH_PYTHON="$2"
      shift 2
      ;;
    -h|--help) research_usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; research_usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$RESEARCH_PYTHON" ]]; then
  echo "Research demo unavailable: Gemma Python is not executable: $RESEARCH_PYTHON" >&2
  exit 2
fi

# This preflight deliberately reads only numeric runtime artifacts and checkpoint
# metadata. The actual runtime performs its own stricter loader and file audit.
"$RESEARCH_PYTHON" - \
  "$RESEARCH_CONFIG" \
  "$RESEARCH_SCENE" \
  "$RESEARCH_BASE_CHECKPOINT" \
  "$RESEARCH_CONTROL_CHECKPOINT" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"Research demo unavailable: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


config = Path(sys.argv[1])
scene = sys.argv[2]
base = Path(sys.argv[3])
control = Path(sys.argv[4])
if not config.is_file():
    fail(f"runtime config is missing: {config}")
if not scene.startswith("scene_") or not scene.removeprefix("scene_").isdigit():
    fail("scene ID must use the opaque scene_NNNNNN form")
if not base.is_dir() or not all(
    (base / name).is_file()
    for name in ("adapter.safetensors", "metadata.json", "runtime_metadata.json")
):
    fail(f"base checkpoint is incomplete: {base}")
map_path = Path("data_gemma4/maps") / scene / "voxel_map.npz"
if not map_path.is_file():
    fail(f"prepared fused map is missing: {map_path}")
if not control.is_dir():
    fail(
        "sealed V66b schema-7 checkpoint has not been published; "
        f"expected {control}"
    )
if control.is_symlink():
    fail("schema-7 checkpoint directory must not be a symbolic link")
inventory = sorted(item.name for item in control.iterdir())
expected_inventory = ["control.safetensors", "runtime_metadata.json"]
if inventory != expected_inventory:
    fail(f"schema-7 checkpoint inventory is not exact: {inventory}")
if any((control / name).is_symlink() for name in expected_inventory):
    fail("schema-7 checkpoint files must not be symbolic links")
try:
    metadata = json.loads((control / "runtime_metadata.json").read_text())
except (OSError, ValueError) as error:
    fail(f"schema-7 metadata is unreadable: {error}")
required = {
    "schema_version": 7,
    "architecture": "always_on_teacher_basis_full_scene_control_v7",
    "always_on_continuous_control": True,
    "complete_scene_prefix_required": True,
    "question_dependent_scene_retrieval": False,
    "environmental_text_inputs": [],
    "training_answers_runtime_loaded": False,
    "answer_class_codebook_runtime_loaded": False,
    "saved_runtime_training_gate_required": True,
    "saved_runtime_training_gate_passed": True,
}
wrong = [key for key, expected in required.items() if metadata.get(key) != expected]
if wrong:
    fail(f"schema-7 sealed runtime contract failed fields: {wrong}")
weights = control / "control.safetensors"
observed_hash = sha256(weights)
if metadata.get("weights_sha256") != observed_hash:
    fail("schema-7 weights hash does not match runtime metadata")
attestation = metadata.get("saved_runtime_training_gate_attestation_sha256")
if not isinstance(attestation, str) or re.fullmatch(r"[0-9a-f]{64}", attestation) is None:
    fail("schema-7 saved-runtime gate attestation is missing")
print("Research demo preflight: PASS")
print(f"  scene: {scene}")
print(f"  schema-7 weights sha256: {observed_hash}")
print(f"  RGB map preview: reports/gemma4/figures/{scene}/map_rgb.png")
print(f"  point cloud: reports/gemma4/figures/{scene}/map_rgb.ply")
PY

if [[ "$RESEARCH_MODE" == "check" ]]; then
  exit 0
fi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

RESEARCH_DEMO_ARGS=(
  --python "$RESEARCH_PYTHON"
  --config "$RESEARCH_CONFIG"
  --scene "$RESEARCH_SCENE"
  --base-checkpoint "$RESEARCH_BASE_CHECKPOINT"
  --control-checkpoint "$RESEARCH_CONTROL_CHECKPOINT"
)
case "$RESEARCH_MODE" in
  interactive) RESEARCH_DEMO_ARGS+=(--interactive) ;;
  leakage) RESEARCH_DEMO_ARGS+=(--leakage) ;;
esac
for RESEARCH_QUESTION in "${RESEARCH_QUESTIONS[@]}"; do
  RESEARCH_DEMO_ARGS+=(--question "$RESEARCH_QUESTION")
done

exec ./scripts/run_schema7_question_control_demo.sh "${RESEARCH_DEMO_ARGS[@]}"
