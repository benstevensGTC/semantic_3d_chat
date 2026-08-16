#!/usr/bin/env bash
set -euo pipefail

V81_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$V81_ROOT"

V81_PYTHON="${V81_PYTHON:-.venv/bin/python}"
V81_CONFIG="configs/experiments/gemma4_v81_strict_fixed_prefix_reader.yaml"
V81_MODE="check"

if [[ $# -gt 1 ]]; then
  echo "Usage: ./scripts/run_v81_strict_fixed_prefix_reader.sh [--check|--seal]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    --check) V81_MODE="check" ;;
    --seal) V81_MODE="seal" ;;
    *)
      echo "Usage: ./scripts/run_v81_strict_fixed_prefix_reader.sh [--check|--seal]" >&2
      exit 2
      ;;
  esac
fi

if [[ ! -x "$V81_PYTHON" ]]; then
  echo "V81 Python is unavailable: $V81_PYTHON" >&2
  exit 2
fi

# This launcher is intentionally model-free and CPU-only. It has no fit,
# Transformers model load, MPS, checkpoint, predictor, or scorer entry point.
V81_ARGS=(--config "$V81_CONFIG")
if [[ "$V81_MODE" == "seal" ]]; then
  V81_ARGS+=(--write-preregistration --write-cpu-preflight)
fi
PYTHONPATH=src "$V81_PYTHON" scripts/preflight_v81_strict_fixed_prefix_reader.py "${V81_ARGS[@]}"
