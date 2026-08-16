#!/usr/bin/env bash
set -euo pipefail

SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SETUP_ROOT"

SETUP_MODE="${1:-all}"
if [[ "$SETUP_MODE" != "all" && "$SETUP_MODE" != "main" && "$SETUP_MODE" != "gemma4" ]]; then
  echo "Usage: bash scripts/setup.sh [all|main|gemma4]" >&2
  exit 2
fi

setup_python() {
  local candidate
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
      'import sys; raise SystemExit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))'; then
      command -v "$candidate"
      return 0
    fi
  done
  echo "Python 3.11 or 3.12 is required." >&2
  return 1
}

setup_main() {
  if command -v uv >/dev/null 2>&1; then
    echo "Installing the locked primary support environment with uv..."
    uv sync --frozen --extra dev --extra mcp
    return
  fi

  local python
  python="$(setup_python)"
  echo "uv is unavailable; creating .venv with $python (fallback is constrained by pyproject.toml, not uv.lock)." >&2
  [[ -x .venv/bin/python ]] || "$python" -m venv .venv
  .venv/bin/python -m pip install -e '.[dev,mcp]'
}

setup_gemma4() {
  local python
  if command -v uv >/dev/null 2>&1; then
    [[ -x .venv-gemma4/bin/python ]] || uv venv --python 3.12 --seed .venv-gemma4
    uv pip install --python .venv-gemma4/bin/python -r requirements-gemma4-probe.txt
    return
  fi

  python="$(setup_python)"
  echo "uv is unavailable; creating the separately pinned Gemma environment with $python." >&2
  [[ -x .venv-gemma4/bin/python ]] || "$python" -m venv .venv-gemma4
  .venv-gemma4/bin/python -m pip install -r requirements-gemma4-probe.txt
}

case "$SETUP_MODE" in
  all)
    setup_main
    setup_gemma4
    ;;
  main) setup_main ;;
  gemma4) setup_gemma4 ;;
esac

echo "Setup complete ($SETUP_MODE). Model weights and prepared demo artifacts remain explicit downloads/copies."
