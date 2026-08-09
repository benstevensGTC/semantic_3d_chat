#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "Python environment missing; run: make setup" >&2
  exit 2
fi

.venv/bin/python scripts/doctor.py "$@"
