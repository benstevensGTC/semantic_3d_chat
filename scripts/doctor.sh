#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x .venv/bin/python ]]; then
  DOCTOR_PYTHON=.venv/bin/python
else
  DOCTOR_PYTHON=""
  for DOCTOR_CANDIDATE in python3.12 python3.11 python3; do
    if command -v "$DOCTOR_CANDIDATE" >/dev/null 2>&1 && \
      "$DOCTOR_CANDIDATE" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
      DOCTOR_PYTHON="$(command -v "$DOCTOR_CANDIDATE")"
      break
    fi
  done
  if [[ -z "$DOCTOR_PYTHON" ]]; then
    echo "Python 3.10+ is unavailable; install Python 3.11 or 3.12, then run: make setup" >&2
    exit 2
  fi
  echo "Project environment is not installed; running bootstrap-only machine inspection with $DOCTOR_PYTHON." >&2
  echo "Install both local environments with: make setup" >&2
fi

"$DOCTOR_PYTHON" scripts/doctor.py "$@"
