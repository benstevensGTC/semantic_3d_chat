#!/usr/bin/env bash
# The Gemma-side measurements, run one at a time: they share the GPU with
# anything else on this machine, and overlapping them only makes both slower.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv-gemma4/bin/python

echo "=== localization: which cell does the object occupy ==="
$PY scripts/lens_eval_rope3d_locate.py 2>&1 | tr '\r' '\n' | grep -vE "Loading weights|^$" | tail -30

echo "=== relations: which is higher, which is nearer ==="
$PY scripts/lens_eval_rope3d_relations.py --per-room 4 2>&1 | tr '\r' '\n' | grep -vE "Loading weights|^$" | tail -40
echo "gemma evals complete"
