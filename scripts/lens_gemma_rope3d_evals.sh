#!/usr/bin/env bash
# The Gemma-side measurements, run one at a time: they share the GPU with
# anything else on this machine, and overlapping them only makes both slower.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv-gemma4/bin/python

# One corpus at a time. Primitive and asset rooms answer different questions,
# and averaging them would answer neither.
for corpus in asset ""; do
  label=${corpus:-primitive}
  suffix=${corpus:+_$corpus}
  flag=${corpus:+--room-prefix $corpus}

  echo "=== [$label] localization: which cell does the object occupy ==="
  $PY scripts/lens_eval_rope3d_locate.py $flag \
      --report "reports/gemma4/metrics/rope3d_locate${suffix:-_primitive}.json" \
      2>&1 | tr '\r' '\n' | grep -vE "Loading weights|^$" | tail -34

  echo "=== [$label] relations: which is higher, which is nearer ==="
  $PY scripts/lens_eval_rope3d_relations.py $flag --per-room 4 \
      --report "reports/gemma4/metrics/rope3d_relations${suffix:-_primitive}.json" \
      2>&1 | tr '\r' '\n' | grep -vE "Loading weights|^$" | tail -44
done
echo "gemma evals complete"
