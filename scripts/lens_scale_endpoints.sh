#!/usr/bin/env bash
# The 19-room points of the scaling curves, at the same step budget as every
# other point on them.
#
# The main sweep names its ablation runs <mode>_rooms19 and its scaling runs the
# same thing, so the scaling loop skips n=19 as already done and the curve's
# most important endpoint silently carries twice the gradient steps of the rest.
# These runs supply that endpoint under its own name.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
STEPS=${STEPS:-2880}
OUT=reports/gemma4/metrics/point_grounding
mkdir -p "$OUT"

run () {
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "=== $tag ==="
  if ! $PY scripts/lens_train_points.py --holdout 8 --exclude-prefix asset \
       --target-steps "$STEPS" --report "$OUT/$tag.json" "$@" \
       2>&1 | grep -v '^  epoch' | tail -14; then
    echo "FAILED $tag"
  fi
}

for mode in rope3d learned_absolute; do
  run "scale19_${mode}" --position-mode "$mode" --train-rooms 19
  run "scale19_relational_${mode}" --task relational --position-mode "$mode" --train-rooms 19
done
echo "endpoints complete"
