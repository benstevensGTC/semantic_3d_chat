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

# A finished report is only reusable if it was measured on the corpus that is
# here now. Rooms are generated into the same directory the sweep reads, so a
# run can predate a corpus that has since grown -- one report survived from a
# 60-room corpus and another from a 73-room one, mid-generation, and the plain
# "file exists, skip it" rule would have kept both and mixed three corpora into
# one table.
POOL=$($PY - <<'PYEND'
from semantic_3d_chat.spatial_lens.grounding_data import available_rooms
rooms = available_rooms()
PREFIX = ""
EXCLUDE = "asset"
if PREFIX:
    rooms = [r for r in rooms if r.startswith(PREFIX)]
if EXCLUDE:
    rooms = [r for r in rooms if not r.startswith(EXCLUDE)]
print(len(rooms))
PYEND
)

corpus_matches () {  # corpus_matches <report path>
  $PY - "$1" "$POOL" <<'PYEND'
import json, sys
try:
    data = json.loads(open(sys.argv[1]).read())
except Exception:
    sys.exit(1)
sys.exit(0 if str(data.get("room_pool")) == sys.argv[2] else 1)
PYEND
}

run () {
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then
    if corpus_matches "$OUT/$tag.json"; then echo "skip $tag"; return; fi
    echo "redo $tag: measured on a different corpus"
    rm -f "$OUT/$tag.json"
  fi
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
