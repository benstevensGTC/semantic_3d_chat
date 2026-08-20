#!/usr/bin/env bash
# The same study, on rooms built from real furniture.
#
# The primitive corpus produced two findings that were plausibly artefacts of
# how those rooms were made: a colour-only control reaching two thirds of
# Gemma's score, which flat-shaded coloured boxes would produce on their own,
# and a relational task no encoding could learn, posed on rooms that never
# contained two of anything. Running the identical study on real assets with
# guaranteed duplicates separates "the method does not work" from "the rooms
# could not show it".
#
# Every run gets the same optimiser-step budget, so no comparison here is
# between one model trained longer than another.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
STEPS=${STEPS:-3600}
HOLDOUT=${HOLDOUT:-30}
OUT=reports/gemma4/metrics/point_grounding_assets
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
PREFIX = "asset"
EXCLUDE = ""
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
  if ! $PY scripts/lens_train_points.py --room-prefix asset --holdout "$HOLDOUT" \
       --target-steps "$STEPS" --report "$OUT/$tag.json" "$@" \
       2>&1 | grep -v '^  epoch' | tail -16; then
    echo "FAILED $tag"
  fi
}

# Naming an object that is in the room. Mostly a semantic task, which is why
# the no-position control does respectably at it.
for mode in rope3d learned_absolute none; do
  run "object_${mode}" --position-mode "$mode"
done
run object_rgb_only --position-mode rope3d --feature-mode rgb

# The form the referring-expression literature uses, and the one the primitive
# rooms could not pose: name a category the room holds more than one of, so
# semantics narrows the field to the cabinets and only distance chooses.
# Chance is one in k, usually a half -- a far tighter line than 1/n_objects.
for mode in rope3d learned_absolute none; do
  run "disambig_${mode}" --task disambiguation --position-mode "$mode"
done
run disambig_rgb_only --task disambiguation --position-mode rope3d --feature-mode rgb
run disambig_rope3d_tokens --task disambiguation --position-mode rope3d --query-mode tokens

# The strictest form: the target is not named at all, only its relation to an
# anchor. This is what sat at chance on the primitive corpus.
for mode in rope3d learned_absolute none; do
  run "relational_${mode}" --task relational --position-mode "$mode"
done
# A mean-pooled phrase cannot express "nearest(shelf)"; keeping the words is
# the likeliest fix, and the only way to know is to run it.
run relational_rope3d_tokens --task relational --position-mode rope3d --query-mode tokens

# Scaling. The axis is rooms; the step budget is fixed, so it is not compute.
for n in 6 12 24 48 90; do
  for mode in rope3d learned_absolute; do
    run "scale_object_${mode}_${n}" --position-mode "$mode" --train-rooms "$n"
  done
  run "scale_disambig_rope3d_${n}" --task disambiguation --position-mode rope3d \
      --query-mode tokens --train-rooms "$n"
done
echo "asset sweep complete"
