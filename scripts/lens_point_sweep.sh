#!/usr/bin/env bash
# What the point-grounding claim rests on, measured one run at a time.
#
# Every run shares the same eight held-out rooms, the same seed, and -- for the
# scaling curve -- the same number of optimiser steps. That last one matters:
# at a fixed epoch count, two rooms get thirteen times fewer gradient steps than
# nineteen do, so a rising curve would be partly measuring training length
# rather than the amount of data.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
EPOCHS=${EPOCHS:-120}
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

run () {  # run <tag> <extra args...>
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then
    if corpus_matches "$OUT/$tag.json"; then echo "skip $tag"; return; fi
    echo "redo $tag: measured on a different corpus"
    rm -f "$OUT/$tag.json"
  fi
  echo "=== $tag ==="
  # Pinned to the primitive corpus: asset rooms appear in the same directory
  # and would otherwise join the pool part-way through a sweep, so early and
  # late runs would be measured on different data.
  if ! $PY scripts/lens_train_points.py --holdout 8 --exclude-prefix asset \
       --report "$OUT/$tag.json" "$@" \
       2>&1 | grep -v '^  epoch' | tail -16; then
    # A crashed run and an unrun one look identical from the output directory,
    # so say which happened.
    echo "FAILED $tag"
  fi
}

# Does the encoding matter, on phrases that name their target?
for mode in rope3d learned_absolute none; do
  run "${mode}_rooms19" --epochs "$EPOCHS" --position-mode "$mode"
done
run rgb_only_rooms19 --epochs "$EPOCHS" --position-mode rope3d --feature-mode rgb

# Augmentation translates the room, which rope3d is invariant to by
# construction and learned_absolute is not. Both need the control, or the
# ablation is measuring the augmentation.
for mode in rope3d learned_absolute; do
  run "${mode}_rooms19_noaug" --epochs "$EPOCHS" --position-mode "$mode" --no-augment
done

# The decisive task: the phrase names no target, only a relation to an anchor,
# so semantics alone cannot resolve it.
for mode in rope3d learned_absolute none; do
  run "relational_${mode}_rooms19" --epochs "$EPOCHS" --task relational --position-mode "$mode"
done
run relational_rgb_only_rooms19 --epochs "$EPOCHS" --task relational \
    --position-mode rope3d --feature-mode rgb

# Scaling, at a fixed compute budget so the x-axis is data and nothing else.
for n in 2 4 8 12 16 19; do
  for mode in rope3d learned_absolute; do
    run "${mode}_rooms$n" --target-steps "$STEPS" --position-mode "$mode" --train-rooms "$n"
  done
done
for n in 4 8 12 16 19; do
  run "relational_rope3d_rooms$n" --target-steps "$STEPS" --task relational \
      --position-mode rope3d --train-rooms "$n"
  run "relational_learned_absolute_rooms$n" --target-steps "$STEPS" --task relational \
      --position-mode learned_absolute --train-rooms "$n"
done
# The first runs of a sweep can predate a change to what gets recorded. Redo
# any run whose report is missing the per-item outcomes the paired comparison
# needs, rather than leaving one arm silently uncomparable.
for path in "$OUT"/*.json; do
  [ -e "$path" ] || continue
  case "$(basename "$path")" in summary.json) continue;; esac
  if ! grep -q '"per_item"' "$path"; then
    echo "re-running $(basename "$path" .json): recorded before per-item outcomes"
    rm -f "$path"
  fi
done
echo "sweep complete"
