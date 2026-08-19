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

run () {  # run <tag> <extra args...>
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "=== $tag ==="
  if ! $PY scripts/lens_train_points.py --holdout 8 --report "$OUT/$tag.json" "$@" \
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
echo "sweep complete"
