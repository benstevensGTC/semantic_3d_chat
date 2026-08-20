#!/usr/bin/env bash
# The same study, on rooms built from real furniture.
#
# The primitive corpus produced two findings that were probably artefacts of how
# it was made: a colour-only control reaching two thirds of Gemma's score, and a
# relational task that no encoding could learn. Coloured boxes make colour
# almost sufficient, and one-of-each rooms make a relation unnecessary. Running
# the identical study on real assets with deliberate duplicates is what
# separates "the method does not work" from "the rooms could not show it".
#
# --room-prefix asset keeps this corpus separate from the primitive one.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
EPOCHS=${EPOCHS:-120}
STEPS=${STEPS:-2880}
HOLDOUT=${HOLDOUT:-15}
OUT=reports/gemma4/metrics/point_grounding_assets
mkdir -p "$OUT"

run () {
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "=== $tag ==="
  if ! $PY scripts/lens_train_points.py --room-prefix asset --holdout "$HOLDOUT" \
       --report "$OUT/$tag.json" "$@" 2>&1 | grep -v '^  epoch' | tail -16; then
    echo "FAILED $tag"
  fi
}

# Position ablation, on phrases that name their target.
for mode in rope3d learned_absolute none; do
  run "${mode}" --epochs "$EPOCHS" --position-mode "$mode"
done
# The control the primitive rooms could not run fairly.
run rgb_only --epochs "$EPOCHS" --position-mode rope3d --feature-mode rgb

# Relational: the target is unnamed, and now the rooms genuinely contain more
# than one of things, so a relation is the only way to say which is meant.
for mode in rope3d learned_absolute none; do
  run "relational_${mode}" --epochs "$EPOCHS" --task relational --position-mode "$mode"
done
run relational_rgb_only --epochs "$EPOCHS" --task relational \
    --position-mode rope3d --feature-mode rgb

# Does keeping the phrase word by word rescue the relational task? A
# mean-pooled query cannot express "nearest(shelf)" at all, so this is the
# single most likely explanation for the primitive corpus's flat result.
run relational_rope3d_tokens --epochs "$EPOCHS" --task relational \
    --position-mode rope3d --query-mode tokens
run relational_learned_absolute_tokens --epochs "$EPOCHS" --task relational \
    --position-mode learned_absolute --query-mode tokens
run rope3d_tokens --epochs "$EPOCHS" --position-mode rope3d --query-mode tokens

# Scaling at a fixed step budget, so the axis is data and not compute.
for n in 4 8 16 24 32 45; do
  for mode in rope3d learned_absolute; do
    run "scale_${mode}_${n}" --target-steps "$STEPS" --position-mode "$mode" --train-rooms "$n"
  done
  run "scale_relational_rope3d_${n}" --target-steps "$STEPS" --task relational \
      --position-mode rope3d --query-mode tokens --train-rooms "$n"
done
echo "asset sweep complete"
