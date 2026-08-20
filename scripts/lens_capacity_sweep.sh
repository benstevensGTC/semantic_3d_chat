#!/usr/bin/env bash
# The knobs the main study left at their defaults.
#
# The headline claim -- that "which cabinet" fails for architectural reasons
# rather than representational ones -- rests on a reader that was never varied:
# 256 dimensions, four layers, a thousand points, and a rotary wavelength of
# eight metres that nobody tuned. If a wider or deeper reader solves the task,
# or if the wavelength was simply wrong for room-sized geometry, that claim is
# false and needs withdrawing.
#
# Every run keeps the same step budget and the same split, so the only thing
# changing is the knob named in the tag.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
STEPS=${STEPS:-3600}
HOLDOUT=${HOLDOUT:-30}
OUT=reports/gemma4/metrics/point_grounding_capacity
mkdir -p "$OUT"

run () {
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "=== $tag ==="
  if ! $PY scripts/lens_train_points.py --room-prefix asset --holdout "$HOLDOUT" \
       --target-steps "$STEPS" --report "$OUT/$tag.json" "$@" \
       2>&1 | grep -v '^  epoch' | tail -14; then
    echo "FAILED $tag"
  fi
}

# Reader capacity, on the task that fails. If depth or width is the missing
# ingredient for multi-hop comparison, it shows up here.
for spec in "256 4" "512 6" "768 8"; do
  set -- $spec
  run "disambig_dim${1}_layers${2}" --task disambiguation --position-mode rope3d \
      --model-dim "$1" --layers "$2"
done

# The rotary wavelength was never tuned. Eight metres is about a room; two is
# about a piece of furniture; thirty-two is far coarser than either.
for cycle in 2.0 4.0 8.0 16.0 32.0; do
  run "disambig_cycle${cycle}" --task disambiguation --position-mode rope3d \
      --metres-per-cycle "$cycle"
  run "object_cycle${cycle}" --position-mode rope3d --metres-per-cycle "$cycle"
done

# More points means finer geometry and a harder needle; both directions matter.
for budget in 512 2048; do
  run "disambig_points${budget}" --task disambiguation --position-mode rope3d \
      --token-budget "$budget"
  run "object_points${budget}" --position-mode rope3d --token-budget "$budget"
done

# Capacity on the task that works, as a control: if width helps nothing
# anywhere, the reader was not the binding constraint on either task.
run "object_dim512_layers6" --model-dim 512 --layers 6 --position-mode rope3d

# Three seeds at one point, so the single-seed curves can be read with a sense
# of how much of their wobble is noise.
for seed in 11 22 33; do
  run "object_seed${seed}" --position-mode rope3d --seed "$seed"
  run "disambig_seed${seed}" --task disambiguation --position-mode rope3d --seed "$seed"
done
echo "capacity sweep complete"
