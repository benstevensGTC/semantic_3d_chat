#!/usr/bin/env bash
# Ablate the position scheme, then scale the training rooms up under each one.
# Every run shares the same eight held-out rooms and the same seed.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
EPOCHS=${EPOCHS:-120}
OUT=reports/gemma4/metrics/point_grounding
mkdir -p "$OUT"

run () {  # run <tag> <extra args...>
  local tag="$1"; shift
  if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "=== $tag ==="
  $PY scripts/lens_train_points.py --epochs "$EPOCHS" --holdout 8 \
      --report "$OUT/$tag.json" "$@" 2>&1 | grep -v '^  epoch' | tail -14
}

# Position ablation at full data.
run rope3d_rooms19            --position-mode rope3d
run learned_absolute_rooms19  --position-mode learned_absolute
run none_rooms19              --position-mode none
run rope3d_rooms19_noaug      --position-mode rope3d --no-augment

# Scaling curves. Does the 3D-relative scheme pull further ahead with more rooms?
for n in 2 4 8 12 16; do
  run "rope3d_rooms$n"           --position-mode rope3d           --train-rooms "$n"
done
for n in 2 4 8 12 16; do
  run "learned_absolute_rooms$n" --position-mode learned_absolute --train-rooms "$n"
done

# Relational phrases name an object by where it is relative to another one, so
# semantics alone cannot resolve them. This is where a position scheme has to
# earn its place rather than ride along on Gemma's features.
for mode in rope3d learned_absolute none; do
  run "relational_${mode}_rooms19" --task relational --position-mode "$mode"
done
for n in 4 8 12 16; do
  run "relational_rope3d_rooms$n" --task relational --position-mode rope3d --train-rooms "$n"
done
echo "sweep complete"
