#!/usr/bin/env bash
# The same localisation test, same rooms, same controls, two model sizes.
#
# E2B ignored the 3D rotary channel entirely: scrambling which place each token
# claimed to be changed nothing. The open question is whether a larger decoder
# can exploit a positional signal it was never trained on. Both runs are
# restricted to the rooms that have a map for each model, so the comparison is
# paired question by question rather than two loosely related numbers.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv-gemma4/bin/python
ROOMS=${ROOMS:-25}
OUT=reports/gemma4/metrics
LOG=$OUT/model_comparison.log
exec > >(tee -a "$LOG") 2>&1
echo "=== model comparison started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

run () {  # run <tag> <model> <cloud>
  local tag="$1" model="$2" cloud="$3"
  if [ -f "$OUT/rope3d_locate_$tag.json" ]; then echo "skip $tag"; return; fi
  echo "=== $tag ($model) ==="
  $PY scripts/lens_eval_rope3d_locate.py \
      --room-prefix asset --rooms "$ROOMS" \
      --model "$model" --cloud-name "$cloud" \
      --report "$OUT/rope3d_locate_$tag.json" \
      2>&1 | tr '\r' '\n' | grep -vE "Loading weights|^$" | tail -30
}

run e4b_25 google/gemma-4-E4B-it point_cloud_e4b.npz
run e2b_25 google/gemma-4-E2B-it point_cloud.npz
echo "model comparison complete"
