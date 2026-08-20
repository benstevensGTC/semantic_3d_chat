#!/usr/bin/env bash
# Everything downstream of the rebuilt corpus, in order, one GPU job at a time.
#
# Phrases are re-cached first: the rebuilt rooms contain different objects, so
# the grounding phrases changed, and a stale cache fails a run rather than
# silently mismatching.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
MAIN=.venv/bin/python
GEMMA=.venv-gemma4/bin/python

step () { echo; echo "########## $* ##########"; echo; }

step "re-cache grounding phrases for the rebuilt corpus"
$GEMMA scripts/lens_cache_phrases.py 2>&1 | tr '\r' '\n' | grep -vE "Loading weights|^$" | tail -3

step "primitive corpus: matched-compute scaling endpoints"
STEPS=2880 ./scripts/lens_scale_endpoints.sh

step "asset corpus: the full study"
STEPS=3600 HOLDOUT=15 ./scripts/lens_asset_sweep.sh

step "Gemma reading the field directly, per corpus"
./scripts/lens_gemma_rope3d_evals.sh

step "figures"
$MAIN scripts/lens_plot_results.py
$MAIN scripts/build_point_grounding_summary.py

echo
echo "EVERYTHING COMPLETE"
