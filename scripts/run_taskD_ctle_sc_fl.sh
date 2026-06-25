#!/usr/bin/env bash
# Task D — CTLE-SC @ FL (UNGATED 2026-06-24 by M4 diagnosis). Build the leak-safe per-fold
# frozen-CTLE cells (stride-1, board dk_ovl base), then route under matched heads and compare.
# Waits for the v14 cascade pair (PIDs in $1 $2) to free GPU before starting the CTLE pretrains.
set -uo pipefail
cd /teamspace/studios/this_studio/PoiMtlNet
export PYTHONPATH=src DISABLE_AMP=1 MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_RAM_HEADROOM_GB=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
export BASELINE_PY="$(which python)"

echo "=== [Task D] waiting for cascade pair (${1:-} ${2:-}) to finish $(date -u) ==="
for pid in "${1:-}" "${2:-}"; do
  [ -n "$pid" ] && while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done
echo "=== [Task D] GPU free; building CTLE cells $(date -u) ==="

for f in 0 1 2 3 4; do
  echo "--- build_ctle_substrate fold $f (stride-1) $(date -u) ---"
  python scripts/baselines/build_ctle_substrate.py --state florida --seed 0 --fold "$f" --stride 1 \
    || { echo "[Task D] cell fold $f FAILED"; }
done

echo "=== [Task D] cells built; matched-head compare $(date -u) ==="
python scripts/closing_data/mac_baseline_compare.py --state florida --baseline ctle \
    --cells-root output --folds 5 --heads cat reg \
  && echo "[Task D] OK $(date -u)" || echo "[Task D] compare FAILED $(date -u)"
ls -la docs/results/closing_data/baseline_compare/florida_ctle.json 2>/dev/null
