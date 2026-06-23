#!/usr/bin/env bash
# STAGED for the A40 lane (cannot run from the H100) — TX STL CAT ceiling (next_gru, seed0 5f, dk_ovl).
# Completes the TX cell: the TX reg ceiling (64.96, fp32) exists; the cat ceiling was never run
# ("cat ceiling NOT run for TX", tx_ba2_s0.json). Mirrors the CA/AL/AZ/FL cat-ceiling recipe.
#
# PREREQUISITE: the TX gated-overlap engine (check2hgi_dk_ovl/texas) must exist first — it is
# MISSING as of 2026-06-23. Build it (needs the v14 design_k TX substrate on disk):
#   PYTHONPATH=src python scripts/mtl_improvement/build_overlap_probe_engine.py texas 1
# Then verify output/check2hgi_dk_ovl/texas/input/next.parquet exists before running this.
#
# Adjust REPO/PY/cache paths for the A40 box. cat is precision-insensitive (fp16-eval OK).
set -uo pipefail
cd "${REPO:-$HOME/PoiMtlNet}"
export PYTHONPATH=src
export MTL_CHUNK_VAL_METRIC=1
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/.inductor_cache_tx_cat}
PY=${PY:-python}

$PY scripts/train.py --task next --state texas --engine check2hgi_dk_ovl \
    --model next_gru --folds 5 --epochs 50 --seed 0 --batch-size 2048 --max-lr 3e-3 \
    --gradient-accumulation-steps 1 --no-checkpoints --compile --tf32

# Score (fold-mean macro-F1 at f1-best epoch):
RD=$(ls -dt results/check2hgi_dk_ovl/texas/next_* 2>/dev/null | head -1)
$PY scripts/closing_data/score_stl_cat_ceiling.py "$RD" --tag texas_cat_ceiling
echo "TX cat ceiling done — rundir: $RD"
