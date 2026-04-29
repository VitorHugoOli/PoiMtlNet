#!/usr/bin/env bash
# Phase 3 — sequential MTL B3 grid (single GPU).
# Runs CA c2hgi → CA hgi → TX c2hgi → TX hgi on GPU 0 with fail-fast abort.
# ETA on A100 (40 GB): ~3-5 h wall-clock.
set -u
cd "$(dirname "$0")/.."

mkdir -p logs/phase3

echo "######## Phase 3 MTL CH18 sequential grid start $(date) ########"

CELLS=(
    "california check2hgi"
    "california hgi"
    "texas      check2hgi"
    "texas      hgi"
)

for cell in "${CELLS[@]}"; do
    read -r STATE ENGINE <<< "$cell"
    if ! bash scripts/run_phase3_mtl_cell.sh "$STATE" "$ENGINE" 0; then
        echo ""
        echo "[abort] $STATE × $ENGINE failed (likely OOM). Skipping remaining cells."
        echo "######## Phase 3 MTL grid ABORTED $(date) ########"
        exit 1
    fi
done

echo ""
echo "######## Phase 3 MTL CH18 sequential grid complete $(date) ########"
echo ""
echo "Next: python3 scripts/finalize_phase3.py"
