#!/usr/bin/env bash
# Phase 3 — parallel MTL B3 grid across multiple GPUs in one pod.
# Auto-detects GPU count and dispatches 4 cells (CA c2hgi, CA hgi, TX c2hgi, TX hgi)
# across available GPUs:
#   - 4+ GPUs: all 4 cells in parallel (~50-80 min wall-clock on A100 40 GB)
#   - 2-3 GPUs: 2 cells per wave (CA wave, then TX wave)
#   - 1 GPU:  falls back to scripts/run_phase3_mtl_grid.sh (sequential)
set -u
cd "$(dirname "$0")/.."

mkdir -p logs/phase3

NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected ${NGPU} GPU(s)"

CELLS=(
    "california check2hgi"
    "california hgi"
    "texas      check2hgi"
    "texas      hgi"
)

run_cell_bg() {
    local STATE="$1" ENGINE="$2" GPU="$3"
    bash scripts/run_phase3_mtl_cell.sh "$STATE" "$ENGINE" "$GPU" &
    local PID=$!
    echo "  → $STATE/$ENGINE on GPU $GPU (PID $PID)"
    eval "PID_${STATE}_${ENGINE}=$PID"
}

wait_all() {
    local rc=0
    for pid in "$@"; do
        wait "$pid" || rc=1
    done
    return $rc
}

if [ "$NGPU" -ge 4 ]; then
    echo "######## Phase 3 MTL CH18 parallel grid start (4 GPU) $(date) ########"
    PIDS=()
    GPU=0
    for cell in "${CELLS[@]}"; do
        read -r STATE ENGINE <<< "$cell"
        bash scripts/run_phase3_mtl_cell.sh "$STATE" "$ENGINE" "$GPU" &
        PIDS+=($!)
        echo "  → $STATE/$ENGINE on GPU $GPU (PID $!)"
        GPU=$((GPU+1))
    done
    echo ""
    echo "All 4 cells launched. Waiting for completion..."
    wait_all "${PIDS[@]}"
    rc=$?
    echo ""
    echo "######## Phase 3 MTL grid complete (rc=$rc) $(date) ########"
    exit $rc
elif [ "$NGPU" -ge 2 ]; then
    echo "######## Phase 3 MTL CH18 parallel grid start (2 GPU, 2 waves) $(date) ########"
    # Wave 1: CA cells
    echo ""; echo "=== Wave 1: CA c2hgi + CA hgi ==="
    bash scripts/run_phase3_mtl_cell.sh california check2hgi 0 &
    PID1=$!
    bash scripts/run_phase3_mtl_cell.sh california hgi       1 &
    PID2=$!
    if ! wait_all "$PID1" "$PID2"; then
        echo "[abort] Wave 1 (CA) failed."
        exit 1
    fi
    # Wave 2: TX cells
    echo ""; echo "=== Wave 2: TX c2hgi + TX hgi ==="
    bash scripts/run_phase3_mtl_cell.sh texas check2hgi 0 &
    PID3=$!
    bash scripts/run_phase3_mtl_cell.sh texas hgi       1 &
    PID4=$!
    wait_all "$PID3" "$PID4"
    rc=$?
    echo ""
    echo "######## Phase 3 MTL grid complete (rc=$rc) $(date) ########"
    exit $rc
else
    echo "[info] Only 1 GPU detected — falling back to sequential grid"
    exec bash scripts/run_phase3_mtl_grid.sh
fi
