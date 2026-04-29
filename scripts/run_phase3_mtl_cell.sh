#!/usr/bin/env bash
# Phase 3 — single MTL B3 cell launcher.
# Pins the cell to a specific GPU via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   bash scripts/run_phase3_mtl_cell.sh STATE ENGINE GPU_ID
#
# Example:
#   bash scripts/run_phase3_mtl_cell.sh california check2hgi 0
#   bash scripts/run_phase3_mtl_cell.sh texas      hgi       3
set -u
cd "$(dirname "$0")/.."

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 STATE ENGINE GPU_ID"
    echo "  STATE   = california | texas"
    echo "  ENGINE  = check2hgi | hgi"
    echo "  GPU_ID  = integer (0..N-1)"
    exit 1
fi

STATE="$1"
ENGINE="$2"
GPU_ID="$3"

export PYTHONPATH=src
export OUTPUT_DIR=$(pwd)/output
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Canonical batch size. Override via MTL_BATCH_SIZE env var if needed.
MTL_BS="${MTL_BATCH_SIZE:-2048}"

UPSTATE=$(echo "$STATE" | tr '[:lower:]' '[:upper:]')
TAG="MTL_B3_${UPSTATE}_${ENGINE}_5f50ep"
LOG="logs/phase3/${TAG}.log"
mkdir -p logs/phase3

echo "================================================================"
echo "[${TAG}] start $(date)"
echo "  STATE=$STATE  ENGINE=$ENGINE  GPU_ID=$GPU_ID  bs=$MTL_BS"
echo "  log=$LOG"
echo "================================================================"

python3 -u scripts/train.py \
    --task mtl --state "$STATE" --engine "$ENGINE" \
    --task-set check2hgi_next_region \
    --model mtlnet_crossattn \
    --mtl-loss static_weight --category-weight 0.75 \
    --reg-head next_getnext_hard --cat-head next_gru \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/${STATE}/region_transition_log.pt" \
    --batch-size "$MTL_BS" \
    --folds 5 --epochs 50 --seed 42 --no-checkpoints \
    > "$LOG" 2>&1
rc=$?

echo "[${TAG}] exit $rc at $(date)"
exit $rc
