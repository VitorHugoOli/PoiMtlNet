#!/usr/bin/env bash
# Post-P0A queue: P1-A B9 + P1-C B10 + GA cross-state.
# All stack on the P4+Cosine champion. Run after P0-A completes.
set -u
WORKTREE="${WORKTREE:-$(pwd)}"
export PYTHONPATH="src"
export DATA_ROOT="${DATA_ROOT:-/workspace/PoiMtlNet/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/PoiMtlNet/output}"
PY="${PY:-/opt/poimtlnet-venv/bin/python}"
cd "${WORKTREE}"
mkdir -p logs

base_flags_fl=(
    --task mtl --task-set check2hgi_next_region
    --state florida --engine check2hgi
    --model mtlnet_crossattn
    --mtl-loss static_weight --category-weight 0.75
    --cat-head next_gru --reg-head next_getnext_hard
    --reg-head-param d_model=256 --reg-head-param num_heads=8
    --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/florida/region_transition_log.pt"
    --task-a-input-type checkin --task-b-input-type region
    --folds 5 --epochs 50 --seed 42
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
    --gradient-accumulation-steps 1
    --alternating-optimizer-step
    --scheduler cosine --max-lr 3e-3
    --min-best-epoch 10
    --per-fold-transition-dir "${OUTPUT_DIR}/check2hgi/florida"
    --no-checkpoints --no-folds-cache
)

base_flags_ga=(
    --task mtl --task-set check2hgi_next_region
    --state georgia --engine check2hgi
    --model mtlnet_crossattn
    --mtl-loss static_weight --category-weight 0.75
    --cat-head next_gru --reg-head next_getnext_hard
    --reg-head-param d_model=256 --reg-head-param num_heads=8
    --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/georgia/region_transition_log.pt"
    --task-a-input-type checkin --task-b-input-type region
    --folds 5 --epochs 50 --seed 42
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
    --gradient-accumulation-steps 1
    --alternating-optimizer-step
    --scheduler cosine --max-lr 3e-3
    --min-best-epoch 10
    --per-fold-transition-dir "${OUTPUT_DIR}/check2hgi/georgia"
    --no-checkpoints --no-folds-cache
)

run() {
    local tag="$1"; shift
    echo ""
    echo "================================================================"
    echo "=== [${tag}] start $(date)"
    echo "================================================================"
    "$PY" -u scripts/train.py "$@" 2>&1 | tee "logs/${tag}.log"
    echo "[${tag}] exit ${PIPESTATUS[0]} at $(date)"
}

# P1-A B9: exempt alpha from weight decay (champion + --alpha-no-weight-decay)
run "f50_p1a_b9_alpha_no_wd_fl" \
    "${base_flags_fl[@]}" --batch-size 2048 --alpha-no-weight-decay

# P1-C B10: 2x alpha steps via half batch (champion + bs=1024)
run "f50_p1c_b10_bs1024_fl" \
    "${base_flags_fl[@]}" --batch-size 1024

# P0-cross-GA: champion at Georgia (cross-state portability check; AL/AZ
# folder IDs not in fetch script, GA substitutes as different region scale).
run "f50_p0_cross_ga_champion" \
    "${base_flags_ga[@]}" --batch-size 2048

echo ""
echo "================================================================"
echo "=== Post-P0A queue COMPLETE $(date) ==="
echo "================================================================"
