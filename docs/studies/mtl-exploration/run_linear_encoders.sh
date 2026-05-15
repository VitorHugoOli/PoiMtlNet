#!/usr/bin/env bash
# Cells B + C: --model-param linear_encoders=true with shared_layer_size in {64, 256}
# AL 25ep seed=42 to discriminate non-linearity factor from d_model factor.

set -u
WORKTREE="$(pwd)"
export PYTHONPATH="src"
export DATA_ROOT="${DATA_ROOT:-${WORKTREE}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKTREE}/output}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="${PY:-${WORKTREE}/.venv/bin/python}"
LOGDIR="${WORKTREE}/docs/studies/mtl-exploration/logs"
mkdir -p "${LOGDIR}"

run_state() {
    local state="$1"; local sls="$2"
    local tag="linear_enc_sls${sls}_${state}_1f25ep_seed42"
    echo "=== [${tag}] start $(date +%H:%M:%S)"
    "$PY" -u scripts/train.py \
        --task mtl --task-set check2hgi_next_region \
        --state "${state}" --engine check2hgi \
        --model mtlnet_crossattn \
        --model-param "shared_layer_size=${sls}" \
        --model-param linear_encoders=true \
        --cat-head next_gru --reg-head next_getnext_hard \
        --reg-head-param d_model=256 --reg-head-param num_heads=8 \
        --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/${state}/region_transition_log.pt" \
        --task-a-input-type checkin --task-b-input-type region \
        --folds 1 --epochs 25 \
        --batch-size 2048 \
        --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
        --gradient-accumulation-steps 1 \
        --per-fold-transition-dir "${OUTPUT_DIR}/check2hgi/${state}" \
        --no-checkpoints --no-folds-cache \
        --min-best-epoch 5 \
        --mtl-loss static_weight --category-weight 0.75 \
        --seed 42 \
        --alternating-optimizer-step \
        --scheduler cosine --max-lr 3e-3 \
        --alpha-no-weight-decay \
        2>&1 | tee "${LOGDIR}/${tag}.log"
    local rc="${PIPESTATUS[0]}"
    echo "[${tag}] exit ${rc} at $(date +%H:%M:%S)"
    return "${rc}"
}

STATE="${1:-alabama}"
# Cell B: linear encoder + d_model=64
run_state "${STATE}" 64 || exit $?
# Cell C: linear encoder + d_model=256
run_state "${STATE}" 256 || exit $?
echo "BC_${STATE}_DONE" > "${LOGDIR}/_bc_${STATE}_done.flag"
