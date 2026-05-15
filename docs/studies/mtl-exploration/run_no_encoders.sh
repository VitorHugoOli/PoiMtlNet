#!/usr/bin/env bash
# mtl-exploration smoke (2026-05-14): B9 recipe with task encoders ablated
# (replaced with nn.Identity, d_model=64 throughout the cross-attn stack).
# 1-fold x 50ep on AL+AZ+FL, seed=42. Single-seed smoke; if interesting,
# extend to 5-fold multi-seed afterward.
#
# Source recipe: docs/NORTH_STAR.md §"Champion — F50 B9"
# Single change: --model-param shared_layer_size=64 + no_task_encoders=true.
#
# Local MPS expected wall-clock (1 fold x 50ep):
#   AL ~12 min, AZ ~18 min, FL ~70 min  =>  ~100 min total

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
    local state="$1"
    local tag="no_enc_${state}_1f50ep_seed42"
    echo "================================================================"
    echo "=== [${tag}] start $(date +%H:%M:%S)"
    echo "================================================================"
    "$PY" -u scripts/train.py \
        --task mtl --task-set check2hgi_next_region \
        --state "${state}" --engine check2hgi \
        --model mtlnet_crossattn \
        --model-param shared_layer_size=64 \
        --model-param no_task_encoders=true \
        --cat-head next_gru --reg-head next_getnext_hard \
        --reg-head-param d_model=256 --reg-head-param num_heads=8 \
        --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/${state}/region_transition_log.pt" \
        --task-a-input-type checkin --task-b-input-type region \
        --folds 1 --epochs 50 \
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
run_state "${STATE}"
