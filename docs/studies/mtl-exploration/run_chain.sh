#!/usr/bin/env bash
# Sequential AL/AZ/FL chain. Two arms:
#   --variant no_encoders : B9 with task encoders replaced by Identity (d_model=64).
#   --variant baseline    : Canonical B9 (default encoders, d_model=256).
# 1-fold x 50 epochs, seed=42, single seed for fair pair-wise comparison.

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

if [ $# -lt 2 ]; then
    echo "usage: run_chain.sh VARIANT STATE [STATE ...]   (VARIANT in: no_encoders, baseline)"
    exit 2
fi
VARIANT="$1"
shift
EPOCHS="${EPOCHS:-50}"

variant_args=()
case "${VARIANT}" in
    no_encoders)
        variant_args=(--model-param shared_layer_size=64 --model-param no_task_encoders=true)
        ;;
    baseline)
        variant_args=()
        ;;
    *) echo "unknown variant: ${VARIANT}"; exit 2 ;;
esac

run_state() {
    local state="$1"
    local tag="${VARIANT}_${state}_1f${EPOCHS}ep_seed42"
    echo "================================================================"
    echo "=== [${tag}] start $(date +%H:%M:%S)"
    echo "================================================================"
    "$PY" -u scripts/train.py \
        --task mtl --task-set check2hgi_next_region \
        --state "${state}" --engine check2hgi \
        --model mtlnet_crossattn \
        ${variant_args[@]+"${variant_args[@]}"} \
        --cat-head next_gru --reg-head next_getnext_hard \
        --reg-head-param d_model=256 --reg-head-param num_heads=8 \
        --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/${state}/region_transition_log.pt" \
        --task-a-input-type checkin --task-b-input-type region \
        --folds 1 --epochs "${EPOCHS}" \
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

for s in "$@"; do
    run_state "${s}" || exit $?
done
