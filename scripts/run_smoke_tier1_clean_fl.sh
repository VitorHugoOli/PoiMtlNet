#!/usr/bin/env bash
# Tier-1 verification smokes: PLE-lite uniform-leak check + TGSTAN
# hidden-leak quantification. Both 5f×50ep on FL with per-fold log_T.
# Per re-run decision matrix (F50_T4_RERUN_DECISION.md), these are
# the two architectural-axis confirmations the paper needs.
set -u
WORKTREE="${WORKTREE:-$(pwd)}"
export PYTHONPATH="src"
export DATA_ROOT="${DATA_ROOT:-/workspace/PoiMtlNet/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/PoiMtlNet/output}"
PY="${PY:-/opt/poimtlnet-venv/bin/python}"
cd "${WORKTREE}"
mkdir -p logs

base_fl=(
    --task mtl --task-set check2hgi_next_region
    --state florida --engine check2hgi
    --mtl-loss static_weight --category-weight 0.75
    --cat-head next_gru
    --reg-head-param d_model=256 --reg-head-param num_heads=8
    --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/florida/region_transition_log.pt"
    --task-a-input-type checkin --task-b-input-type region
    --folds 5 --epochs 50 --seed 42 --batch-size 2048
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
    --gradient-accumulation-steps 1
    --scheduler constant
    --min-best-epoch 5
    --per-fold-transition-dir "${OUTPUT_DIR}/check2hgi/florida"
    --no-checkpoints --no-folds-cache
)

run() {
    local tag="$1"; shift
    echo "================================================================"
    echo "=== [${tag}] start $(date)"
    echo "================================================================"
    "$PY" -u scripts/train.py "$@" 2>&1 | tee "logs/${tag}.log"
    echo "[${tag}] exit ${PIPESTATUS[0]} at $(date)"
}

# 1. PLE-lite clean — representative architectural alt to confirm uniform leak
#    Pre-C4 leaky: PLE +0.25 vs H3-alt at >=ep5. Expect tied/positive clean.
run "f50_p0e_ple_clean_fl" \
    "${base_fl[@]}" \
    --model mtlnet_ple \
    --reg-head next_getnext_hard

# 2. TGSTAN clean — quantify the per-sample-gate amplifier risk
#    Audit's #2 flag: gate=sigmoid(MLP) routes leak selectively to val.
#    If TGSTAN clean is much lower than TGSTAN leaky (>20 pp), the gate
#    is amplifying selectively and any prior TGSTAN claims are bogus.
run "f50_p0d_tgstan_clean_fl" \
    "${base_fl[@]}" \
    --model mtlnet_crossattn \
    --reg-head next_tgstan

echo ""
echo "================================================================"
echo "=== Tier-1 smoke queue COMPLETE $(date) ==="
echo "================================================================"
