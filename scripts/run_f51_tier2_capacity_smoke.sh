#!/usr/bin/env bash
# F51 Tier 2 — encoder/backbone capacity smoke (1 fold × 50 ep × bs=2048).
#
# Mission (from F50_NORTH_STAR_DEEP_EXPLORATION_PROMPT.md §3 Tier 2):
# "Reg-side encoder saturation (D5 finding) is partly a capacity bottleneck.
#  Larger reg encoder might delay saturation."
#
# Smoke pass first; promote to 5-fold paper-grade only if smoke shifts the
# reg-best epoch past ep 5–6 OR raises the reg plateau by >0.5 pp vs B9
# seed=42 fold-1 reference (reg top10 ~63.53, reg-best ep 6).
#
# Sweep: 7 capacity knobs × ~3 levels each = ~21 smokes × ~3.3 min ≈ 70 min.
# Stays bound by Tier 1 GPU availability — run AFTER Tier 1 multi-seed completes.

set -u
WORKTREE="${WORKTREE:-$(pwd)}"
export PYTHONPATH="src"
export DATA_ROOT="${DATA_ROOT:-/workspace/PoiMtlNet/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/PoiMtlNet/output}"
PY="${PY:-/opt/poimtlnet-venv/bin/python}"
cd "${WORKTREE}"
mkdir -p logs

# Base B9 recipe at FL, 1 fold, seed=42 (the reference seed).
base_b9_smoke=(
    --task mtl --task-set check2hgi_next_region
    --state florida --engine check2hgi
    --model mtlnet_crossattn
    --cat-head next_gru --reg-head next_getnext_hard
    --reg-head-param d_model=256 --reg-head-param num_heads=8
    --reg-head-param "transition_path=${OUTPUT_DIR}/check2hgi/florida/region_transition_log.pt"
    --task-a-input-type checkin --task-b-input-type region
    --folds 1 --epochs 50 --seed 42
    --batch-size 2048
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
    --gradient-accumulation-steps 1
    --per-fold-transition-dir "${OUTPUT_DIR}/check2hgi/florida"
    --no-checkpoints --no-folds-cache
    --min-best-epoch 5
    --mtl-loss static_weight --category-weight 0.75
    --alternating-optimizer-step
    --scheduler cosine --max-lr 3e-3
    --alpha-no-weight-decay
)

run() {
    local tag="$1"; shift
    echo "================================================================"
    echo "=== [${tag}] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "================================================================"
    "$PY" -u scripts/train.py "$@" 2>&1 | tee "logs/${tag}.log"
    echo "[${tag}] exit ${PIPESTATUS[0]} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# 1. encoder_layer_size sweep — current 256 → {128, 384, 512}
for v in 128 384 512; do
    run "f51_t2_encoder_layer_size_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "encoder_layer_size=${v}"
done

# 2. num_encoder_layers sweep — current 2 → {1, 3, 4}
for v in 1 3 4; do
    run "f51_t2_num_encoder_layers_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "num_encoder_layers=${v}"
done

# 3. encoder_dropout sweep — current 0.1 → {0.05, 0.2, 0.3}
for v in 0.05 0.2 0.3; do
    run "f51_t2_encoder_dropout_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "encoder_dropout=${v}"
done

# 4. shared_layer_size sweep — current 256 → {128, 384, 512}.
# NOTE: must match encoder_layer_size for the cross-attn block dim alignment.
for v in 128 384 512; do
    run "f51_t2_shared_layer_size_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "shared_layer_size=${v}" \
        --model-param "encoder_layer_size=${v}"
done

# 5. num_crossattn_blocks sweep — current 2 → {1, 3, 4}.
# F52 said cross-attn mixing is dead at FL → expect tied with B9.
for v in 1 3 4; do
    run "f51_t2_num_crossattn_blocks_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "num_crossattn_blocks=${v}"
done

# 6. num_crossattn_heads sweep — current 4 → {2, 8, 16}
for v in 2 8 16; do
    run "f51_t2_num_crossattn_heads_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "num_crossattn_heads=${v}"
done

# 7. crossattn_ffn_dim sweep — current 256 → {128, 512, 1024}.
# F52 said FFN+LN is the productive part — does width help?
for v in 128 512 1024; do
    run "f51_t2_crossattn_ffn_dim_${v}" \
        "${base_b9_smoke[@]}" \
        --model-param "crossattn_ffn_dim=${v}"
done

echo ""
echo "================================================================"
echo "=== F51 Tier 2 capacity smoke COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "================================================================"
