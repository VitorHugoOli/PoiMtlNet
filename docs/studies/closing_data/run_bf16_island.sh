#!/usr/bin/env bash
# bf16 + fp32-attention-island validation (A40). Champion-G, bf16 autocast arm
# (MTL_AUTOCAST_BF16=1 + fp32 eval) with the STAN fp32-attn island (MTL_STAN_FP32_ATTN=1).
# MTL_STRICT=1 → a non-finite step FAILS LOUD (so a grad-NaN aborts, doesn't silently skip).
# Usage: run_bf16_island.sh <state> [folds] [epochs] [extra train.py args...]
# Reports: non-finite skip count + champG cat/reg vs RESULTS_BOARD §1.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
STATE="$1"; FOLDS="${2:-5}"; EPOCHS="${3:-50}"; shift; shift 2>/dev/null || true; shift 2>/dev/null || true
OUT=docs/studies/closing_data/bf16_island_runs/${STATE}
mkdir -p "$OUT"
V14=check2hgi_design_k_resln_mae_l0_1
# Istanbul uses engine check2hgi (overlap in-place); others use dk_ovl
ENG=check2hgi_dk_ovl; PFDIR="output/$V14/$STATE"
[ "$STATE" = "istanbul" ] && { ENG=check2hgi; PFDIR="output/check2hgi/$STATE"; }

export PYTHONPATH=src
export MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1      # bf16 train, fp32 eval (board bf16 arm)
export MTL_STAN_FP32_ATTN=${MTL_STAN_FP32_ATTN:-1}                            # the island under test (toggle: MTL_STAN_FP32_ATTN=0)
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1 MTL_RAM_HEADROOM_GB=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_bf16_${STATE}
ts() { date -u +%H:%M:%S; }
LOG="$OUT/champG_bf16.log"
echo "[$(ts)] START $STATE bf16+island ($FOLDS folds x $EPOCHS ep, eng=$ENG)" | tee "$LOG"
S=$SECONDS
python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$ENG" \
    --state "$STATE" --seed 0 --epochs "$EPOCHS" --folds "$FOLDS" --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "$PFDIR" --no-checkpoints "$@" >> "$LOG" 2>&1
RC=$?; echo "[$(ts)] END $STATE bf16 rc=$RC wall=$((SECONDS-S))s" | tee -a "$LOG"
echo "==== non-finite skips (MTL_STRICT would abort on one; 0 = clean) ====" | tee -a "$LOG"
grep -ciE "non-finite|nan|skip.*step|guard_finite" "$LOG" | sed 's/^/skip-related log lines: /' | tee -a "$LOG"
grep -iE "non-finite|NaN detected|abort" "$LOG" | tail -3 | tee -a "$LOG"
RD=$(ls -dt results/$ENG/$STATE/mtlnet_*ep${EPOCHS}_* 2>/dev/null | head -1)
echo "rundir=$RD" | tee -a "$LOG"
[ -n "$RD" ] && python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag ${STATE}_bf16_island 2>&1 | grep -iE "cat macro|reg FULL|per-fold" | tee -a "$LOG"
echo "[$(ts)] DONE $STATE bf16+island" | tee -a "$LOG"
