#!/usr/bin/env bash
# pipeline_audit — aligned re-tune PASS 1b (advisor-panel additions, 2026-07-02).
# The methodology critic found pass-1's lr5e4 arm targets the WRONG param group: the
# cross-attention blocks — the medium of the self-pairing shortcut — live in the SHARED
# optimizer group (helpers.py setup_per_head_optimizer), untouched by --cat-lr. These two
# arms attack the shortcut medium directly:
#   shlr5e4 — --shared-lr 5e-4 (halved peak LR on the cross-attn/shared group)
#   shdrop30 — --model-param shared_dropout=0.30 (2x dropout inside the cross-attn blocks)
# Trajectory-analyst prior: both predicted to fail (deficit is a CEILING: base's un-annealed
# ep25 value already exceeds aligned's all-time peak; aligned train-f1 79.5 vs base 69.4 at
# equal-or-lower val = memorization via the own-window read; derange ≡ base shows the free
# fix is "don't self-pair"). Running them makes the closure airtight: schedule, cat-LR,
# shared-LR, weight decay, AND capacity attacks all tried.
set -u
source /home/vitor.oliveira/.venv/bin/activate
cd "$(dirname "$0")/../../.."
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/pipeline_audit/aligned_retune; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
st=alabama

run_cell() {  # arm seed extra_flags
  local arm="$1" sd="$2" extra="$3"
  local cd_="$OUT/${arm}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_pipeline_audit_${st}"
  unset MTL_ALIGNED_DERANGE MTL_TRAIN_DIAGNOSTICS MTL_NO_TRAIN_DIAGNOSTICS MTL_PROFILE_JSON
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
    --aligned-pairing \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints $extra > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "${RD:-}" ]; then
    python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag retune_${arm}_s${sd} > "$cd_/score.txt" 2>&1 || true
    cat=$(grep -oE "cat macro-F1 \(diag-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
    reg=$(grep -oE "reg FULL top10_acc \(indist-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${arm}\t${sd}\t${cat:--}\t${reg:--}\t${wall}\t${rc}\t$(basename ${RD:-NONE})" >> "$SUMMARY"
  echo "[retune1b] ${arm} s${sd} rc=$rc wall=${wall}s cat=${cat:--} reg=${reg:--}"
}

echo "[retune1b] start $(date -u +%H:%M:%S)"
for SD in 0 1; do
  run_cell shlr5e4  "$SD" "--shared-lr 5e-4"
  run_cell shdrop30 "$SD" "--model-param shared_dropout=0.30"
done
echo "[retune1b] DONE $(date -u +%H:%M:%S)"
