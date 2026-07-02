#!/usr/bin/env bash
# pipeline_audit — "fair re-tune for aligned" sweep (2026-07-02, user hypothesis).
# Battery finding: aligned peaks ep 21-26 (base 32-50) then overfits — the champion recipe
# (50-ep OneCycle, wd 0.05, cat-lr 1e-3) was tuned FOR random pairing, and aligned's best
# checkpoint is selected MID-SCHEDULE at high LR (never sees the annealing tail).
# Arms (all = aligned recipe +), AL, seeds {0,1} first pass:
#   ep25   — schedule-matched horizon: OneCycle fully anneals where aligned naturally peaks
#   wd10   — stronger AdamW regularization (0.05 → 0.10) to delay the overfit peak
#   lr5e4  — halved cat-head peak LR (per-head OneCycle live since v17)
#   combo  — ep25 + wd10 + lr5e4
# Pre-registered win condition: tuned-aligned cat >= base - 0.5 AND reg >= base - 0.2 (paired
# per seed vs the battery base arm) -> extend winner to seeds {7,100} + an alcond variant
# (the prize: aligned+cond beating base reg with cat parity). Else: aligned is not salvageable
# by recipe tuning at AL; champion random pairing stands.
set -u
source /home/vitor.oliveira/.venv/bin/activate
cd "$(dirname "$0")/../../.."
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/pipeline_audit/aligned_retune; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
[ -f "$SUMMARY" ] || echo -e "arm\tseed\tcat\treg\twall\trc\trundir" > "$SUMMARY"
st=alabama

run_cell() {  # arm seed epochs extra_flags
  local arm="$1" sd="$2" ep="$3" extra="$4"
  local cd_="$OUT/${arm}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_pipeline_audit_${st}"
  unset MTL_ALIGNED_DERANGE MTL_TRAIN_DIAGNOSTICS MTL_NO_TRAIN_DIAGNOSTICS MTL_PROFILE_JSON
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed "$sd" --epochs "$ep" --folds 5 --batch-size 8192 \
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
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep${ep}_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "${RD:-}" ]; then
    python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag retune_${arm}_s${sd} > "$cd_/score.txt" 2>&1 || true
    cat=$(grep -oE "cat macro-F1 \(diag-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
    reg=$(grep -oE "reg FULL top10_acc \(indist-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${arm}\t${sd}\t${cat:--}\t${reg:--}\t${wall}\t${rc}\t$(basename ${RD:-NONE})" >> "$SUMMARY"
  echo "[retune] ${arm} s${sd} rc=$rc wall=${wall}s cat=${cat:--} reg=${reg:--}"
}

echo "[retune] start $(date -u +%H:%M:%S)"
for SD in 0 1; do
  run_cell ep25  "$SD" 25 ""
  run_cell wd10  "$SD" 50 "--weight-decay 0.10"
  run_cell lr5e4 "$SD" 50 "--cat-lr 5e-4"
  run_cell combo "$SD" 25 "--weight-decay 0.10 --cat-lr 5e-4"
done
echo "[retune] PASS1 DONE $(date -u +%H:%M:%S)"
column -t "$SUMMARY"
