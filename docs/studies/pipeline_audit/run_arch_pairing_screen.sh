#!/usr/bin/env bash
# pipeline_audit — arch × pairing SCREEN (2026-07-02, user hypothesis).
# ⚠ SCREENING-GRADE ONLY (seed 0 × 5 folds; NOT board-comparable): the reg head here is
# next_stan_flow (prior off) not the champion dualtower head, so absolute numbers do NOT
# compare to the battery. The readout is WITHIN-arch: sign of Δ(aligned − base) per arch.
# Hypothesis: the aligned-pairing harm needs CHANNEL CAPACITY to memorize instance detail;
# cross-stitch's α-matrix scalar mixing (Misra et al. 2016) is a tiny-capacity channel that
# might harvest semantic pairing without the shortcut. xstitch (stitch→cross-attn hybrid)
# retains the full-capacity attention read → expected to behave like crossattn (aligned hurts).
set -u
source /home/vitor.oliveira/.venv/bin/activate
cd "$(dirname "$0")/../../.."
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/pipeline_audit/arch_pairing_screen; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
[ -f "$SUMMARY" ] || echo -e "arch\tpairing\tseed\tcat\treg\twall\trc\trundir" > "$SUMMARY"
st=alabama; sd=0

run_cell() {  # arch pairing_flag_or_empty tag
  local model="$1" pflag="$2" tag="$3"
  local cd_="$OUT/${tag}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_pipeline_audit_${st}"
  unset MTL_ALIGNED_DERANGE MTL_TRAIN_DIAGNOSTICS MTL_NO_TRAIN_DIAGNOSTICS MTL_PROFILE_JSON
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model "$model" --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints $pflag > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "${RD:-}" ]; then
    python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag archscreen_${tag} > "$cd_/score.txt" 2>&1 || true
    cat=$(grep -oE "cat macro-F1 \(diag-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
    reg=$(grep -oE "reg FULL top10_acc \(indist-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
  fi
  local pairing="base"; [ -n "$pflag" ] && pairing="aligned"
  echo -e "${model}\t${pairing}\t${sd}\t${cat:--}\t${reg:--}\t${wall}\t${rc}\t$(basename ${RD:-NONE})" >> "$SUMMARY"
  echo "[archscreen] ${tag} rc=$rc wall=${wall}s cat=${cat:--} reg=${reg:--}"
}

echo "[archscreen] start $(date -u +%H:%M:%S)"
run_cell mtlnet_crossstitch        ""                  cstitch_base
run_cell mtlnet_crossstitch        "--aligned-pairing" cstitch_aligned
run_cell mtlnet_crossattn_xstitch  ""                  xstitch_base
run_cell mtlnet_crossattn_xstitch  "--aligned-pairing" xstitch_aligned
echo "[archscreen] DONE $(date -u +%H:%M:%S)"
column -t "$SUMMARY"
