#!/usr/bin/env bash
# FL pct_start screen — does reshaping OneCycle warmup recover the bs=8192 cat dip?
# bs=8192, FL, seed 0, 1-fold screen. pct_start ∈ {0.30(control), 0.40, 0.50}.
# Champion-G held. Success = cat >= base-2048 FL cat (78.34 diag-best 1f) AND reg flat-up AND 0 NaN.
# SEQUENTIAL exclusive (timing not the axis here; cat recovery is).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/pct_start_fl_runs
mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "pct_start\tcat_diagbest\treg_diagbest\tbest_ep_cat\twall_s\tnan\trc" > "$SUMMARY"

run_cell() {
  local ps="$1"; local cd="$OUT/ps${ps}"; mkdir -p "$cd"; local log="$cd/run.log"; local S=$SECONDS
  ( export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_psfl_${ps}"
    python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
      --state florida --seed 0 --epochs 50 --folds 1 --batch-size 8192 \
      --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
      --cat-head next_gru --reg-head next_stan_flow_dualtower \
      --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
      --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
      --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
      --scheduler onecycle --max-lr 3e-3 --pct-start "$ps" \
      --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
      --model mtlnet_crossattn_dualtower --compile --tf32 \
      --per-fold-transition-dir "output/$V14/florida" --no-checkpoints
  ) > "$log" 2>&1
  local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -dt results/$OVL/florida/mtlnet_*bs8192_ep50_* 2>/dev/null | head -1)
  local cat="-" reg="-" bec="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag psfl_$ps 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    bec=$(echo "$sc" | grep -oE "epochs=\[[0-9]+\]" | head -1)
  fi
  echo -e "${ps}\t${cat:--}\t${reg:--}\t${bec:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[psfl] ps=$ps done rc=$rc wall=${wall}s cat=$cat reg=$reg best_ep=$bec nan=$nan"
}

echo "[psfl] FL bs=8192 pct_start screen: 0.30 0.40 0.50 (1-fold, seq). base-2048 FL 1f cat=78.34 reg=75.58"
for ps in 0.30 0.40 0.50; do run_cell "$ps"; done
echo "[psfl] ALL DONE"
echo "==== PCT_START FL SCREEN (target: cat >= 78.34) ===="
column -t "$SUMMARY"
