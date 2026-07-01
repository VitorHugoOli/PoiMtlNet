#!/usr/bin/env bash
# Batch-size × LR screen sweep — AL champion-G fp32, 2 folds × 50 ep, seed 0.
# 7 cells (base + 4k/8k × none/sqrt/linear), run MAX_PAR at a time, score each → summary table.
# Quality is the trustworthy axis (GPU may be shared); wall-clock recorded but contention-sensitive.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/batch_sweep_runs
mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1; STATE=alabama
FOLDS="${1:-2}"; EPOCHS="${2:-50}"; MAX_PAR="${3:-2}"
SUMMARY="$OUT/summary.tsv"
echo -e "tag\tbs\tmax_lr\tcat_lr\treg_lr\tshared_lr\tcat_macroF1\treg_acc10\twall_s\trc" > "$SUMMARY"

# tag:bs:max_lr:cat_lr:reg_lr:shared_lr
CELLS=(
  "base:2048:3e-3:1e-3:3e-3:1e-3"
  "4k_none:4096:3e-3:1e-3:3e-3:1e-3"
  "4k_sqrt:4096:4.243e-3:1.414e-3:4.243e-3:1.414e-3"
  "4k_lin:4096:6e-3:2e-3:6e-3:2e-3"
  "8k_none:8192:3e-3:1e-3:3e-3:1e-3"
  "8k_sqrt:8192:6e-3:2e-3:6e-3:2e-3"
  "8k_lin:8192:12e-3:4e-3:12e-3:4e-3"
)

run_cell() {
  IFS=':' read -r tag bs mlr clr rlr slr <<< "$1"
  local cd="$OUT/$tag"; mkdir -p "$cd"; local log="$cd/run.log"
  local S=$SECONDS
  ( export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_bsweep_${tag}"
    python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
      --state "$STATE" --seed 0 --epochs "$EPOCHS" --folds "$FOLDS" --batch-size "$bs" \
      --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
      --cat-head next_gru --reg-head next_stan_flow_dualtower \
      --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
      --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
      --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
      --scheduler onecycle --max-lr "$mlr" --cat-lr "$clr" --reg-lr "$rlr" --shared-lr "$slr" \
      --model mtlnet_crossattn_dualtower --compile --tf32 \
      --per-fold-transition-dir "output/$V14/$STATE" --no-checkpoints --run-id "bsweep_${tag}" --per-fold-seed
  ) > "$log" 2>&1
  local rc=$?; local wall=$((SECONDS-S))
  local RD; RD=$(ls -d results/$OVL/$STATE/*_bsweep_${tag} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    read -r cat reg < <(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag bsweep_$tag 2>/dev/null \
      | grep -oE "= [0-9.]+ ±" | grep -oE "[0-9.]+" | head -2 | tr '\n' ' ')
  fi
  echo -e "${tag}\t${bs}\t${mlr}\t${clr}\t${rlr}\t${slr}\t${cat:--}\t${reg:--}\t${wall}\t${rc}" >> "$SUMMARY"
  echo "[sweep] $tag done rc=$rc wall=${wall}s cat=$cat reg=$reg"
}

echo "[sweep] $((${#CELLS[@]})) cells, $FOLDS folds x $EPOCHS ep, max_par=$MAX_PAR"
running=0
for c in "${CELLS[@]}"; do
  run_cell "$c" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[sweep] ALL DONE"
echo "==== SUMMARY (vs base; board AL §1 cat 63.56 / reg 69.81) ===="
column -t "$SUMMARY"
