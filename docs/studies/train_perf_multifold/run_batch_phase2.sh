#!/usr/bin/env bash
# Batch-size Phase-2 confirmation — base / 4k_none / 8k_none (LR UNCHANGED) at 5-fold,
# seed-0, SEQUENTIAL (board protocol → directly comparable to RESULTS_BOARD §1).
# States AL + AZ. Quality is the axis (GPU may be shared). 2 cells run concurrently.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/batch_phase2_runs
mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
MAX_PAR="${1:-2}"
SUMMARY="$OUT/summary.tsv"
echo -e "state\ttag\tbs\tcat_macroF1\treg_acc10\twall_s\trc" > "$SUMMARY"

# state:tag:bs  (LR always unchanged: max 3e-3 / cat 1e-3 / reg 3e-3 / shared 1e-3)
JOBS=(
  "alabama:base:2048" "alabama:4k_none:4096" "alabama:8k_none:8192"
  "arizona:base:2048" "arizona:4k_none:4096" "arizona:8k_none:8192"
)

run_job() {
  IFS=':' read -r st tag bs <<< "$1"
  local cd="$OUT/${st}_${tag}"; mkdir -p "$cd"; local log="$cd/run.log"; local S=$SECONDS
  ( export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_p2_${st}_${tag}"
    python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
      --state "$st" --seed 0 --epochs 50 --folds 5 --batch-size "$bs" \
      --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
      --cat-head next_gru --reg-head next_stan_flow_dualtower \
      --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
      --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
      --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
      --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
      --model mtlnet_crossattn_dualtower --compile --tf32 \
      --per-fold-transition-dir "output/$V14/$st" --no-checkpoints
  ) > "$log" 2>&1
  local rc=$?; local wall=$((SECONDS-S))
  local RD; RD=$(ls -dt results/$OVL/$st/mtlnet_*ep50_* 2>/dev/null | head -1)
  local cat="-" reg="-"
  [ -n "$RD" ] && read -r cat reg < <(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag p2_${st}_$tag 2>/dev/null \
    | grep -oE "= [0-9.]+ ±" | grep -oE "[0-9.]+" | head -2 | tr '\n' ' ')
  echo -e "${st}\t${tag}\t${bs}\t${cat:--}\t${reg:--}\t${wall}\t${rc}" >> "$SUMMARY"
  echo "[p2] ${st}/${tag} done rc=$rc wall=${wall}s cat=$cat reg=$reg"
}

echo "[p2] ${#JOBS[@]} jobs (AL+AZ × base/4k/8k), 5f×50ep seq, max_par=$MAX_PAR"
running=0
for j in "${JOBS[@]}"; do
  run_job "$j" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[p2] ALL DONE"
echo "==== PHASE-2 SUMMARY (board §1: AL 63.56/69.81 · AZ 63.39/59.34) ===="
column -t "$SUMMARY"
