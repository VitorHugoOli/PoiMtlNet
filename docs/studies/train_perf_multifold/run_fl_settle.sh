#!/usr/bin/env bash
# FL SETTLE — the FL batch result was n=1 (1 fold). Run FL × {base 2048, 8k 8192} × seeds {0,1,7,100}
# × 5-fold = n=20/cell to test whether the cat -0.58 (and reg +0.34) survive past single-fold noise.
# Quality is the axis (deterministic) → base+8k run CONCURRENT per seed (FL base ~12GB + 8k ~18GB < 45GB free).
# PID-keyed rundir capture (race-free). fp32 (MTL_DISABLE_AMP=1, large-C FL).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/fl_settle_runs
mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "tag\tbs\tseed\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

run_one() {
  local tag="$1" bs="$2" sd="$3"
  local cd_="$OUT/${tag}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  ( export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_flset_${tag}_s${sd}"
    python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
      --state florida --seed "$sd" --epochs 50 --folds 5 --batch-size "$bs" \
      --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
      --cat-head next_gru --reg-head next_stan_flow_dualtower \
      --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
      --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
      --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
      --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
      --model mtlnet_crossattn_dualtower --compile --tf32 \
      --per-fold-transition-dir "output/$V14/florida" --no-checkpoints
  ) > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  # base/8k use different bs globs (no collision); seeds are sequential so newest-per-bs == this run.
  local RD; RD=$(ls -dt results/$OVL/florida/mtlnet_*bs${bs}_ep50_* 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag flset_${tag}_s$sd 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${tag}\t${bs}\t${sd}\t${RD##*_}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[flset] ${tag}/s${sd} rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan rd=$(basename ${RD:-NONE})"
}

# Concurrent base+8k per seed; seeds sequential. Scoring deferred to a clean post-pass (avoid ls -dt race).
for sd in 0 1 7 100; do
  echo "[flset] === seed $sd: base + 8k concurrent ==="
  run_one base 2048 "$sd" &
  P1=$!
  run_one 8k   8192 "$sd" &
  P2=$!
  wait $P1 $P2
done
echo "[flset] ALL RUNS DONE — see post-score pass (run_fl_settle_score.sh) for race-free per-rundir means"
column -t "$SUMMARY"
