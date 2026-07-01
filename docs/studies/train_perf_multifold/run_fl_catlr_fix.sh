#!/usr/bin/env bash
# FL CAT-LR FIX — test the research mechanism (agent ade63a19): FL cat regresses at bs8192 because the
# harder reg head (C=4703) captures the shared backbone once gradient noise drops. The surgical fix is to
# scale ONLY the cat head's LR (√-scale for 4x batch), leaving reg-lr untouched so the reg gain is preserved.
# FL, bs=8192, seed0, 5-fold, fp32. Target to BEAT: FL base bs2048 cat 79.83 / reg 75.58 (settle).
# ref8k bs8192 baseline ~78.76 cat / 75.92 reg. 2-wide (FL 8k ~18GB → 36GB < 45GB free). PID-keyed scoring.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
OUT=$D/fl_catlr_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "cell\textra\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"
MAX_PAR="${1:-2}"

# name:extra-flags (appended; override the baked --cat-lr 1e-3 / --category-weight 0.75 / pct default)
CELLS=(
  "ref8k:"
  "catlr2e3:--cat-lr 2e-3"
  "catlr1.5e3:--cat-lr 1.5e-3"
  "cw0.80:--category-weight 0.80"
  "pct045:--pct-start 0.45"
)

run_cell() {
  local cell="$1" extra="$2"
  local cd_="$OUT/${cell}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_flcl_${cell}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state florida --seed 0 --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/florida" --no-checkpoints $extra > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/florida/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag flcl_$cell 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${cell}\t${extra}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[flcatlr] ${cell} (${extra:-none}) rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[flcatlr] ${MAX_PAR}-wide. Target: beat ref8k 78.76 cat toward base-2048 79.83, keep reg ~75.9"
running=0
for c in "${CELLS[@]}"; do
  IFS=':' read -r cell extra <<< "$c"
  run_cell "$cell" "$extra" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[flcatlr] ALL DONE"
echo "==== FL CAT-LR FIX (bs8192 seed0 5f). ref8k 78.76/75.92; base-2048 target 79.83/75.58 ===="
column -t "$SUMMARY"
