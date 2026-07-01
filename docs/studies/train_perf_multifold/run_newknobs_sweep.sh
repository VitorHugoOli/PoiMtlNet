#!/usr/bin/env bash
# NEW-KNOBS sweep at bs=8192 (web-research agent a46de69c top toggle-now levers).
# Tests low-risk quality levers BEYOND the coupled sweep (epochs/wd/pct were already covered):
#   category_weight re-balance (cat<->reg tradeoff shifts at 4x batch), logit-adjustment on cat
#   (macro-F1-aligned), large-batch stabilizers (adam beta2=0.95, grad-clip 0.5), + a stabilizer combo.
# AL+AZ, seed-0, 5-fold. ref8k = champion bs8192 (cw0.75, beta2 0.999, clip 1.0). Champion-G otherwise held.
# Waits for the coupled sweep to finish first. 3-wide parallel (quality axis; PID-keyed scoring, race-free).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
OUT=$D/newknobs_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "state\tcell\textra\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"
MAX_PAR="${1:-3}"

# wait for the coupled sweep to finish (clean GPU)
echo "[newknobs] waiting for coupled sweep to finish..."
for i in $(seq 1 360); do
  if grep -q "\[sweep\] ALL DONE" "$D/coupled_sweep_runs/DRIVER.log" 2>/dev/null; then echo "[newknobs] coupled sweep done at t=${i}min — starting"; break; fi
  sleep 60
done

# name:extra-flags  (appended AFTER the baked --category-weight 0.75, so cw overrides win)
CELLS=(
  "ref8k:"
  "cw0.70:--category-weight 0.70"
  "cw0.80:--category-weight 0.80"
  "logitadj:--logit-adjust-tau 1.0"
  "gradclip05:--max-grad-norm 0.5"
  "beta2_095:--adam-beta2 0.95"
  "stabcombo:--adam-beta2 0.95 --max-grad-norm 0.5"
)

run_cell() {
  local st="$1" cell="$2" extra="$3"
  local cd_="$OUT/${st}_${cell}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_nk_${st}_${cell}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed 0 --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints $extra > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag nk_${st}_$cell 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${st}\t${cell}\t${extra}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[newknobs] ${st}/${cell} (${extra:-none}) rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[newknobs] ${MAX_PAR}-wide parallel; ref8k = champion bs8192 baseline"
running=0
for st in alabama arizona; do
  for c in "${CELLS[@]}"; do
    IFS=':' read -r cell extra <<< "$c"
    run_cell "$st" "$cell" "$extra" &
    running=$((running+1))
    [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
  done
done
wait
echo "[newknobs] ALL DONE"
echo "==== NEW-KNOBS SWEEP (bs=8192, seed0 5f). ref8k = champion baseline. ===="
column -t "$SUMMARY"
