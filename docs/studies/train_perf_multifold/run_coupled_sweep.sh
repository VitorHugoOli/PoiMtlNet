#!/usr/bin/env bash
# AL/AZ COUPLED-HYPERPARAM SWEEP at bs=8192 (user-take B + advisor a7e80a72).
# Tests the hyperparameters that COUPLE to batch size beyond LR — primarily the OneCycle optimizer
# STEP-BUDGET (epochs), plus weight_decay and pct_start — measuring BOTH quality AND wall.
# Reference cell = ref8k (ep50/wd0.05/default pct). Champion-G otherwise held. seed 0, 5-fold.
# Waits for the FL settle run to finish first (clean GPU → honest wall timing). SEQUENTIAL (clean wall).
# PID-keyed rundir capture (race-free).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
OUT=$D/coupled_sweep_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "state\tcell\tepochs\twd\tpct\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

# --- wait for FL settle to finish (clean GPU) ---
echo "[sweep] waiting for FL settle to finish before starting (clean wall timing)..."
for i in $(seq 1 480); do  # up to 8h
  if grep -q "ALL RUNS DONE" "$D/fl_settle_runs/DRIVER.log" 2>/dev/null; then echo "[sweep] FL done at t=${i}min — starting"; break; fi
  sleep 60
done

# cell:epochs:wd:pct   (pct '-' = use OneCycle default 0.3)
CELLS=(
  "ref8k:50:0.05:-"      # 8k baseline reference (same-session apples-to-apples)
  "ep75:75:0.05:-"       # step-budget restore (advisor #1, highest info)
  "ep65:65:0.05:-"       # step-budget dose-response (#2)
  "wd0.10:50:0.10:-"     # stronger reg / per-step (#3)
  "wd0.025:50:0.025:-"   # weaker reg (#4)
  "ps0.40:50:0.05:0.40"  # warmup shape (#5)
  "ep65wd025:65:0.025:-" # combo (#6)
)

run_cell() {
  local st="$1" cell="$2" ep="$3" wd="$4" pct="$5"
  local cd_="$OUT/${st}_${cell}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  local pctflag=""; [ "$pct" != "-" ] && pctflag="--pct-start $pct"
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_cs_${st}_${cell}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed 0 --epochs "$ep" --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 $pctflag --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --weight-decay "$wd" \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep${ep}_*_${pid} 2>/dev/null | head -1)
  [ -z "$RD" ] && RD=$(ls -dt results/$OVL/$st/mtlnet_*bs8192_ep${ep}_* 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag cs_${st}_$cell 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${st}\t${cell}\t${ep}\t${wd}\t${pct}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[sweep] ${st}/${cell} ep=$ep wd=$wd pct=$pct rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

for st in alabama arizona; do
  for c in "${CELLS[@]}"; do IFS=':' read -r cell ep wd pct <<< "$c"; run_cell "$st" "$cell" "$ep" "$wd" "$pct"; done
done
echo "[sweep] ALL DONE"
echo "==== COUPLED SWEEP (bs=8192, seed0 5f). ref8k = ep50/wd0.05. Quality + wall. ===="
column -t "$SUMMARY"
