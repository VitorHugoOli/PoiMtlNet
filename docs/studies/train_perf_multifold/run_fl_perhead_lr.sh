#!/usr/bin/env bash
# FL PER-HEAD LR validation — the mechanism-correct FL fix, enabled by MTL_ONECYCLE_PER_HEAD_LR.
# Mechanism (agent ade63a19): FL cat regresses at bs8192 because the reg head (C=4703) captures the shared
# backbone once gradient noise drops. LEVER: LOWER the reg head's OneCycle peak (reg_max 3e-3 → 2.5/2.0e-3) to
# reduce reg's backbone dominance, holding cat+shared at 3e-3 → does FL cat recover toward base-2048 79.83
# WITHOUT giving up the reg gain (~75.9)? FL bs8192 seed0 5-fold, per-head ON, PID-keyed.
# Waits for the current FL fix run (cw0.80/pct045) to finish first. 2-wide (FL 8k ~18GB → 36GB < free).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
OUT=$D/fl_perhead_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "cell\treglr\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"
MAX_PAR="${1:-2}"

echo "[flph] waiting for the current FL fix run (cw0.80/pct045) to finish..."
for i in $(seq 1 360); do
  grep -q "\[flcatlr\] ALL DONE" "$D/fl_catlr_runs/DRIVER.log" 2>/dev/null && { echo "[flph] prior FL run done at t=${i}min"; break; }
  sleep 60
done

# cell:reglr (cat+shared held at 3e-3; per-head ON makes reg peak = reglr)
CELLS=(
  "ctrl:3e-3"      # per-head ON all 3e-3 = uniform → must == ref8k 78.76 (validates ON-equal at FL)
  "reg2.5e3:2.5e-3"
  "reg2e3:2e-3"
)

run_cell() {
  IFS=':' read -r cell rlr <<< "$1"
  local cd_="$OUT/${cell}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export MTL_ONECYCLE_PER_HEAD_LR=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_flph_${cell}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state florida --seed 0 --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 3e-3 --reg-lr "$rlr" --shared-lr 3e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/florida" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/florida/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag flph_$cell 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${cell}\t${rlr}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[flph] ${cell} reg_lr=$rlr rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[flph] ${MAX_PAR}-wide. ref8k 78.76/77.42; base-2048 target 79.83/75.58. Lower reg peak → recover cat?"
running=0
for c in "${CELLS[@]}"; do
  run_cell "$c" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[flph] ALL DONE"; column -t "$SUMMARY"