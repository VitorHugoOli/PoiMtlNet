#!/usr/bin/env bash
# FL PER-HEAD LR isolation — cleanly separate the cat-LR lever from the shared-LR lever.
# perhead (cat1/reg3/shared1) recovered FL cat +0.96 but lowered BOTH cat AND shared LR. These two cells
# isolate which one matters:
#   cat_only    (cat1/reg3/shared3) — lowers ONLY the cat-head LR. If this recovers → cat-LR-overshoot.
#   shared_only (cat3/reg3/shared1) — lowers ONLY the shared-backbone LR. If this recovers → backbone/drift
#                                     (consistent with a reg-capture style effect).
# FL bs8192 seed0 5-fold, per-head ON, PID-keyed. Waits for fl_perhead_runs to finish. 2-wide.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
OUT=$D/fl_isolate_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "cell\tcatlr\treglr\tsharedlr\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"
MAX_PAR="${1:-2}"

echo "[fliso] waiting for fl_perhead_runs to finish..."
for i in $(seq 1 480); do
  grep -q "\[flph\] ALL DONE" "$D/fl_perhead_runs/DRIVER.log" 2>/dev/null && { echo "[fliso] fl_perhead done at t=${i}min"; break; }
  sleep 60
done

CELLS=(
  "cat_only:1e-3:3e-3:3e-3"      # lower ONLY cat-head LR
  "shared_only:3e-3:3e-3:1e-3"   # lower ONLY shared-backbone LR
)

run_cell() {
  IFS=':' read -r cell clr rlr slr <<< "$1"
  local cd_="$OUT/${cell}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export MTL_ONECYCLE_PER_HEAD_LR=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_fliso_${cell}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state florida --seed 0 --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr "$clr" --reg-lr "$rlr" --shared-lr "$slr" \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/florida" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/florida/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag fliso_$cell 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${cell}\t${clr}\t${rlr}\t${slr}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[fliso] ${cell} lr=$clr/$rlr/$slr rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[fliso] ${MAX_PAR}-wide. ref8k 78.76; perhead(cat+shared↓) recovered to 79.72. cat_only vs shared_only isolate."
running=0
for c in "${CELLS[@]}"; do
  run_cell "$c" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[fliso] ALL DONE"; column -t "$SUMMARY"