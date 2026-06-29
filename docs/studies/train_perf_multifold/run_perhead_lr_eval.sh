#!/usr/bin/env bash
# PER-HEAD LR evaluation — validate the new MTL_ONECYCLE_PER_HEAD_LR fix at AL, then (queued) the real FL test.
# Validates: (1) flag OFF = current champion (uniform 3e-3); (2) flag ON + equal LRs == OFF (parity → impl correct);
# (3) flag ON + per-head 1e-3/3e-3/1e-3 = the INTENDED per-head LR — does it help/hurt AL?
# bs=8192, seed0, 5-fold, PID-keyed. AL runs 1-wide (alongside the FL cw0.80/pct045 cells → keep VRAM safe).
# STATE/MAXPAR via args: run_perhead_lr_eval.sh <state> <max_par>
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
STATE="${1:-alabama}"; MAX_PAR="${2:-1}"
OUT=$D/perhead_lr_${STATE}_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "cell\tperhead\tcatlr\treglr\tsharedlr\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

# name:perhead(0/1):catlr:reglr:sharedlr
CELLS=(
  "off_champ:0:1e-3:3e-3:1e-3"     # flag OFF → uniform 3e-3 (= current champion ref8k). baseline
  "on_equal:1:3e-3:3e-3:3e-3"      # flag ON, all equal → [3e-3,3e-3,3e-3] = uniform. PARITY: must == off_champ
  "on_perhead:1:1e-3:3e-3:1e-3"    # flag ON, champion ratios → [1e-3,3e-3,1e-3] = intended per-head LR
)

run_cell() {
  IFS=':' read -r cell ph clr rlr slr <<< "$1"
  local cd_="$OUT/${cell}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_ph_${STATE}_${cell}"
  if [ "$ph" = "1" ]; then export MTL_ONECYCLE_PER_HEAD_LR=1; else unset MTL_ONECYCLE_PER_HEAD_LR; fi
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$STATE" --seed 0 --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr "$clr" --reg-lr "$rlr" --shared-lr "$slr" \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$STATE" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/$STATE/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag ph_${STATE}_$cell 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${cell}\t${ph}\t${clr}\t${rlr}\t${slr}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[perhead-$STATE] ${cell} ph=$ph lr=$clr/$rlr/$slr rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[perhead-$STATE] ${MAX_PAR}-wide. off_champ=baseline; on_equal must==off_champ (parity); on_perhead=intended"
running=0
for c in "${CELLS[@]}"; do
  run_cell "$c" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[perhead-$STATE] ALL DONE"; column -t "$SUMMARY"
echo "PARITY CHECK: off_champ vs on_equal should be byte-identical (validates the per-group impl)."