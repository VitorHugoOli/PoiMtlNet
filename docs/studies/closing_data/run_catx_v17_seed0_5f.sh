#!/usr/bin/env bash
# CA/TX v17 SEED-0, 5-FOLD cell (A40, serial, --profile).
# The board-comparable seed-0 5f cell for the two big states, on the champion v17 recipe
# (v16 + bs8192 + per-head cat-lr). Seeds {1,7,100} go to the H100 (run_catx_v17_n20_h100.sh).
# Serial: each big-state cell ~26-29GB VRAM peak (2-wide won't fit 46GB) + ~50GB host-RAM at
# fold construction. Measured (2026-07-01): CA ~59 min/fold, TX ~66 min/fold -> ~10.5h total.
# fp32 mandatory (Ampere bf16 grad-NaN at large C); dataset stays CPU-resident (NO MTL_DATASET_GPU).
set -u
source /home/vitor.oliveira/.venv/bin/activate   # no python on PATH by default (a40-wk)
cd "$(dirname "$0")/../../.."   # repo root
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/closing_data/catx_v17_seed0_5f; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"; echo -e "state\tseed\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

run_cell() {  # state seed
  local st="$1" sd="$2"
  local cd_="$OUT/${st}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export MTL_RAM_HEADROOM_GB=24
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_catxv17s0_${st}"
  export MTL_PROFILE_JSON="$cd_/profile.json"
  echo "[s0_5f] launch $st/s$sd  $(date -u +%H:%M:%S)"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --profile \
    --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)|out of memory|OutOfMemory" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "${RD:-}" ]; then
    local sc; sc=$(PYTHONPATH=src python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag catxv17s0_${st} 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 .*= *[0-9.]+" | grep -oE "[0-9.]+ *±" | grep -oE "[0-9.]+" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc .*= *[0-9.]+" | grep -oE "[0-9.]+ *±" | grep -oE "[0-9.]+" | head -1)
  fi
  echo -e "${st}\t${sd}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[s0_5f] ${st}/s${sd} rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan rd=$(basename ${RD:-NONE})"
}

echo "[s0_5f] CA+TX v17 seed-0 5-fold (serial, --profile, fp32 bs8192 per-head)"
run_cell california 0
run_cell texas 0
echo "[s0_5f] DONE"; cat "$SUMMARY"
