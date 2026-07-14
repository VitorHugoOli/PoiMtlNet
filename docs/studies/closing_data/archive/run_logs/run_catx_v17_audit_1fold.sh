#!/usr/bin/env bash
# CA/TX v17 AUDIT — ONE fold each (fold 0 of the canonical 5-split), seed 0, --profile.
# Purpose: sanity-check the v17 recipe (bs8192 + per-head cat-lr) runs end-to-end at the
# BIG states on the A40 and produces a sane fold-0 number, WITHOUT committing to the full
# n=20 (which goes to the H100 — see run_catx_v17_n20_h100.sh). Serial (1 cell at a time):
# each big-state fold is ~22GB VRAM / ~50GB peak host-RAM at construction; serial keeps both
# comfortably under the A40 46GB / 125GB. fp32 mandatory (bf16 grad-NaN at large C).
set -u
source /home/vitor.oliveira/.venv/bin/activate   # no python on PATH by default (a40-wk)
cd "$(dirname "$0")/../../.."   # repo root
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/closing_data/catx_v17_audit; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"; echo -e "state\tseed\tfold\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

run_cell() {  # state seed
  local st="$1" sd="$2"
  local cd_="$OUT/${st}_s${sd}_f0"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export MTL_RAM_HEADROOM_GB=24
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_catxv17audit_${st}"
  export MTL_PROFILE_JSON="$cd_/profile.json"
  echo "[audit] launch $st/s$sd fold0  $(date -u +%H:%M:%S)"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --profile --only-fold 0 \
    --state "$st" --seed "$sd" --epochs 50 --batch-size 8192 \
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
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag catxv17audit_${st}_f0 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat[= ]+[0-9.]+" | grep -oE "[0-9.]+" | head -1)
    reg=$(echo "$sc" | grep -oE "reg[= ]+[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  fi
  echo -e "${st}\t${sd}\t0\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[audit] ${st}/s${sd} fold0 rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan rd=$(basename ${RD:-NONE})"
}

echo "[audit] CA/TX v17 fold-0 audit (serial, --profile, fp32 bs8192 per-head)"
run_cell california 0
run_cell texas 0
echo "[audit] DONE"; cat "$SUMMARY"
