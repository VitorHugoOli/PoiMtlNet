#!/usr/bin/env bash
# pipeline_audit — AL v17 baseline, ONE fold (fold 0 of the canonical 5-split), seed 0, --profile.
# Purpose: authoritative per-section profiler baseline (data/forward/backward/train_metric/eval,
# batch/s, GPU util, peak mem) for the pipeline-audit perf phase. Mirrors
# docs/studies/closing_data/run_catx_v17_audit_1fold.sh run_cell exactly (same env + recipe),
# state=alabama. fp32 board protocol (MTL_DISABLE_AMP=1) for comparability with RESULTS_BOARD A40 rows.
set -u
source /home/vitor.oliveira/.venv/bin/activate   # no python on PATH by default (a40-wk)
cd "$(dirname "$0")/../../.."   # repo root
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/pipeline_audit/runs; mkdir -p "$OUT"

TAG="${1:-al_v17_f0_baseline}"
st=alabama; sd=0
cd_="$OUT/${TAG}"; mkdir -p "$cd_"; log="$cd_/run.log"; S=$SECONDS
export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_pipeline_audit_${st}"
export MTL_PROFILE_JSON="$cd_/profile.json"
echo "[audit] launch $st/s$sd fold0 tag=$TAG $(date -u +%H:%M:%S)"
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
pid=$!; wait $pid; rc=$?; wall=$((SECONDS-S))
nan=$(grep -ciE "non-finite (grad|loss)|out of memory|OutOfMemory" "$log" 2>/dev/null || echo 0)
RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
echo "[audit] $st/s$sd fold0 rc=$rc wall=${wall}s nan=$nan rundir=${RD:-NONE}" | tee "$cd_/summary.txt"
if [ -n "${RD:-}" ]; then
  echo "rundir=$RD" >> "$cd_/summary.txt"
  python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag pipeline_audit_${TAG} 2>&1 | tee -a "$cd_/summary.txt" || true
fi
echo "[audit] DONE $TAG"
