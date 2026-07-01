#!/usr/bin/env bash
# pipeline_audit — post-fix gates: (1) eager 8-ep parity arm (must be byte-identical
# to the PRE-edit al_parity_eager_off artifacts), (2) full 5-fold compiled AL v17
# champion run (final quality gate vs board expectations).
set -u
source /home/vitor.oliveira/.venv/bin/activate
cd "$(dirname "$0")/../../.."
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/pipeline_audit/runs; mkdir -p "$OUT"
st=alabama; sd=0

common_env() {
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_pipeline_audit_${st}"
  unset MTL_NO_TRAIN_DIAGNOSTICS
}

run_arm() {  # tag epochs compile_flags fold_flags
  local tag="$1" epochs="$2" compileflags="$3" foldflags="$4"
  local cd_="$OUT/${tag}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  common_env
  export MTL_PROFILE_JSON="$cd_/profile.json"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --profile $foldflags \
    --state "$st" --seed "$sd" --epochs "$epochs" --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple $compileflags \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$log" 2>&1
  local rc=$?; local wall=$((SECONDS-S))
  local RD; RD=$(ls -dt results/$OVL/$st/mtlnet_*ep${epochs}_* 2>/dev/null | head -1)
  echo "[gate] $tag rc=$rc wall=${wall}s rundir=${RD:-NONE}" | tee "$cd_/summary.txt"
  [ -n "${RD:-}" ] && echo "$RD" > "$cd_/rundir.txt"
  if [ -n "${RD:-}" ] && [ "$epochs" = "50" ]; then
    python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag pipeline_audit_${tag} 2>&1 | tee -a "$cd_/summary.txt" || true
  fi
}

echo "[gate] post-fix gates start $(date -u +%H:%M:%S)"
run_arm postfix_parity_eager 8  ""                 "--only-fold 0"
run_arm postfix_5fold        50 "--compile --tf32" ""
echo "[gate] DONE $(date -u +%H:%M:%S)"
