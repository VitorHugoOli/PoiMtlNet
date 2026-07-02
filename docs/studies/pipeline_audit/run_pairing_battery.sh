#!/usr/bin/env bash
# pipeline_audit — AL pairing-decomposition battery (2026-07-01, user request).
# Question: WHY does aligned pairing hurt AL (G0.1 advisory −4.77 cat, seed-0)?
# Design: 5 arms × seeds {0,1,7,100} × 5 folds, all on the SAME code/session/device:
#   base    — champion v17 (independent shuffles; 2×bs distinct windows per step)
#   aligned — --aligned-pairing (joint loader, shared perm; bs distinct windows/step)
#   derange — --aligned-pairing + MTL_ALIGNED_DERANGE=1 (SAME joint machinery/perm,
#             task-b rolled by 1 → random-like pairing at aligned-arm diversity).
#             aligned vs derange = pairing SEMANTICS; derange vs base = loader
#             structure / per-step sample diversity.
#   alcond  — aligned + cond_coupling=posterior cond_dim=7 cond_inject=add
#             (the untested cell: per-sample cat→reg conditioning WITH alignment)
#   cond    — cond_coupling WITHOUT alignment (R-CC's historical confounded form;
#             MTL_STRICT relaxed for this arm — the new cond-guard hard-fails it
#             by design; kept as the quantified-confound control)
# Serial; ~10.5 min/run compiled ⇒ ~3.5 h. Seed-major so seed-0 gives an early read.
set -u
source /home/vitor.oliveira/.venv/bin/activate
cd "$(dirname "$0")/../../.."
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/pipeline_audit/pairing_battery; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
[ -f "$SUMMARY" ] || echo -e "arm\tseed\tcat\treg\twall\trc\trundir" > "$SUMMARY"
st=alabama

COND="--reg-head-param cond_coupling=posterior --reg-head-param cond_dim=7 --reg-head-param cond_inject=add"

run_cell() {  # arm seed extra_flags strict derange
  local arm="$1" sd="$2" extra="$3" strict="$4" derange="$5"
  local cd_="$OUT/${arm}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_pipeline_audit_${st}"
  if [ "$strict" = "1" ]; then export MTL_STRICT=1; else unset MTL_STRICT; fi
  if [ "$derange" = "1" ]; then export MTL_ALIGNED_DERANGE=1; else unset MTL_ALIGNED_DERANGE; fi
  unset MTL_TRAIN_DIAGNOSTICS MTL_NO_TRAIN_DIAGNOSTICS MTL_PROFILE_JSON
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints $extra > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "${RD:-}" ]; then
    python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag pairing_${arm}_s${sd} > "$cd_/score.txt" 2>&1 || true
    cat=$(grep -oE "cat macro-F1 \(diag-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
    reg=$(grep -oE "reg FULL top10_acc \(indist-best\) *= *[0-9.]+" "$cd_/score.txt" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${arm}\t${sd}\t${cat:--}\t${reg:--}\t${wall}\t${rc}\t$(basename ${RD:-NONE})" >> "$SUMMARY"
  echo "[battery] ${arm} s${sd} rc=$rc wall=${wall}s cat=${cat:--} reg=${reg:--}"
}

echo "[battery] start $(date -u +%H:%M:%S)"
for SD in 0 1 7 100; do
  run_cell base    "$SD" ""                            1 0
  run_cell aligned "$SD" "--aligned-pairing"           1 0
  run_cell derange "$SD" "--aligned-pairing"           1 1
  run_cell alcond  "$SD" "--aligned-pairing $COND"     1 0
  run_cell cond    "$SD" "$COND"                       0 0
done
echo "[battery] DONE $(date -u +%H:%M:%S)"
column -t "$SUMMARY"
