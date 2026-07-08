#!/usr/bin/env bash
# H3 step 3: Istanbul on dk_ovl+v17 — MTL n=20 + STL cat ceiling n=20 + STL reg ceiling n=20.
# Interleaved by seed so s0 forms the SANITY GATE (beats-cat/matches-reg must replicate; the doc says STOP if it flips).
# Small state (520 regions) → cat ceiling uses the small-state recipe bs2048@0.005 (per user 2026-07-02). fp32 board-invariant.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=src
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
BASE=docs/studies/closing_data/v17_completion/h3_istanbul
OUT=$BASE/step3_runs; mkdir -p "$OUT"
DRV=$BASE/step3_DRIVER.log; : > "$DRV"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DRV"; }

mtl_run(){ local SD=$1 log="$OUT/mtl_s${SD}.log"
  export MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1 MTL_ONECYCLE_PER_HEAD_LR=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_h3ist_mtl_s${SD}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state istanbul --seed "$SD" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/istanbul" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?
  local RD; RD=$(ls -d results/$OVL/istanbul/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  [ $rc -ne 0 -o -z "$RD" ] && { log "  MTL s${SD} FAIL rc=$rc rd='$RD'"; return 1; }
  local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$SD" --tag h3ist_mtl_s${SD} 2>/dev/null)
  local cat reg
  cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  echo "$cat" > "$OUT/mtl_cat_s${SD}.txt"; echo "$reg" > "$OUT/mtl_reg_s${SD}.txt"
  log "  MTL s${SD} DONE cat=$cat reg=$reg"
}

cat_ceil(){ local SD=$1 log="$OUT/cat_s${SD}.log"
  MTL_NO_TRAIN_DIAGNOSTICS=1 python scripts/train.py --task next --state istanbul --engine "$OVL" \
    --model next_gru --folds 5 --epochs 50 --seed "$SD" \
    --batch-size 2048 --max-lr 0.005 --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?
  local RD; RD=$(ls -dt results/$OVL/istanbul/next_*_${pid} 2>/dev/null | head -1)
  [ $rc -ne 0 -o -z "$RD" ] && { log "  CAT s${SD} FAIL rc=$rc"; return 1; }
  python scripts/closing_data/score_stl_cat_ceiling.py "$RD" --tag h3ist_cat_s${SD} >> "$log" 2>&1
  local v; v=$(python -c "import json;print(json.load(open('$RD/stl_cat_ceiling_score.json'))['cat_macro_f1_mean'])")
  echo "$v" > "$OUT/cat_ceil_s${SD}.txt"; log "  CAT ceiling s${SD} DONE = $v"
}

reg_ceil(){ local SD=$1 log="$OUT/reg_s${SD}.log"
  export MTL_CHUNK_VAL_METRIC=1
  python -u scripts/p1_region_head_ablation.py --state istanbul --heads next_stan_flow \
    --input-type region --region-emb-source "$V14" \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --engine-override "$OVL" --folds 5 --epochs 50 --seed "$SD" --target region \
    --compile --tf32 --tag istanbul_ovl_stl_reg_s${SD} > "$log" 2>&1
  local rc=$?; [ $rc -ne 0 ] && { log "  REG s${SD} FAIL rc=$rc"; return 1; }
  local v; v=$(python -c "import json;d=json.load(open('docs/results/P1/region_head_istanbul_region_5f_50ep_istanbul_ovl_stl_reg_s${SD}.json'));print(round(d['heads']['next_stan_flow']['aggregate']['top10_acc_mean']*100,4))")
  echo "$v" > "$OUT/reg_ceil_s${SD}.txt"; log "  REG ceiling s${SD} DONE = $v"
}

log "H3 step3 START — MTL + cat + reg ceilings, n=20, dk_ovl+v17"
for SD in 0 1 7 100; do
  log "=== seed $SD ==="
  mtl_run "$SD" || true
  cat_ceil "$SD" || true
  reg_ceil "$SD" || true
  if [ "$SD" = "0" ]; then
    MC=$(cat "$OUT/mtl_cat_s0.txt" 2>/dev/null); MR=$(cat "$OUT/mtl_reg_s0.txt" 2>/dev/null)
    CC=$(cat "$OUT/cat_ceil_s0.txt" 2>/dev/null); RC=$(cat "$OUT/reg_ceil_s0.txt" 2>/dev/null)
    DC=$(python -c "print(round(float('${MC:-nan}')-float('${CC:-nan}'),2))" 2>/dev/null)
    DR=$(python -c "print(round(float('${MR:-nan}')-float('${RC:-nan}'),2))" 2>/dev/null)
    log "GATE s0: MTLcat=$MC STLcat=$CC Δcat=$DC | MTLreg=$MR STLreg=$RC Δreg=$DR"
    log "GATE check: Δcat should be >0 (beats), Δreg should be >~-2 (matches/beats). If flipped, INSPECT before trusting {1,7,100}."
  fi
done
log "H3 step3 COMPLETE"

python - "$OUT" <<'PY' 2>>"$DRV" | tee -a "$DRV"
import os,statistics as st,sys
OUT=sys.argv[1]
def col(pfx):
    v={}
    for sd in (0,1,7,100):
        f=f"{OUT}/{pfx}_s{sd}.txt"
        if os.path.exists(f):
            try: v[sd]=float(open(f).read().strip())
            except: pass
    return v
mc,mr,cc,rc=col("mtl_cat"),col("mtl_reg"),col("cat_ceil"),col("reg_ceil")
def m(d): return st.mean(list(d.values())) if d else float('nan')
print("\n==== Istanbul dk_ovl+v17 (n=20) ====")
print(f"MTL cat  {m(mc):.3f} (n={len(mc)})   STL cat ceiling {m(cc):.3f} (n={len(cc)})   Δcat {m(mc)-m(cc):+.2f}")
print(f"MTL reg  {m(mr):.3f} (n={len(mr)})   STL reg ceiling {m(rc):.3f} (n={len(rc)})   Δreg {m(mr)-m(rc):+.2f}")
PY
