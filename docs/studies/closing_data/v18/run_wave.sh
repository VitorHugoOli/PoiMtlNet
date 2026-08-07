#!/usr/bin/env bash
# v18 — run ONE wave (one seed) across all six states and all three model families.
#
#   bash docs/studies/closing_data/v18/run_wave.sh <SEED>
#
# Families, all on the v18 engine, recipe frozen at v17:
#   cat   dedicated next-category   train.py --task next --model next_gru  (per-state bs/lr)
#   reg   dedicated next-region     p1_region_head_ablation.py --heads next_stan_flow
#   joint v17 MTL                   train.py --task mtl (bs8192, per-head cat-lr 1e-3)
#
# Protocol notes that are NOT free choices:
#   - fp32 (MTL_DISABLE_AMP=1) + --compile --tf32 everywhere: this is what every published v17
#     comparand used. bf16 grad-NaNs on this A40 at CA/TX class counts, and a non-compiled v18
#     differenced against a compiled v17 would cross protocols.
#   - MTL_ONECYCLE_PER_HEAD_LR=1 is what makes --cat-lr/--reg-lr take effect at all. Without it
#     the scalar max_lr broadcasts to every group and the run is silently v16, not v17.
#   - --canon none + the full explicit recipe: auto-canon would pin the v14 substrate and
#     MTL_STRICT=1 would hard-fail the wrong-substrate guard.
#
# Resumable + idempotent: a cell whose sidecar exists is SKIPPED, loudly. Rundirs are captured by
# the launched PID, never by newest-mtime globbing (two jobs run concurrently by design).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SEED=${1:?usage: run_wave.sh <SEED>}
ENG=check2hgi_v18
V14=check2hgi_design_k_resln_mae_l0_1
BASE=docs/studies/closing_data/v18
SIDE=docs/results/closing_data/v18
LOGS=$BASE/logs/wave_s${SEED}
RUNNING=$BASE/logs/running.jsonl
DRV=$BASE/logs/wave_s${SEED}_DRIVER.log
mkdir -p "$LOGS" "$SIDE" "$BASE/logs"
SHA=$(git rev-parse HEAD)
RAM_FLOOR_GB=80          # a CA/TX MTL dataset build peaks ~66 GB host RAM; box is shared, 125 GB

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }
phase(){ python "$BASE/status_update.py" --phase "wave${SEED}" >/dev/null 2>&1 || true; }
memavail_gb(){ awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo; }
flag(){ echo "$*" >> "$BASE/logs/verify_flags.jsonl"; log "[VERIFY] $*"; }

wait_ram(){
  local a
  a=$(memavail_gb)
  local waited=0
  while [ "$a" -lt "$RAM_FLOOR_GB" ]; do
    [ $((waited % 600)) -eq 0 ] && log "    waiting for RAM: ${a} GB < ${RAM_FLOOR_GB} GB floor (shared box)"
    sleep 60; waited=$((waited+60)); a=$(memavail_gb)
  done
}

mark_running(){
  local st=$1; local fam=$2; local pid=$3
  echo "{\"state\":\"$st\",\"seed\":$SEED,\"family\":\"$fam\",\"pid\":$pid,\"started_at\":\"$(date -Iseconds)\"}" >> "$RUNNING"
  phase
}

# per-state dedicated-category recipe (CEILINGS_N20_FINAL.md; istanbul per h3_istanbul driver)
cat_bs(){ case $1 in alabama|istanbul) echo 2048 ;; *) echo 8192 ;; esac; }

sidecar_write(){  # state family wall rundir cat reg extra_json
  local st=$1; local fam=$2; local wall=$3; local rd=$4; local cv=$5; local rv=$6
  python - "$SIDE/${st}_s${SEED}_${fam}.json" "$st" "$SEED" "$fam" "$wall" "$rd" "$cv" "$rv" "$SHA" <<'PY'
import json, sys
out, st, sd, fam, wall, rd, cv, rv, sha = sys.argv[1:10]
def f(x):
    try: return float(x)
    except Exception: return None
json.dump({
    "state": st, "seed": int(sd), "family": fam,
    "wall_seconds": int(float(wall)), "rundir": rd or None,
    "cat": f(cv), "reg": f(rv),
    "commit_sha": sha,
    "v18_config": {"engine": "check2hgi_v18", "forward_only": True, "in_channels": 15,
                   "node_layout": ["canonical_11", "continuous_time_4"], "repr_seed": 42},
    "protocol": {"precision": "fp32 (MTL_DISABLE_AMP=1)", "compile": True, "tf32": True,
                 "folds": 5, "epochs": 50,
                 "cat_metric": "macro-F1 at f1-best epoch (diag-best)",
                 "reg_metric": "top10_acc_indist * (1 - ood_fraction) * 100 at indist-best epoch"},
}, open(out, "w"), indent=2)
print(out)
PY
}

# ------------------------------------------------------------------ family (a): dedicated cat
cell_cat(){
  local st=$1
  local side="$SIDE/${st}_s${SEED}_cat.json"
  if [ -f "$side" ]; then log "  SKIP $st s$SEED cat (sidecar exists)"; return 0; fi
  local bs; bs=$(cat_bs "$st")
  local lg="$LOGS/${st}_cat.log"
  local t0=$SECONDS
  log "  START $st s$SEED cat (bs=$bs lr=0.005)"
  MTL_NO_TRAIN_DIAGNOSTICS=1 python scripts/train.py --task next --state "$st" --engine "$ENG" \
    --model next_gru --embedding-dim 64 --folds 5 --epochs 50 --seed "$SEED" \
    --batch-size "$bs" --max-lr 0.005 --compile --tf32 --no-checkpoints > "$lg" 2>&1 &
  local pid=$!
  mark_running "$st" cat "$pid"
  wait $pid; local rc=$?
  local rd; rd=$(ls -d results/$ENG/$st/next_*_${pid} 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$rd" ]; then log "  FAIL  $st s$SEED cat rc=$rc rd='$rd' (see $lg)"; return 1; fi
  python scripts/closing_data/score_stl_cat_ceiling.py "$rd" --tag v18_${st}_cat_s${SEED} >> "$lg" 2>&1
  local v; v=$(python -c "import json;print(json.load(open('$rd/stl_cat_ceiling_score.json'))['cat_macro_f1_mean'])" 2>/dev/null)
  [ -z "$v" ] && { log "  FAIL  $st s$SEED cat: no score"; return 1; }
  sidecar_write "$st" cat $((SECONDS-t0)) "$rd" "$v" "" >/dev/null
  log "  DONE  $st s$SEED cat = $v  ($((SECONDS-t0))s)"
  phase
}

# ------------------------------------------------------------------ family (b): dedicated reg
cell_reg(){
  local st=$1
  local side="$SIDE/${st}_s${SEED}_reg.json"
  if [ -f "$side" ]; then log "  SKIP $st s$SEED reg (sidecar exists)"; return 0; fi
  local lg="$LOGS/${st}_reg.log"
  local tag="v18_${st}_reg_s${SEED}"
  local t0=$SECONDS
  log "  START $st s$SEED reg"
  export MTL_CHUNK_VAL_METRIC=1
  python -u scripts/p1_region_head_ablation.py --state "$st" --heads next_stan_flow \
    --input-type region --region-emb-source "$V14" \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --engine-override "$ENG" --folds 5 --epochs 50 --seed "$SEED" --target region \
    --max-lr 0.003 --compile --tf32 --tag "$tag" > "$lg" 2>&1 &
  local pid=$!
  mark_running "$st" reg "$pid"
  wait $pid; local rc=$?
  local rj="docs/results/P1/region_head_${st}_region_5f_50ep_${tag}.json"
  if [ $rc -ne 0 ] || [ ! -f "$rj" ]; then log "  FAIL  $st s$SEED reg rc=$rc (see $lg)"; return 1; fi
  local v; v=$(python -c "import json;d=json.load(open('$rj'));print(round(d['heads']['next_stan_flow']['aggregate']['top10_acc_mean']*100,4))" 2>/dev/null)
  [ -z "$v" ] && { log "  FAIL  $st s$SEED reg: no score"; return 1; }
  sidecar_write "$st" reg $((SECONDS-t0)) "$rj" "" "$v" >/dev/null
  log "  DONE  $st s$SEED reg = $v  ($((SECONDS-t0))s)"
  phase
}

# ------------------------------------------------------------------ family (c): joint v17 MTL
cell_joint(){
  local st=$1
  local side="$SIDE/${st}_s${SEED}_joint.json"
  if [ -f "$side" ]; then log "  SKIP $st s$SEED joint (sidecar exists)"; return 0; fi
  case $st in florida|california|texas) wait_ram ;; esac
  local lg="$LOGS/${st}_joint.log"
  local t0=$SECONDS
  log "  START $st s$SEED joint (bs8192, per-head cat-lr 1e-3, fp32, compile)"
  export MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1 MTL_ONECYCLE_PER_HEAD_LR=1
  export MTL_RAM_HEADROOM_GB=12
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_v18_${st}_s${SEED}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$ENG" \
    --state "$st" --seed "$SEED" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$lg" 2>&1 &
  local pid=$!
  mark_running "$st" joint "$pid"
  wait $pid; local rc=$?
  local rd; rd=$(ls -d results/$ENG/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$rd" ]; then log "  FAIL  $st s$SEED joint rc=$rc rd='$rd' (see $lg)"; return 1; fi
  # diag-best (Table-3 convention)
  python scripts/closing_data/a40_score_matched.py "$rd" --seed "$SEED" --tag v18_${st}_joint_s${SEED} > "$LOGS/${st}_joint_score.txt" 2>&1
  # joint-best (the single served checkpoint, JOINT_BEST_SCORING.md)
  python scripts/closing_data/score_joint_best.py "$rd" --seed "$SEED" --tag v18_${st}_joint_s${SEED} >> "$LOGS/${st}_joint_score.txt" 2>&1 || true
  local cv rv
  cv=$(python -c "import json;print(json.load(open('$rd/a40_matched_score.json'))['cat_macro_f1_mean'])" 2>/dev/null)
  rv=$(python -c "import json;print(json.load(open('$rd/a40_matched_score.json'))['reg_full_top10_mean'])" 2>/dev/null)
  [ -z "$cv" ] && { log "  FAIL  $st s$SEED joint: no score"; return 1; }
  sidecar_write "$st" joint $((SECONDS-t0)) "$rd" "$cv" "$rv" >/dev/null
  log "  DONE  $st s$SEED joint cat=$cv reg=$rv  ($((SECONDS-t0))s)"
  # §6.6 per-state sanity — v17 published values (CEILINGS_N20_FINAL.md)
  python - "$st" "$cv" "$rv" <<'PY' | tee -a "$DRV" >> "$BASE/logs/verify_flags.jsonl"
import sys
V17 = {"alabama": (64.54, 69.80), "arizona": (65.83, 59.56), "florida": (79.85, 77.42),
       "california": (77.05, 65.69), "texas": (77.24, 67.06), "istanbul": (63.33, 75.44)}
st, cv, rv = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
c17, r17 = V17[st]
if abs(cv - c17) < 5:
    print(f"{st}: cat {cv:.2f} is within 5 pp of v17 {c17:.2f} -- INVESTIGATE the forward-only path")
if abs(rv - r17) > 2:
    print(f"{st}: reg moved {rv-r17:+.2f} pp vs v17 {r17:.2f} (>2 pp) -- INVESTIGATE")
PY
  phase
}

run_state(){   # all three families for one state, sequentially
  local st=$1
  log "== $st (seed $SEED) =="
  cell_cat   "$st" || true
  cell_reg   "$st" || true
  cell_joint "$st" || true
}

# ================================================================================================
log "===== v18 WAVE seed=$SEED START — commit ${SHA:0:8} ====="
phase

# Small states run 2-wide. Large states (FL/TX/CA) run strictly 1-wide: their MTL dataset build
# peaks ~66 GB host RAM, and catx_v17_seed0_5f/RESULTS.md records 2-wide as infeasible there.
log "--- small states 2-wide: istanbul, alabama, arizona"
run_state istanbul & A=$!
run_state alabama  & B=$!
wait $A; wait $B
run_state arizona

log "--- large states 1-wide: florida, texas, california"
for st in florida texas california; do
  run_state "$st"
done

log "===== v18 WAVE seed=$SEED COMPLETE ====="
phase
python "$BASE/status_update.py" --phase "wave${SEED}" | tee -a "$DRV"
