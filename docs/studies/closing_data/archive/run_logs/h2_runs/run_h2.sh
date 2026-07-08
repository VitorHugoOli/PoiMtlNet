#!/usr/bin/env bash
# H2 — v17 STL cat ceiling fan-out (A40).
# Recipe (v17, matched to the MTL cat lever): next_gru, bs8192, OneCycle peak max_lr=1e-3, ep50, 5f.
#   default_next max_lr=0.01 -> --max-lr 1e-3 lowers the cat-head peak to match --onecycle-per-head-lr --cat-lr 1e-3.
#   (A40.md's literal --onecycle-per-head-lr/--cat-lr is MTL-only and is REJECTED for a single-task next run.)
# 5 Gowalla states x seeds {0,1,7,100}. Istanbul deferred to H3 (no dk_ovl substrate yet).
# Sequential (cheap; avoids the concurrent-rundir race). rundir captured by child PID.
set -u
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate

ENGINE=check2hgi_dk_ovl
COLL=docs/results/closing_data/h2_v17_cat_ceiling
RUNS=docs/studies/closing_data/v17_completion/h2_runs
mkdir -p "$COLL" "$RUNS"
DRIVER="$RUNS/DRIVER.log"
: > "$DRIVER"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DRIVER"; }

log "H2 v17 STL cat ceiling fan-out START (engine=$ENGINE, bs8192, max_lr=1e-3, ep50, 5f)"
for ST in alabama arizona florida california texas; do
  for S in 0 1 7 100; do
    RL="$RUNS/${ST}_s${S}.log"
    log "RUN  ${ST} s${S} -> $RL"
    MTL_NO_TRAIN_DIAGNOSTICS=1 python scripts/train.py --task next --state "$ST" --engine "$ENGINE" \
        --model next_gru --folds 5 --epochs 50 --seed "$S" \
        --batch-size 8192 --max-lr 1e-3 --no-checkpoints > "$RL" 2>&1 &
    CHILD=$!
    wait $CHILD; EC=$?
    RD=$(ls -dt results/${ENGINE}/${ST}/next_*_${CHILD} 2>/dev/null | head -1)
    if [ "$EC" -ne 0 ] || [ -z "$RD" ]; then
      log "FAIL ${ST} s${S} exit=$EC rundir='$RD' (see $RL)"; continue
    fi
    python scripts/closing_data/score_stl_cat_ceiling.py "$RD" \
        --tag "${ST}_s${S}_stl_cat_ceiling_v17" >> "$RL" 2>&1
    if [ -f "$RD/stl_cat_ceiling_score.json" ]; then
      cp "$RD/stl_cat_ceiling_score.json" "$COLL/${ST}_s${S}.json"
      F1=$(python -c "import json;print(json.load(open('$COLL/${ST}_s${S}.json'))['cat_macro_f1_mean'])")
      log "DONE ${ST} s${S} cat_macro_f1=${F1}  rundir=$(basename "$RD")"
    else
      log "FAIL ${ST} s${S} scoring produced no sidecar (see $RL)"
    fi
  done
done
log "H2 v17 STL cat ceiling fan-out COMPLETE"

# Aggregate: per-state n=4 mean + per-seed table
python - "$COLL" >> "$DRIVER" 2>&1 <<'PY'
import json, glob, os, sys, statistics as st
coll=sys.argv[1]
rows={}
for f in sorted(glob.glob(os.path.join(coll,"*_s*.json"))):
    b=os.path.basename(f)[:-5]  # state_sSEED
    st_, sd=b.rsplit("_s",1)
    v=json.load(open(f))["cat_macro_f1_mean"]
    rows.setdefault(st_,{})[int(sd)]=v
print("\n=== H2 v17 STL cat ceiling — summary ===")
print(f"{'state':10} {'s0':>7} {'s1':>7} {'s7':>7} {'s100':>7} {'mean':>8} {'std':>7}")
agg={}
for s_ in ["alabama","arizona","florida","california","texas"]:
    d=rows.get(s_,{})
    vals=[d.get(k) for k in (0,1,7,100)]
    have=[x for x in vals if x is not None]
    m=st.mean(have) if have else float('nan')
    sd=st.pstdev(have) if len(have)>1 else 0.0
    agg[s_]=(m,sd,len(have))
    cells=" ".join(f"{(x if x is not None else float('nan')):7.2f}" for x in vals)
    print(f"{s_:10} {cells} {m:8.2f} {sd:7.2f}  (n={len(have)})")
json.dump({k:{'mean':round(v[0],4),'std':round(v[1],4),'n':v[2]} for k,v in agg.items()},
          open(os.path.join(coll,"_aggregate.json"),"w"), indent=2)
print("wrote", os.path.join(coll,"_aggregate.json"))
PY
