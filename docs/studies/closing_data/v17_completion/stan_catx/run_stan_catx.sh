#!/usr/bin/env bash
# Faithful STAN CA/TX, v6 recipe + patience-reduced (quality-neutral: best_epoch ~3 << 10).
# 5 folds, seed 0, 200ep cap, bs2048, bf16+compile, d-model 128. RAM watchdog protects the shared box.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src
BASE=docs/studies/closing_data/v17_completion/stan_catx
DRV=$BASE/stan_DRIVER.log; : > "$DRV"
FLOOR_KB=10000000
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }
watchdog(){ local pid=$1; while kill -0 "$pid" 2>/dev/null; do
    a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$a" -lt "$FLOOR_KB" ] && { log "[watchdog] MemAvailable ${a}KB < floor — SIGKILL $pid"; kill -9 "$pid"; return 1; }
    sleep 5; done; }

for ST in texas california; do
  log "STAN $ST START (v6, patience 10, 5f)"
  python research/baselines/stan/train.py --state "$ST" --seed 0 --epochs 200 --folds 5 \
      --batch-size 2048 --amp bf16 --compile --d-model 128 --patience 10 --tag v6_p10 \
      > "$BASE/stan_${ST}.log" 2>&1 &
  pid=$!; watchdog "$pid" & wpid=$!
  wait "$pid"; rc=$?; kill "$wpid" 2>/dev/null || true
  nan=$(grep -ciE "nan|non-finite" "$BASE/stan_${ST}.log" 2>/dev/null || echo 0)
  log "STAN $ST DONE rc=$rc nan=$nan"
  [ "$rc" -ne 0 ] && { log "ABORT $ST (rc=$rc)"; exit $rc; }
done
log "STAN ALL DONE — aggregating + floor compare"

python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, statistics as st
floors={"texas":{"best_simple":54.94,"markov1":35.60},"california":{"best_simple":52.09,"markov1":31.46}}
print("\n==== Faithful STAN CA/TX (v6, patience10, 5f, seed0) ====")
for ST in ["texas","california"]:
    try:
        d=json.load(open(f"docs/results/baselines/faithful_stan_{ST}_5f_200ep_v6_p10.json"))
        pf=d["per_fold"]; acc=[r["top10_acc"]*100 for r in pf]; be=[r.get("best_epoch") for r in pf]
        m=st.mean(acc); s=st.pstdev(acc)
        f=floors[ST]
        verdict="CLEARS best-simple" if m>f["best_simple"] else ("clears markov-1 only" if m>f["markov1"] else "BELOW FLOOR")
        print(f"{ST:11} Acc@10 = {m:.2f} ± {s:.2f}  best_ep={be}  | floor best-simple {f['best_simple']} / markov {f['markov1']} -> {verdict}")
    except Exception as e:
        print(f"{ST:11} (no result: {e})")
PY
