#!/usr/bin/env bash
# Faithful STAN CA/TX — RESUMABLE fold-by-fold (robust to the shared-box RAM-hog neighbor).
# Each fold is its own process writing its own _fold{k}.json → a kill costs <=1 fold; re-invoking skips done folds.
# NO self-watchdog (STAN ~18GB is a good citizen; let the kernel OOM-killer target the real 80GB hog if it comes to that).
# Warm/shared inductor cache across folds (the eval's free, zero-numerical-effect win → folds 1-4 skip cold recompiles).
# Recipe = v6 + patience 10 (quality-neutral): 200ep cap, bs2048, bf16+compile, d-model 128, seed 0.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src
export TORCHINDUCTOR_CACHE_DIR=/home/vitor.oliveira/.inductor_cache_stan_catx   # warm cache (eval win)
BASE=docs/studies/closing_data/v17_completion/stan_catx
DRV=$BASE/stan_foldwise_DRIVER.log
RES=docs/results/baselines
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

run_fold(){ # $1=state $2=fold
  local ST=$1 F=$2 J="$RES/faithful_stan_${ST}_5f_200ep_v6_p10_fold${F}.json"
  if [ -f "$J" ]; then log "SKIP ${ST} fold${F} (exists)"; return 0; fi
  local attempt
  for attempt in 1 2; do
    log "RUN  ${ST} fold${F} (attempt ${attempt})"
    python research/baselines/stan/train.py --state "$ST" --seed 0 --epochs 200 --folds 5 \
        --only-fold "$F" --batch-size 2048 --amp bf16 --compile --d-model 128 --patience 10 \
        --tag v6_p10 > "$BASE/stan_${ST}_fold${F}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$J" ]; then
      local a10; a10=$(python -c "import json;print(round(json.load(open('$J'))['per_fold'][0]['top10_acc']*100,3))" 2>/dev/null)
      log "DONE ${ST} fold${F} Acc@10=${a10}"; return 0
    fi
    log "FAIL ${ST} fold${F} attempt ${attempt} rc=${rc} (likely neighbor OOM kill; see stan_${ST}_fold${F}.log)"
  done
  log "GIVEUP ${ST} fold${F} after 2 attempts — will remain for a later resume"; return 1
}

log "STAN foldwise START (resumable, warm cache, no self-watchdog)"
for ST in texas california; do
  for F in 0 1 2 3 4; do run_fold "$ST" "$F" || true; done
done
log "STAN foldwise PASS COMPLETE — aggregating"

python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, glob, statistics as st
RES="docs/results/baselines"
floors={"texas":54.94,"california":52.09}
print("\n==== Faithful STAN CA/TX (v6 p10, foldwise) ====")
for ST in ["texas","california"]:
    js=sorted(glob.glob(f"{RES}/faithful_stan_{ST}_5f_200ep_v6_p10_fold*.json"))
    acc=[]
    for f in js:
        d=json.load(open(f)); acc.append(d["per_fold"][0]["top10_acc"]*100)
    if len(acc)==5:
        m=st.mean(acc); s=st.pstdev(acc); fl=floors[ST]
        print(f"{ST:11} Acc@10 = {m:.2f} ± {s:.2f}  (5f)  | floor {fl} -> {'CLEARS ✅' if m>fl else 'BELOW ✗'}")
    else:
        print(f"{ST:11} INCOMPLETE ({len(acc)}/5 folds: {[round(a,2) for a in acc]}) — re-run driver to fill")
PY
