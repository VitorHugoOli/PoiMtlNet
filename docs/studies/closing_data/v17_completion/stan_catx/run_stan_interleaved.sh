#!/usr/bin/env bash
# Faithful STAN CA/TX — RESUMABLE, INTERLEAVED fold order (early partial mean for BOTH states).
# Same robust design as run_stan_foldwise.sh (per-fold JSON, resumable skip, 2x retry, warm inductor cache, NO
# self-watchdog). Only the ORDER differs: alternate TX/CA so each state accumulates folds early.
# Recipe = v6 + patience 10 (quality-neutral): 200ep cap, bs2048, bf16+compile, d-model 128, seed 0.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src
export TORCHINDUCTOR_CACHE_DIR=/home/vitor.oliveira/.inductor_cache_stan_catx
BASE=docs/studies/closing_data/v17_completion/stan_catx
DRV=$BASE/stan_interleaved_DRIVER.log
RES=docs/results/baselines
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

# interleaved: TX0/TX1 already done (skipped), then alternate CA/TX so both states fill early.
JOBS=("texas:0" "texas:1" "california:0" "texas:2" "california:1" "texas:3" "california:2" "texas:4" "california:3" "california:4")

run_fold(){ local ST=$1 F=$2; local J="$RES/faithful_stan_${ST}_5f_200ep_v6_p10_fold${F}.json"
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
    log "FAIL ${ST} fold${F} attempt ${attempt} rc=${rc} (see stan_${ST}_fold${F}.log)"
  done
  log "GIVEUP ${ST} fold${F} after 2 attempts"; return 1
}

# running partial-mean report after each completed fold (so we see both states early)
partial(){ python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, glob, statistics as st
RES="docs/results/baselines"; floors={"texas":54.94,"california":52.09}
out=[]
for ST in ["texas","california"]:
    acc=[json.load(open(f))["per_fold"][0]["top10_acc"]*100 for f in sorted(glob.glob(f"{RES}/faithful_stan_{ST}_5f_200ep_v6_p10_fold*.json"))]
    if acc:
        m=st.mean(acc); tag="CLEARS" if m>floors[ST] else "below"
        out.append(f"{ST}={m:.2f}({len(acc)}f,{tag})")
print("  PARTIAL: "+" | ".join(out) if out else "  PARTIAL: none")
PY
}

log "STAN interleaved START (resumable, warm cache, no self-watchdog)"
for job in "${JOBS[@]}"; do
  run_fold "${job%%:*}" "${job##*:}" || true
  partial
done
log "STAN interleaved COMPLETE"
partial
