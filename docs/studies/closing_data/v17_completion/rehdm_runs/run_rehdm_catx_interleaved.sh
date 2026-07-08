#!/usr/bin/env bash
# ReHDM CA/TX — RESUMABLE, INTERLEAVED seed order (early partial mean for BOTH states),
# same robust design as run_stan_interleaved.sh: each (state, seed) is its own process
# writing its own _run0.json → a kill costs <=1 seed; re-invoking skips done seeds. 2x retry.
# NO self-watchdog (shared-box lesson: let the kernel OOM-killer target a real hog, not our job).
# Recipe = faithful v4 (zero-init Eq.9), 5 seeds (42-46, match AL v4) x 50ep, bs64, auto-fp32 (n_regions>3000).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
RES=docs/results/baselines
DRV=$BASE/rehdm_catx_DRIVER.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

# interleaved: alternate TX/CA per seed so both states accumulate results in lockstep.
JOBS=("texas:42" "california:42" "texas:43" "california:43" "texas:44" "california:44" "texas:45" "california:45" "texas:46" "california:46")

run_one(){ local ST=$1; local S=$2
  local TAG="REHDM_${ST}_catx_s${S}"
  local J="$RES/${TAG}_run0.json"
  if [ -f "$J" ]; then log "SKIP ${ST} s${S} (exists)"; return 0; fi
  local attempt
  for attempt in 1 2; do
    log "RUN  ${ST} s${S} (attempt ${attempt})"
    python -u -m research.baselines.rehdm.train --state "$ST" --folds 1 --seed "$S" --epochs 50 \
        --tag "$TAG" > "$BASE/rehdm_${ST}_s${S}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ] && [ -f "$J" ]; then
      local a10; a10=$(python -c "import json;print(round((json.load(open('$J'))['test'] or {}).get('acc@10',0)*100,3))" 2>/dev/null)
      log "DONE ${ST} s${S} acc@10=${a10}"; return 0
    fi
    log "FAIL ${ST} s${S} attempt ${attempt} rc=${rc} (see rehdm_${ST}_s${S}.log)"
  done
  log "GIVEUP ${ST} s${S} after 2 attempts"; return 1
}

partial(){ python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, glob, statistics as st
RES="docs/results/baselines"; out=[]
for ST in ["texas","california"]:
    acc=[]
    for f in sorted(glob.glob(f"{RES}/REHDM_{ST}_catx_s*_run0.json")):
        t=(json.load(open(f))['test'] or {}).get('acc@10')
        if t is not None: acc.append(t*100)
    if acc:
        m=st.mean(acc); s=st.pstdev(acc) if len(acc)>1 else 0.0
        out.append(f"{ST}={m:.2f}±{s:.2f}({len(acc)}seed)")
print("  PARTIAL: "+" | ".join(out) if out else "  PARTIAL: none")
PY
}

log "ReHDM CA/TX interleaved START (resumable, no self-watchdog, faithful v4, auto-fp32)"
for job in "${JOBS[@]}"; do
  run_one "${job%%:*}" "${job##*:}" || true
  partial
done
log "ReHDM CA/TX interleaved COMPLETE"
partial
