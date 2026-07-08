#!/usr/bin/env bash
# A2-azfl: ReHDM AZ/FL v4 re-run (version-uniform row). The cited AZ/FL ReHDM numbers are v2-code; AL is v4
# (65.38). This re-runs AZ + FL on the corrected v4 code so the paper's ReHDM row is all-v4 (drops the version
# caveat). Runs BEFORE A1, as parallel as feasible (32-core box, low load → 2-wide + concurrent with the CA/TX
# + Istanbul ReHDM jobs is fine). Resumable (skip if summary exists). Faithful v4 (zero-init Eq.9), 5 seeds 42-46.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
DRV="$BASE/a2_azfl_DRIVER.log"; : > "$DRV"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

# ---- 1. build AZ + FL ReHDM ETL if missing (CPU; vectorized, no OOM) ----
for st in arizona florida; do
  if [ -f "output/baselines/rehdm/$st/inputs.parquet" ]; then log "ETL $st exists (skip)"; continue; fi
  log "ETL $st build START"
  python -m research.baselines.rehdm.etl --state "$st" > "$BASE/etl_${st}.log" 2>&1
  rc=$?
  n=$(python -c "import pandas as pd;print(len(pd.read_parquet('output/baselines/rehdm/$st/inputs.parquet')))" 2>/dev/null || echo '?')
  log "ETL $st DONE rc=$rc rows=$n"
  [ "$rc" -ne 0 ] && { log "ETL $st FAILED — abort A2-azfl"; exit 1; }
done

# ---- 2. run AZ + FL v4 training 2-wide (concurrent) ----
run_state(){ local st=$1
  local TAG="REHDM_${st}_v4_5seeds_50ep"
  local SUM="docs/results/baselines/${TAG}_summary.json"
  if [ -f "$SUM" ]; then log "SKIP $st (summary exists)"; return 0; fi
  log "RUN  $st v4 5s x 50ep"
  python -u -m research.baselines.rehdm.train --state "$st" --folds 5 --seed 42 --epochs 50 \
      --tag "$TAG" > "$BASE/rehdm_${st}_v4.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -f "$SUM" ]; then
    local a; a=$(python -c "import json;d=json.load(open('$SUM'));print(f\"acc@10={d['test_acc@10_mean']*100:.2f}±{d['test_acc@10_std']*100:.2f}\")" 2>/dev/null)
    log "DONE $st v4 $a"
  else
    log "FAIL $st v4 rc=$rc (see rehdm_${st}_v4.log)"
  fi
}

log "A2-azfl START — AZ + FL ReHDM v4, 2-wide"
run_state arizona & AZ=$!
run_state florida & FL=$!
wait $AZ; wait $FL
log "A2-azfl COMPLETE — AZ/FL now all-v4 (version-uniform ReHDM row)"
python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, glob
print("\n==== ReHDM v4 (all-v4 row) ====")
rows={"alabama":"65.38 (AL, v4)"}
for st in ["arizona","florida"]:
    fs=glob.glob(f"docs/results/baselines/REHDM_{st}_v4_5seeds_50ep_summary.json")
    if fs:
        d=json.load(open(fs[0])); rows[st]=f"{d['test_acc@10_mean']*100:.2f}±{d['test_acc@10_std']*100:.2f}"
for s,v in rows.items(): print(f"  {s:9} {v}")
print("  (old v2-code row was AL 66.06 / AZ 54.65 / FL 65.68 — now superseded by v4)")
PY
