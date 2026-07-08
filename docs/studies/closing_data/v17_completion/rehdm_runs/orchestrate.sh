#!/usr/bin/env bash
# Wait for the TX timing smoke to finish, record its per-epoch dt, then launch the AL full run (faithful/corrected code).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
DRV=$BASE/orchestrate.log; : > "$DRV"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

log "waiting for TX timing smoke to finish..."
while [ ! -f docs/results/baselines/SMOKE_tx_timing_run0.json ]; do sleep 10; done
sleep 3
txdt=$(grep -aoE "ep=1/1 .*dt=[0-9.]+s" "$BASE/smoke_tx.log" | grep -aoE "dt=[0-9.]+s" | head -1)
txnr=$(grep -aoE "n_regions=[0-9]+" "$BASE/smoke_tx.log" | head -1)
txtr=$(grep -aoE "train=[0-9]+" "$BASE/smoke_tx.log" | head -1)
log "TX smoke DONE: per-epoch $txdt ($txtr trajectories, $txnr)"

log "launching AL full run (faithful/corrected code, 5 seeds x 50 epochs, auto-bf16 since AL n_regions<3000)"
python -u -m research.baselines.rehdm.train --state alabama --folds 5 --epochs 50 \
    --tag REHDM_al_v3_faithful_5seeds_50ep > "$BASE/al_full.log" 2>&1
rc=$?
log "AL full run DONE rc=$rc"
if [ -f docs/results/baselines/REHDM_al_v3_faithful_5seeds_50ep_summary.json ]; then
  python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json
d=json.load(open("docs/results/baselines/REHDM_al_v3_faithful_5seeds_50ep_summary.json"))
print(f"  AL faithful: test_acc@10 = {d.get('test_acc@10_mean'):.4f} ± {d.get('test_acc@10_std') or 0:.4f}  mrr={d.get('test_mrr_mean'):.4f}")
print(f"  (OLD v2 code was 66.06 ± 0.98)")
PY
fi
log "ORCHESTRATE COMPLETE"
