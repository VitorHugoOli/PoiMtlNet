#!/usr/bin/env bash
# A2 bucketization investigation on AL: isolate the -6pp cause + test the noisy-init fix.
#  v1 A2off      (REHDM_ST_BUCKETS=0)   — A3/A6/B2 on, Eq.9 OFF: isolates A2 from A3/A6.
#  v2 A2zeroinit (REHDM_ST_ZERO_INIT=1) — Eq.9 ON but st-embeddings zero-init (learned, not noise).
# Compare vs A2on-random=60.16 (have) and old-code-A2off=66.06.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
DRV=$BASE/a2_investigate.log; : > "$DRV"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

log "v1: A2off (REHDM_ST_BUCKETS=0) — AL 3 seeds"
REHDM_ST_BUCKETS=0 python -u -m research.baselines.rehdm.train --state alabama --folds 3 --epochs 50 \
    --tag REHDM_al_A2off_3seeds > "$BASE/al_A2off.log" 2>&1
log "v1 done rc=$?"

log "v2: A2on-zeroinit (REHDM_ST_ZERO_INIT=1) — AL 3 seeds"
REHDM_ST_ZERO_INIT=1 python -u -m research.baselines.rehdm.train --state alabama --folds 3 --epochs 50 \
    --tag REHDM_al_A2zero_3seeds > "$BASE/al_A2zero.log" 2>&1
log "v2 done rc=$?"

log "=== A2 investigation comparison (AL test acc@10) ==="
python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json,glob,statistics as st
def mean(tag):
    fs=sorted(glob.glob(f"docs/results/baselines/{tag}_run*.json"))
    a=[(json.load(open(f)).get('test') or {}).get('acc@10') for f in fs]
    a=[x for x in a if x is not None]
    return (st.mean(a)*100, st.pstdev(a)*100, len(a)) if a else (None,None,0)
rows=[("A2 OFF (buckets=0, isolate)","REHDM_al_A2off_3seeds"),
      ("A2 ON zero-init","REHDM_al_A2zero_3seeds"),
      ("A2 ON random-init (current)","REHDM_al_v3_faithful_5seeds_50ep")]
print(f"  {'variant':32} {'acc@10':>10} {'n':>3}")
for name,tag in rows:
    m,s,n=mean(tag)
    print(f"  {name:32} {m:6.2f}±{s:4.2f} {n:>3d}" if m else f"  {name:32} {'--':>10}")
print(f"  {'A2 OFF (old code, ref)':32} {'66.06±0.98':>10}   5")
print("\n  READ: if A2off≈66 → A2 is the sole -6pp cause. If A2zero≈66 → noisy init was the bug (zero-init = correct faithful Eq.9). If A2zero still ≈60 → Eq.9 genuinely hurts as we understand it.")
PY
log "A2 INVESTIGATION COMPLETE"
