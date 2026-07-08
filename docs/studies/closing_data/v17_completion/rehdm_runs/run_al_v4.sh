#!/usr/bin/env bash
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
DRV=$BASE/al_v4_DRIVER.log; : > "$DRV"
echo "[$(date '+%F %T')] AL v4 (zero-init default) 5 seeds x 50ep START" | tee -a "$DRV"
python -u -m research.baselines.rehdm.train --state alabama --folds 5 --epochs 50 \
    --tag REHDM_al_v4_faithful_5seeds_50ep > "$BASE/al_v4.log" 2>&1
echo "[$(date '+%F %T')] AL v4 DONE rc=$?" | tee -a "$DRV"
python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json
d=json.load(open("docs/results/baselines/REHDM_al_v4_faithful_5seeds_50ep_summary.json"))
print(f"  AL FAITHFUL (v4, zero-init Eq.9): test acc@10 = {d['test_acc@10_mean']*100:.2f} ± {d['test_acc@10_std']*100:.2f}  mrr={d['test_mrr_mean']:.4f}")
print(f"  (v3 random-init buggy=60.16 | A2off=64.97 | old-code=66.06)")
PY
echo "[$(date '+%F %T')] AL v4 COMPLETE" | tee -a "$DRV"
