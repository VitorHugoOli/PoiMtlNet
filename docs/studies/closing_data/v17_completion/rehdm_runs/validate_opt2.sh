#!/usr/bin/env bash
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
DRV=$BASE/validate_opt2_DRIVER.log; : > "$DRV"
echo "[$(date '+%F %T')] AL 2-seed opt validation (eval workers reverted to 2) START" | tee -a "$DRV"
python -u -m research.baselines.rehdm.train --state alabama --folds 2 --epochs 50 \
    --tag REHDM_al_OPT2_validate > "$BASE/al_opt2_validate.log" 2>&1
echo "[$(date '+%F %T')] done rc=$?" | tee -a "$DRV"
python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, re
base={42:0.6418056918547596, 43:0.6506378802747792}
print("  seed  opt_acc@10          baseline           BIT-EXACT?")
ok=True
for r in [0,1]:
    d=json.load(open(f"docs/results/baselines/REHDM_al_OPT2_validate_run{r}.json"))
    s=d['seed']; t=(d['test'] or {})['acc@10']; b=base[s]; m=(t==b); ok=ok and m
    print(f"  {s}    {t:.16f}  {b:.16f}  {'YES ✅' if m else 'NO ✗'}")
print(f"\n  VERDICT: {'BIT-EXACT — collate opt is quality-neutral ✅' if ok else 'DIVERGED ✗'}")
dts=[float(x) for x in re.findall(r'dt=([0-9.]+)s', open(f"{BASE}/al_opt2_validate.log".replace('$BASE','docs/studies/closing_data/v17_completion/rehdm_runs')).read())]
if dts: print(f"  AL per-epoch: opt mean={sum(dts)/len(dts):.2f}s (baseline v4 ~5.5s)")
PY
