#!/usr/bin/env bash
# ReHDM Istanbul (faithful v4, zero-init Eq.9) — 5 seeds (42-46, matching AL/CA/TX) x 50ep, auto-bf16 (n_regions 290<3000).
# Separate process from the CA/TX driver (different state, no collision). Small state (~106k trajs) → ~25-40min.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src PYTHONUNBUFFERED=1
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
DRV=$BASE/rehdm_istanbul_DRIVER.log; : > "$DRV"
echo "[$(date '+%F %T')] ReHDM Istanbul v4 5s x 50ep START (concurrent w/ CA s42; small/bf16)" | tee -a "$DRV"
python -u -m research.baselines.rehdm.train --state istanbul --folds 5 --seed 42 --epochs 50 \
    --tag REHDM_istanbul_v4_5seeds_50ep > "$BASE/rehdm_istanbul.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] ReHDM Istanbul DONE rc=$rc" | tee -a "$DRV"
if [ -f docs/results/baselines/REHDM_istanbul_v4_5seeds_50ep_summary.json ]; then
  python -c "import json;d=json.load(open('docs/results/baselines/REHDM_istanbul_v4_5seeds_50ep_summary.json'));print(f\"  Istanbul ReHDM: test acc@10 = {d['test_acc@10_mean']*100:.2f} +/- {d['test_acc@10_std']*100:.2f} (mrr {d['test_mrr_mean']:.4f})\")" | tee -a "$DRV"
fi
echo "[$(date '+%F %T')] REHDM ISTANBUL COMPLETE" | tee -a "$DRV"
