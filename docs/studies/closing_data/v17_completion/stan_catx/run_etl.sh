#!/usr/bin/env bash
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src
BASE=docs/studies/closing_data/v17_completion/stan_catx
for ST in texas california; do
  echo "[$(date '+%F %T')] ETL $ST START"
  python -m research.baselines.stan.etl --state $ST > "$BASE/etl_${ST}.log" 2>&1
  rc=$?
  n=$(python -c "import pandas as pd;print(len(pd.read_parquet('output/baselines/stan/$ST/inputs.parquet')))" 2>/dev/null || echo "?")
  echo "[$(date '+%F %T')] ETL $ST DONE rc=$rc windows=$n"
done
echo "[$(date '+%F %T')] ETL ALL DONE"
