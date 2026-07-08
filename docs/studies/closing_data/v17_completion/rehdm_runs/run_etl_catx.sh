#!/usr/bin/env bash
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src
BASE=docs/studies/closing_data/v17_completion/rehdm_runs
FLOOR_KB=15000000
watchdog(){ local pid=$1; while kill -0 "$pid" 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$a" -lt "$FLOOR_KB" ] && { echo "[watchdog] MemAvailable ${a}KB < floor — SIGKILL $pid" | tee -a "$BASE/etl_catx.log"; kill -9 "$pid"; return 1; }
  sleep 3; done; }
for ST in texas california; do
  echo "[$(date '+%F %T')] ReHDM ETL $ST START" | tee -a "$BASE/etl_catx.log"
  python -m research.baselines.rehdm.etl --state "$ST" > "$BASE/etl_${ST}.log" 2>&1 &
  pid=$!; watchdog "$pid" & wpid=$!; wait "$pid"; rc=$?; kill "$wpid" 2>/dev/null || true
  n=$(python -c "import pandas as pd;print(len(pd.read_parquet('output/baselines/rehdm/$ST/inputs.parquet')))" 2>/dev/null || echo "?")
  echo "[$(date '+%F %T')] ReHDM ETL $ST DONE rc=$rc rows=$n" | tee -a "$BASE/etl_catx.log"
  [ "$rc" -ne 0 ] && { echo "  ABORT $ST — see etl_${ST}.log"; exit "$rc"; }
done
echo "[$(date '+%F %T')] ReHDM ETL CA+TX ALL DONE" | tee -a "$BASE/etl_catx.log"
