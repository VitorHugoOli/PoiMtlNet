#!/usr/bin/env bash
# Memory-bounded STAN ETL (--streaming) for TX + CA, with a RAM watchdog that SIGKILLs the
# run if MemAvailable drops below the floor — protects the shared box from a repeat OOM.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=/home/vitor.oliveira/PoiMtlNet:/home/vitor.oliveira/PoiMtlNet/src
BASE=docs/studies/closing_data/v17_completion/stan_catx
FLOOR_KB=12000000   # 12 GB: abort if MemAvailable falls below this

watchdog() {  # $1 = pid to guard
  local pid=$1
  while kill -0 "$pid" 2>/dev/null; do
    avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$avail" -lt "$FLOOR_KB" ]; then
      echo "[watchdog] MemAvailable ${avail}KB < floor ${FLOOR_KB}KB — SIGKILL $pid" | tee -a "$BASE/etl_streaming.log"
      kill -9 "$pid" 2>/dev/null
      return 1
    fi
    sleep 3
  done
}

for ST in texas california; do
  echo "[$(date '+%F %T')] ETL(stream) $ST START" | tee -a "$BASE/etl_streaming.log"
  python -m research.baselines.stan.etl --state "$ST" --streaming --chunk-users 2000 \
      > "$BASE/etl_${ST}_stream.log" 2>&1 &
  epid=$!
  watchdog "$epid" & wpid=$!
  wait "$epid"; rc=$?
  kill "$wpid" 2>/dev/null || true
  n=$(python -c "import pandas as pd;print(len(pd.read_parquet('output/baselines/stan/$ST/inputs.parquet')))" 2>/dev/null || echo "?")
  peak=$(awk '/MemAvailable/{print int((125829120-$2)/1048576)"GB used-ish"}' /proc/meminfo)
  echo "[$(date '+%F %T')] ETL(stream) $ST DONE rc=$rc windows=$n (mem now ~$peak)" | tee -a "$BASE/etl_streaming.log"
  [ "$rc" -ne 0 ] && { echo "[$(date '+%F %T')] ABORT (rc=$rc) — see etl_${ST}_stream.log"; exit $rc; }
done
echo "[$(date '+%F %T')] ETL(stream) ALL DONE" | tee -a "$BASE/etl_streaming.log"
