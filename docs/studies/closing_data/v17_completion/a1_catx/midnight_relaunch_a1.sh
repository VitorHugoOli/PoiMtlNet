#!/usr/bin/env bash
# Midnight-BRT conditional A1 relaunch: at 00:00 BRT (03:00 UTC), once the GPU is idle, launch A1 1-WIDE.
# (Smoke 2026-07-10 proved 2-wide fits RAM but is GPU-bound → no throughput benefit → stay 1-wide.)
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
BASE=docs/studies/closing_data/v17_completion/a1_catx
LOG="$BASE/midnight_launcher.log"
log(){ echo "[$(date -u '+%F %T UTC') | $(TZ=America/Sao_Paulo date '+%T BRT')] $*" | tee -a "$LOG"; }

# 1. sleep until the next 03:00 UTC == 00:00 BRT
now=$(date -u +%s); target=$(date -u -d 'today 03:00' +%s); [ "$now" -ge "$target" ] && target=$(date -u -d 'tomorrow 03:00' +%s)
log "launcher armed. Sleeping $((target-now))s (~$(( (target-now)/3600 ))h $(( ((target-now)%3600)/60 ))m) until midnight BRT."
sleep $((target-now))
log "midnight BRT reached — polling GPU until idle (only relaunch when nothing is running on the GPU)."

# 2. poll until the GPU is idle (no compute-apps AND <500 MiB used)
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); used=${used:-9999}
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  if [ "${apps:-1}" -eq 0 ] && [ "$used" -lt 500 ]; then
    log "GPU IDLE (used=${used}MiB, apps=${apps}) → launching A1 1-wide."; break
  fi
  log "GPU busy (used=${used}MiB, apps=${apps}) — re-check in 5min."; sleep 300
done

# 3. launch the A1 orchestrator (1-wide; its own ReHDM-wait passes instantly + RAM gate ≥80GB)
setsid nohup bash "$BASE/run_a1_catx_n20.sh" > /dev/null 2>&1 < /dev/null &
log "A1 orchestrator launched (1-wide, {1,7,100})."
