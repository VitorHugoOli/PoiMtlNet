#!/usr/bin/env bash
# System watchdog: GPU memory/util, per-process GPU mem, system RAM, and OOM-event detection.
# Observe + log + flag (does NOT kill anything). Tracks peaks. Poll every 30s.
# Usage: monitor_oom_ram.sh [max_minutes] [watch_dir_glob]
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
MAXMIN="${1:-840}"            # default 14h
WATCH="${2:-docs/studies/train_perf_multifold}"
LOG=docs/studies/train_perf_multifold/SYSTEM_MONITOR.log
: > "$LOG"
gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
ram_total=$(free -m | awk '/^Mem:/{print $2}')
peak_gpu=0; peak_ram_used=0
# only watch the CURRENTLY-active run dirs (not historical), and baseline the OOM count so pre-existing
# OOM tracebacks in old logs don't false-alarm — only a NEW OOM (count increase) fires.
OOM_DIRS="$WATCH/fl_settle_runs $WATCH/coupled_sweep_runs"
oom_seen=$(grep -rilE "outofmemory|out of memory|cuda error: out of memory" $OOM_DIRS 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] monitor START — GPU total ${gpu_total}MiB, RAM total ${ram_total}MiB, max ${MAXMIN}min" | tee -a "$LOG"

for i in $(seq 1 $((MAXMIN*2))); do   # 2 polls/min
  ts=$(date '+%F %T')
  read gu gf gutil < <(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | head -1 | tr ',' ' ')
  read rt ru ra < <(free -m | awk '/^Mem:/{print $2, $3, $7}')
  swused=$(free -m | awk '/^Swap:/{print $3}')
  nproc=$(pgrep -fc "scripts/train.py" 2>/dev/null || echo 0)
  (( gu > peak_gpu )) && peak_gpu=$gu
  (( ru > peak_ram_used )) && peak_ram_used=$ru
  # OOM event detection: scan recent training logs + dmesg (best-effort, no sudo)
  newoom=$(grep -rilE "outofmemory|out of memory|cuda error: out of memory" $OOM_DIRS 2>/dev/null | wc -l)
  killoom=$(dmesg 2>/dev/null | grep -ciE "killed process|oom-kill" || echo 0)
  flag=""
  (( gf < 1500 )) && flag="${flag} [WARN gpu_free=${gf}MiB<1.5G]"
  (( ra < 4000 )) && flag="${flag} [WARN ram_avail=${ra}MiB<4G]"
  (( swused > 2000 )) && flag="${flag} [WARN swap=${swused}MiB]"
  if (( newoom > oom_seen )); then flag="${flag} [!! NEW OOM in a run.log !!]"; oom_seen=$newoom; fi
  (( killoom > 0 )) && flag="${flag} [!! dmesg oom-kill x${killoom} !!]"
  # log every 5min (10 polls) unless a flag fires (then always)
  if (( i % 10 == 1 )) || [ -n "$flag" ]; then
    echo "[$ts] GPU ${gu}/${gpu_total}MiB used (free ${gf}, util ${gutil}%) | RAM used ${ru}/${rt} avail ${ra} swap ${swused} | train.py x${nproc} | peak_gpu ${peak_gpu} peak_ram ${peak_ram_used}${flag}" >> "$LOG"
  fi
  sleep 30
done
echo "[$(date '+%F %T')] monitor END — PEAK gpu ${peak_gpu}MiB / ram_used ${peak_ram_used}MiB / OOM events ${oom_seen}" | tee -a "$LOG"
