#!/usr/bin/env bash
# Watchdog for the CA/TX v17 run — huge states, RAM/CPU/VRAM OOM watch. Observe + log + flag (no kill).
# Poll 30s. Flags: host-RAM avail low, swap growing (the real OOM tell for big CPU-resident datasets), VRAM low.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
LOG=docs/studies/closing_data/catx_v17_runs/MONITOR.log
: > "$LOG"
ram_total=$(free -m | awk '/^Mem:/{print $2}')
gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
peak_ram=0; peak_gpu=0; peak_swap=0
echo "[$(date '+%F %T')] catx monitor START — RAM ${ram_total}MiB, GPU ${gpu_total}MiB, cpus $(nproc)" | tee -a "$LOG"
for i in $(seq 1 2400); do   # up to 20h
  ts=$(date '+%F %T')
  read rt ru ra < <(free -m | awk '/^Mem:/{print $2,$3,$7}')
  swused=$(free -m | awk '/^Swap:/{print $3}')
  read gu gf < <(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits | head -1 | tr ',' ' ')
  load=$(awk '{print $1}' /proc/loadavg)
  nproc_t=$(pgrep -fc "scripts/train.py" 2>/dev/null || echo 0)
  # top train.py by RSS (GB)
  rss=$(ps -eo rss,args | grep "[s]cripts/train.py" | awk '{s+=$1} END{printf "%.1f", s/1048576}')
  (( ru > peak_ram )) && peak_ram=$ru; (( gu > peak_gpu )) && peak_gpu=$gu; (( swused > peak_swap )) && peak_swap=$swused
  flag=""
  (( ra < 15000 )) && flag="${flag} [WARN ram_avail=${ra}MiB<15G]"
  (( swused > 4000 )) && flag="${flag} [WARN swap=${swused}MiB — OOM risk]"
  (( gf < 1500 )) && flag="${flag} [WARN gpu_free=${gf}MiB<1.5G]"
  grep -rilE "out of memory|outofmemory|Killed process|oom-kill" docs/studies/closing_data/catx_v17_runs/*/run.log 2>/dev/null | head -1 >/dev/null && flag="${flag} [!! OOM in a run.log !!]"
  if (( i % 6 == 1 )) || [ -n "$flag" ]; then
    echo "[$ts] RAM used ${ru}/${rt} avail ${ra} | swap ${swused} | train.py RSS ${rss}GB x${nproc_t} | GPU ${gu}/${gpu_total} free ${gf} | load ${load} | peaks ram ${peak_ram} gpu ${peak_gpu} swap ${peak_swap}${flag}" >> "$LOG"
  fi
  sleep 30
done
echo "[$(date '+%F %T')] catx monitor END — PEAK ram ${peak_ram}MiB / gpu ${peak_gpu}MiB / swap ${peak_swap}MiB" | tee -a "$LOG"
