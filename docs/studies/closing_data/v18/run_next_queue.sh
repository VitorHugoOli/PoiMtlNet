#!/usr/bin/env bash
# v18 queue: wait for the florida trunk arm, then run the ALABAMA MTL rows 3+7,
# then resume the paused stage-1 dedicated sweep (which skips its 17 completed arms).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
BASE=docs/studies/closing_data/v18
log(){ echo "[$(date '+%F %T')] QUEUE: $*" | tee -a "$BASE/logs/queue_DRIVER.log"; }
log "waiting for the florida trunk arm"
while pgrep -f run_trunk_florida >/dev/null 2>&1; do sleep 60; done
log "florida trunk arm finished -> starting ALABAMA MTL rows 3+7"
bash "$BASE/run_mtl_alabama.sh" >> "$BASE/logs/queue_stdout.log" 2>&1
log "MTL rows 3+7 done -> resuming the stage-1 dedicated sweep (skips completed arms)"
bash "$BASE/run_stage1_sweep.sh" >> "$BASE/logs/queue_stdout.log" 2>&1
log "QUEUE COMPLETE"
