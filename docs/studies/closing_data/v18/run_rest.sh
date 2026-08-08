#!/usr/bin/env bash
# v18 — remaining rows in ONE process, in order: 10 -> 5 -> 6 -> 3b.
#
# NO pgrep-based waiting anywhere. The previous chain deadlocked because `pgrep -f run_queueN.sh`
# also matched a health-monitor process that carried those script names in its own argv, so the
# waiter never saw its predecessor exit (queue2 finished 08:07, queue3 was still "waiting" at 13:22
# -- ~5 h of idle GPU). Sequential execution in one process cannot have that failure mode.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
BASE=docs/studies/closing_data/v18
D=$BASE/logs/rest_DRIVER.log
log(){ echo "[$(date '+%F %T')] REST: $*" | tee -a "$D"; }

log "===== START: rows 10, 5, 6, 3b ====="
log "row 10 — MTL batch size at alabama"
bash "$BASE/run_mtl_bs_alabama.sh" >> "$BASE/logs/rest_stdout.log" 2>&1
log "row 10 done"

log "rows 5, 6, 3b — TX dedicated (1 fold), then FL MTL cat-LR (1 fold)"
SKIP_WAIT=1 bash "$BASE/run_queue4.sh" >> "$BASE/logs/rest_stdout.log" 2>&1
log "===== REST COMPLETE — rows 10, 5, 6, 3b done; row 9 next ====="
