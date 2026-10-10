#!/usr/bin/env bash
# STAN baseline completion for Ch.5's region table (knowladge, author-authorized 2026-10-10). One job at a time.
#   (1) AZ: all 5 folds rerun, recipe of the printed cell (README_FAITHFUL_STAN.md "How to run"): fp32 + compile,
#       --folds 5 --epochs 200 --seed 0 --patience 20, NEW tag v5_compiled_rerun20261010 (never overwrites the printed JSON).
#   (2) CA folds 2,3,4 and (3) TX fold 4: STATUS.md "To RESUME" recipe (--only-fold k --amp bf16 --compile --d-model 128
#       --patience 10 --tag v6_p10), order CA2, TX4, CA3, CA4, skip-if-JSON, up to 2 attempts, no self-watchdog.
# Runs in the detached worktree /dados/poimtlnet/wt_stan (origin/main 93a1a6c0), so every JSON lands on /dados; reads the
# existing STAN ETL inputs from ~/PoiMtlNet/output/baselines/stan/<state>/ (read-only, no rebuild).
# Starts only after the JISA campaign logs CAMPAIGN ALL DONE and no train.py runs; then the occupancy gate.
# Every step goes to /dados/poimtlnet/stan_rerun/STAN_LEDGER.txt; a watcher logs temperature/load/boot every 60 s.
WT=/dados/poimtlnet/wt_stan; OUT=/dados/poimtlnet/stan_rerun; mkdir -p "$OUT"
L=$OUT/STAN_LEDGER.txt; JL=/dados/poimtlnet/output_jisa/_logs/LEDGER.txt
log(){ echo "$* $(date '+%F %T')" >> "$L"; sync "$L"; }
BOOT=$(uptime -s)
# 1) wait for the JISA campaign to finish
until awk '/^CAMPAIGN RUN regime=eqw/{f=0} /^CAMPAIGN ALL DONE/{f=1} END{exit !f}' "$JL" && ! pgrep -f "[s]cripts/train\.py" >/dev/null; do sleep 60; done
log "STAN RERUN wait over: JISA CAMPAIGN ALL DONE, no train.py running"
bash /dados/poimtlnet/jisa_scripts/gate_check.sh STAN "$L" || exit 1
# 2) watcher: temperatures (hwmon), load, MemAvailable, boot id; stops with the chain
( while [ ! -f "$OUT/.chain_done" ]; do
    t=$(for h in /sys/class/hwmon/hwmon*; do n=$(cat $h/name 2>/dev/null); for f in $h/temp*_input; do [ -f "$f" ] && echo -n "$n:$(( $(cat $f) / 1000 ))C "; done; done)
    echo "$(date '+%F %T') boot=$(uptime -s) load=$(cut -d' ' -f1 /proc/loadavg) avail_GB=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo) $t" >> "$OUT/watch.log"
    sleep 60; done ) &
cd "$WT" || exit 1
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=$WT:$WT/src OUTPUT_DIR=/home/vitor.oliveira/PoiMtlNet/output TORCHINDUCTOR_CACHE_DIR=$OUT/inductor_cache
log "STAN RUN wt=$WT head=$(git -C $WT rev-parse HEAD) train_md5=$(md5sum research/baselines/stan/train.py | cut -c1-32) model_md5=$(md5sum research/baselines/stan/model.py | cut -c1-32) python=$(python -V 2>&1) torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null) inputs: az=$(md5sum $OUTPUT_DIR/baselines/stan/arizona/inputs.parquet | cut -c1-12) ca=$(md5sum $OUTPUT_DIR/baselines/stan/california/inputs.parquet | cut -c1-12) tx=$(md5sum $OUTPUT_DIR/baselines/stan/texas/inputs.parquet | cut -c1-12)"
summ(){ python - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for i, f in enumerate(d["per_fold"]):
    print("  fold_entry%d: Acc@10=%.4f Acc@5=%.4f Acc@1=%.4f MRR=%.4f macroF1=%.4f best_epoch=%s n_epochs_ran=%s" % (
        i, f["top10_acc"], f["top5_acc"], f["accuracy"], f["mrr"], f["f1"], f.get("best_epoch"), f.get("n_epochs_ran")))
PY
}
# (1) Arizona, 5 folds in one process (as the printed cell)
CMD_AZ="python -m research.baselines.stan.train --state arizona --folds 5 --epochs 200 --seed 0 --patience 20 --compile --tag v5_compiled_rerun20261010"
J_AZ=docs/results/baselines/faithful_stan_arizona_5f_200ep_v5_compiled_rerun20261010.json
log "STAN AZ START cmd: $CMD_AZ"; t0=$(date +%s)
$CMD_AZ > "$OUT/stan_arizona_5f_rerun.log" 2>&1; rc=$?
log "STAN AZ END rc=$rc wall=$(( $(date +%s) - t0 ))s json=$WT/$J_AZ"
[ -f "$J_AZ" ] && summ "$J_AZ" >> "$L"
[ "$(uptime -s)" != "$BOOT" ] && { log "STAN STOPPED: reboot detected"; touch "$OUT/.chain_done"; exit 2; }
# (2)/(3) CA/TX folds
for job in california:2 texas:4 california:3 california:4; do
  ST=${job%%:*}; F=${job##*:}; J=docs/results/baselines/faithful_stan_${ST}_5f_200ep_v6_p10_fold${F}.json
  if [ -f "$J" ]; then log "STAN SKIP $ST fold$F (exists: $J)"; continue; fi
  CMD="python research/baselines/stan/train.py --state $ST --seed 0 --epochs 200 --folds 5 --only-fold $F --batch-size 2048 --amp bf16 --compile --d-model 128 --patience 10 --tag v6_p10"
  for attempt in 1 2; do
    log "STAN $ST fold$F START attempt=$attempt cmd: $CMD"; t0=$(date +%s)
    $CMD > "$OUT/stan_${ST}_fold${F}_a${attempt}.log" 2>&1; rc=$?
    log "STAN $ST fold$F END rc=$rc wall=$(( $(date +%s) - t0 ))s json=$WT/$J"
    [ "$(uptime -s)" != "$BOOT" ] && { log "STAN STOPPED: reboot detected"; touch "$OUT/.chain_done"; exit 2; }
    if [ $rc -eq 0 ] && [ -f "$J" ]; then summ "$J" >> "$L"; break; fi
  done
done
log "STAN RERUN ALL DONE"; touch "$OUT/.chain_done"
