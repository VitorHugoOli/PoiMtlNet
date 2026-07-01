#!/usr/bin/env bash
# FL seed-0 early-stop: once seed-0 (5-fold, n=5) base+8k are scored, if 8k cat matches-or-exceeds
# base (within TOL pp), the FL "regression" worry is resolved at proper n=5 → STOP remaining seeds
# {1,7,100}, and release the queued coupled sweep (which waits on the "ALL RUNS DONE" marker).
# If 8k cat is meaningfully BELOW base, let all 4 seeds run to characterize it.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
D=docs/studies/train_perf_multifold
SUM=$D/fl_settle_runs/summary.tsv
DRV=$D/fl_settle_runs/DRIVER.log
TOL="${1:-0.20}"   # "matches" tolerance in pp on cat
echo "[earlystop] watching $SUM for seed-0 base+8k (TOL=${TOL}pp on cat)"
for i in $(seq 1 600); do  # up to 10h
  if grep -q "ALL RUNS DONE" "$DRV" 2>/dev/null; then echo "[earlystop] FL driver already done — nothing to do"; exit 0; fi
  bcat=$(awk -F'\t' '$1=="base" && $3=="0" && $5!="-" {print $5}' "$SUM" 2>/dev/null | head -1)
  kcat=$(awk -F'\t' '$1=="8k"   && $3=="0" && $5!="-" {print $5}' "$SUM" 2>/dev/null | head -1)
  breg=$(awk -F'\t' '$1=="base" && $3=="0" {print $6}' "$SUM" 2>/dev/null | head -1)
  kreg=$(awk -F'\t' '$1=="8k"   && $3=="0" {print $6}' "$SUM" 2>/dev/null | head -1)
  if [ -n "$bcat" ] && [ -n "$kcat" ]; then
    echo "[earlystop] seed-0 done: base cat=$bcat reg=$breg | 8k cat=$kcat reg=$kreg"
    decision=$(python3 -c "print('STOP' if $kcat >= $bcat - $TOL else 'CONTINUE')")
    echo "[earlystop] cat delta = $(python3 -c "print(round($kcat-$bcat,3))") pp → decision: $decision"
    if [ "$decision" = "STOP" ]; then
      echo "[earlystop] 8k matches/exceeds base at n=5 → stopping seeds {1,7,100}"
      pkill -f "run_fl_settle.sh" 2>/dev/null || true
      pkill -f "scripts/train.py.*--state florida" 2>/dev/null || true
      sleep 3
      echo "[flset] ALL RUNS DONE (early-stop: seed-0 8k cat ${kcat} matches/exceeds base ${bcat} at n=5; seeds 1,7,100 skipped)" >> "$DRV"
      echo "[earlystop] released coupled sweep; FL settled at seed-0 n=5."
    else
      echo "[earlystop] 8k below base by >${TOL}pp at n=5 → letting all 4 seeds run for full characterization."
    fi
    exit 0
  fi
  sleep 60
done
echo "[earlystop] timed out waiting for seed-0"
