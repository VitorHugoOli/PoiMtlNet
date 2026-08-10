#!/usr/bin/env bash
# v18_2 A40 wave — the four SMALL states, seeds 1/7/100, seed-major.
#
# Division of labour under the current plan: the A40 keeps istanbul, alabama, arizona and
# florida (free), while california and texas run on rented A100s in their own Modal lanes.
# This script therefore never touches CA or TX.
#
# SEED-MAJOR, not state-major. If the run is cut short, a complete seed across all four
# states degrades honestly to n=10 or n=15 with the convention stated; a half-finished
# seed does not. Within a seed, cheapest state first so the earliest hours bank the most
# cells (arizona 0.45 h -> alabama 0.47 h -> istanbul 0.87 h -> florida 2.98 h).
#
# HARVEST=0 here: the A40 writes straight into the repo, so there is nothing to bring back.
#
#   usage: run_wave_a40.sh [seed ...]        default: 1 7 100
set -uo pipefail
REPO=${REPO:-$(pwd)}
LANE="$REPO/docs/studies/closing_data/v18_2/scripts/run_lane.sh"
OUT=${OUT:-docs/results/closing_data/v18_2}
ENG=${ENG:-check2hgi_v18}
V14=${V14:-check2hgi_design_k_resln_mae_l0_1}
SEEDS=${*:-1 7 100}
STATES="arizona alabama istanbul florida"      # cheapest first, measured
mkdir -p "$OUT/logs"
wlog(){ echo "[$(date -Is)] WAVE $*" | tee -a "$OUT/logs/wave_a40.log"; }

wlog "===== v18_2 A40 wave: seeds [$SEEDS] x [$STATES] ====="
wlog "california and texas are NOT run here — they are on the Modal lanes."
for SEED in $SEEDS; do
  wlog "--- seed $SEED ---"
  for ST in $STATES; do
    # Per-cell guard. run_lane.sh skips on its own sidecar, but four seed-1 reg cells were
    # produced by the earlier v18 driver and have no v18_2 sidecar, so they would be redone.
    # Write the sidecar for anything already present in docs/results/P1 before starting.
    p1=$(ls -t docs/results/P1/*"${ST}"*"reg_s${SEED}"*.json 2>/dev/null | head -1)
    side="$OUT/${ST}_s${SEED}_reg.json"
    if [ -n "$p1" ] && [ ! -f "$side" ]; then
      printf '{"state":"%s","seed":%s,"family":"reg","rundir":"%s","note":"pre-existing from the v18 driver"}\n' \
        "$ST" "$SEED" "$p1" > "$side"
      wlog "  adopted pre-existing reg cell $ST s$SEED -> $p1"
    fi
    t0=$SECONDS
    HARVEST=0 bash "$LANE" "$ST" "$SEED" "$ENG" "$V14" "$OUT"
    wlog "  $ST s$SEED done in $((SECONDS-t0))s"
  done
done
wlog "===== A40 WAVE COMPLETE ====="
