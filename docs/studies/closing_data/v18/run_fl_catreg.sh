#!/usr/bin/env bash
# florida cat+reg at seeds 7 and 100 -- QUEUED behind the running v18 regeneration.
#
# WHY QUEUED AND NOT NOW. The A40 is at 100% utilisation and 87 C with SW_THERMAL already
# active while it runs the seed-1 wave. Starting these cells alongside it would slow both and
# push the card further into thermal throttle, and run_wave.sh deliberately runs large states
# 1-wide because a CA/TX MTL dataset build peaks ~66 GB host RAM. So: wait for the regen driver
# to exit (that covers its first pass, its retry pass and make_results), let the card cool, then
# run. Nothing here touches the wave.
#
# WHY THESE FOUR CELLS. florida's JOINT is already at n=20 (seeds 0/1/7/100 -- s7 and s100 were
# produced on rented hardware). Its dedicated arm only reaches seeds 0/1, so the symmetric
# pairing in score_all correctly holds florida's section-1 contrast at n=10 and those two paid-for
# joint cells cannot be cited. These four cells are exactly what converts them: they take
# florida's section-1 to n=20 for ~1.9 h of a card that is otherwise idle.
#
# OUT is the BOARD directory, not v18_2: score_all.py hardcodes docs/results/closing_data/v18,
# so a sidecar written anywhere else is invisible to the aggregator.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet || exit 3
source /home/vitor.oliveira/.venv/bin/activate

WAVE_PID=${WAVE_PID:?set WAVE_PID to the run_regen.sh pid}
LANE=docs/studies/closing_data/v18_2/scripts/run_lane.sh
OUT=docs/results/closing_data/v18
ENG=check2hgi_v18
V14=check2hgi_design_k_resln_mae_l0_1
LOG=docs/studies/closing_data/v18/logs/fl_catreg_queue.log
mkdir -p "$(dirname "$LOG")"
log(){ echo "[$(date -Is)] FLQ: $*" | tee -a "$LOG"; }

log "===== queued: florida cat+reg seeds 7,100 -- waiting on run_regen pid $WAVE_PID ====="
while kill -0 "$WAVE_PID" 2>/dev/null; do sleep 120; done
log "regen driver exited; wave complete"

# Thermal courtesy. Every wall measured on this card while hot is slower than its own reference,
# and monitor.sh has caught it at 88 C with SW_THERMAL. Give it a chance to settle first.
for i in $(seq 30); do
  T=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  [ "${T:-99}" -lt 80 ] && { log "card at ${T}C -- starting"; break; }
  log "card at ${T}C, waiting for <80C ($i/30)"; sleep 60
done

# Do not start a cell into a RAM-starved box; the wave's neighbours can still be holding memory.
for i in $(seq 30); do
  A=$(awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo)
  [ "${A:-0}" -ge 40 ] && { log "RAM available ${A} GB -- ok"; break; }
  log "only ${A} GB RAM available, waiting ($i/30)"; sleep 60
done

RC=0
for SEED in 7 100; do
  log "--- florida s$SEED cat,reg ---"
  t0=$SECONDS
  CELLS=cat,reg bash "$LANE" florida "$SEED" "$ENG" "$V14" "$OUT" || RC=1
  log "--- florida s$SEED done in $((SECONDS-t0))s ---"
done

log "regenerating the board (score_all -> make_results -> status)"
.venv/bin/python docs/studies/closing_data/v18/make_results.py   >>"$LOG" 2>&1 || log "make_results FAILED"
.venv/bin/python docs/studies/closing_data/v18/status_update.py --phase wave1 >>"$LOG" 2>&1 || true

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import json
d = json.load(open("docs/studies/closing_data/v18/data/v18_results.json"))
c = d["cells"]["florida"]
for k in ("stl_cat_paired", "joint_cat_paired", "stl_reg_paired", "joint_reg_paired"):
    v = c.get(k)
    if v:
        print(f"FLQ: florida {k:18} mean={v['mean']} n_seeds={v['n_seeds']} n={v['n']}")
print("FLQ: delta_cat =", c.get("delta_cat_vs_own_ceiling"),
      " delta_reg =", c.get("delta_reg_vs_own_ceiling"))
PY

log "===== FLORIDA cat+reg QUEUE COMPLETE (rc=$RC) ====="
exit $RC
