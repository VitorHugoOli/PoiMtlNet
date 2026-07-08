#!/usr/bin/env bash
# Detached closeout: remaining cat n=20 (CA/TX @ bs8192@0.005) THEN reg n=20 top-up. Runs under setsid so it
# survives session/harness events (the harness-managed background jobs kept getting killed at the CA stage).
# No harness notifications when detached → poll DRIVER.logs + the STAMP file below.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
BASE=docs/studies/closing_data/v17_completion
STAMP="$BASE/finalize_all.stamp"
echo "START $(date '+%F %T')" > "$STAMP"

echo "[finalize_all] CAT CA/TX @ bs8192@0.005 ..." >> "$STAMP"
bash "$BASE/cat_ceiling_sweep/sweep.sh" "california texas" "0 1 7 100" "8192:0.005" >> "$STAMP" 2>&1
echo "CAT_DONE $(date '+%F %T')" >> "$STAMP"

echo "[finalize_all] REG top-up {1,7,100} x5 ..." >> "$STAMP"
bash "$BASE/reg_topup/finalize_reg.sh" >> "$STAMP" 2>&1
echo "REG_DONE $(date '+%F %T')" >> "$STAMP"
echo "ALL_DONE $(date '+%F %T')" >> "$STAMP"
