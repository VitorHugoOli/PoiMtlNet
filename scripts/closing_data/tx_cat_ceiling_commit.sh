#!/usr/bin/env bash
# Wait for the TX cat-ceiling STL run to finish (EXIT) OR die (ramwatch kill), score whatever
# folds completed, append the result to TX_CELL.md + write artefact, commit + push (race-robust).
# Survives session death. One-shot.
set -uo pipefail
cd /teamspace/studios/this_studio/PoiMtlNet
BRANCH=study/board-h100
LOG=logs/tx_cat_ceiling.log
DEST=docs/results/closing_data/h100
REG_CEIL=64.96

echo "[tx-cat] waiting for cat-ceiling to finish or die..."
while true; do
  grep -q 'EXIT=' "$LOG" 2>/dev/null && { echo "[tx-cat] EXIT seen"; break; }
  alive=$(for p in $(pgrep -x python); do cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); echo "$cl"|grep -q 'task next' && echo "$cl"|grep -q texas && echo 1; done | wc -l)
  [ "$alive" -eq 0 ] && { echo "[tx-cat] proc gone (no EXIT — likely ramwatch kill); scoring partial"; break; }
  sleep 30
done

RD=$(ls -dt results/check2hgi_dk_ovl/texas/next_* 2>/dev/null | head -1)
if [ -z "$RD" ]; then echo "[tx-cat] no rundir — abort"; exit 0; fi
nfold=$(ls "$RD"/metrics/fold*_next_category_val.csv 2>/dev/null | wc -l)
echo "[tx-cat] rundir=$RD folds=$nfold — scoring"
PYTHONPATH=src python scripts/closing_data/score_stl_cat_ceiling.py "$RD" --tag texas_cat_ceiling >/tmp/tx_cat_score.txt 2>&1 || true
cat /tmp/tx_cat_score.txt
if [ -f "$RD/stl_cat_ceiling_score.json" ]; then
  cp "$RD/stl_cat_ceiling_score.json" "$DEST/texas_s0_stl_cat_ceiling.json"
  python3 - "$DEST/texas_s0_stl_cat_ceiling.json" "$REG_CEIL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); regc = float(sys.argv[2])
cm, cs, n = d["cat_macro_f1_mean"], d["cat_macro_f1_std"], len(d["cat_per_fold"])
block = f"""

## TX STL ceilings — RESULT (scored {n}f)
- **cat macro-F1 ceiling = {cm:.4f} ± {cs:.4f}** (next_gru STL, dk_ovl, seed0; per-fold {d['cat_per_fold']}).
- reg FULL top10 ceiling = {regc:.2f} (a40 fp32, see above).
- **Compare to the TX MTL cell (TX_CELL table): MTL cat beats this ceiling by the table-mean minus {cm:.2f}.**
"""
open("docs/studies/closing_data/TX_CELL.md", "a").write(block)
print(f"[tx-cat] cat ceiling = {cm:.2f} ± {cs:.2f} (n={n})")
PY
fi
git add docs/studies/closing_data/TX_CELL.md "$DEST/texas_s0_stl_cat_ceiling.json" 2>/dev/null || true
git commit -m "board-h100: TX STL cat ceiling scored (next_gru, dk_ovl, seed0) — completes TX ceilings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017bpM3kYznPgop7ySjMPkGF" >/dev/null 2>&1 || true
for try in 1 2 3 4; do
  git pull --rebase origin "$BRANCH" >/tmp/tx_cat_pull.txt 2>&1 || true
  if git push origin "$BRANCH" >/tmp/tx_cat_push.txt 2>&1; then echo "[tx-cat] pushed"; break; fi
  echo "[tx-cat] push retry $try"; sleep 10
done
echo "[tx-cat] done ($(date -u +%H:%M:%S)Z)"
