#!/usr/bin/env bash
# Best-vs-best STL cat ceiling sweep (advisor-panel design, 2026-07-02).
# Finds the true STL cat ceiling = max over {batch x LR} of the n=20 mean, LR frozen globally after an AL+AZ screen.
# Recipe pinned to the STL next_gru / next_category ceiling: engine check2hgi_dk_ovl, ep50, 5f, --no-checkpoints.
# Only moving axes = --batch-size and --max-lr (OneCycle peak). rundir captured by child PID (race-safe). Resumable
# (skips an arm whose JSON already exists).
#
# Usage: sweep.sh "<states csv>" "<seeds csv>" "<arms: bs:lr bs:lr ...>"
#   e.g. sweep.sh "alabama arizona" "0 1" "8192:0.01 8192:0.02 8192:0.04 8192:0.08 2048:0.005 2048:0.01 2048:0.02"
set -u
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate

STATES="${1:?states}"; SEEDS="${2:?seeds}"; ARMS="${3:?arms}"
ENGINE=check2hgi_dk_ovl
BASE=docs/studies/closing_data/v17_completion/cat_ceiling_sweep
COLL="$BASE/sweep_results"; LOGS="$BASE/logs"
mkdir -p "$COLL" "$LOGS"
DRIVER="$BASE/DRIVER.log"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DRIVER"; }

log "SWEEP START states=[$STATES] seeds=[$SEEDS] arms=[$ARMS]"
for ARM in $ARMS; do
  BS="${ARM%%:*}"; LR="${ARM##*:}"
  for ST in $STATES; do
    for S in $SEEDS; do
      OUT="$COLL/${ST}_bs${BS}_lr${LR}_s${S}.json"
      if [ -f "$OUT" ]; then log "SKIP ${ST} bs${BS} lr${LR} s${S} (exists)"; continue; fi
      RL="$LOGS/${ST}_bs${BS}_lr${LR}_s${S}.log"
      log "RUN  ${ST} bs${BS} lr${LR} s${S}"
      MTL_NO_TRAIN_DIAGNOSTICS=1 python scripts/train.py --task next --state "$ST" --engine "$ENGINE" \
          --model next_gru --folds 5 --epochs 50 --seed "$S" \
          --batch-size "$BS" --max-lr "$LR" --no-checkpoints > "$RL" 2>&1 &
      CHILD=$!; wait $CHILD; EC=$?
      RD=$(ls -dt results/${ENGINE}/${ST}/next_*_${CHILD} 2>/dev/null | head -1)
      if [ "$EC" -ne 0 ] || [ -z "$RD" ]; then log "FAIL ${ST} bs${BS} lr${LR} s${S} exit=$EC rd='$RD' (see $RL)"; continue; fi
      python scripts/closing_data/score_stl_cat_ceiling.py "$RD" --tag "${ST}_bs${BS}_lr${LR}_s${S}" >> "$RL" 2>&1
      if [ -f "$RD/stl_cat_ceiling_score.json" ]; then
        cp "$RD/stl_cat_ceiling_score.json" "$OUT"
        F1=$(python -c "import json;print(json.load(open('$OUT'))['cat_macro_f1_mean'])")
        log "DONE ${ST} bs${BS} lr${LR} s${S} f1=${F1}"
      else
        log "FAIL ${ST} bs${BS} lr${LR} s${S} no sidecar (see $RL)"
      fi
    done
  done
done
log "SWEEP COMPLETE"
