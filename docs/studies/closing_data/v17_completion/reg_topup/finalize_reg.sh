#!/usr/bin/env bash
# Reg-ceiling n=20 top-up (dk_ovl overlap, next_stan_flow, prior-OFF).
# Recipe = the board template (a40_task2_tx_reg.sh Cell A): p1_region_head_ablation.py, region, freeze_alpha=True
# alpha_init=0.0 (prior INERT → log_T has zero effect), --engine-override dk_ovl, --region-emb-source v14, max_lr 0.003.
# Because the prior is inert we OMIT --per-fold-transition-dir (CA/TX lack seeded log_T at {1,7,100}; the file is
# irrelevant to the output). A parity gate re-runs AL s0 (no dir) and REQUIRES it to reproduce the board 0.6999 before
# any {1,7,100} run — if it doesn't, the inert-omit assumption is wrong and we STOP.
set -uo pipefail
REPO=/home/vitor.oliveira/PoiMtlNet; cd "$REPO"; export PYTHONPATH=src
PY=/home/vitor.oliveira/.venv/bin/python
export MTL_CHUNK_VAL_METRIC=1 MTL_COMPILE_DYNAMIC=1
export TORCHINDUCTOR_CACHE_DIR=/home/vitor.oliveira/.inductor_cache_board
V14=check2hgi_design_k_resln_mae_l0_1; OVL=check2hgi_dk_ovl; EP=50; F=5
BASE=docs/studies/closing_data/v17_completion/reg_topup
COLL=docs/results/closing_data/reg_ceiling_n20; LOGS="$BASE/logs"
mkdir -p "$COLL" "$LOGS"
DRV="$BASE/DRIVER.log"; : > "$DRV"
log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DRV"; }

# run_reg <state> <seed> <tag> -> writes P1 json, echoes aggregate top10_acc*100
run_reg(){ local ST=$1 SD=$2 TAG=$3; local RL="$LOGS/${ST}_s${SD}.log"
  $PY -u scripts/p1_region_head_ablation.py --state "$ST" --heads next_stan_flow \
      --input-type region --region-emb-source "$V14" \
      --override-hparams freeze_alpha=True alpha_init=0.0 \
      --engine-override "$OVL" \
      --folds "$F" --epochs "$EP" --seed "$SD" --target region \
      --compile --tf32 --tag "$TAG" > "$RL" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && { log "  FAIL ${ST} s${SD} rc=$rc (see $RL)"; tail -8 "$RL" | sed 's/^/    /'; return 1; }
  local J="docs/results/P1/region_head_${ST}_region_${F}f_${EP}ep_${TAG}.json"
  $PY -c "import json;d=json.load(open('$J'));print(round(d['heads']['next_stan_flow']['aggregate']['top10_acc_mean']*100,4))"
}

log "REG top-up START (inert-omit mode; parity gate on AL s0 vs board 69.99)"
PAR=$(run_reg alabama 0 alabama_ovl_stl_reg_parity_nodir) || exit 1
log "PARITY alabama s0 (no dir) top10=${PAR}  (board 69.99)"
python3 -c "import sys; d=abs(float('${PAR}')-69.99); sys.exit(0 if d<=0.6 else 1)" || { log "PARITY FAIL (|Δ|>0.6) — inert-omit assumption WRONG. STOP, investigate."; exit 2; }
log "PARITY OK — proceeding with {1,7,100} × 5 states (no dir, inert)."

for ST in alabama arizona florida california texas; do
  for SD in 1 7 100; do
    TAG="${ST}_ovl_stl_reg_topup_s${SD}"
    OUT="$COLL/${ST}_s${SD}.txt"
    [ -f "$OUT" ] && { log "SKIP ${ST} s${SD} (exists)"; continue; }
    log "RUN  ${ST} s${SD}"
    V=$(run_reg "$ST" "$SD" "$TAG") || continue
    echo "$V" > "$OUT"; log "DONE ${ST} s${SD} top10=${V}"
  done
done
log "REG top-up COMPLETE"

# aggregate n=20 (seed0 from board file + {1,7,100} topup)
$PY - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, glob, statistics as st
COLL="docs/results/closing_data/reg_ceiling_n20"
board_s0={"alabama":69.99,"arizona":59.40,"florida":76.71,"california":63.48,"texas":64.96}
print("\n=== REG ceiling n=20 (dk_ovl) ===")
print(f"{'state':11} {'s0(board)':>9} {'s1':>7} {'s7':>7} {'s100':>7} {'n=20 mean':>10}")
for s in ["alabama","arizona","florida","california","texas"]:
    vals={0:board_s0[s]}
    for sd in (1,7,100):
        import os
        f=f"{COLL}/{s}_s{sd}.txt"
        if os.path.exists(f): vals[sd]=float(open(f).read().strip())
    allv=[vals[k] for k in sorted(vals)]
    m=st.mean(allv)
    cells=" ".join(f"{vals.get(k,float('nan')):7.2f}" for k in (0,1,7,100))
    print(f"{s:11} {cells} {m:10.3f}  (n_seed={len(allv)})")
PY
