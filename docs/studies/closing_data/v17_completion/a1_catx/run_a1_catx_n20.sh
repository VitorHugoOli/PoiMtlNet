#!/usr/bin/env bash
# A1 (ex-H1): CA/TX v17 MTL n=20 top-up — seeds {1,7,100} (seed-0 exists in catx_v17_seed0_5f/).
# Robust for the SHARED A40 box + ~1.5 d duration:
#   - waits for the running ReHDM jobs to finish + stops the ReHDM CA/TX loop (frees ~65 GB the MTL dataset needs)
#   - 1-WIDE serial (never 2-wide: each CA/TX MTL dataset build peaks ~66 GB host RAM — 2-wide OOMs)
#   - RAM-gated: waits until MemAvailable >= RAM_FLOOR_GB before each cell (the July-1 OOM was this guard firing)
#   - resumable: skips a cell whose stable sidecar exists; 2x retry; NO self-watchdog (let the kernel target a real hog)
# Recipe = the v17 champion, copied VERBATIM from ../../run_catx_v17_n20.sh (fp32, bs8192, per-head cat-lr,
#   --canon none + MTL_STRICT, geom_simple, compile+tf32). Scored with the matched scorer, same as seed-0.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
BASE=docs/studies/closing_data/v17_completion/a1_catx
SIDE=docs/results/closing_data/catx_v17_n20            # stable per-cell score sidecars (resumability + n=20 agg)
mkdir -p "$BASE" "$SIDE"
DRV="$BASE/a1_DRIVER.log"
RAM_FLOOR_GB=80                                        # ~66 GB peak + ~14 buffer; box is 125 GB total
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }
memavail_gb(){ awk '/MemAvailable/{printf "%.0f",$2/1048576}' /proc/meminfo; }

# interleaved so CA and TX fill in lockstep (like the seed-0 driver)
JOBS=("california:1" "texas:1" "california:7" "texas:7" "california:100" "texas:100")

wait_ram(){ # block until the box has enough free RAM for one CA/TX MTL dataset build
  local a; a=$(memavail_gb); local waited=0
  while [ "$a" -lt "$RAM_FLOOR_GB" ]; do
    [ $((waited % 600)) -eq 0 ] && log "  waiting for RAM: ${a} GB avail < ${RAM_FLOOR_GB} GB floor (box shared)"
    sleep 60; waited=$((waited+60)); a=$(memavail_gb)
  done
}

run_cell(){ local st=$1; local sd=$2
  local side="$SIDE/${st}_s${sd}.json"
  if [ -f "$side" ]; then log "SKIP ${st} s${sd} (sidecar exists)"; return 0; fi
  wait_ram
  local cd_="$BASE/${st}_s${sd}"; mkdir -p "$cd_"; local rlog="$cd_/run.log"
  local attempt
  for attempt in 1 2; do
    log "RUN  ${st} s${sd} (attempt ${attempt}, ${a:-?}->$(memavail_gb) GB avail)"
    export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export MTL_RAM_HEADROOM_GB=12          # 1-wide → smaller headroom than the 2-wide driver's 24
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_a1_${st}_s${sd}"
    python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
      --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
      --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
      --cat-head next_gru --reg-head next_stan_flow_dualtower \
      --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
      --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
      --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
      --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
      --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
      --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$rlog" 2>&1 &
    local pid=$!; wait $pid; local rc=$?
    local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
    if [ "$rc" -eq 0 ] && [ -n "$RD" ]; then
      python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag a1_${st}_s$sd > "$cd_/score.txt" 2>&1
      if [ -f "$RD/a40_matched_score.json" ]; then
        cp "$RD/a40_matched_score.json" "$side"
        local c; c=$(python -c "import json;d=json.load(open('$side'));print(f\"cat={d.get('cat_macro_f1_mean','?')} reg={d.get('reg_full_top10_mean','?')}\")" 2>/dev/null)
        log "DONE ${st} s${sd} $c  rd=$(basename "$RD")"
        return 0
      fi
    fi
    local oom; oom=$(grep -ciE "out of memory|OutOfMemory|MemoryError|Killed" "$rlog" 2>/dev/null || echo 0)
    log "FAIL ${st} s${sd} attempt ${attempt} rc=${rc} oom=${oom} (see $rlog)"
    sleep 30
  done
  log "GIVEUP ${st} s${sd} after 2 attempts"; return 1
}

aggregate(){ python - <<'PY' 2>>"$DRV" | tee -a "$DRV"
import json, glob, statistics as st
SIDE="docs/results/closing_data/catx_v17_n20"
# seed-0 canonical (from catx_v17_seed0_5f matched scores)
S0={"california":(77.0422,65.694),"texas":(77.2272,67.0723)}
print("\n==== A1: CA/TX v17 MTL — n=20 progress (seed-0 + {1,7,100}) ====")
for stt in ["california","texas"]:
    cats=[S0[stt][0]]; regs=[S0[stt][1]]; seeds=["0(canon)"]
    for f in sorted(glob.glob(f"{SIDE}/{stt}_s*.json")):
        d=json.load(open(f)); cats.append(d.get('cat_macro_f1_mean')); regs.append(d.get('reg_full_top10_mean'))
        seeds.append(f.split('_s')[-1].replace('.json',''))
    n=len(cats)
    print(f"  {stt:11} n_seeds={n} {seeds}  cat {st.mean(cats):.3f}±{st.pstdev(cats) if n>1 else 0:.3f}  reg {st.mean(regs):.3f}±{st.pstdev(regs) if n>1 else 0:.3f}")
PY
}

# ---- wait for ALL running ReHDM jobs (CA s42, Istanbul, A2-azfl AZ+FL), then free the box ----
log "A1 orchestrator START — waiting for ReHDM CA s42 + Istanbul + A2-azfl(AZ,FL) to finish before freeing the box"
while [ ! -f docs/results/baselines/REHDM_california_catx_s42_run0.json ] || \
      [ ! -f docs/results/baselines/REHDM_istanbul_v4_5seeds_50ep_summary.json ] || \
      [ ! -f docs/results/baselines/REHDM_arizona_v4_5seeds_50ep_summary.json ] || \
      [ ! -f docs/results/baselines/REHDM_florida_v4_5seeds_50ep_summary.json ]; do sleep 60; done
log "ReHDM CA s42 + Istanbul + AZ/FL done — stopping the ReHDM CA/TX interleaved loop (keeps banked seeds; frees RAM)"
pkill -f run_rehdm_catx_interleaved.sh 2>/dev/null || true
pkill -f "rehdm.train.*--state california" 2>/dev/null || true
pkill -f "rehdm.train.*--state texas" 2>/dev/null || true
sleep 20
log "box freed: $(memavail_gb) GB avail. Launching A1 {1,7,100} 1-wide, RAM floor ${RAM_FLOOR_GB} GB."

for j in "${JOBS[@]}"; do
  run_cell "${j%%:*}" "${j##*:}" || true
  aggregate
done
log "A1 COMPLETE — fold {1,7,100} into catx_v17_seed0_5f/RESULTS.md (n=20), update RESULTS_BOARD §1 + CEILINGS_N20_FINAL (drop CA/TX seed-0 asterisks)"
aggregate
