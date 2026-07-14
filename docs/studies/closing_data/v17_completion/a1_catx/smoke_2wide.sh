#!/usr/bin/env bash
# 2-WIDE SMOKE: can 1 CA + 1 TX v17 MTL run concurrently (each dataset build peaks ~66 GB host RAM) and is
# there a throughput benefit vs 1-wide? Truncated A1 recipe: --only-fold 0 --epochs 3 (full dataset still built,
# so the ~66 GB RAM peak is representative). Samples MemAvailable throughout. NON-destructive smoke.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
BASE=docs/studies/closing_data/v17_completion/a1_catx/smoke2wide
mkdir -p "$BASE"
DRV="$BASE/DRIVER.log"; : > "$DRV"
log(){ echo "[$(date '+%T')] $*" | tee -a "$DRV"; }

# background RAM sampler (min MemAvailable = the tightest point)
( echo "min_avail_gb=999"; while true; do
    a=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo)
    echo "$(date '+%T') $a"; sleep 4
  done ) > "$BASE/ram.log" 2>&1 &
SAMPLER=$!

run_one(){ local st=$1
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export MTL_RAM_HEADROOM_GB=2                # permissive: let a REAL OOM (not the guard) reveal the limit
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_smoke2w_${st}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed 1 --epochs 3 --folds 5 --only-fold 0 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$BASE/smoke_${st}.log" 2>&1
  echo "  ${st} rc=$? " >> "$DRV"
}

log "2-wide smoke START (CA + TX concurrent, only-fold 0, 3 epochs). start RAM avail=$(awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo) GB"
run_one california & CA=$!
run_one texas & TX=$!
wait $CA; rc_ca=$?
wait $TX; rc_tx=$?
kill $SAMPLER 2>/dev/null || true

log "both finished (rc_ca=$rc_ca rc_tx=$rc_tx). Analysis:"
python - <<'PY' 2>&1 | tee -a "$DRV"
import re
BASE="docs/studies/closing_data/v17_completion/a1_catx/smoke2wide"
# min RAM avail during the run
vals=[float(l.split()[1]) for l in open(f"{BASE}/ram.log") if len(l.split())==2 and l.split()[1].replace('.','').isdigit()]
print(f"  MIN MemAvailable during 2-wide = {min(vals):.1f} GB (of 125.6)  -> {'FITS' if min(vals)>3 else 'NEAR-OOM'}")
# per-epoch dt each state; OOM check
for st in ["california","texas"]:
    t=open(f"{BASE}/smoke_{st}.log").read()
    oom = bool(re.search(r"out of memory|OutOfMemory|MemoryError|Killed|non-finite", t, re.I))
    dts=[float(x) for x in re.findall(r"dt=([0-9.]+)s", t)]
    eps=re.findall(r"Fold \d/5 completed in ([0-9.]+)s", t)
    # tqdm rate
    rate=re.findall(r"([0-9.]+)batch/s", t); rate=rate[-1] if rate else "?"
    print(f"  {st:11} oom={oom}  per-epoch dt={dts if dts else 'n/a'}  last_batch_rate={rate}/s  fold-time={eps or 'n/a'}")
print("\n  1-WIDE baseline (CA fold ~3595s/50ep => ~72s/epoch). If 2-wide per-epoch <~1.6x the 1-wide, 2-wide WINS on throughput.")
PY
log "2-wide smoke COMPLETE"
