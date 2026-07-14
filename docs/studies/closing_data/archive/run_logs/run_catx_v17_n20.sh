#!/usr/bin/env bash
# CA/TX n=20 {0,1,7,100} for the v17 champion (bs8192 + per-head cat-lr 1e-3), completing the board.
# HUGE states (CA C=8501, TX C=6553): fp32 mandatory (MTL_DISABLE_AMP=1); dataset stays CPU-resident via the
# auto-fit (NEVER MTL_DATASET_GPU=1 → ~31GB OOM). 2-wide = 1 CA + 1 TX concurrent (balanced VRAM ~20GB each,
# ~40GB < 46). The OOM/RAM watchdog (monitor_oom_ram.sh) runs alongside. PID-keyed scoring. --canon none +
# explicit v17 recipe (the dk_ovl wrong-substrate guard hard-fails --canon v17 under MTL_STRICT).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/closing_data
OUT=$D/catx_v17_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "state\tseed\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"
MAX_PAR="${1:-2}"

# wait for the CA overlap engine to be built (california next_region.parquet)
echo "[catx] waiting for CA overlap build (california next_region.parquet)..."
for i in $(seq 1 60); do
  [ -f output/$OVL/california/input/next_region.parquet ] && { echo "[catx] CA overlap ready at t=${i}min"; break; }
  sleep 60
done
# verify both provenances are gated before training
for st in california texas; do
  python3 -c "import json; d=json.load(open('output/$OVL/$st/input/next_build_provenance.json')); assert d['stride']==1 and d['emit_tail']==False and d['min_sequence_length']==10, d; print('[catx] $st provenance OK', {k:d[k] for k in ('stride','emit_tail','min_sequence_length')})" || { echo "[catx] $st provenance BAD — abort"; exit 1; }
done

# interleave CA+TX so a 2-wide wave = 1 CA + 1 TX (balanced VRAM/RAM)
JOBS=()
for sd in 0 1 7 100; do JOBS+=("california:$sd" "texas:$sd"); done

run_job() {
  IFS=':' read -r st sd <<< "$1"
  local cd_="$OUT/${st}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export MTL_RAM_HEADROOM_GB=24            # host-RAM guard headroom (2 big CPU-resident datasets)
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_catxv17_${st}_s${sd}"
  export MTL_PROFILE_JSON="$cd_/profile.json"   # transient per-fold profiler report (--profile)
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --profile \
    --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --checkpoint-selector geom_simple --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)|out of memory|OutOfMemory" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag catxv17_${st}_s$sd 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${st}\t${sd}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[catx] ${st}/s${sd} rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan rd=$(basename ${RD:-NONE})"
}

echo "[catx] ${#JOBS[@]} runs (CA+TX × {0,1,7,100}), max_par=$MAX_PAR, v17 bs8192 fp32 per-head"
running=0
for j in "${JOBS[@]}"; do
  run_job "$j" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[catx] ALL DONE"; column -t "$SUMMARY"
python - "$SUMMARY" << 'PY'
import sys,csv,statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['cat'] not in ('-','')]
cells={}
for r in rows:
    try: c=float(r['cat']); g=float(r['reg'])
    except: continue
    cells.setdefault(r['state'],[]).append((c,g))
print("\n==== CA/TX v17 n=20 cell means ====")
for s,v in sorted(cells.items()):
    cs=[x[0] for x in v]; gs=[x[1] for x in v]
    print(f"{s:11} n={len(v)}  cat {st.mean(cs):.3f}±{st.pstdev(cs) if len(cs)>1 else 0:.3f}  reg {st.mean(gs):.3f}±{st.pstdev(gs) if len(gs)>1 else 0:.3f}")
PY
