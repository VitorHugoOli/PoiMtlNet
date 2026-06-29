#!/usr/bin/env bash
# n=20 {0,1,7,100} confirmation of the per-head cat-lr 1e-3 recipe (the new win) vs the current champion.
#   NEW recipe  = bs8192 + MTL_ONECYCLE_PER_HEAD_LR=1, cat-lr 1e-3 / reg-lr 3e-3 / shared-lr 3e-3 (cat-LR lowered).
#   FL anchor   = bs2048 champion (per-head OFF, uniform 3e-3) — need seeds {1,7,100}; seed-0=79.83 from settle.
# Small-state baselines already at n=20 (AL bs2048 63.55 / AZ 63.57; AL bs8192 63.90 / AZ 64.31).
# 5-fold per seed, board protocol, PID-keyed scoring. Small states first (2-wide), then FL.
# Quality is the axis (deterministic) → OK to share the GPU. Results saved to closing_data at the end.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
D=docs/studies/train_perf_multifold
OUT=$D/n20_perhead_runs; mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/summary.tsv"
echo -e "state\trecipe\tbs\tperhead\tcatlr\treglr\tsharedlr\tseed\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"
MAX_PAR="${1:-2}"

# state:recipe:bs:perhead:catlr:reglr:sharedlr:seed
JOBS=()
for sd in 0 1 7 100; do JOBS+=("alabama:new:8192:1:1e-3:3e-3:3e-3:$sd"); done
for sd in 0 1 7 100; do JOBS+=("arizona:new:8192:1:1e-3:3e-3:3e-3:$sd"); done
for sd in 1 7 100;   do JOBS+=("florida:base:2048:0:1e-3:3e-3:1e-3:$sd"); done
for sd in 0 1 7 100; do JOBS+=("florida:new:8192:1:1e-3:3e-3:3e-3:$sd"); done

run_job() {
  IFS=':' read -r st rec bs ph clr rlr slr sd <<< "$1"
  local cd_="$OUT/${st}_${rec}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  if [ "$ph" = "1" ]; then export MTL_ONECYCLE_PER_HEAD_LR=1; else unset MTL_ONECYCLE_PER_HEAD_LR; fi
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_n20ph_${st}_${rec}_s${sd}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size "$bs" \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr "$clr" --reg-lr "$rlr" --shared-lr "$slr" \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$st" --no-checkpoints > "$log" 2>&1 &
  local pid=$!; wait $pid; local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -d results/$OVL/$st/mtlnet_*bs${bs}_ep50_*_${pid} 2>/dev/null | head -1)
  local cat="-" reg="-"
  if [ -n "$RD" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag n20ph_${st}_${rec}_s$sd 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${st}\t${rec}\t${bs}\t${ph}\t${clr}\t${rlr}\t${slr}\t${sd}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[n20ph] ${st}/${rec}/s${sd} rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[n20ph] ${#JOBS[@]} runs, max_par=$MAX_PAR (small states first, then FL)"
running=0
for j in "${JOBS[@]}"; do
  run_job "$j" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[n20ph] ALL DONE"; column -t "$SUMMARY"
python - "$SUMMARY" << 'PY'
import sys,csv,statistics as st
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['cat'] not in ('-','')]
cells={}
for r in rows:
    try: c=float(r['cat']); g=float(r['reg'])
    except: continue
    cells.setdefault((r['state'],r['recipe']),[]).append((c,g,r['seed']))
print("\n==== n=20 cell means (over seeds) ====")
for (s,rec),v in sorted(cells.items()):
    cs=[x[0] for x in v]; gs=[x[1] for x in v]
    cm=st.mean(cs); gm=st.mean(gs)
    csd=st.pstdev(cs) if len(cs)>1 else 0; gsd=st.pstdev(gs) if len(gs)>1 else 0
    print(f"{s:9}{rec:6} n={len(v)} seeds={sorted(x[2] for x in v)}  cat {cm:.4f}±{csd:.3f}  reg {gm:.4f}±{gsd:.3f}")
PY
