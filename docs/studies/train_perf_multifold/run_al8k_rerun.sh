#!/usr/bin/env bash
# AL 8k re-run (the n=20 cell that OOM'd during FL overlap). 4 seeds, 5-fold, SEQUENTIAL.
# PID-based rundir capture (rundir suffix == train.py PID) → race-free scoring.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/n20_batch_runs
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
SUMMARY="$OUT/al8k_rescore.tsv"
echo -e "seed\trundir_pid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

for sd in 0 1 7 100; do
  cd_="$OUT/alabama_8k_s${sd}_rerun"; mkdir -p "$cd_"; log="$cd_/run.log"; S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_al8k_s${sd}"
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state alabama --seed "$sd" --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/alabama" --no-checkpoints > "$log" 2>&1 &
  pid=$!; wait $pid; rc=$?; wall=$((SECONDS-S))
  nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  RD=$(ls -d results/$OVL/alabama/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
  cat="-" reg="-"
  if [ -n "$RD" ]; then
    sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag al8k_s$sd 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat macro-F1 \(diag-best\)[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
    reg=$(echo "$sc" | grep -oE "reg FULL top10_acc[^=]*= [0-9.]+" | grep -oE "[0-9.]+$" | head -1)
  fi
  echo -e "${sd}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[al8k] seed=$sd rc=$rc wall=${wall}s pid=$pid cat=$cat reg=$reg nan=$nan rd=$(basename ${RD:-NONE})"
done
echo "[al8k] DONE"; column -t "$SUMMARY"
python - "$SUMMARY" << 'PY'
import sys,csv,statistics as st
r=[x for x in csv.DictReader(open(sys.argv[1]),delimiter='\t') if x['cat'] not in ('-','')]
c=[float(x['cat']) for x in r]; g=[float(x['reg']) for x in r]
if c: print(f"\nAL 8k n={len(c)}  cat {st.mean(c):.4f}±{st.pstdev(c) if len(c)>1 else 0:.3f}  reg {st.mean(g):.4f}±{st.pstdev(g) if len(g)>1 else 0:.3f}")
PY
