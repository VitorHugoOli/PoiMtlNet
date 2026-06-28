#!/usr/bin/env bash
# n=20 confirmation — is the bs=8192 small-state cat gain real (outside multi-seed CI)?
# AL+AZ × {base 2048, 8k 8192} × seeds {0,1,7,100} × 5-fold = 16 runs (n=20 folds/cell).
# Champion-G held; LR UNCHANGED. Quality is deterministic → GPU may be shared. max_par cells concurrent.
# Champion prior is INERT (freeze_alpha/alpha_init=0, KD off) → MTL_SKIP_INERT_LOGT default-on,
# so NO per-fold seeded log_T needed (verified Phase 4b byte-identical).
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/n20_batch_runs
mkdir -p "$OUT"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
MAX_PAR="${1:-2}"
SUMMARY="$OUT/summary.tsv"
echo -e "state\ttag\tbs\tseed\tcat_macroF1\treg_acc10\twall_s\tnan\trc" > "$SUMMARY"

JOBS=()
for st in alabama arizona; do
  for tb in "base:2048" "8k:8192"; do
    for sd in 0 1 7 100; do JOBS+=("$st:$tb:$sd"); done
  done
done

run_job() {
  IFS=':' read -r st tag bs sd <<< "$1"
  local cd="$OUT/${st}_${tag}_s${sd}"; mkdir -p "$cd"; local log="$cd/run.log"; local S=$SECONDS
  ( export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_n20_${st}_${tag}_s${sd}"
    python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
      --state "$st" --seed "$sd" --epochs 50 --folds 5 --batch-size "$bs" \
      --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
      --cat-head next_gru --reg-head next_stan_flow_dualtower \
      --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
      --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
      --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
      --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
      --model mtlnet_crossattn_dualtower --compile --tf32 \
      --per-fold-transition-dir "output/$V14/$st" --no-checkpoints
  ) > "$log" 2>&1
  local rc=$?; local wall=$((SECONDS-S))
  local nan; nan=$(grep -ciE "non-finite (grad|loss)" "$log" 2>/dev/null || echo 0)
  local RD; RD=$(ls -dt results/$OVL/$st/mtlnet_*bs${bs}_ep50_*seed${sd}* 2>/dev/null | head -1)
  [ -z "$RD" ] && RD=$(ls -dt results/$OVL/$st/mtlnet_*bs${bs}_ep50_* 2>/dev/null | head -1)
  local cat="-" reg="-"
  [ -n "$RD" ] && read -r cat reg < <(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag n20_${st}_${tag}_s${sd} 2>/dev/null \
    | grep -oE "= [0-9.]+ ±" | grep -oE "[0-9.]+" | head -2 | tr '\n' ' ')
  echo -e "${st}\t${tag}\t${bs}\t${sd}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[n20] ${st}/${tag}/s${sd} done rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan"
}

echo "[n20] ${#JOBS[@]} runs (AL+AZ × base/8k × 4 seeds), 5f×50ep, max_par=$MAX_PAR"
running=0
for j in "${JOBS[@]}"; do
  run_job "$j" &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[n20] ALL DONE"
echo "==== N=20 SUMMARY — per-cell mean over {0,1,7,100} computed by aggregate below ===="
column -t "$SUMMARY"
python - "$SUMMARY" << 'PY'
import sys, csv, statistics as st
rows=list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
cells={}
for r in rows:
    try: c=float(r['cat_macroF1']); g=float(r['reg_acc10'])
    except: continue
    cells.setdefault((r['state'],r['tag']),[]).append((c,g))
print("\n==== n=20 cell means (over seeds) ====")
print(f"{'state':10}{'tag':6}{'n':>3}  cat_mean±sd      reg_mean±sd")
for (s,t),v in sorted(cells.items()):
    cs=[x[0] for x in v]; gs=[x[1] for x in v]
    cm=st.mean(cs); gm=st.mean(gs)
    csd=st.pstdev(cs) if len(cs)>1 else 0; gsd=st.pstdev(gs) if len(gs)>1 else 0
    print(f"{s:10}{t:6}{len(v):>3}  {cm:.4f}±{csd:.3f}   {gm:.4f}±{gsd:.3f}")
PY
