#!/usr/bin/env bash
# CA/TX v17 n=20 — H100 driver (the documented home for the overlap-MTL board).
# Runs champion v17 (= v16 + bs8192 + per-head cat-lr 1e-3) MTL on the gated stride-1 overlap
# engine check2hgi_dk_ovl, for CA + TX × seeds {0,1,7,100} × 5 folds = the n=20 board top-up.
#
# WHY H100, not the A40: CA/TX stride-1 overlap MTL at fp32 is ~52 min/epoch on the A40
# (measured 2026-07-01) -> ~9 days/cell, ~72 days for 8 cells. Infeasible on the A40. The
# H100 (80GB, faster fp32/TF32) both speeds each cell AND fits >2 concurrent big-state cells,
# so set MAX_PAR up to the VRAM budget (each cell ~22-26GB; 80GB -> 3-wide safe).
#
# PRECISION: keep true fp32 (MTL_DISABLE_AMP=1) to match the fp32 board TX cell and avoid the
# bf16-at-large-C question entirely. (H100 bf16 corroborated TX to 0.03pp, but fp32 is the
# apples-to-apples comparand for the n=20 board.)
#
# PREREQS on the H100 box (verify before launch):
#   1. torch 2.11.0+cu128, the repo checked out at this commit, venv active.
#   2. The v14 substrate present for CA + TX:
#        output/check2hgi_design_k_resln_mae_l0_1/{california,texas}/{embeddings,region_embeddings,poi_embeddings}.parquet
#   3. The gated overlap engine built for CA + TX (stride=1, emit_tail=false, min_seq=10):
#        output/check2hgi_dk_ovl/{california,texas}/input/next_region.parquet
#        Build if missing:  python scripts/mtl_improvement/build_overlap_probe_engine.py <state> 1 10
#        Verify provenance: input/next_build_provenance.json -> {stride:1, emit_tail:false, min_sequence_length:10}
#   4. The champion is prior-OFF + KD-off -> per-fold log_T is INERT (MTL_SKIP_INERT_LOGT default-on),
#        so NO region_transition_log_*.pt files are needed. (--per-fold-transition-dir is harmless; the
#        inert-skip drops the load. If you ever activate the prior, build seeded per-fold log_T first.)
#
# Usage:  bash run_catx_v17_n20_h100.sh [MAX_PAR]   (default 3)
set -u
source /home/vitor.oliveira/.venv/bin/activate 2>/dev/null || true   # adjust for the H100 box venv path
cd "$(dirname "$0")/../../.."   # repo root
MAX_PAR="${1:-3}"
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
OUT=docs/studies/closing_data/catx_v17_n20_h100; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"; echo -e "state\tseed\tpid\tcat\treg\twall\tnan\trc" > "$SUMMARY"

for st in california texas; do
  prov="output/$OVL/$st/input/next_build_provenance.json"
  [ -f "$prov" ] || { echo "[h100] MISSING overlap engine for $st ($prov) — build it first (see header)"; exit 2; }
  python - "$prov" "$st" <<'PY' || exit 2
import json,sys
p=json.load(open(sys.argv[1]))
ok=p.get("stride")==1 and p.get("emit_tail") is False and p.get("min_sequence_length")==10
print(f"[h100] {sys.argv[2]} provenance {'OK' if ok else 'BAD'} {{stride:{p.get('stride')}, emit_tail:{p.get('emit_tail')}, min_seq:{p.get('min_sequence_length')}}}")
sys.exit(0 if ok else 3)
PY
done

run_cell() {  # state seed
  local st="$1" sd="$2"
  local cd_="$OUT/${st}_s${sd}"; mkdir -p "$cd_"; local log="$cd_/run.log"; local S=$SECONDS
  export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_ONECYCLE_PER_HEAD_LR=1 MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export MTL_RAM_HEADROOM_GB=24
  export TORCHINDUCTOR_CACHE_DIR="$HOME/.inductor_cache_catxv17h100_${st}_s${sd}"
  export MTL_PROFILE_JSON="$cd_/profile.json"
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
  if [ -n "${RD:-}" ]; then
    local sc; sc=$(python scripts/closing_data/a40_score_matched.py "$RD" --seed "$sd" --tag catxv17h100_${st}_s$sd 2>/dev/null)
    cat=$(echo "$sc" | grep -oE "cat[= ]+[0-9.]+" | grep -oE "[0-9.]+" | head -1)
    reg=$(echo "$sc" | grep -oE "reg[= ]+[0-9.]+" | grep -oE "[0-9.]+" | head -1)
  fi
  echo -e "${st}\t${sd}\t${pid}\t${cat:--}\t${reg:--}\t${wall}\t${nan}\t${rc}" >> "$SUMMARY"
  echo "[h100] ${st}/s${sd} rc=$rc wall=${wall}s cat=$cat reg=$reg nan=$nan rd=$(basename ${RD:-NONE})"
}

JOBS=(); for st in california texas; do for sd in 0 1 7 100; do JOBS+=("$st $sd"); done; done
echo "[h100] ${#JOBS[@]} runs (CA+TX × {0,1,7,100}), max_par=$MAX_PAR, v17 bs8192 fp32 per-head"
running=0
for j in "${JOBS[@]}"; do
  run_cell $j &
  running=$((running+1))
  [ "$running" -ge "$MAX_PAR" ] && { wait -n 2>/dev/null || wait; running=$((running-1)); }
done
wait
echo "[h100] DONE"; cat "$SUMMARY"
python - "$SUMMARY" <<'PY'
import sys,statistics as st
rows=[l.split("\t") for l in open(sys.argv[1]).read().splitlines()[1:] if l.strip()]
by={}
for r in rows:
    if len(r)<6 or r[3] in("","-"): continue
    try: c=float(r[3]); g=float(r[4])
    except: continue
    by.setdefault(r[0],([],[]))[0].append(c); by[r[0]][1].append(g)
for s,(cs,gs) in by.items():
    print(f"{s:11} n={len(cs)}  cat {st.mean(cs):.3f}±{st.pstdev(cs) if len(cs)>1 else 0:.3f}  reg {st.mean(gs):.3f}±{st.pstdev(gs) if len(gs)>1 else 0:.3f}")
PY
