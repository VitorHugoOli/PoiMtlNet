#!/usr/bin/env bash
# FL champion-G MTL, SEQUENTIAL (single process, --folds 5), for the seq-vs-2parallel timing A/B.
# Mirrors scripts/closing_data/board_h100_mtl.sh (fp32 arm) for the a40-wk box. Profiler on.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
OUT=docs/studies/train_perf_multifold/fl_seq_runs
mkdir -p "$OUT"
V14=check2hgi_design_k_resln_mae_l0_1
OVL=check2hgi_dk_ovl
STATE=florida

export PYTHONPATH=src
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_fl_seq
export MTL_DISABLE_AMP=1           # fp32 (A40 big-state requirement)
export MTL_PROFILE=1               # run profiler: per-fold section timing + throughput
export MTL_PROFILE_JSON="$OUT/profile.json"

echo "[seq start] $(date -u +%H:%M:%S)" | tee "$OUT/champG.log"
T0=$SECONDS
python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state "$STATE" --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 --profile \
    --per-fold-transition-dir output/"$V14"/"$STATE" --no-checkpoints >> "$OUT/champG.log" 2>&1
RC=$?; WALL=$((SECONDS-T0))
echo "[seq end rc=$RC TOTAL_WALL=${WALL}s ($((WALL/60))m)] $(date -u +%H:%M:%S)" | tee -a "$OUT/champG.log"
RD=$(ls -dt results/"$OVL"/"$STATE"/mtlnet_*ep50_* | head -1)
echo "rundir=$RD" | tee -a "$OUT/champG.log"
python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag fl_champG_seq 2>&1 | tee "$OUT/champG.score.txt"
cp -f "$RD/a40_matched_score.json" "$OUT/champG.score.json" 2>/dev/null
echo "=== FL SEQ DONE ===" | tee -a "$OUT/champG.log"
