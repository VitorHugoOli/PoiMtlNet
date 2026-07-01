#!/usr/bin/env bash
# AL baseline validation (A40, fp32 board protocol) — champion-G MTL + STL cat + STL reg ceilings.
# Mirrors scripts/closing_data/board_h100_mtl.sh (fp32 arm) + board_h100_ceiling.sh (cat/reg),
# adapted for the a40-wk box (venv + /home path). Logs each cell + wall-clock; captures the NEW rundir; scores.
# Targets (RESULTS_BOARD §1b A40 same-device): MTL cat 63.25 / reg 69.65 ; STL cat 55.87 ; STL reg 69.99.
# Output: docs/studies/train_perf_multifold/baseline_runs/{champG,cat,reg}.{log,score.json}
# Optional arg $1 = tag suffix (default "baseline"); lets a post-change A/B run write to a separate prefix.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate

TAG="${1:-baseline}"
OUT=docs/studies/train_perf_multifold/${TAG}_runs
mkdir -p "$OUT"
V14=check2hgi_design_k_resln_mae_l0_1
OVL=check2hgi_dk_ovl
STATE=alabama

export PYTHONPATH=src
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board_al
# P1 (the vectorized STAN mask) is the PERF default and stays ON — we do NOT pin
# MTL_STAN_LEGACY_MASK. Phase-5c validation showed the legacy mask does NOT actually deliver
# bit-exact compiled reproduction: with a FRESH inductor cache, mask on==off (the compiled
# number is governed by the cache/compile SESSION, not the mask gate; mask=1 only restores the
# slower graph-break path). The deterministic ground truth is EAGER (the parity harness);
# compiled numbers are within fold-std. See log.md §Phase 5c.

ts() { date -u +%H:%M:%S; }
# newest rundir matching glob $1 that is newer than marker file $2
newest_new() { find $(dirname "$1") -maxdepth 1 -type d -name "$(basename "$1")" -newer "$2" 2>/dev/null | sort | tail -1; }

# ── Cell 1: champion-G MTL (fp32) ───────────────────────────────────────────────
MARK=$(mktemp); echo "[$(ts)] START champion-G MTL (fp32)" | tee "$OUT/champG.log"; S=$SECONDS
( export MTL_DISABLE_AMP=1
  python scripts/train.py --task mtl --canon none \
    --task-set check2hgi_next_region --engine "$OVL" \
    --state "$STATE" --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/$STATE" --no-checkpoints
) >> "$OUT/champG.log" 2>&1
RC=$?; echo "[$(ts)] END champion-G MTL rc=$RC wall=$((SECONDS-S))s" | tee -a "$OUT/champG.log"
MTL_RD=$(newest_new "results/$OVL/$STATE/mtlnet_*ep50_*" "$MARK"); rm -f "$MARK"
echo "rundir=$MTL_RD" | tee -a "$OUT/champG.log"
[ -n "$MTL_RD" ] && python scripts/closing_data/a40_score_matched.py "$MTL_RD" --seed 0 --tag al_champG_${TAG} 2>&1 | tee "$OUT/champG.score.txt" && cp -f "$MTL_RD/a40_matched_score.json" "$OUT/champG.score.json" 2>/dev/null

# ── Cell 2: STL category ceiling (next_gru, board precision = default) ───────────
MARK=$(mktemp); echo "[$(ts)] START STL cat ceiling" | tee "$OUT/cat.log"; S=$SECONDS
( python scripts/train.py --task next --state "$STATE" --engine "$OVL" \
    --model next_gru --folds 5 --epochs 50 --seed 0 \
    --batch-size 2048 --max-lr 3e-3 --gradient-accumulation-steps 1 --no-checkpoints \
    --compile --tf32
) >> "$OUT/cat.log" 2>&1
RC=$?; echo "[$(ts)] END STL cat ceiling rc=$RC wall=$((SECONDS-S))s" | tee -a "$OUT/cat.log"
CAT_RD=$(newest_new "results/$OVL/$STATE/next_*ep50_*" "$MARK"); rm -f "$MARK"
echo "rundir=$CAT_RD" | tee -a "$OUT/cat.log"
[ -n "$CAT_RD" ] && python scripts/closing_data/score_stl_cat_ceiling.py "$CAT_RD" --tag alabama_cat_ceiling 2>&1 | tee "$OUT/cat.score.txt" && cp -f "$CAT_RD/stl_cat_ceiling_score.json" "$OUT/cat.score.json" 2>/dev/null

# ── Cell 3: STL region ceiling (p1 next_stan_flow a0, fp32) ──────────────────────
echo "[$(ts)] START STL reg ceiling (fp32)" | tee "$OUT/reg.log"; S=$SECONDS
( export MTL_DISABLE_AMP=1
  python scripts/p1_region_head_ablation.py --state "$STATE" --heads next_stan_flow \
    --input-type region --region-emb-source "$V14" \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --engine-override "$OVL" \
    --per-fold-transition-dir "output/$V14/$STATE" \
    --folds 5 --epochs 50 --seed 0 --target region \
    --compile --tf32 --tag "${STATE}_ovl_stl_reg_${TAG}"
) >> "$OUT/reg.log" 2>&1
RC=$?; echo "[$(ts)] END STL reg ceiling rc=$RC wall=$((SECONDS-S))s" | tee -a "$OUT/reg.log"

echo "[$(ts)] ALL DONE" | tee -a "$OUT/champG.log"
