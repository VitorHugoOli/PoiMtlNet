#!/usr/bin/env bash
# A40-5-Istanbul: P6 cascade-variant coverage. v17 recipe (user decision) = the EXACT H3 Istanbul MTL command
# (its own comparand) + the 5 b4 cascade pins (cond_coupling posterior/softmax/add/detach + disable_cross_attn).
# Only the coupling topology differs vs the H3 v17 comparand (cat 63.32 / reg 75.41 → joint 69.10). seed 0 × 5f.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate
export PYTHONPATH=src
export MTL_DISABLE_AMP=1 MTL_CHUNK_VAL_METRIC=1 MTL_COMPILE_DYNAMIC=1 MTL_ONECYCLE_PER_HEAD_LR=1  # NO MTL_STRICT (cond-guard hard-fails on unaligned cond_coupling; matches AL/AZ/FL cascade; guard is preflight-only, not math)
export TORCHINDUCTOR_CACHE_DIR=/home/vitor.oliveira/.inductor_cache_a40_5_casc
OVL=check2hgi_dk_ovl; V14=check2hgi_design_k_resln_mae_l0_1
BASE=docs/studies/closing_data/v17_completion/a40_5_cascade
DRV=$BASE/DRIVER.log; : > "$DRV"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$DRV"; }

log "A40-5-Istanbul cascade START (v17 recipe = H3 cmd + 5 pins, seed 0, 5f)"
python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region --engine "$OVL" \
    --state istanbul --seed 0 --epochs 50 --folds 5 --batch-size 8192 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir "output/$V14/istanbul" --no-checkpoints \
    --reg-head-param cond_coupling=posterior --reg-head-param cond_signal=softmax \
    --reg-head-param cond_inject=add --reg-head-param cond_detach=True \
    --model-param disable_cross_attn=True > "$BASE/cascade_run.log" 2>&1 &
pid=$!; wait $pid; rc=$?
RD=$(ls -d results/$OVL/istanbul/mtlnet_*bs8192_ep50_*_${pid} 2>/dev/null | head -1)
nan=$(grep -ciE "non-finite" "$BASE/cascade_run.log" 2>/dev/null || echo 0)
log "cascade DONE rc=$rc nan=$nan rundir=$(basename "${RD:-NONE}")"
[ $rc -ne 0 -o -z "$RD" ] && { log "FAIL — see cascade_run.log"; exit 1; }

log "scoring (matched scorer) + Δjoint vs H3 comparand"
python scripts/closing_data/a40_score_matched.py "$RD" --seed 0 --tag istanbul_cascade_v17_s0 | tee -a "$DRV"
python - "$RD" <<'PY' 2>>"$DRV" | tee -a "$DRV"
import sys, json, math, glob
rd=sys.argv[1]
sc=json.load(open(f"{rd}/a40_matched_score.json"))
casc_cat=sc.get("cat_macro_f1_mean") or sc.get("cat_macro_f1")
casc_reg=sc.get("reg_full_top10_mean") or sc.get("reg_full_top10")
comp_cat, comp_reg = 63.3179, 75.4066   # H3 Istanbul v17 MTL seed-0 comparand
cj=math.sqrt(casc_cat*casc_reg); pj=math.sqrt(comp_cat*comp_reg)
print(f"\n==== A40-5 Istanbul cascade vs parallel (v17, dk_ovl, seed0 5f) ====")
print(f"{'':10} {'cat':>8} {'reg':>8} {'joint':>8}")
print(f"{'cascade':10} {casc_cat:8.2f} {casc_reg:8.2f} {cj:8.2f}")
print(f"{'parallel':10} {comp_cat:8.2f} {comp_reg:8.2f} {pj:8.2f}")
print(f"{'Δ':10} {casc_cat-comp_cat:+8.2f} {casc_reg-comp_reg:+8.2f} {cj-pj:+8.2f}")
print(f"\nVERDICT: {'TIE holds (|Δjoint|≤0.5, far below fold-std)' if abs(cj-pj)<0.5 else 'TIE BREAKS — STOP + report (result, not bug)'}")
PY
log "A40-5-Istanbul cascade COMPLETE"
