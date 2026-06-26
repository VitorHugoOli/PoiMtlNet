#!/usr/bin/env bash
# Fast metric-parity harness for the slimming A/B gates.
#
# Runs the champion MTL path on a SMALL config (alabama — the smallest available
# check2hgi_dk_ovl substrate; 2 folds × 8 epochs; EAGER fp32 → deterministic + ~90s)
# and captures the per-fold VAL metric CSVs. A behavior-preserving refactor must keep
# these BYTE-IDENTICAL. (Istanbul has no check2hgi_dk_ovl substrate, so AL is the floor.)
#
# Usage:
#   parity_check.sh run  <tag> [extra train.py args...]   # run + capture → /tmp/parity/<tag>/
#   parity_check.sh diff <tagA> <tagB>                     # byte-compare two captures
#
# Exercises: cross-attn dualtower + next_gru(cat) + next_stan_flow_dualtower(reg, C=1109>256 →
# the S1 streaming train-metric + S2 chunked val-metric + OOD-restricted Acc@K paths) + the
# joint-selector math + per-fold log_T prior. Pass `--log-t-kd-weight 0.2` to also exercise KD.
set -uo pipefail
cd /home/vitor.oliveira/PoiMtlNet
source /home/vitor.oliveira/.venv/bin/activate >/dev/null 2>&1
export PYTHONPATH=src MTL_DISABLE_AMP=1 MTL_STRICT=1 MTL_CHUNK_VAL_METRIC=1
V14=check2hgi_design_k_resln_mae_l0_1
PDIR=/tmp/parity
mkdir -p "$PDIR"

CMD="${1:?run|diff|run_stl}"; shift
if [ "$CMD" = "run_stl" ]; then
  # Single-task (STL cat-ceiling) config — exercises _run_next / next_cv / _single_task_train
  # (the runner-merge A/B that the MTL harness doesn't cover). 2 folds × 8 ep, eager fp32.
  TAG="${1:?tag}"; shift
  python scripts/train.py --task next --state alabama --engine check2hgi_dk_ovl \
    --model next_gru --folds 2 --epochs 8 --seed 0 --batch-size 2048 \
    --max-lr 3e-3 --gradient-accumulation-steps 1 --no-checkpoints "$@" \
    > "$PDIR/${TAG}.log" 2>&1
  RC=$?
  RD=$(ls -dt results/check2hgi_dk_ovl/alabama/next_*ep8_* 2>/dev/null | head -1)
  rm -rf "$PDIR/$TAG"; mkdir -p "$PDIR/$TAG"
  [ -n "$RD" ] && cp "$RD"/metrics/fold*_val.csv "$PDIR/$TAG/" 2>/dev/null
  n=$(ls "$PDIR/$TAG"/*.csv 2>/dev/null | wc -l)
  echo "[parity:run_stl] tag=$TAG rc=$RC captured=$n CSVs (rundir=$RD)"
elif [ "$CMD" = "run" ]; then
  TAG="${1:?tag}"; shift
  python scripts/train.py --task mtl --canon none --task-set check2hgi_next_region \
    --engine check2hgi_dk_ovl --state alabama --seed 0 --epochs 8 --only-folds 0,1 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 --no-reg-class-weights --no-cat-class-weights \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --per-fold-transition-dir "output/$V14/alabama" \
    --no-checkpoints --run-id "parity_${TAG}" --per-fold-seed "$@" > "$PDIR/${TAG}.log" 2>&1
  RC=$?
  RD=$(ls -d results/check2hgi_dk_ovl/alabama/mtlnet_*parity_$(printf '%s' "$TAG" | tr 'A-Z' 'a-z') 2>/dev/null | head -1)
  rm -rf "$PDIR/$TAG"; mkdir -p "$PDIR/$TAG"
  [ -n "$RD" ] && cp "$RD"/metrics/fold*_val.csv "$PDIR/$TAG/" 2>/dev/null
  # Also capture the selection record (diagnostic_best_epochs + primary_checkpoint epoch/metrics),
  # stripping run-varying timing, so the diff covers the joint-selector / checkpoint path too.
  if [ -n "$RD" ]; then
    for fi in "$RD"/folds/fold*_info.json; do
      [ -f "$fi" ] || continue
      python3 - "$fi" "$PDIR/$TAG/$(basename "$fi" .json).sel.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
pc = d.get("primary_checkpoint", {}) or {}
# The joint-SELECTOR-dependent record: which epoch the selector picked + its per-task
# metrics. (diagnostic_best_epochs is selector-independent and carries run-varying `time`.)
sel = {"primary_epoch": pc.get("epoch"), "primary_task_metrics": pc.get("task_metrics")}
json.dump(sel, open(sys.argv[2], "w"), indent=2, sort_keys=True)
PY
    done
  fi
  n=$(ls "$PDIR/$TAG"/*.csv 2>/dev/null | wc -l); s=$(ls "$PDIR/$TAG"/*.sel.json 2>/dev/null | wc -l)
  echo "[parity:run] tag=$TAG rc=$RC captured=$n CSVs + $s sel digests (rundir=$RD)"
  [ "$n" -eq 4 ] || { echo "[parity:run] WARN expected 4 val CSVs (2 folds × 2 tasks); check $PDIR/${TAG}.log"; tail -5 "$PDIR/${TAG}.log"; }
elif [ "$CMD" = "diff" ]; then
  A="$1"; B="$2"
  if diff -rq "$PDIR/$A" "$PDIR/$B" >/dev/null 2>&1; then
    echo "✅ PARITY: $A == $B (val metrics byte-identical)"
  else
    echo "❌ PARITY FAIL: $A vs $B"; diff -rq "$PDIR/$A" "$PDIR/$B"
  fi
fi
