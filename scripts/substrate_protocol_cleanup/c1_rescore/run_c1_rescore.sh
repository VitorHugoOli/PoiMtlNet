#!/usr/bin/env bash
# substrate-protocol-cleanup Tier C1 re-score (modality-bug fix, 2026-05-29).
# Re-train AL + AZ with --save-task-best-snapshots (seed=42, 5 folds,
# H3-alt small-state recipe), then re-score each fold with the FIXED
# route_task_best.py (which now rebuilds val loaders with task_b=region).
# Score + DELETE each state's snapshots before the next state to stay under
# the disk budget. DONE markers + per-step logs.
set -u

REPO=/home/vitor.oliveira/PoiMtlNet
PY="$REPO/.venv/bin/python"
TC="$REPO/docs/results/substrate_protocol_cleanup/tier_c"
LOGDIR="$REPO/scripts/substrate_protocol_cleanup/c1_rescore"
cd "$REPO" || exit 1

MIN_GPU_MIB=10000
MIN_DISK_GB=3

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
disk_free_gb() { df -BG "$REPO" | tail -1 | awk '{gsub("G","",$4); print $4}'; }

gate() {
  local g; g=$(gpu_free); local d; d=$(disk_free_gb)
  echo "[gate] gpu_free=${g}MiB disk_free=${d}GB"
  if [ "$g" -lt "$MIN_GPU_MIB" ]; then echo "[gate] FAIL gpu<${MIN_GPU_MIB}"; return 1; fi
  if [ "$d" -lt "$MIN_DISK_GB" ]; then echo "[gate] FAIL disk<${MIN_DISK_GB}"; return 1; fi
  return 0
}

run_state() {
  local state=$1
  local snapdir="$REPO/results/check2hgi/$state/task_best_snapshots"
  local cfg="$REPO/results/check2hgi/$state/config.json"
  local outdir="$TC/$state/C1_route"
  mkdir -p "$outdir"

  echo "===== [$state] TRAIN $(date -u) ====="
  if ! gate; then echo "[$state] GATE FAIL pre-train, abort"; return 1; fi

  "$PY" scripts/train.py --task mtl --task-set check2hgi_next_region \
      --state "$state" --engine check2hgi --seed 42 \
      --epochs 50 --folds 5 --batch-size 2048 \
      --model mtlnet_crossattn \
      --mtl-loss static_weight --category-weight 0.75 \
      --scheduler constant \
      --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
      --cat-head next_gru --reg-head next_getnext_hard \
      --task-a-input-type checkin --task-b-input-type region \
      --per-fold-transition-dir "output/check2hgi/$state" \
      --save-task-best-snapshots --no-checkpoints \
      > "$LOGDIR/train_${state}.log" 2>&1
  local rc=$?
  echo "[$state] train rc=$rc"
  if [ $rc -ne 0 ]; then echo "[$state] TRAIN FAILED rc=$rc"; return 1; fi

  if [ ! -d "$snapdir" ]; then echo "[$state] NO snapdir $snapdir"; return 1; fi
  echo "[$state] snapshots:"; ls -la "$snapdir" | tail -20
  cp "$cfg" "$outdir/config.json"

  echo "===== [$state] SCORE $(date -u) ====="
  for f in 1 2 3 4 5; do
    "$PY" scripts/route_task_best.py \
        --snapshots-dir "$snapdir" \
        --fold "$f" \
        --config "$cfg" \
        --output-json "$outdir/route_fold${f}.json" \
        > "$outdir/route_fold${f}.log" 2>&1
    echo "[$state] scored fold $f rc=$?"
  done

  echo "===== [$state] DELETE snapshots $(date -u) ====="
  rm -rf "$snapdir"
  echo "[$state] disk_free=$(disk_free_gb)GB after delete"
  touch "$TC/${state}_C1_RESCORE_DONE"
  return 0
}

echo "##### C1 RE-SCORE START $(date -u) #####"
run_state alabama
AL_RC=$?
echo "alabama done rc=$AL_RC disk=$(disk_free_gb)GB"
run_state arizona
AZ_RC=$?
echo "arizona done rc=$AZ_RC disk=$(disk_free_gb)GB"

echo "##### running analyzer #####"
"$PY" scripts/substrate_protocol_cleanup/analyze_tier_c1.py > "$LOGDIR/analyze.log" 2>&1
echo "analyzer rc=$?"

touch "$TC/TIER_C1_RESCORE_ALL_DONE"
echo "##### C1 RE-SCORE END $(date -u) AL_RC=$AL_RC AZ_RC=$AZ_RC #####"
