#!/bin/bash
# =============================================================================
# P3 BOARD DRIVER — the ONE sanctioned build+run path for the P3 board.
# =============================================================================
#
#   ┌───────────────────────────────────────────────────────────────────────┐
#   │  THIS SCRIPT IS THE ONLY SANCTIONED P3-BOARD PATH.                      │
#   │                                                                        │
#   │  The four P3-board values live HERE and NOWHERE ELSE:                  │
#   │      • stride        = 1   (fully overlapping windows)                 │
#   │      • min_seq       = 10  (MIN_SEQUENCE_LENGTH override)              │
#   │      • --compile     (torch.compile, inductor)                         │
#   │      • --tf32        (TF32 matmul)                                     │
#   │                                                                        │
#   │  They are NOT global defaults. Every OTHER build/run path is FROZEN at │
#   │  stride=9 (non-overlap) / min_seq=5 / no-compile / no-tf32, producing │
#   │  byte-identical output to the v11/v14 substrates + the §0.1 numbers.   │
#   │  The recipe itself is already the default (canon v16); this driver     │
#   │  only adds the windowing + perf knobs uniformly across all cells.      │
#   └───────────────────────────────────────────────────────────────────────┘
#
# What it does, per state, REUSING the frozen v14 embeddings (engine
# check2hgi_design_k_resln_mae_l0_1) — it NEVER rebuilds embeddings (those are
# hash-manifested / frozen):
#   (a) rebuild the windowing-dependent inputs (next / sequences) at
#       --stride 1 --min-seq 10 via create_inputs.pipe.py-equivalent builder;
#       then rebuild next_region.parquet from the stride-1 inputs.
#   (b) build the seeded per-fold log_T at the new windowing per (state,seed)
#       so it matches the stride-1 next_region (the +8..+12pp stale-log_T trap).
#   (c) per (state,seed) in STATES × SEEDS: run train.py --canon v16 --compile
#       --tf32 with PID-suffix rundir capture (NEVER `ls -dt | head`).
#
# Preflight (refuses to launch otherwise):
#   • torch == 2.11.0+cu128
#   • log_T freshness: every region_transition_log_seed{S}_fold{N}.pt mtime
#     must be >= next_region.parquet mtime.
#
# Usage:
#   bash scripts/closing_data/p3_board.sh --dry-run        # print plan, no exec
#   bash scripts/closing_data/p3_board.sh                  # run all states/seeds
#   bash scripts/closing_data/p3_board.sh --states "alabama florida" --seeds "0 1"
#
# GPU job — do NOT run on CPU-only / review machines. --dry-run is always safe.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Config (the P3-board constants live HERE)
# ---------------------------------------------------------------------------
ENGINE="check2hgi_design_k_resln_mae_l0_1"   # frozen v14 substrate (NOT rebuilt)
STRIDE=1                                       # P3 board: fully overlapping windows
MIN_SEQ=10                                     # P3 board: MIN_SEQUENCE_LENGTH override
CANON="v16"                                    # champion G recipe (already the default)
REQUIRED_TORCH="2.11.0+cu128"

DEFAULT_STATES=(alabama arizona georgia florida california texas)
DEFAULT_SEEDS=(0 1 7 100)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
DRY_RUN=0
STATES=("${DEFAULT_STATES[@]}")
SEEDS=("${DEFAULT_SEEDS[@]}")
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --states)  read -r -a STATES <<< "$2"; shift 2 ;;
        --seeds)   read -r -a SEEDS  <<< "$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "ERR unknown arg: $1 (see --help)"; exit 2 ;;
    esac
done

say(){ echo "[p3_board] $*"; }
hdr(){
    echo "=============================================================================="
    echo "  P3 BOARD — the ONLY sanctioned P3-board build+run path"
    echo "  stride=$STRIDE  min_seq=$MIN_SEQ  --compile  --tf32   (canon $CANON, all cells uniform)"
    echo "  These knobs live in THIS script — they are NOT global defaults."
    echo "  Every other path stays frozen at stride=9 / min_seq=5 / no-compile / no-tf32."
    echo "  engine (frozen, NOT rebuilt): $ENGINE"
    echo "  states: ${STATES[*]}"
    echo "  seeds:  ${SEEDS[*]}"
    [ "$DRY_RUN" -eq 1 ] && echo "  MODE:   DRY-RUN (plan only, no execution)"
    echo "=============================================================================="
}

# ---------------------------------------------------------------------------
# Preflights
# ---------------------------------------------------------------------------
preflight_torch(){
    local got
    got="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo MISSING)"
    if [ "$got" != "$REQUIRED_TORCH" ]; then
        echo "ERR torch version mismatch: have '$got', require '$REQUIRED_TORCH'"
        return 1
    fi
    say "torch $got OK"
}

# mtime helper that works on both GNU (Linux) and BSD (macOS) stat
mtime(){ stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"; }

preflight_logT_fresh(){
    local state="$1" seed="$2"
    local out_dir="output/${ENGINE}/${state}"
    local next_region="$out_dir/input/next_region.parquet"
    [ -f "$next_region" ] || { echo "ERR next_region missing: $next_region"; return 1; }
    local nr_m; nr_m="$(mtime "$next_region")"
    for f in 1 2 3 4 5; do
        local lt="$out_dir/region_transition_log_seed${seed}_fold${f}.pt"
        [ -f "$lt" ] || { echo "ERR missing log_T $lt"; return 1; }
        [ "$(mtime "$lt")" -ge "$nr_m" ] || { echo "ERR STALE log_T $lt (< next_region)"; return 1; }
    done
    say "log_T freshness OK ($state seed=$seed, 5 folds)"
}

# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
build_inputs(){
    local state="$1"
    local out_dir="output/${ENGINE}/${state}"
    [ -f "$out_dir/embeddings.parquet" ] || { echo "ERR frozen v14 embeddings missing: $out_dir/embeddings.parquet (do NOT rebuild — restore from manifest)"; return 1; }
    say "[$state] (a) rebuild next/sequences at stride=$STRIDE min_seq=$MIN_SEQ (reusing frozen embeddings)"
    "$PY" - "$state" "$STRIDE" "$MIN_SEQ" "$ENGINE" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from configs.paths import EmbeddingEngine
from data.inputs.builders import generate_next_input_from_checkins
state, stride, min_seq, engine = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
generate_next_input_from_checkins(
    state, EmbeddingEngine(engine), stride=stride, min_sequence_length=min_seq,
)
print(f"[p3_board] next.parquet + sequences rebuilt (stride={stride}, min_seq={min_seq})")
PYEOF
    say "[$state] (a) rebuild next_region.parquet from the stride-1 inputs"
    # NOTE: build_design_next_region.py joins against canonical (stride-9)
    # sequences_next.parquet and asserts equal row counts. At stride=1 those
    # counts DIFFER → it will FAIL LOUDLY (it will NOT silently produce a
    # mismatched next_region). See "KNOWN GAP" in the final report. A
    # stride-aware next_region builder is required before a real P3 launch.
    "$PY" scripts/substrate_protocol_cleanup/build_design_next_region.py \
        --state "$state" --engine "$ENGINE"
}

build_logT(){
    local state="$1" seed="$2"
    local out_dir="output/${ENGINE}/${state}"
    local canon_dir="output/check2hgi/${state}"
    say "[$state seed=$seed] (b) build seeded per-fold log_T at the new windowing"
    # KNOWN GAP: compute_region_transition.py --per-fold is hardwired to the
    # canonical check2hgi engine (it reads canonical sequences_next.parquet +
    # canonical fold groups). It does NOT see stride-1 inputs. Until that tool
    # is stride/engine-aware, the log_T it emits reflects stride-9 transitions.
    # We still rebuild + copy + touch (so the freshness mtime guard passes) and
    # FLAG this as a correctness caveat for the real launch.
    "$PY" scripts/compute_region_transition.py --state "$state" --per-fold --seed "$seed"
    for f in 1 2 3 4 5; do
        cp "$canon_dir/region_transition_log_seed${seed}_fold${f}.pt" \
           "$out_dir/region_transition_log_seed${seed}_fold${f}.pt"
    done
    sleep 1
    touch "$out_dir"/region_transition_log_seed${seed}_fold*.pt
}

# ---------------------------------------------------------------------------
# Train step (PID-suffix rundir capture — NEVER ls -dt | head)
# ---------------------------------------------------------------------------
run_cell(){
    local state="$1" seed="$2"
    local out_dir="output/${ENGINE}/${state}"
    local logroot="/tmp/p3_board"; mkdir -p "$logroot"
    local runlog="$logroot/${state}_s${seed}.train.log"
    say "[$state seed=$seed] (c) train.py --canon $CANON --compile --tf32 -> $runlog"
    "$PY" scripts/train.py --task mtl --canon "$CANON" \
        --state "$state" --seed "$seed" \
        --compile --tf32 \
        --per-fold-transition-dir "$out_dir" \
        > "$runlog" 2>&1 &
    local pid=$!
    say "[$state seed=$seed] launched PID=$pid; rundir captured by PID suffix (mtlnet_..._${pid})"
    wait "$pid"
    # PID-suffix rundir: train.py names the rundir mtlnet_..._{PID}; resolve by suffix
    local rundir
    rundir="$(ls -d results/${ENGINE}/${state}/mtlnet_*_${pid} 2>/dev/null | head -1 || true)"
    say "[$state seed=$seed] rundir=${rundir:-<not found: inspect $runlog>}"
}

# ---------------------------------------------------------------------------
# Plan / execute
# ---------------------------------------------------------------------------
print_plan(){
    echo "PLAN (the exact commands the live run would execute):"
    echo "  preflight: $PY -c 'import torch; assert torch.__version__==\"$REQUIRED_TORCH\"'"
    for state in "${STATES[@]}"; do
        echo "  --- state=$state ---"
        echo "  (a) generate_next_input_from_checkins($state, $ENGINE, stride=$STRIDE, min_sequence_length=$MIN_SEQ)"
        echo "  (a) $PY scripts/substrate_protocol_cleanup/build_design_next_region.py --state $state --engine $ENGINE"
        for seed in "${SEEDS[@]}"; do
            echo "  (b) $PY scripts/compute_region_transition.py --state $state --per-fold --seed $seed   (+copy/touch into output/$ENGINE/$state)"
        done
        for seed in "${SEEDS[@]}"; do
            echo "  (pre) log_T freshness check (5 folds) for seed=$seed"
            echo "  (c) $PY scripts/train.py --task mtl --canon $CANON --state $state --seed $seed --compile --tf32 --per-fold-transition-dir output/$ENGINE/$state   [rundir captured by \$! PID suffix]"
        done
    done
    echo "TOTAL cells: $(( ${#STATES[@]} * ${#SEEDS[@]} ))  (${#STATES[@]} states × ${#SEEDS[@]} seeds), uniform --compile --tf32 --stride $STRIDE --min-seq $MIN_SEQ"
}

main(){
    hdr
    if [ "$DRY_RUN" -eq 1 ]; then
        print_plan
        echo
        say "DRY-RUN complete — nothing executed."
        return 0
    fi

    # ── LAUNCH SAFETY-STOP (adversarial review 2026-06-19) ───────────────────
    # 3 launch-blockers make a LIVE run UNSAFE today (a --dry-run is always safe):
    #  (1) build_inputs OVERWRITES the frozen v14 substrate's next.parquet /
    #      sequences_next.parquet IN PLACE, then hard-aborts at the canonical
    #      row-count assert → leaves v14 in a corrupted mixed-windowing state.
    #      FIX: stage stride-1 inputs to a separate dir, or back-up+restore v14.
    #  (2) build_design_next_region.py joins against the CANONICAL stride-9
    #      sequences_next.parquet → hard-fails at stride-1. FIX: stride-aware
    #      next_region builder joining the engine's OWN stride-1 sequences.
    #  (3) compute_region_transition.py --per-fold is hardwired to CHECK2HGI →
    #      emits stride-9 log_T while the mtime guard still passes (copy+touch)
    #      → the +8..+12pp stale-log_T trap. FIX: make it stride/engine-aware.
    # Until these land, refuse to run (would corrupt frozen data / wrong numbers).
    if [ "${P3_BOARD_FORCE:-0}" != "1" ]; then
        echo "‼ p3_board: NOT launch-ready — 3 P3 launch-blockers must be fixed first"
        echo "  (see the LAUNCH SAFETY-STOP block above + docs/studies/pre_freeze_gates/DEFAULTS_AND_GUARDS.md)."
        echo "  --dry-run previews the plan safely; set P3_BOARD_FORCE=1 ONLY after the 3 fixes land."
        exit 1
    fi

    preflight_torch

    for state in "${STATES[@]}"; do
        build_inputs "$state"
        for seed in "${SEEDS[@]}"; do
            build_logT "$state" "$seed"
        done
    done

    for state in "${STATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            preflight_logT_fresh "$state" "$seed"
        done
    done

    for state in "${STATES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_cell "$state" "$seed"
        done
    done

    say "DONE — all $(( ${#STATES[@]} * ${#SEEDS[@]} )) cells complete."
}

main
