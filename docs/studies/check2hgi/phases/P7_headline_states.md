# Phase 7 — Headline States (FL, CA, TX)

**Goal:** produce the paper's final headline numbers on the three large Gowalla states (Florida, California, Texas), with 5-fold cross-validation on user-disjoint splits at fair LR. This phase is **parallelizable across machines** — each state is independent and can run on its own hardware. Results feed directly into the paper's main table.

**Status as of 2026-04-20:** FL 5-fold STL cat + FL 5-fold MTL cross-attn currently running on the Mac M2 Pro (see `STATUS_REPORT_2026-04-20_v2.md`). CA + TX have raw data but need Check2HGI embeddings generated first.

## 0. What this phase delivers

Per-state table (to be merged into `results/BASELINES_AND_BEST_MTL.md` and the paper):

| Block | Measurement | Purpose |
|---|---|---|
| **Simple baselines** | Majority, top-K-popular, user-history, Markov k=1..9 region | Paper floor (context-matched) |
| **STL cat ceiling** | Check2HGI STL next_category 5f×50ep max_lr=0.01 | STL cat reference |
| **STL reg ceiling** | Check2HGI STL next_region GRU 5f×50ep max_lr=0.003 | STL reg reference |
| **MTL champion** | cross-attn + pcgrad 5f×50ep max_lr=0.003 | Primary paper claim (Δ cat, Δ reg vs STL) |
| **MTL λ=0 decomposition** | cross-attn + static_weight cat=0.0 5f×50ep max_lr=0.003 | Architectural-overhead decomposition (CH-M8: cat-enabled transfer) |

All with seed=42, user-disjoint `StratifiedGroupKFold`, per-task modality (cat: checkin, reg: region).

## 1. Prerequisites per state

**Available now (skip generation step):** FL.
**Need generation:** CA, TX.

To generate Check2HGI embeddings + inputs for a new state:

1. Edit `pipelines/embedding/check2hgi.pipe.py` — in `STATES` dict, uncomment target state:
   ```python
   STATES = {
       'California': {'shapefile': Resources.TL_CA},  # or 'Texas': {'shapefile': Resources.TL_TX},
   }
   ```
2. Run:
   ```bash
   cd <worktree>
   export PYTHONPATH=src
   export DATA_ROOT="<path-to-ingred>/data"
   export OUTPUT_DIR="<path-to-output>"  # or /tmp/check2hgi_data
   /path/to/python pipelines/embedding/check2hgi.pipe.py
   ```
3. Then generate region inputs:
   ```bash
   /path/to/python pipelines/create_inputs_check2hgi.pipe.py --state california
   ```
4. Verify outputs:
   ```
   ${OUTPUT_DIR}/check2hgi/california/
     ├── embeddings.parquet
     ├── poi_embeddings.parquet
     ├── region_embeddings.parquet
     └── input/
         ├── next.parquet
         ├── next_poi.parquet
         └── next_region.parquet
   ```

**Estimated embedding-gen time:** ~1-2 h per state on MPS/M2. Scales with check-in count (TX ~200K, CA ~300K rows).

## 2. Per-state execution plan (5 runs per state)

Each state needs these 5 runs, in this order (allows partial results to be useful even if interrupted):

### 2.1 Simple baselines (~5 min, CPU-only)

Produces `docs/studies/check2hgi/results/P0/simple_baselines/<state>/next_region.json` + `next_category.json`.

```bash
# Substitute <state> with: florida, california, or texas
STATE=<state>

cd <worktree>
export PYTHONPATH=src
export DATA_ROOT="<path>/data"
export OUTPUT_DIR="<output-path>"

/path/to/python scripts/compute_simple_baselines.py --state ${STATE} --task next_region
/path/to/python scripts/compute_simple_baselines.py --state ${STATE} --task next_category
```

**Outputs:** `P0/simple_baselines/<state>/{next_category,next_region}.json`. Markov k=1..9 included (since 2026-04-20 extension).

### 2.2 STL next-category 5f×50ep (~3-5 h depending on state size)

Uses `--task next` with the default `next_mtl` transformer head for next-category
prediction (NOT `--task category`, which is flat POI classification).

```bash
/path/to/python -u scripts/train.py \
    --task next --engine check2hgi \
    --state ${STATE} \
    --folds 5 --epochs 50 --seed 42 \
    --task-input-type checkin \
    --max-lr 0.01 \
    --gradient-accumulation-steps 1 --no-checkpoints
```

Then archive the summary JSON:
```bash
LATEST=$(ls -dt <worktree>/results/check2hgi/${STATE}/*_lr*_bs*_ep50_* | head -1)
cp "${LATEST}/summary/full_summary.json" \
   docs/studies/check2hgi/results/P4/${STATE}_stl_cat_fairlr_5f50ep.json
```

**Expected:** cat F1 ~42-66% depending on state (AL 38.58, AZ 42.08, FL ~63).

### 2.3 STL reg GRU 5f×50ep (~3-5 h)

```bash
/path/to/python -u scripts/train.py \
    --task next --engine check2hgi \
    --state ${STATE} \
    --folds 5 --epochs 50 --seed 42 \
    --task-input-type region \
    --next-head next_gru \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints
```

Archive as `P4/${STATE}_stl_reg_gru_fairlr_5f50ep.json`.

**Expected:** reg A@10 ~55-70% depending on state (AL 56.94, FL 68.33).

### 2.4 MTL cross-attn + pcgrad 5f×50ep (~6-10 h) — HEADLINE

```bash
/path/to/python -u scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state ${STATE} --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints
```

Archive as `P4/${STATE}_mtl_crossattn_pcgrad_fairlr_5f50ep.json`.

**This is the paper's headline.** Expected: cat exceeds STL (AL +0, AZ +1, FL +3); reg regresses by 5-11 pp.

### 2.5 MTL λ=0 cross-attn decomposition 5f×50ep (~3-5 h)

```bash
/path/to/python -u scripts/train.py \
    --task mtl --task-set check2hgi_next_region \
    --state ${STATE} --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.0 \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints
```

Archive as `P4/${STATE}_mtl_crossattn_lambda0_fairlr_5f50ep.json`.

**Gives the decomposition:** `architectural_overhead = STL_reg - lambda0_reg`, `transfer = full_MTL_reg - lambda0_reg`. This is CH-M8's numerical backbone.

## 3. Per-state total compute

| State | Rows | Regions | Baselines | STL cat | STL reg | MTL | λ=0 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Florida (FL) | 127,340 | 4,702 | 5 min | ~3 h | ~3 h | ~8 h | ~4 h | **~18 h** |
| California (CA) | ~300K (est.) | ~6K (est.) | 5 min | ~7 h | ~7 h | ~20 h | ~10 h | **~44 h + 2 h embed-gen** |
| Texas (TX) | ~200K (est.) | ~5K (est.) | 5 min | ~5 h | ~5 h | ~14 h | ~7 h | **~31 h + 1.5 h embed-gen** |

Runs are sequential per state but **fully parallelizable across machines.**

## 4. Machine assignment (parallel execution)

Assign one state per machine. Each machine independent — no shared state.

| State | Machine | Notes |
|---|---|---|
| **FL** | Mac M2 Pro (this machine) | 2/5 runs in progress; continue here |
| **CA** | Machine B | Must run embedding-gen first |
| **TX** | Machine C | Must run embedding-gen first |

Each machine needs: clone of worktree, `.venv` with pytorch+mps (or cuda), `data/checkins/<State>.parquet` + TIGER shapefile, and `$OUTPUT_DIR` pointing to local fast storage (not SSD if Thunderbolt-attached).

## 5. Launch script (drop-in per machine)

```bash
#!/usr/bin/env bash
# P7 launcher — run on one machine, one state.
# Usage: STATE=florida bash p7_launcher.sh
#
# Before running: ensure Check2HGI embeddings for $STATE exist at
# ${OUTPUT_DIR}/check2hgi/${STATE}/input/next_region.parquet

set -u
STATE="${STATE:?set STATE=florida|california|texas}"
WORKTREE="${WORKTREE:-$(pwd)}"
export PYTHONPATH="src"
export DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
export OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY="${PY:-python}"

DEST="${WORKTREE}/docs/studies/check2hgi/results/P4"
mkdir -p "${DEST}"

archive_summary() {
    local dest_name="$1"
    local latest
    latest=$(ls -dt "${WORKTREE}/results/check2hgi/${STATE}/"*_lr*_bs*_ep50_* 2>/dev/null | head -1)
    if [ -n "${latest}" ] && [ -f "${latest}/summary/full_summary.json" ]; then
        cp "${latest}/summary/full_summary.json" "${DEST}/${dest_name}.json"
        echo "[P7] saved → ${DEST}/${dest_name}.json"
    else
        echo "[P7] WARNING: no summary JSON for ${dest_name}"
    fi
}

run_with_retry() {
    local tag="$1" dest_name="$2"; shift 2
    for attempt in 1 2 3; do
        echo ""
        echo "=== [${tag}] attempt ${attempt} at $(date) ==="
        "$PY" -u scripts/train.py "$@"
        rc=$?
        echo "[${tag}] exit ${rc} at $(date)"
        if [ $rc -eq 0 ]; then
            archive_summary "${dest_name}"
            return 0
        fi
        sleep 30
    done
    echo "[${tag}] FAILED 3× — continuing"
}

# ========== 2.1 Simple baselines (fast; runs first so headline baselines are ready early) ==========
"$PY" scripts/compute_simple_baselines.py --state "${STATE}" --task next_region
"$PY" scripts/compute_simple_baselines.py --state "${STATE}" --task next_category

# ========== 2.2 STL cat 5f × 50ep ==========
run_with_retry "${STATE}_STL_cat" "${STATE}_stl_cat_fairlr_5f50ep" \
    --task category --engine check2hgi \
    --state "${STATE}" \
    --folds 5 --epochs 50 --seed 42 \
    --max-lr 0.01 \
    --gradient-accumulation-steps 1 --no-checkpoints

# ========== 2.3 STL reg GRU 5f × 50ep ==========
run_with_retry "${STATE}_STL_reg" "${STATE}_stl_reg_gru_fairlr_5f50ep" \
    --task next --engine check2hgi \
    --state "${STATE}" \
    --folds 5 --epochs 50 --seed 42 \
    --task-input-type region \
    --next-head next_gru \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints

# ========== 2.4 MTL cross-attn + pcgrad 5f × 50ep (HEADLINE) ==========
run_with_retry "${STATE}_MTL_crossattn" "${STATE}_mtl_crossattn_pcgrad_fairlr_5f50ep" \
    --task mtl --task-set check2hgi_next_region \
    --state "${STATE}" --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss pcgrad \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints

# ========== 2.5 MTL λ=0 cross-attn decomposition 5f × 50ep ==========
run_with_retry "${STATE}_MTL_lambda0" "${STATE}_mtl_crossattn_lambda0_fairlr_5f50ep" \
    --task mtl --task-set check2hgi_next_region \
    --state "${STATE}" --engine check2hgi \
    --folds 5 --epochs 50 --seed 42 \
    --task-a-input-type checkin --task-b-input-type region \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.0 \
    --max-lr 0.003 \
    --gradient-accumulation-steps 1 --no-checkpoints

echo ""
echo "=== P7 ${STATE} complete at $(date) ==="
```

Save as `scripts/p7_launcher.sh` (will commit with this phase plan).

## 6. Monitoring while running

Fast health-check (per machine):
```bash
# Is training still running?
ps aux | grep 'train.py' | grep -v grep

# Latest per-fold log lines
tail -3 <log-file>

# Per-fold completion pattern (expected every ~10-25 min depending on state/experiment)
grep "Fold [0-9]/5 completed" <log-file>

# Per-fold metrics summary
grep -E "next_category |next_region " <log-file>
```

**Known infra issues** (see `research/KNOWN_INFRA_ISSUES.md`):
- If Thunderbolt-SSD-mounted codebase flakes: use `OUTPUT_DIR=/tmp/...` for data/outputs, keep venv+code on local SSD.
- `--no-checkpoints` is wired to the MTL path (fix 10889ba 2026-04-20) — no accidental model-save during long runs.

## 7. Result aggregation (post-execution)

When all 3 states × 5 runs = 15 JSONs are in `docs/studies/check2hgi/results/P4/`, regenerate the paper table:

```bash
/path/to/python scripts/aggregate_p7_table.py  # TO BE WRITTEN after first state's data lands
```

**Paper table layout** (will be the body of `BASELINES_AND_BEST_MTL.md` §final):

```
| Row | Method                 | FL cat F1 | FL reg A@10 | CA cat F1 | CA reg A@10 | TX cat F1 | TX reg A@10 |
|-----|------------------------|-----------|-------------|-----------|-------------|-----------|-------------|
| 1   | Markov-1-region (floor)|    —      |      65.05  |     —     |       ?     |     —     |       ?     |
| 2   | Markov-9-region (match)|    —      |      54.10  |     —     |       ?     |     —     |       ?     |
| 3   | Check2HGI STL cat      |   63.17   |       —     |     ?     |       —     |     ?     |       —     |
| 4   | Check2HGI STL reg GRU  |    —      |      68.33  |     —     |       ?     |     —     |       ?     |
| 5   | MTL cross-attn+pcgrad  |  66.46 🚀 |     57.60   |     ?🚀   |       ?     |     ?🚀   |       ?     |
| 6   | MTL λ=0 (decomposition)|  12.50    |     43.40   |     ?     |       ?     |     ?     |       ?     |
|     | Δ cat (MTL − STL)      | **+3.29** |      —      | **?**     |       —     | **?**     |       —     |
|     | Δ reg (MTL − STL)      |    —      |   **−10.73**|     —     |  **?**      |     —     |  **?**      |
|     | Arch overhead          |    —      |     24.93   |     —     |       ?     |     —     |       ?     |
|     | Cat-enabled transfer   |    —      |    +14.20   |     —     |       ?     |     —     |       ?     |
```

## 8. Acceptance criteria for shipping the paper

- FL, CA, TX all have all 5 runs above completed (3 × 5 = 15 JSONs).
- Paper table (§7) reports mean ± std for all cells with n=5 folds.
- The monotone scale-curve trend (cat Δ grows with data, reg Δ widens with class count) holds across all 5 states (AL, AZ, FL, CA, TX).
- At least FL cat Δ (+3.29 or revised value) has non-overlapping σ with zero.

## 9. Risks + mitigations

| Risk | Mitigation |
|---|---|
| MPS OOM on large-state MTL | Fallback: reduce batch 2048 → 1024, or use `--gradient-accumulation-steps 2`. Don't change hparams otherwise. |
| Machine crashes mid-run | Retry wrapper in launcher handles 3 attempts; restart from the failed step manually if all 3 fail. |
| Embedding gen differs between machines | Pin seed in `embeddings/check2hgi/check2hgi.py`; should be deterministic. |
| Results vary across machines (hardware) | Shouldn't matter at this precision — σ around 1-3 pp dominates hardware noise. If suspect, re-run one state on primary machine and compare. |
| CA/TX Check2HGI training fails | Fall back to AL+AZ+FL only (3-state scale-curve is already sufficient for BRACIS). |

## 10. Dependencies + preconditions

- **Code version:** commit `8674656` or later (Markov k=9 extension). Ideally same `worktree-check2hgi-mtl` branch across all 3 machines to avoid analysis drift.
- **Python env:** Python 3.12, pytorch ≥ 2.4 with MPS or CUDA, cvxpy (for Nash), fvcore (FLOPs profiler; optional).
- **Data:** `data/checkins/<State>.parquet`, `data/miscellaneous/tl_2022_*_tract_*.shp` (for FL, CA, TX).
- **Disk:** ~10-20 GB for outputs per state.

## 11. What this phase does NOT include

Out-of-scope for P7 (deferred or skipped):
- **Multi-seed n=15** — can be a P8 if reviewers push back. +15× compute.
- **Cross-attn λ=0 on FL** — only dselectk λ=0 was run. If we want exact decomposition, add a 6th run. But dselectk λ=0 and cross-attn λ=0 were within σ on AL (51.87 vs 52.27), so dselectk is a fine proxy.
- **Hybrid arch, Nash-MTL, gated-skip, AdaShare** — all rejected as null on AZ; no reason to re-test at scale.
- **Additional embeddings (DGI, HGI)** — not needed for headline; paper is about Check2HGI.
- **POI-granularity next-POI** — out of scope for this study.

## Entry point

To launch on a new machine after embeddings are ready:
```bash
git checkout worktree-check2hgi-mtl
cd <worktree>
git pull
chmod +x scripts/p7_launcher.sh
STATE=california WORKTREE=$(pwd) DATA_ROOT=/path/to/data OUTPUT_DIR=/path/to/output PY=/path/to/python bash scripts/p7_launcher.sh > p7_california.log 2>&1 &
```
