# HANDOFF — REDUCED board · **A40** (CUDA, Ampere) · branch `study/board-a40`

> Deadline-grade 1-seed board (`RUN_MATRIX_REDUCE.md`). The A40 is the **stable** card → it owns the single
> longest run, **TX** (FL moved to the H100 to co-schedule with the small states, 2026-06-23). **1 seed (0) × 5
> folds.** It is currently finishing the M2 Pro lane's **POI2Vec** (~2-3 h left); when that ends, leave ONLY TX
> here. Incremental commits + own PR.

## 0 · SCOPE — A40 owns: **TX only** (full cell set). FL went to the H100 (parallel with the small states).

## 1 · STEP 0 — finish POI2Vec (the M2 Pro lane work that migrated here)
Let the running POI2Vec build complete + commit its cells (it belongs to the **m2pro** PR #30 lane — commit on
`study/board-m2pro` or hand the artifacts back per that lane's manifest). Do NOT start the TX GPU cell until
POI2Vec frees the card.

> 🚧 **HARD BLOCKER (audited 2026-06-24).** `output/check2hgi_dk_ovl/texas/input/next.parquet` is **MISSING** on
> disk → BOTH the TX cat-ceiling and the TX MTL Cell B fail immediately until the TX overlap engine is built. The
> `build_overlap_probe_engine.py texas 1` step below IS the gate — it needs the **v14 design_k TX substrate**
> (`output/check2hgi_design_k_resln_mae_l0_1/texas/`) on disk first. This is the real gate on the TX run, not any
> code change.
>
> ✅ **Precision = fp32** (consistent with AL/AZ/CA). The H100 gate already ran: **bf16 ≈ fp32 (Δ≤0.12 pp)** and,
> because overlap runs are data-bound (GPU util ~8-25%), bf16 buys ~0 wall-clock → use fp32 unless A40 memory
> forces bf16. ✅ **Merged to main (#33): the `perf(mtl)` 4.5× wide-head fix** (GPU-resident train-metric +
> batched collate) — this is what makes TX Cell B **~11 h instead of ~780 h** on the 8501-wide reg head. Pull
> main before launching.

## 2 · STEP 1 — TX prep (CPU; can run during POI2Vec / before the gate verdict)
```bash
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
# REQUIRED FIRST (next.parquet is missing): build the TX overlap engine from the v14 design_k TX substrate
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py texas 1
PYTHONPATH=src .venv/bin/python scripts/compute_region_transition.py --state texas --per-fold --seed 0
# TX STL reg ceiling — REUSE prior 64.96 fp32; STL cat ceiling (NEW tooling on main, #33):
#   bash scripts/closing_data/a40_tx_cat_ceiling.sh   (next_gru, seed0 5f, dk_ovl; scored by score_stl_cat_ceiling.py)
```

## 3 · STEP 2 — TX MTL (LAST — the single most expensive cell, ~11 h)
The prior TX result (`tx_ba2_s0.json` Δreg −2.41, `best_epochs=[4,50,4,4,5]`) is a **fp16 ep30-collapse artifact
→ VOID**. Re-run in the chosen precision (after the H100 posts the gate verdict — bf16 or fp32): the §1-H100
champion-G command, `--state texas`. Auto-fit dataset; **NEVER `MTL_DATASET_GPU=1`** (forces ~31 GB copies →
OOM); `MTL_CHUNK_VAL_METRIC=1` is mandatory at TX's 4.09M-row overlap scale.

## 4 · PROCESS / PINS / STOP
- Branch `study/board-a40`; **incremental commits** (per cell + result JSON + finding); push as you go.
- log_T freshness preflight before every `--per-fold-transition-dir` run; `MTL_STRICT=1`.
- **STOP for the user:** any OOM / freshness / leak-guard failure; a TX MTL that still NaN-collapses under bf16.
- Do NOT run AL/AZ/FL/CA (H100) or the baselines (Macs). Do NOT merge.
