# HANDOFF — REDUCED board · **A40** (CUDA, Ampere) · branch `study/board-a40`

> Deadline-grade 1-seed board (`RUN_MATRIX_REDUCE.md`). The A40 is the **stable** card → it owns the two
> **heavy** states **FL** and **TX**. **1 seed (0) × 5 folds.** It is currently finishing the M2 Pro lane's
> **POI2Vec** (~2-3 h left); when that ends, leave ONLY this board work here. Incremental commits + own PR.

## 0 · SCOPE — A40 owns: FL, TX (full cell set each). FL was unassigned in the user's sketch → placed here so
the two heaviest states sit on the stable card and the H100 keeps the small states + CA. Confirm if you'd rather
swap FL to the H100.

## 1 · STEP 0 — finish POI2Vec (the M2 Pro lane work that migrated here)
Let the running POI2Vec build complete + commit its cells (it belongs to the **m2pro** PR #30 lane — commit on
`study/board-m2pro` or hand the artifacts back per that lane's manifest). Do NOT start GPU board cells that
contend for the card until POI2Vec frees it.

## 2 · STEP 1 — FL (start the precision-FREE parts during the POI2Vec wait)
The STL ceilings need NO precision decision → run them while POI2Vec finishes / while the H100 settles the gate.
```bash
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py florida 1
PYTHONPATH=src .venv/bin/python scripts/compute_region_transition.py --state florida --per-fold --seed 0
# STL reg ceiling (fp32) — REUSE prior 76.71 if a valid 5f artifact exists; else p1_region_head_ablation.py (as H100 §2)
# STL cat ceiling — train.py --task next --engine check2hgi_dk_ovl --state florida --seed 0 --folds 5 --cat-head next_gru --compile --tf32
```
**FL MTL** (after the H100 posts the gate verdict — bf16 or fp32): the §1-H100 champion-G command with
`--state florida`, the chosen precision prefix. FL fp32 anchor: fold-1 reg **77.71 > ceiling 76.71** (the gap
closed/reversed — expect FL MTL to meet/beat both ceilings).

## 3 · STEP 2 — TX (LAST — the single most expensive cell, ~11 h MTL)
The prior TX result (`tx_ba2_s0.json` Δreg −2.41, `best_epochs=[4,50,4,4,5]`) is a **fp16 ep30-collapse artifact
→ VOID**. Re-run in the chosen precision. Build TX overlap engine + log_T; STL reg ceiling **64.96 fp32 —
REUSE**; champion-G MTL `--state texas`. Auto-fit dataset; **NEVER `MTL_DATASET_GPU=1`** (forces ~31 GB copies →
OOM); `MTL_CHUNK_VAL_METRIC=1` is mandatory at TX's 4.09M-row overlap scale.

## 4 · PROCESS / PINS / STOP
- Branch `study/board-a40`; **incremental commits** (per cell + result JSON + finding); push as you go.
- log_T freshness preflight before every `--per-fold-transition-dir` run; `MTL_STRICT=1`.
- **STOP for the user:** any OOM / freshness / leak-guard failure; a TX MTL that still NaN-collapses under bf16.
- Do NOT run AL/AZ/CA (H100) or the baselines (Macs). Do NOT merge.
