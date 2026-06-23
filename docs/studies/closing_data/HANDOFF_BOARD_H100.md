# HANDOFF — REDUCED board · **H100** (CUDA, Hopper) · branch `study/board-h100`

> Deadline-grade 1-seed board (`RUN_MATRIX_REDUCE.md`). The H100 carries the **precision gate** + the small/mid
> states + **CA last**. **1 seed (0) × 5 folds** everywhere. Incremental commits + own PR; do NOT touch main or
> another lane. The prior fp16 results on this branch are VOID (CA_MTL_DIVERGENCE.md) but the branch is mergeable
> as a record — we reset/re-run on top.

## 0 · SCOPE — H100 owns: AL, AZ, CA (full cell set each) + the precision gate
Run in this order (cheap → expensive). Each state = MTL champion-G + its STL ceilings, seed 0 × 5f.

## 1 · ⚡ STEP 0 — precision gate (bf16 vs fp32) at AL — RUN FIRST, BLOCKS all MTL
This doubles as the AL MTL cell. Two arms, identical except the env prefix; both scored fp32 by `r0_matched_rescore.py`.
```bash
# env (per-box, once)
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
# build AL overlap engine + seeded log_T (CPU; ~minutes)
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py alabama 1
PYTHONPATH=src .venv/bin/python scripts/compute_region_transition.py --state alabama --per-fold --seed 0

# Arm X — bf16 (proposed default):
MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1 \
  .venv/bin/python scripts/train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_dk_ovl \
    --state alabama --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/alabama --no-checkpoints
# Arm Y — full fp32 (reference): same line, prefix  MTL_DISABLE_AMP=1  instead.
```
**Decision rule:** `|Δcat|,|Δreg| ≤ 0.05 pp` (4 dp, per-fold + mean) ⇒ board standardizes **bf16**; else **fp32**.
**STOP and post the 4-dp table for the user.** Keep the chosen arm's AL run as the AL MTL result. (AL fp32
anchors: reg ceiling 69.98; fp32 MTL reg ≈ 69.80.)

## 2 · STEP 1 — AL ceilings (precision-free; can run during/after the gate)
```bash
# STL cat ceiling
.venv/bin/python scripts/train.py --task next --engine check2hgi_dk_ovl --state alabama --seed 0 \
    --folds 5 --cat-head next_gru --compile --tf32   # macro-F1 ceiling
# STL reg ceiling (fp32 already) — REUSE prior 69.98 if a valid 5f artifact exists; else:
.venv/bin/python scripts/p1_region_head_ablation.py --state alabama --region-emb-source check2hgi_dk_ovl \
    --region-head next_stan_flow --seed 0 --folds 5 --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/alabama
```

## 3 · STEP 2 — AZ (repeat STEP 0-Arm-chosen MTL + STEP 1 ceilings, state=arizona)
Build the AZ overlap engine + log_T first (as §1). MTL in the chosen precision (no second A/B). AZ regions=1547.

## 4 · STEP 3 — CA (LAST — heaviest, restart-risky)
CA (8501 regions, 3.17M rows) is where fp16 collapsed at ep30. With **bf16 it clears ep30 in ~40 min** — but the
studio restarts every ~1-2 h, so **start CA FIRST in a fresh window** and let it run. Build CA overlap engine +
log_T, then the chosen-precision MTL + STL ceilings (reg ceiling **63.48 fp32 — REUSE**, don't re-run). Auto-fit
dataset; **NEVER `MTL_DATASET_GPU=1`** (OOM). If a restart truncates a fold, resume that fold only.

## 5 · PROCESS / PINS / STOP
- Branch `study/board-h100`; **incremental commits** (per cell + result JSON + 1-line finding); push as you go.
- Every `--per-fold-transition-dir` run: **log_T freshness preflight** (`src/data/log_t_freshness.py`); log_T
  mtime > the overlap `next_region.parquet` mtime. `MTL_STRICT=1` hard-fails a stale/ungated build.
- **STOP for the user:** the precision-gate table (§1); any OOM / freshness / leak-guard failure; any MTL run
  that still NaN-collapses under bf16 (would mean the fix is insufficient — escalate).
- Do NOT run FL/TX (A40) or the baselines (Macs). Do NOT merge.
