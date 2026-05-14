# Execution Plan — Fair-hyperparameter ablation + Arizona scale validation

**Date:** 2026-04-18, after the max_lr=0.003 finding revealed HP mismatch.

## Why this plan exists

The prior ablation (7 intervention families, ~2 days of compute) was run at `max_lr=0.001` for MTL while STL was tuned at `max_lr=0.003` (region) and `max_lr=0.01` (cat). The first LR-correction experiment (step 7, partial) closed +4.25 pp of the region gap with a single HP change — meaning **most ablation conclusions may not survive fair hyperparameters**.

Separately, all experiments so far are on Alabama (10K rows). Florida (127K) is the headline state, and we saw MTL lift category on FL (+1.61 pp) even at the wrong LR. The AL findings may under-predict MTL potential at scale.

This plan (a) re-verifies the ablation under fair HPs and (b) adds Arizona (26K rows, 2× AL) as a mid-scale validation before committing FL compute.

## Phase 1 — Complete max_lr sweep (in progress, ~20 min remaining)

Already-running (bg `bauhto2o2`): dselectk+pcgrad AL 5f × 50ep at `max_lr ∈ {0.003, 0.01}`.

- max_lr=0.003 result: **cat 36.95 / reg 53.13 / Δm −8.64%** (+4.25 pp reg over max_lr=0.001).
- max_lr=0.01 result: pending.

**Decision rule:** pick the max_lr that gives best Δm as the "fair" value for Phase 2+.
- If max_lr=0.01 > max_lr=0.003: fair LR = 0.01.
- Else (expected): fair LR = 0.003.

## Phase 2 — AL reruns at fair max_lr (~100 min, sequential)

5 configs, chained background run. Each ~20 min.

| # | Config | Flag diff from baseline |
|---|--------|-------------------------|
| R1 | λ=0.0 isolation | `--mtl-loss static_weight --category-weight 0.0 --max-lr X` |
| R2 | mtlnet baseline | `--model mtlnet --max-lr X` |
| R3 | Cross-attention | `--model mtlnet_crossattn --max-lr X` |
| R4 | MTLoRA r=8 | `--model mtlnet_dselectk --model-param lora_rank=8 --max-lr X` |
| R5 | λ=0.5 equal-weight | `--mtl-loss static_weight --category-weight 0.5 --max-lr X` |

All: AL 5f × 50ep, seed 42, per-task modality, GRU region head, pcgrad (except R1/R5).

**Deliverables after Phase 2:**
- Rewritten FINAL_ABLATION_SUMMARY with fair-HP numbers.
- Revised paper narrative (likely: "HP calibration dominates; at fair HPs gap is ≤4 pp, not 8 pp").
- Fair-HP AL champion identified for use in Phases 3–4.

## Phase 3 — Arizona STL baselines (~65 min)

Adds the second point on the scale curve (AL → AZ → FL).

| # | Baseline | Command sketch |
|---|----------|----------------|
| A1 | Arizona STL next-category (Check2HGI) | `scripts/train.py --task next --state arizona --engine check2hgi --folds 5 --epochs 50 --seed 42 --no-checkpoints` (~60 min) |
| A2 | Arizona STL next-region (GRU, standalone) | `scripts/p1_region_head_ablation.py --state arizona --heads next_gru --folds 5 --epochs 50 --input-type region` (~3–5 min) |
| A3 | Arizona STL next-region (TCN, fast alt) | same with `--heads next_tcn_residual` (~3–5 min) |

## Phase 4 — Arizona MTL replication (~2 h)

Run the **top 2 fair-HP AL configs** from Phase 2 on Arizona, at the fair max_lr.

Baseline MTL: dselectk+pcgrad on Arizona at fair max_lr (for context).

Then the top 2 according to Phase 2 results — likely:
- Champion for cat: probably cross-attn (if fair-HP result stays positive) or mtlnet baseline.
- Champion for reg: probably MTLoRA r=8 or dselectk+pcgrad.

Each config: Arizona 5f × 50ep, fair max_lr, per-task modality. Expect ~40 min per config (AZ is 2× AL data).

**Deliverables after Phase 4:**
- Scale-curve data: AL → AZ for each key intervention.
- Verified finding: does the "fair-HP MTL closes X pp" pattern hold or break at 2× data?

## Phase 5 — Florida replication (~2 h)

Run the **Arizona-verified champion** on Florida, 1f × 50ep at fair max_lr.

- FL MTL best (cat): expected similar or better than current 64.78 (the FL MTL we already have was at unfair LR).
- FL MTL best (reg): critical test — does reg gap shrink below the current −11 pp?

## Phase 6 — Paper rewrite

With scale-curve data in hand:

1. **Rewrite FINAL_ABLATION_SUMMARY** with AL + AZ tables.
2. **Revise the paper's central claim** based on what survives fair HPs.
3. **Decide if per-task parameter groups** (different max_lr per branch) is worth implementing as a final fix. Gated on whether residual gap > 2 pp after fair single-LR MTL.

## Total compute estimate

| Phase | Compute | Cumulative |
|------:|--------:|-----------:|
| 1 (remaining max_lr=0.01) | 20 min | 20 min |
| 2 (AL 5 reruns) | 100 min | 2 h |
| 3 (AZ baselines) | 65 min | 3.1 h |
| 4 (AZ MTL, 2–3 configs) | 120 min | 5.1 h |
| 5 (FL champion) | 120 min | 7.1 h |
| 6 (rewrite, no compute) | — | — |

**~7 h** of unattended compute from now. Each phase commits + pushes on completion.

## Early-exit conditions

- **Phase 2 R1 (λ=0.0 at fair LR)**: if architectural overhead shrinks to < 2 pp, the paper's "overhead" claim dies and we focus on HP-calibration as the headline methodological finding.
- **Phase 4 Arizona**: if AZ results diverge from AL, stop and investigate before Phase 5.
- **Phase 5 Florida**: if FL doesn't replicate the AL/AZ pattern, paper leads with "scale-dependent MTL behavior" story instead.

## Abort conditions

- MPS / SSD instability: persist by redirecting `OUTPUT_DIR` to `/tmp` (already pattern we use).
- Any rerun gives > 3 pp SHIFT relative to prior: verify with a repeat (same seed, same config) before publishing.
