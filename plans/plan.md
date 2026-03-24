# BRACIS 2026 ENIAC Master Plan (Merged)

Last update: 2026-02-06  
Target submission date (working assumption): 2026-05-26

## Context

- Goal: define the most important execution order to maximize article quality and reduce publication risk.
- ENIAC 2026 CFP is not published yet; `2026-05-26` is the internal hard deadline.

## Publication Goal

Deliver a credible, reproducible paper with:

- Full baseline matrix for Florida, Texas, California:
  - `POIRGNN`, `PGC`, `Next (legacy)`, `Havana`, `MTLNet (legacy)`.
- Improved Next model (autotune + reproducible gains).
- Final MTLNet with MoE and optimizer comparisons.
- Leakage-safe split protocol and stable reproducibility package.

## Critical Path (Most Important Order)

### Priority 1 - Validity Blockers (must be fixed before final experiments)

- [ ] Fix fusion feature truncation (dynamic dimensions must be preserved end-to-end).
  - Current risk: fusion files have larger dimensions, but loaders/folds truncate to `64`.
- [ ] Refactor MTL input handling to support separate category and next dimensions.
  - Current risk: training still uses legacy single `INPUT_DIM`.
- [ ] Normalize category task protocol across engines (avoid check-in vs POI mismatch bias).
  - Current risk: Check2HGI category setup is not directly comparable to DGI/HGI.
- [ ] Implement leakage-safe split module and integrate in pipelines (`train/val/test` artifacts).
  - Current risk: split design exists in docs but not in code.
- [ ] Fix training correctness bugs:
  - `train_f1` logging bug (`0.0`) in Next/Category trainers.
  - optional scheduler called without guard.
  - MTL learning-rate and gradient-accumulation controls not correctly applied.

### Priority 2 - Baseline Availability + Reproducibility Infrastructure

- [ ] Add/adapt missing baseline implementations: `POIRGNN`, `PGC`, `Havana`.
- [ ] Define canonical config and command per model/state.
- [ ] Replace manual commented pipeline lists with matrix-driven execution config.
- [ ] Add deterministic run metadata:
  - split IDs, seeds, runtime, params/FLOPs, commit hash.

### Priority 3 - Baseline Matrix v1 (FL/TX/CA)

- [ ] Run all requested baseline models for Florida.
- [ ] Run all requested baseline models for Texas.
- [ ] Run all requested baseline models for California.
- [ ] Publish v1 comparison table (`macro-F1`, `accuracy`, `std`, runtime, FLOPs).

### Priority 4 - Next Model Improvement Track

- [ ] Apply critical fixes from `src/train/next/UPDATE.md`.
- [ ] Run controlled HPO/autotune.
- [ ] Evaluate architecture variants and lock best configuration.
- [ ] Rerun best Next configuration on FL/TX/CA with frozen split protocol.

### Priority 5 - MTL Optimizer Benchmark Track

- [ ] Complete PCGrad implementation.
- [ ] Add method factory/selector (NashMTL, PCGrad, GradNorm, Naive at minimum).
- [ ] Run fair benchmark with identical data splits, seeds, and budget.
- [ ] Report mean/std and convergence behavior.

### Priority 6 - Final MTLNet (MoE) Track

- [ ] Implement MoE block in MTLNet shared representation.
- [ ] Add gating config and ablation flags (`FiLM`, `MoE`, `FiLM+MoE`).
- [ ] Compare against legacy MTLNet and best non-MoE setup.

### Priority 7 - Final Analysis + Paper Package

- [ ] Freeze final results table and plots.
- [ ] Run final reproducibility rerun (at least one full-state rerun).
- [ ] Prepare methods text for:
  - split protocol,
  - fusion dimensions,
  - optimizer protocol,
  - MoE architecture.
- [ ] Write reproducibility appendix (split IDs, seeds, compute budget, commands).

## Key Risks To Control

- Invalid fusion results due to hidden dimensional truncation.
- Leakage or unfair comparison due to inconsistent split/task protocols.
- Missing baselines causing weak positioning in related work/experiments.
- Incomplete optimizer support blocking MTL claims.
- High regression risk from test debt in core training/ETL.

## Timeline (Execution Order with Dates)

### Phase A - Protocol Hardening (2026-02-06 to 2026-03-01)

- [ ] Finish Priority 1 validity blockers.
- [ ] Exit criteria:
  - no fusion truncation,
  - leakage-safe splits wired,
  - training metrics trustworthy.

### Phase B - Baselines + Infra (2026-03-02 to 2026-03-20)

- [ ] Finish Priority 2.
- [ ] Start Priority 3 runs.
- [ ] Exit criteria:
  - missing baselines integrated,
  - matrix execution is reproducible and automated.

### Phase C - Model Improvement Experiments (2026-03-21 to 2026-04-20)

- [ ] Finish Priority 4 and Priority 5.
- [ ] Start Priority 6 implementation and ablations.
- [ ] Exit criteria:
  - best Next fixed,
  - at least 3 MTL optimization methods compared,
  - first MoE ablation table available.

### Phase D - Final Matrix + Consolidation (2026-04-21 to 2026-05-10)

- [ ] Complete final FL/TX/CA matrix with frozen protocol.
- [ ] Consolidate confidence intervals and convergence analysis.

### Phase E - Writing and Submission (2026-05-11 to 2026-05-26)

- [ ] Finish Priority 7.
- [ ] Internal technical review + final rerun.
- [ ] Submit ENIAC paper by 2026-05-26.

## Definition of Done (Article-Ready)

- [ ] All required baselines completed for FL/TX/CA with reproducible commands.
- [ ] Fusion and split protocols are technically valid and documented.
- [ ] Improved Next result is reproducible and compared to legacy.
- [ ] Final MTLNet MoE is implemented and benchmarked.
- [ ] At least 3 MTL optimizer methods are compared under a fair protocol.
- [ ] Final tables/figures and methods text are ready for submission.
