# Future Work — Substrate-adaptive MTL loss balancing (F2)

**Date drafted:** 2026-05-20
**Source study:** [`docs/studies/canonical_improvement/`](../studies/canonical_improvement/) Tier 6 CORRECTION (2026-05-19) + [`docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`](../studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md) §F2
**Sequencing:** deferred until after `mtl-protocol-fix` lands; pre-requisite for `mtl_architecture_revisit.md`'s loss-balancing axis.

## What's deferred

A rigorous, leak-free re-evaluation of MTL loss-balancing methods that were previously demoted or never run cleanly under per-fold log_T:

| Method | Current status | Why revisit |
|---|---|---|
| **NashMTL** | Demoted to `static_weight(w_cat=0.75)` due to cvxpy/ECOS solver instability | Solver instability was small-state-specific; FL is well-conditioned. May stabilise reg through ep 50. |
| **GradNorm** | Scaffolded in `src/losses/gradnorm.py`; never multi-seed under per-fold log_T | Adaptive gradient-magnitude balance may prevent reg destabilisation. |
| **PCGrad** | Scaffolded in `src/losses/pcgrad.py`; never multi-seed under per-fold log_T | Project conflicting task gradients; tackles destructive interference directly. |
| **FAMO** | Launcher exists (F50 T1.3); never under per-fold seed-tagged log_T | Adaptive loss balancing; lightweight. |
| **Aligned-MTL** | Launcher exists (F50 T1.4); never under per-fold seed-tagged log_T | Principal-component-aligned gradients; targets reg-cat conflict directly. |
| **Per-task LR scheduling** | Not implemented | Drop reg_lr aggressively after ep ~10 once reg peaks; keep cat_lr cosine to ep 50. |
| **Gradient masking** | Not implemented | Detect reg-head plateau via val loss; freeze shared-backbone reg-side gradients afterward. |

## Why deferred

The `mtl-protocol-fix` study's F1 selector fix may already extract the reg capacity at a single checkpoint. If it does, F2 work (loss balancing) is reduced from "must close 10-pp gap" to "polish the single checkpoint." The scope and EV depend on what F1 reveals.

## Acceptance criterion

When picked up:

1. **Pre-flight gate** — the residual reg gap that survives F1 fix is quantified at every state (AL/AZ/FL/CA/TX). If <2 pp residual: F2 is low-EV polish; if >2 pp residual: F2 is load-bearing.
2. **Methodology** — every method tested under per-fold seed-tagged log_T (no pre-2026-05-15 leak); 5 seeds × 5 folds minimum at FL; AL/AZ for cross-state.
3. **Three-frontier reporting** — every variant reports MTL @ best joint, MTL @ best disjoint, STL ceiling.
4. **Promotion bar** — paired Wilcoxon p ≤ 0.05 on reg at FL pooled n=20, no state Δ_reg ≤ −0.5 pp, no Δ_cat ≤ −1.0 pp at any state.

## Cost (estimated)

- NashMTL FL multi-seed: ~5-6 GPU-h.
- GradNorm/PCGrad/FAMO/Aligned-MTL × FL multi-seed: ~4-5 GPU-h each = ~20 GPU-h total.
- Per-task LR / gradient-masking: ~3-4 GPU-h each.
- **Total estimated: 30-40 GPU-h** for a full methodical sweep.

## Live docs the work would touch

- `src/losses/` — implementation updates / new flags
- `src/training/runners/mtl_cv.py` — loss-balancer integration
- `docs/results/RESULTS_TABLE.md` — new section for loss-balancer comparison
- `docs/CHANGELOG.md` — timeline entry

## Pointers

- Predecessor analysis: [`docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`](../studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md) §F2
- Existing scaffolded losses: `src/losses/{nash_mtl.py, gradnorm.py, pcgrad.py, focal.py, naive.py}`
- User's MTL-loss observation: [`docs/studies/mtl-exploration/considerations.md`](../studies/mtl-exploration/considerations.md) ("static loss weight are not the most simple")
- Leak-fix point that gates this work: `src/training/runners/mtl_cv.py` (n_splits guard, 2026-05-15)
