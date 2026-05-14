# Phase P5 — MTL architecture × MTL optimiser grid (ported from legacy P1)

**Gates:** P3 + P4 complete.
**Purpose:** Replicate the legacy study's architecture × optimiser grid on the new check2HGI track so we have apples-to-apples MTL-design evidence for the paper.

## Design

Same factors as legacy P1 but applied to the check2HGI `{next_category, next_region}` task pair instead of the legacy `{category, next}` pair.

### Architectures (5)

| Short | Model | Notes |
|-------|-------|-------|
| `base` | `mtlnet` | FiLM + shared residual backbone (current default) |
| `cgc22` | `mtlnet_cgc` | 2 shared experts / 2 task experts |
| `cgc21` | `mtlnet_cgc` | 2 shared / 1 task |
| `mmoe4` | `mtlnet_mmoe` | 4 experts, MMoE gating |
| `dsk42` | `mtlnet_dselectk` | 4 experts, 2 selectors, temp=0.5 |

**Prerequisite:** `MTLnetCGC`, `MTLnetMMoE`, `MTLnetDSelectK`, `MTLnetPLE` must be parameterised with `task_set` the same way `MTLnet` was (P1-b). These variants currently hardcode `category_x, next_x` in their `_mix` internals — they need a sequential-task-A path. Port is mechanical but ~150 LOC per variant.

### MTL optimisers (target: 20; priority 6)

| Category | Priority-1 (run first) | Priority-2 (full grid) |
|----------|------------------------|------------------------|
| Static   | `equal_weight`, `static_weight` | `uncertainty_weighting` |
| Loss-based | `nash_mtl` (default), `dwa` | `famo`, `bayesagg_mtl`, `excess_mtl`, `stch` |
| Gradient-based | `cagrad`, `aligned_mtl` | `pcgrad`, `gradnorm`, `db_mtl`, `fairgrad`, `graddrop`, `mgda`, `moco`, `sdmgrad`, `imtl_h`, `imtl_l` |

Start with the 6 priority-1 optimisers; extend to full 20 once those stabilise.

### States

Alabama (primary), Arizona (replication — supports cross-state generalisation check).

### Budget

| Stage | Runs per state | Config | Est. time/run | Total/state |
|-------|---------------|--------|---------------|-------------|
| P5a — screen | 5 × 20 = 100 | 1 fold, 10 epochs | ~1 min | ~100 min |
| P5b — promote top-10 | 10 | 2 folds, 15 epochs | ~3 min | ~30 min |
| P5c — confirm top-5 | 5 | 5 folds, 50 epochs | ~22 min | ~110 min |
| **Total** | ~115 runs | | | **~4 h / state** |

Matches legacy P1 budget. AL + AZ ≈ 8h sequential.

### Effective batch size

**Decision:** all candidates at `gradient_accumulation_steps=1, batch_size=4096`. Matches legacy P1 exactly.

### Seed

Seed 42 for screen + promotion; multi-seed (123, 2024) for top-3 confirmation only.

## Test IDs

`P5_<state>_<stage>_<arch>_<optim>_<seed>` — e.g. `P5_AL_screen_dsk42_al_seed42`.

## Claims touched

Matches legacy P1's claim mapping but relocated to check2HGI:

- **CH14** (new, check2HGI mirror of legacy C02) — On check2HGI `{next_category, next_region}`, gradient-surgery optimisers (cagrad, aligned_mtl) beat equal-weight by ≥ 2 p.p. joint_acc1 at matched batch.
- **CH15** (legacy C05 mirror) — Expert-gating archs (cgc, mmoe, dsk) beat base `mtlnet` with FiLM-only by ≥ 0.05 joint_acc1 averaged across optimisers.
- **CH16** (legacy C04 mirror) — Winning (arch, optim) pair on check2HGI differs from winning pair on HGI from the legacy track.

Add these to `CLAIMS_AND_HYPOTHESES.md` after this phase doc lands.

## Decision gate → P6

Proceed to P6 only if:
1. P5c has a single clear winner OR a tie of 2–3 within seed variance.
2. The winner has a sensible profile — `val_f1_next_category ≥ 0.20` AND `val_acc1_next_region ≥ 2 × majority-class`.
3. AL is complete; AZ nice-to-have but can lag by 1 day.

If `equal_weight` wins on check2HGI: major paper implication (Xin et al. "adaptive doesn't help" replicates beyond single-source) — pause and re-plan before P6.

## Surprises to watch

| Symptom | Interpretation |
|---------|----------------|
| `uncertainty_weighting` top-5 | Check2HGI's scale imbalance may not require gradient surgery |
| `base` + `cagrad` beats `cgc` + `eq` | Optimiser > architecture effect on this data |
| AL winner ≠ AZ winner | Dataset-dependent; flag for single-state caveat |
| FL results diverge from AL/AZ (when we run P5 on FL) | Region-cardinality effect; document as limitation |

## Outputs

- `docs/studies/check2hgi/results/P5/` — ~230 test dirs (115 × 2 states)
- Heatmaps per state: `results/P5/heatmap_AL.png`, `heatmap_AZ.png`
- Summary: `results/P5/SUMMARY.md` — champion (arch, optim) pair + justification
- Claim statuses for CH14, CH15, CH16

## Differences vs legacy P1 worth noting in the paper

1. **Label space:** legacy P1 used {category (7 classes), next (7 classes)}. P5 uses {next_category (7), next_region (~1109 AL / ~4703 FL)}. The gradient-scale imbalance is much larger on check2HGI.
2. **Input granularity:** legacy used POI-level embeddings (fusion, hgi). P5 uses check-in-level embeddings. Encoder receives richer per-sample context but less stable POI-level structure.
3. **Primary metric:** legacy used macro-F1 for both heads. P5 uses macro-F1 (next_category) + Acc@1 (next_region), and joint = mean(val_f1_next_category, val_acc1_next_region).

These differences are why P5 is a separate phase and not a trivial rerun of P1.
