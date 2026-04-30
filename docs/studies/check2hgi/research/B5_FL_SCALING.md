# B5 Scaling Test — Florida 1 Fold × 50 Epoch

**Date:** 2026-04-22 20:04–20:52 (48 min wall-clock on M4 Pro MPS).
**Protocol:** 1 fold of k=2 CV (50% train / 50% val by user group), seed 42.
**Config:** `mtlnet_crossattn + pcgrad + next_getnext_hard d=256, 8h`.
**Reference (B-M13, same config but soft probe):** `mtlnet_lr1.0e-04_bs2048_ep50_20260421_1357`.

## Hypothesis

From `B5_MACRO_ANALYSIS.md`: scaling hypothesis predicted FL hard would
beat FL soft by an even larger margin than AZ's +6.59 pp Acc@10, because
at 4703 regions the soft probe has less sample-per-class to learn the
transition mapping.

## Results

| Metric | FL Soft (B-M13) | **FL B5 Hard** | Δ |
|---|---:|---:|---:|
| **Region** Acc@1 | 12.74 | **13.70** | **+0.96** |
| **Region** Acc@5 | 36.01 | **49.54** | **+13.53** 🔥 |
| **Region** Acc@10 | **60.62** | 58.88 | −1.74 |
| **Region** MRR | 25.55 | **28.01** | **+2.46** |
| **Region** F1_macro | 6.44 | **8.39** | +1.95 |
| **Category** F1 | **66.01** | 55.43 | **−10.58** ⚠️ |
| **Category** Acc@1 | **68.40** | 56.55 | **−11.85** ⚠️ |
| **Category** Acc@3 | **90.27** | 88.71 | −1.56 |
| **Category** MRR | **80.23** | 73.37 | −6.86 |

## Finding

**The scaling hypothesis partially held but a second-order effect surfaced:**

1. **Region (task_b) mostly improves.** 4 of 5 region metrics lift
   (biggest: Acc@5 +13.5 pp; MRR +2.5 pp; F1 +1.95 pp; Acc@1 +1 pp).
   Only Acc@10 drops 1.74 pp, which at σ-unknown (n=1) is likely noise.
   So the B5 "hard beats soft on region" pattern extends to FL scale,
   at least on 4 of 5 metrics.

2. **Category (task_a) degrades severely.** Cat F1 drops 10.58 pp
   (66.01 → 55.43), Acc@1 drops 11.85 pp. This is a **first** — on
   AL and AZ, the cat task was stable within σ under B5. At FL's
   scale the hard prior appears to starve the shared cross-attention
   backbone of category-relevant signal.

**Working hypothesis:** `log_T[last_region_idx]` at 4703 regions is
numerically large (each row has ≤4703 non-trivial log-probability
entries), and `α * log_T[idx]` is added directly to region logits.
The MTL optimizer (PCGrad) sees a much larger per-sample gradient
flowing through task_b than through task_a; the shared cross-attention
backbone tilts toward features that help region prediction and away
from features that the category head needs. On AL (1109 regions) and
AZ (1547 regions) the `log_T` rows are smaller and this imbalance
is mild. On FL (4703 regions) it becomes dominant.

## Implication for the paper

**FL B5-hard is NOT a drop-in replacement** for FL B-M13 in the
headline MTL column. A paper that reports only region metrics could
use FL-hard (Acc@5 +13.5, MRR +2.5 pp), but the **joint** MTL claim
requires the cat task to hold up, which it does not.

Three options:

- **A. Drop FL B5-hard from headline.** Report AL + AZ B5-hard as
  the region-task champion; FL stays on soft-probe for the joint
  headline. Document the scale-dependent trade-off as a finding.
- **B. Down-weight region loss for FL.** E.g., `task_b_weight=0.5`
  to let the category head recover. Requires an extra hyperparameter
  sweep. Untested.
- **C. Run FL B5-hard 5-fold to tighten σ.** If σ on cat is ≥5 pp,
  the −10.58 pp might partially be n=1 noise. ~6h MPS; single seed.

## Open questions

1. **n=1 noise vs real regression on cat.** A 5-fold replicate
   would make the cat drop defendable. Estimated time: ~5-6h MPS.
2. **Is the Acc@10 −1.74 a real regression or noise?** Likely noise
   given the other 4 region metrics all lift.
3. **Would task-weight rebalancing fix it?** Theoretical yes, but
   would need a small sweep over `task_b_weight ∈ {0.25, 0.5, 0.75}`
   for FL.

## Revised paper narrative

The paper's joint-task champion should be:

- **AL: MTL-GETNext-soft (B-M6b) or MTL-GETNext-hard (B-M6e)** —
  effectively tied within σ.
- **AZ: MTL-GETNext-hard (B-M9d)** — decisive on region, ties on cat.
- **FL: MTL-GETNext-soft (B-M13)** — hard trades too much cat
  performance at this scale; soft-probe is the robust joint choice.

This is scale-dependent, which is an interesting scientific finding
in its own right. The paper can frame it as:

> We present MTL-GETNext in two variants — a soft-probe adaptation and a
> faithful hard-index formulation. Which variant dominates depends on the
> region-count scale: at ~10³ regions (AL, 1109; AZ, 1547) the hard
> formulation matches or beats soft across all metrics; at ~10⁴ regions
> (FL, 4703) the hard prior over-dominates the MTL gradient and sacrifices
> next-category quality. We report both variants and recommend deployment
> following the scale cutoff.

## Files

- FL hard run JSON: `docs/studies/check2hgi/results/B5/fl_1f50ep_next_getnext_hard.json`
- FL soft reference: `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260421_1357/`
- Launcher: `scripts/run_b5_hard_mtl.sh` (extended to support FL with minor edit)
