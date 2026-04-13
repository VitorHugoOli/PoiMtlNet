# MTLnet Improvement Plan v2: Validity-First Staged Upgrade

## Summary

The previous plan was useful as a survey, but it is partially stale and should
not drive the next implementation steps unchanged. The most important
correction is that the current `MTLnet` does not collapse the next-task
sequence before the shared backbone: `next_encoder`, `FiLM`, `shared_layers`,
and `NextHeadMTL` all operate on `[batch, seq, hidden]`. The category head is
also now `CategoryHeadTransformer`, not the older ensemble head.

The priority is:

1. Make experiments scientifically trustworthy and diagnostic.
2. Replace expensive NashMTL only if diagnostics show it is helping.
3. Test a small sequence-aware MoE/CGC architecture against the strongest
   simple baseline.

This plan intentionally avoids jumping directly to broad architecture searches.
The current next task remains next-category classification over 7 classes; a
future true next-POI ranking head should be staged as a separate target change.

## Critical Findings

- Reject the old "sequence collapse" premise. Do not build Phase 2 around
  fixing a nonexistent collapse bug.
- The immediate risks are evaluation validity and training behavior:
  - Real-data artifacts may live outside this worktree.
  - `gradient_accumulation_steps=2` must be honored by the runner.
  - A deployable MTL model needs one joint checkpoint, not two incompatible
    per-task checkpoints.
- NashMTL is theoretically reasonable, but expensive and likely overkill for 2
  tasks unless gradient cosine logs show sustained conflict.
- Literature and community practice both warn that complex MTL optimizers often
  fail to beat tuned simple scalarization.
- LibMTL-style MMoE/CGC/PLE are standard architecture choices, but many
  reference implementations assume single-input or vision/resnet-style models.
  This repo is multi-input and sequence-aware, so the architecture must preserve
  task-specific encoders and sequence dimensions.

## Phase 0: Diagnostic Validity

### Goals

- Establish a trustworthy baseline.
- Make task tradeoffs visible.
- Prevent checkpoint/reporting ambiguity.

### Required Changes

- Implement and expose these MTL losses through the existing loss registry:
  - `equal_weight`: `L = next_loss + category_loss`
  - `static_weight`: category weight grid `{0.25, 0.5, 0.75}`
  - `uncertainty_weighting`
  - `random_weight` / `rlw`
  - `famo`
- Implement or explicitly disable gradient accumulation. Default behavior should
  honor `gradient_accumulation_steps=2` because the config already exposes it.
- Log shared-gradient cosine similarity once per epoch, or at a fixed low-cost
  interval.
- Log per-task shared-gradient norms, loss ratio, loss weights, and wall-clock
  time per fold.
- Track one primary joint checkpoint:
  - `joint_score = 0.5 * next_macro_f1 + 0.5 * category_macro_f1`
  - Store and evaluate the deployable model state selected by this score.
- Keep per-task best epochs as diagnostics only.
- Record Pareto-front membership for analysis, but do not use Pareto selection
  as the first checkpointing policy.

### Required Reports

- Primary metrics from the joint checkpoint.
- Diagnostic per-task best epochs clearly labeled as diagnostic.
- Per-task classification reports from the same joint checkpoint.
- Fold-level diagnostics:
  - `grad_cosine_shared`
  - `grad_norm_next_shared`
  - `grad_norm_category_shared`
  - `loss_ratio_next_to_category`
  - `loss_weight_next`
  - `loss_weight_category`

## Phase 1: Simple Optimizer Baselines

Run these before further CGC tuning:

1. Current baseline: `mtlnet + nash_mtl`
2. `mtlnet + equal_weight`
3. `mtlnet + static_weight` with category weights `{0.25, 0.5, 0.75}`
4. `mtlnet + uncertainty_weighting`
5. `mtlnet + random_weight`
6. `mtlnet + famo`

Promotion rule:

- Start with 1 fold, 3 epochs for diagnostics.
- Promote only the best two simple strategies to 2 folds, 5 epochs.
- Run full 5-fold CV only for the current baseline and the best simple
  optimizer unless there is a clear reason to expand.

Decision rule:

- If shared-gradient cosine is near zero or positive most of the time, prefer
  simple scalarization over gradient-solver methods.
- If one task dominates or cosine is persistently negative, consider stronger
  conflict-aware methods later, such as CAGrad or Aligned-MTL.
- Do not add CAGrad or Aligned-MTL in Phase 1 by default.

## Phase 2: Sequence-Aware CGC-Lite

Only test MMoE-lite and CGC-lite against the strongest simple baseline from
Phase 1.

### Architecture

- Preserve the current category encoder, next encoder, and task heads.
- Replace only FiLM plus the shared residual backbone.
- Start with sequence-aware MMoE-lite as the lower-complexity architecture:
  - `4` shared experts
  - one gate per task
  - no task-specific experts
- Use shared experts plus task-specific experts:
  - `2` shared experts
  - `1` category expert
  - `1` next expert
- Each expert uses the current `ResidualBlock` stack shape.
- Experts must accept both `[B, D]` and `[B, S, D]` tensors.
- Gates should operate over the candidate experts for each task.

### Diagnostics

- Log gate entropy for each task.
- Do not add gate entropy regularization initially.
- Add entropy regularization only if gates collapse and the simple baseline is
  otherwise competitive.

### Decision Rule

CGC-lite is worth keeping only if it improves the joint score, or improves one
task materially without an unacceptable regression in the other task. If it only
adds parameters and shifts performance from next-category to category, it should
not become the default architecture.

## Phase 3: Next-POI Upgrade Path

Do not change the target during Phase 0 or Phase 1.

Current target:

- `next_category`: 7-class classification.

Future target:

- `next_poi`: true POI ranking or classification.

Preparation now:

- Keep config names and interfaces explicit about `next_category` versus
  `next_poi`.
- Avoid hard-coding assumptions that prevent a future POI vocabulary head.
- When the future target changes, evaluate spatio-temporal attention and
  transition-graph methods separately from the current MTL optimizer work.

## Test Plan

### Unit Tests

- Loss registry creates `equal_weight`, `static_weight`,
  `uncertainty_weighting`, and `famo`.
- Each loss returns stable weights and backpropagates on two toy losses.
- CGC-lite forward pass returns `[B, 7]` for both category and next-category
  inputs.
- MMoE-lite forward pass returns `[B, 7]` for both category and next-category
  inputs.
- Expert gates have correct shapes for category `[B, D]` and next `[B, S, D]`.
- `gradient_accumulation_steps=2` produces one optimizer step every 2 batches,
  and scheduler steps only with optimizer steps.
- Joint checkpoint selection stores one deployable model state.

### Integration Tests

- Existing MTL synthetic training still decreases loss.
- Saved summaries identify the primary joint checkpoint separately from
  diagnostic per-task best epochs.
- Smoke run, assuming local data exists:

```bash
PYTHONPATH=src python scripts/train.py --task mtl --state florida --engine dgi --epochs 1 --folds 1
```

### Experiment Protocol

- Run diagnostics first on 1 fold, 3 epochs.
- Promote only the best two loss strategies to 2 folds, 5 epochs.
- Test CGC-lite only against the strongest simple baseline.
- Run full 5-fold CV only for the current baseline, best simple optimizer, and
  best CGC-lite variant.

## Current Evidence

On the initial real-data `dgi + alabama`, 50-epoch, 2-fold run:

- `mtlnet + nash_mtl` slightly outperformed CGC/equal on next-category macro F1.
- `mtlnet_cgc + equal_weight` slightly improved category macro F1 but degraded
  next-category macro F1.
- Gradient cosine values were near zero rather than showing sustained strong
  conflict.

Interpretation: do not tune CGC first. Complete the simple optimizer baseline
stage and make reporting unambiguous before spending more work on architecture.

## Assumptions

- Scientific validity is the main priority.
- Phase 1 keeps the existing 7-class `next_category` target.
- No broad experiment matrix should run until short staged runs justify it.
- Real-data artifacts may live outside this worktree.

## References

- Ma et al., "Modeling Task Relationships in Multi-task Learning with
  Multi-gate Mixture-of-Experts", KDD 2018.
- Tang et al., "Progressive Layered Extraction: A Novel Multi-Task Learning
  Model for Personalized Recommendations", RecSys 2020.
- Navon et al., "Multi-Task Learning as a Bargaining Game", ICML 2022.
- Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning",
  NeurIPS 2021.
- Liu et al., "FAMO: Fast Adaptive Multitask Optimization", NeurIPS 2023.
- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses",
  CVPR 2018.
- Senushkin et al., "Independent Component Alignment for Multi-Task Learning",
  CVPR 2023.
- Yang et al., "GETNext: Trajectory Flow Map Enhanced Transformer for Next POI
  Recommendation", SIGIR 2022.
- Luo et al., "STAN: Spatio-Temporal Attention Network for Next Location
  Recommendation", WWW 2021.
- Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even
  Help?", NeurIPS 2022.
