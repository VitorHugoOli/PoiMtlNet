# New MTL Candidate Analysis (2026-04-13)

## Context

This analysis identifies new MTL optimizers and architectures for the MTLnet
framework, based on a comprehensive literature review cross-referenced against
the LibMTL benchmark library, the Awesome-Multi-Objective-Deep-Learning list,
and recent publications at NeurIPS 2023-2024, ICML 2024, CVPR 2023, and ICLR
2025.

### What We Already Have

**Optimizers (17):** equal_weight, static_weight, uncertainty_weighting, uw_so,
random_weight, famo, fairgrad, bayesagg_mtl, go4align, excess_mtl, stch,
db_mtl, nash_mtl, pcgrad, gradnorm, focal, naive.

**Architectures (4):** MTLnet (FiLM + shared residual), MTLnetCGC, MTLnetMMoE,
MTLnetDSelectK.

### Current Best Results (Alabama, 2-fold, 15-epoch promoted)

- **HGI:** CGC s2t2 + equal_weight (joint 0.4855)
- **DGI:** DSelectK e4k2 + db_mtl (joint 0.3337)

### Key Observations From Ablation

1. Gradient cosine between tasks is near zero -- minimal sustained conflict.
2. Simple weighting (equal_weight) beat complex solvers on HGI.
3. db_mtl is consistently competitive across engines/architectures.
4. Engine choice (DGI vs HGI) changes the winner more than arch/loss choice.

---

## Selected New Candidates

### OPTIMIZER 1: CAGrad (Conflict-Averse Gradient Descent)

**Paper:** Liu et al., "Conflict-Averse Gradient Descent for Multi-task
Learning", NeurIPS 2021.
**Code:** https://github.com/Cranial-XIX/CAGrad
**LibMTL:** Supported (verified).

**Algorithm:** Finds an update direction that maximizes the worst-case local
improvement of any task within a neighborhood of the average gradient. For 2
tasks, the subproblem has a closed-form solution via the Gram matrix of
per-task gradients. Uses scipy.minimize on a (n_tasks-1)-dimensional
simplex, which for n_tasks=2 is a scalar optimization.

**Core computation:**
1. Compute per-task gradients g_1, g_2 over shared parameters.
2. Build Gram matrix G = [g_i . g_j].
3. Solve: maximize min_i (g_bar + lambda * sum_j w_j g_j) . g_i,
   subject to ||sum_j w_j g_j|| <= c * ||g_bar||.
4. Return g_bar + lambda * w* as the combined gradient.

**Why it fits:**
- The plan explicitly names CAGrad as a candidate for Phase 1 if gradient
  cosine shows conflict (Section "Decision rule").
- Even with near-zero cosine, CAGrad is a principled middle ground between
  equal_weight (no gradient manipulation) and NashMTL (expensive solver).
- For 2 tasks the computational overhead is negligible -- single scalar
  optimization per step.
- Provably converges to a minimum over the average loss.
- Subsumes both GD (c=0) and MGDA (c->inf) as special cases.

**Hyperparameters:** `c` (conflict-aversion radius, default 0.4), `rescale`
mode (0/1/2).

**Complexity vs existing:** Cheaper than NashMTL (no cvxpy/ECOS). Comparable
to PCGrad but with stronger theoretical guarantees and the c parameter gives
controllable task balance.

---

### OPTIMIZER 2: Aligned-MTL (Independent Component Alignment)

**Paper:** Senushkin et al., "Independent Component Alignment for Multi-Task
Learning", CVPR 2023.
**Code:** https://github.com/SamsungLabs/MTL (LibMTL integration available).
**LibMTL:** Supported (verified).

**Algorithm:** Decomposes the gradient matrix via eigendecomposition to align
principal components, making aligned gradients orthogonal and of equal
magnitude. Then sums with pre-defined task weights.

**Core computation:**
1. Compute per-task gradients G = [g_1; g_2] (2 x D matrix).
2. Compute Gram matrix M = G G^T (2x2 for our case).
3. Eigendecompose M: eigenvalues lambda, eigenvectors V.
4. Filter by condition number tolerance.
5. Compute alignment weights alpha = (V diag(1/sqrt(lambda)) * sqrt(lambda_min) V^T) * ones.
6. Return sum_i alpha_i * g_i.

**Why it fits:**
- The plan mentions Aligned-MTL as a candidate alongside CAGrad.
- For 2 tasks, the eigendecomposition is on a 2x2 matrix -- essentially free.
- Unlike NashMTL, it guarantees convergence to an optimum with **pre-defined
  task weights**, giving explicit control over the task tradeoff.
- Addresses both gradient conflict AND gradient dominance simultaneously
  via condition number control.
- No external solvers needed (pure PyTorch).

**Hyperparameters:** None required (the alignment is fully determined by
the gradient geometry). Optional: pre-defined task weights for asymmetric
objectives.

**Complexity vs existing:** Cheaper than NashMTL, comparable to PCGrad.
The 2x2 eigendecomposition adds negligible overhead.

---

### OPTIMIZER 3: DWA (Dynamic Weight Average)

**Paper:** Liu et al., "End-to-End Multi-Task Learning with Attention",
CVPR 2019.
**LibMTL:** Supported (verified).

**Algorithm:** Adjusts task weights based on the rate of change of each
task's loss over consecutive epochs. Tasks whose loss decreased less get
higher weight in the next epoch.

**Core computation:**
1. Track loss history: L_t(i) for each task i at epoch t.
2. Compute relative change: r_t(i) = L_t(i) / L_{t-1}(i).
3. Compute weights: w_t(i) = n_tasks * exp(r_t(i) / T) / sum_j exp(r_t(j) / T).
4. Return sum_i w_t(i) * L_i.

**Why it fits:**
- Extremely simple, zero gradient computation overhead.
- Good sanity-check baseline between static weighting and gradient-based
  methods. If DWA matches complex gradient solvers, gradient manipulation
  isn't helping.
- Temperature T controls sensitivity: high T -> equal weighting, low T ->
  aggressive rebalancing.
- No per-step gradient computation needed -- only uses loss values.
- The ablation showed equal_weight won on HGI. DWA is the natural next
  step: adaptive but still loss-based, not gradient-based.

**Hyperparameters:** `temperature` (default 2.0).

**Complexity vs existing:** Cheapest possible adaptive method. No gradient
computation, no solver. Just loss ratio + softmax.

---

### ARCHITECTURE 1: PLE (Progressive Layered Extraction)

**Paper:** Tang et al., "Progressive Layered Extraction (PLE): A Novel
Multi-Task Learning Model for Personalized Recommendations", RecSys 2020.
**LibMTL:** Supported (verified).

**Architecture:** Stacks multiple CGC layers progressively. Each level's
experts receive the gated outputs from the previous level, enabling
progressive refinement of shared and task-specific representations.

**Why it fits:**
- CGC s2t2 is already the best architecture on HGI (joint 0.4855). PLE
  is the natural multi-level extension.
- Our CGCLiteLayer already implements a single CGC level. PLE stacks
  2-3 of these with inter-level gating.
- The existing `_build_shared_backbone` hook in MTLnet makes this a
  clean subclass -- override with a stacked CGC block.
- Experts already handle both [B,D] and [B,S,D] tensors.
- PLE explicitly addresses the "seesaw phenomenon" where improving one
  task degrades another -- directly relevant to our next vs category
  tradeoff.

**Implementation plan:**
1. Add `PLELiteLayer` to `_components.py`: stack N CGCLiteLayers where
   each level's input is the gated output of the previous level.
2. Add `MTLnetPLE` to `src/models/mtl/mtlnet_ple/model.py` following
   the CGC pattern.
3. Parameters: `num_levels` (default 2), plus CGC params per level.

**Complexity vs existing:** ~2x parameters of CGC s2t2 for 2 levels.
The multi-level gating adds minimal compute -- the experts dominate.

---

## Candidates Considered and Rejected

### SDMGrad (NeurIPS 2023)
Direction-oriented multi-objective gradient. **Rejected:** Requires 3
separate gradient computations per step (double sampling for unbiased
estimates). For our small 2-task setup, the theoretical benefits don't
justify the 3x gradient cost. CAGrad provides similar conflict-avoidance
benefits with a single gradient computation.

### MoCo (ICLR 2023)
Multi-objective collaborative learning with EMA gradient estimates +
learned quadratic weights. **Rejected:** Conceptually similar to DB-MTL
(which we already have) -- both use EMA-smoothed gradients and learned
task weights. Adding MoCo would create redundancy without clear
differentiation in the ablation.

### MGDA (NeurIPS 2018)
Multiple gradient descent algorithm. **Rejected:** Superseded by CAGrad
(which includes MGDA as a special case with c->inf). MGDA also has known
issues with imbalanced solutions where one task dominates.

### GradDrop (NeurIPS 2020)
Random sign-based gradient dropout. **Rejected:** Stochastic and hard to
diagnose. Not shown to consistently outperform PCGrad or CAGrad in
benchmarks.

### GradVac (ICLR 2021)
Gradient vaccine with magnitude matching. **Rejected:** Similar to PCGrad
but adds EMA-based cosine similarity tracking. Already have PCGrad;
incremental benefit unclear for 2 tasks.

### IMTL (ICLR 2021)
Impartial multi-task learning. **Rejected:** Enforces equal projection of
aggregated gradient onto each task gradient. Subsumed theoretically by
Aligned-MTL which provides a more general condition-number-based approach.

### AutoLambda (TMLR 2022)
Meta-learning task weights using validation set. **Rejected:** Requires
validation loss at every training step for the meta-update. Expensive and
complicates the training loop significantly. Not compatible with gradient
accumulation.

### HoME (Kuaishou 2024)
Hierarchical mixture of experts with 3 expert classes. **Rejected:**
Designed for 6+ tasks in industrial recommendation. For 2 tasks, the
hierarchy (global -> group -> task) collapses to CGC/PLE. Adds complexity
without benefit.

### Cross-stitch Networks (CVPR 2016)
Soft parameter sharing between task columns. **Rejected:** Conceptually
interesting but old and shown to underperform MMoE/CGC in modern
benchmarks. Also designed for symmetric architectures where both tasks
have the same input shape, which doesn't match our setup.

### MTAN (CVPR 2019)
Multi-task attention network. **Rejected:** Designed for dense prediction
(semantic segmentation, depth estimation). Attention mechanisms are
pixel-level, not transferable to our embedding-based classification setup.

---

## Implementation Priority

| Priority | Name | Type | Rationale |
|----------|------|------|-----------|
| 1 | CAGrad | Optimizer | Principled, cheap, plan-referenced |
| 2 | Aligned-MTL | Optimizer | Complementary to CAGrad, no hyperparams |
| 3 | DWA | Optimizer | Cheapest adaptive baseline, good diagnostic |
| 4 | PLE | Architecture | Natural extension of winning CGC |

## Suggested Ablation Candidates

After implementation, add to the ablation matrix:

```
# New optimizer candidates (on base mtlnet)
cagrad            -- mtlnet + cagrad (c=0.4)
cagrad_c02        -- mtlnet + cagrad (c=0.2, less conflict-averse)
aligned_mtl       -- mtlnet + aligned_mtl
dwa               -- mtlnet + dwa (T=2.0)
dwa_t1            -- mtlnet + dwa (T=1.0, more aggressive)

# New optimizer candidates (on best architectures)
arch_cgc_s2t2_cagrad       -- CGC(s=2,t=2) + cagrad
arch_cgc_s2t2_aligned_mtl  -- CGC(s=2,t=2) + aligned_mtl
arch_dselectk_cagrad       -- DSelectK(e=4,k=2) + cagrad

# New architecture candidates
arch_ple_l2_equal          -- PLE(levels=2, s=2, t=2) + equal_weight
arch_ple_l2_db_mtl         -- PLE(levels=2, s=2, t=2) + db_mtl
arch_ple_l2_cagrad         -- PLE(levels=2, s=2, t=2) + cagrad
arch_ple_l3_equal          -- PLE(levels=3, s=2, t=1) + equal_weight
```

## References

1. Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning",
   NeurIPS 2021. https://arxiv.org/abs/2110.14048
2. Senushkin et al., "Independent Component Alignment for Multi-Task
   Learning", CVPR 2023. https://arxiv.org/abs/2305.19000
3. Liu et al., "End-to-End Multi-Task Learning with Attention", CVPR 2019.
   https://arxiv.org/abs/1803.10704
4. Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
   Learning Model for Personalized Recommendations", RecSys 2020.
   https://dl.acm.org/doi/10.1145/3383313.3412236
5. Lin et al., "LibMTL: A Python Library for Multi-Task Learning", JMLR
   2023. https://github.com/median-research-group/LibMTL
6. Lin et al., "Awesome-Multi-Objective-Deep-Learning", 2025.
   https://github.com/Baijiong-Lin/Awesome-Multi-Objective-Deep-Learning
7. Xiao et al., "Direction-oriented Multi-objective Learning: Simple and
   Provable Stochastic Algorithms", NeurIPS 2023 (SDMGrad -- rejected).
8. Chen et al., "Three-Way Trade-Off in Multi-Objective Learning", NeurIPS
   2023 (MoDo -- rejected).
9. Fernando et al., "Mitigating Gradient Bias in Multi-objective Learning",
   ICLR 2023 (MoCo -- rejected).
10. Navon et al., "Multi-Task Learning as a Bargaining Game", ICML 2022
    (Nash-MTL -- already implemented).
11. Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning
    Even Help?", NeurIPS 2022 (motivates simple baselines).
