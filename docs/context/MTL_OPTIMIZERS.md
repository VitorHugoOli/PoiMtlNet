# MTL Loss Weighting & Gradient Balancing Methods

All methods operate on the same interface: given per-task losses
`[L_next, L_category]`, produce either a weighted scalar loss for
standard backpropagation, or directly manipulate gradients on shared
parameters.

The registry (`src/losses/registry.py`) contains 20 canonical methods
+ 3 aliases. Methods are grouped by approach.

---

## Static / Simple Baselines

### Equal Weight
**Key:** `equal_weight` | **Paper:** — | **Type:** Static

`L = L_next + L_category`. No adaptation. The strongest baseline in our
experiments — winner on HGI (joint=0.4855 with CGC s2t2).

---

### Static Weight
**Key:** `static_weight` | **Paper:** — | **Type:** Static

`L = (1 - w) × L_next + w × L_category`. Fixed user-specified weight.
Tested with category weights {0.25, 0.50, 0.75}.

---

### Random Weight (RLW)
**Key:** `random_weight` (alias: `rlw`) | **Type:** Stochastic

Samples weights from Dirichlet(α) each step. Cheap stochastic baseline
to check whether any adaptive method beats random weight noise.

**Reference:** Lin et al., "Reasonable Effectiveness of Random Weighting",
TMLR 2022.

---

## Loss-Based Adaptive Methods

### Dynamic Weight Average (DWA)
**Key:** `dwa` | **Type:** Loss-based

Weights tasks by loss rate of change: `w_i ∝ exp(L_{t-1}(i) / L_{t-2}(i) / T)`.
Tasks whose loss decreased less get higher weight. Temperature T controls
sensitivity. Zero gradient computation overhead.

**Reference:** Liu et al., "End-to-End Multi-Task Learning with Attention",
CVPR 2019. https://arxiv.org/abs/1803.10704

---

### Uncertainty Weighting (UW)
**Key:** `uncertainty_weighting` | **Type:** Learned

Learns a log-variance parameter σ²_i per task:
`L = Σ (L_i / 2σ²_i + log σ_i)`. Tasks with higher uncertainty
(harder to predict) automatically get lower weight. The most-cited
learned task weighting method in the MTL literature.

**Reference:** Kendall et al., "Multi-Task Learning Using Uncertainty
to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.
https://arxiv.org/abs/1705.07115

---

### Soft Optimal Uncertainty Weighting (UW-SO)
**Key:** `uw_so` | **Type:** Learned

Variant of UW using softmax temperature-based inverse-loss weighting.
Temperature parameter controls sharpness.

---

### FAMO (Fast Adaptive Multitask Optimization)
**Key:** `famo` | **Type:** Learned

Learns task weight logits via Adam optimizer, minimizing a per-task loss
normalization objective. Fast (no gradient computation), but
underperformed in our experiments (joint=0.270 vs equal's 0.301).

**Reference:** Liu et al., "FAMO: Fast Adaptive Multitask Optimization",
NeurIPS 2023.

---

## Gradient-Based Methods

### PCGrad (Projecting Conflicting Gradients)
**Key:** `pcgrad` | **Type:** Gradient manipulation

Projects conflicting task gradients: if `g_i · g_j < 0`, subtracts the
conflicting component. Sets `.grad` directly (no weighted loss).

**Reference:** Yu et al., "Gradient Surgery for Multi-Task Learning",
NeurIPS 2020. https://arxiv.org/abs/2001.06782

---

### GradNorm
**Key:** `gradnorm` | **Type:** Gradient manipulation

Balances gradient magnitudes across tasks by learning task weights that
equalize the gradient norms at a shared layer. Uses a reference ratio
based on relative training rates.

**Reference:** Chen et al., "GradNorm: Gradient Normalization for
Adaptive Loss Balancing in Deep Multitask Networks", ICML 2018.
https://arxiv.org/abs/1711.02257

---

### NashMTL
**Key:** `nash_mtl` | **Type:** Gradient solver

Formulates gradient aggregation as a Nash bargaining game. Solves a
convex optimization (via cvxpy/ECOS) to find task weights at a Nash
equilibrium. Expensive per step due to the solver.

Historical note: the NashMTL solver was broken (returning [1,1]) until
2026-04-10 due to a missing cvxpy ECOS dependency hidden by a bare
except. All prior MTL tuning was measured against NaiveLoss-equivalent
behavior.

**Reference:** Navon et al., "Multi-Task Learning as a Bargaining Game",
ICML 2022. https://arxiv.org/abs/2202.01017

---

### CAGrad (Conflict-Averse Gradient Descent)
**Key:** `cagrad` | **Type:** Gradient manipulation

Finds an update direction that maximizes the worst-case local
improvement of any task within a ball of radius `c × ||g_bar||` around
the average gradient. For 2 tasks, the subproblem is a scalar
optimization (cheap). Sets `.grad` directly.

Subsumes GD (c=0) and MGDA (c→∞) as special cases. Hyperparameter `c`
controls conflict-aversion strength.

**Reference:** Liu et al., "Conflict-Averse Gradient Descent for Multi-
task Learning", NeurIPS 2021. https://arxiv.org/abs/2110.14048

---

### Aligned-MTL (Independent Component Alignment)
**Key:** `aligned_mtl` | **Type:** Gradient manipulation

Decomposes the gradient matrix via eigendecomposition. Aligns principal
components to make gradients orthogonal and of equal magnitude. For 2
tasks, the eigendecomposition is on a 2×2 matrix (negligible cost).
No hyperparameters. Sets `.grad` directly.

Provably converges to an optimum with pre-defined task weights, giving
explicit control over the task tradeoff.

**Reference:** Senushkin et al., "Independent Component Alignment for
Multi-Task Learning", CVPR 2023. https://arxiv.org/abs/2305.19000

---

### DB-MTL (Dual-Balancing)
**Key:** `db_mtl` | **Type:** Gradient-based (EMA)

Balances tasks by smoothed gradient/loss signals: computes per-task
gradients of log-losses over shared parameters, applies EMA smoothing
(`beta`, `beta_sigma`), then weights tasks inversely to their gradient
norms. Consistently competitive across engines and architectures.

**Reference:** "Dual-Balancing for Multi-Task Learning", arXiv 2024.

---

### FairGrad
**Key:** `fairgrad` | **Type:** Gradient-based (Gram matrix)

Gradient Gram matrix matching with iterative solver. Seeks task weights
that equalize gradient interactions. Parameter `alpha` controls fairness
curve shape.

**Reference:** "Fair Resource Allocation in Multi-Task Learning",
ICML 2024.

---

### BayesAgg-MTL
**Key:** `bayesagg_mtl` (alias: `bayesagg`) | **Type:** Gradient-based

Bayesian gradient-uncertainty aggregation with diagonal approximation
and EMA tracking. Weights tasks by inverse gradient uncertainty.

**Reference:** "Bayesian Gradient Aggregation for Multi-Task Learning",
ICML 2024.

---

### GO4Align
**Key:** `go4align` | **Type:** Loss + gradient

Risk-aware weighting with task interaction signals. Tracks loss-ratio
risk via EMA and modulates weights using historical correlation between
tasks.

**Reference:** "GO4Align: Group-Oriented Alignment for Multi-Task
Learning", NeurIPS 2024.

---

### ExcessMTL
**Key:** `excess_mtl` (alias: `excessmtl`) | **Type:** Gradient-based

Robust weighting from gradient excess risk. Uses a robust step size to
handle noisy gradient estimates.

**Reference:** "Excess Risk MTL", ICML 2024.

---

### STCH (Smooth Tchebycheff)
**Key:** `stch` | **Type:** Loss-based (scalarization)

Smooth Tchebycheff scalarization with nadir vector normalization and
warmup. Approximates the Tchebycheff (minimax) scalarization with a
smooth surrogate.

**Reference:** "Smooth Tchebycheff Scalarization for Multi-Objective
Optimization", ICML 2024.

---

### Naive
**Key:** `naive` | **Type:** Loss-based

Dynamic alpha/beta weighted sum with clamped adjustment. Legacy method.

---

### Focal Loss
**Key:** `focal` | **Type:** Task-specific (not MTL balancing)

`(1-p_t)^γ × CE` — down-weights easy examples. Used for class imbalance,
not task balancing. Applied per-task as the base criterion, not as the
MTL loss.

**Reference:** Lin et al., "Focal Loss for Dense Object Detection",
ICCV 2017.

---

## Empirical Ranking (DGI, Alabama, 1-fold, 10 epochs)

| Rank | Method | Joint | Type |
|------|--------|-------|------|
| 1 | fairgrad (α=2) | 0.310 | Gradient |
| 2 | db_mtl | 0.303 | Gradient (EMA) |
| 3 | **equal_weight** | **0.301** | Static |
| 4 | excess_mtl | 0.301 | Gradient |
| 5 | stch | 0.301 | Loss scalarization |
| 6 | nash_mtl | 0.300 | Gradient solver |
| 7 | uncertainty_weighting | 0.299 | Learned |
| 8 | go4align | 0.291 | Loss + gradient |
| 9 | famo | 0.270 | Learned |
| 10 | bayesagg_mtl | 0.268 | Gradient |

No adaptive method significantly outperforms equal weighting.

---

## References (consolidated)

1. Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018.
2. Yu et al., "Gradient Surgery for Multi-Task Learning" (PCGrad), NeurIPS 2020.
3. Chen et al., "GradNorm", ICML 2018.
4. Liu et al., "Conflict-Averse Gradient Descent" (CAGrad), NeurIPS 2021.
5. Navon et al., "Multi-Task Learning as a Bargaining Game" (NashMTL), ICML 2022.
6. Senushkin et al., "Independent Component Alignment" (Aligned-MTL), CVPR 2023.
7. Liu et al., "FAMO: Fast Adaptive Multitask Optimization", NeurIPS 2023.
8. Liu et al., "End-to-End Multi-Task Learning with Attention" (DWA), CVPR 2019.
9. Lin et al., "Reasonable Effectiveness of Random Weighting" (RLW), TMLR 2022.
10. Xin et al., "Do Current Multi-Task Optimization Methods Even Help?", NeurIPS 2022.
