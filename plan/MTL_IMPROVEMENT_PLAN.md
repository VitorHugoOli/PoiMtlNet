# MTLnet Improvement Plan: Architecture & Optimization

> **Goal**: Improve the multi-task learning model for joint POI category classification and next-POI prediction by upgrading both the architecture (how tasks share representations) and the optimization strategy (how task losses are balanced).

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Identified Bottlenecks & Weaknesses](#2-identified-bottlenecks--weaknesses)
3. [Architecture Improvements](#3-architecture-improvements)
   - 3.1 [MMoE (Multi-gate Mixture-of-Experts)](#31-mmoe-multi-gate-mixture-of-experts)
   - 3.2 [PLE (Progressive Layered Extraction)](#32-ple-progressive-layered-extraction)
   - 3.3 [CGC (Customized Gate Control)](#33-cgc-customized-gate-control)
   - 3.4 [Cross-Stitch Networks](#34-cross-stitch-networks)
   - 3.5 [MTAN (Multi-Task Attention Network)](#35-mtan-multi-task-attention-network)
   - 3.6 [DSelect-k](#36-dselect-k)
4. [Architecture Comparison Matrix](#4-architecture-comparison-matrix)
5. [Optimization Strategy Improvements](#5-optimization-strategy-improvements)
   - 5.1 [Current NashMTL Analysis](#51-current-nashmtl-analysis)
   - 5.2 [FAMO (Fast Adaptive Multitask Optimization)](#52-famo-fast-adaptive-multitask-optimization)
   - 5.3 [CAGrad (Conflict-Averse Gradient Descent)](#53-cagrad-conflict-averse-gradient-descent)
   - 5.4 [Aligned-MTL](#54-aligned-mtl)
   - 5.5 [MoCo (Mitigating Gradient Bias)](#55-moco-mitigating-gradient-bias)
   - 5.6 [DB-MTL (Dual-Balancing Multi-Task Learning)](#56-db-mtl-dual-balancing-multi-task-learning)
   - 5.7 [IMTL (Impartial Multi-Task Learning)](#57-imtl-impartial-multi-task-learning)
   - 5.8 [Uncertainty Weighting (UW)](#58-uncertainty-weighting-uw)
   - 5.9 [Random Loss Weighting (RLW)](#59-random-loss-weighting-rlw)
6. [Optimizer Comparison Matrix](#6-optimizer-comparison-matrix)
7. [Critical Insight: Do MTL Optimizers Even Help?](#7-critical-insight-do-mtl-optimizers-even-help)
8. [Recommendations & Implementation Roadmap](#8-recommendations--implementation-roadmap)
9. [Experiment Plan](#9-experiment-plan)
10. [References](#10-references)

---

## 1. Current Architecture Analysis

### 1.1 Model Topology

The current `MTLnet` follows a **hard parameter sharing** architecture with **FiLM modulation**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CURRENT MTLnet                               │
│                                                                     │
│  Category Input [B,1,64]     Next Input [B,9,64]                    │
│        │                          │                                 │
│  ┌─────▼─────┐            ┌──────▼──────┐                           │
│  │ Category   │            │   Next      │    Task-Specific          │
│  │ Encoder    │            │  Encoder    │    Encoders (MLP)         │
│  │ (2-layer)  │            │ (2-layer)   │                           │
│  └─────┬─────┘            └──────┬──────┘                           │
│        │ [B,256]                 │ [B,256]                          │
│        │                         │                                  │
│  ┌─────▼─────┐            ┌──────▼──────┐                           │
│  │ FiLM      │            │   FiLM      │    Task-Conditioned       │
│  │ (task=0)  │            │  (task=1)   │    Modulation             │
│  └─────┬─────┘            └──────┬──────┘                           │
│        │                         │                                  │
│  ┌─────▼─────────────────────────▼──────┐                           │
│  │      Shared Residual Backbone        │    Shared Processing      │
│  │      (4 ResidualBlocks)              │    (Same weights)         │
│  └─────┬─────────────────────────┬──────┘                           │
│        │                         │                                  │
│  ┌─────▼─────┐            ┌──────▼──────┐                           │
│  │ Category  │            │   Next      │    Task-Specific          │
│  │ Ensemble  │            │ Transformer │    Heads                  │
│  │ Head      │            │   Head      │                           │
│  └─────┬─────┘            └──────┬──────┘                           │
│        │ [B,7]                   │ [B,7]                            │
└────────┴─────────────────────────┴──────────────────────────────────┘
```

### 1.2 Key Observations from Code Analysis

| Component | Details |
|-----------|---------|
| **Feature size** | 64 (embedding dim) |
| **Shared layer size** | 256 |
| **Encoder** | 2-layer MLP per task: `64 → 256 → 256` |
| **FiLM** | `γ(task_emb) * x + β(task_emb)` — lightweight affine modulation |
| **Shared backbone** | 4 ResidualBlocks, each with 2 FC sublayers + LayerNorm + LeakyReLU |
| **Category head** | 3-path ensemble (depths 2,3,4) → concat → combiner → 7 classes |
| **Next head** | Sinusoidal PE → Transformer encoder (4 layers, 8 heads) → attention pooling → 7 classes |
| **Dropout** | Encoder: 0.1, Shared: 0.15, Cat head: 0.5, Next head: 0.35 |

### 1.3 How Task Data Flows

**Critical asymmetry between tasks**:
- **Category task**: Receives a single embedding `[B, 1, 64]`, encoder reduces to `[B, 256]` → **no sequence dimension enters shared layers**
- **Next task**: Receives a window `[B, 9, 64]`, encoder reduces each step independently to `[B, 256]` → **sequence dimension is lost before shared layers**

This means:
1. The shared backbone processes **1D vectors** `[B, 256]` for both tasks
2. The Next task's temporal information is **already collapsed** before reaching shared layers
3. The Next head's Transformer operates on the shared output, not on sequential embeddings

**This is a significant limitation**: the shared layers never see temporal patterns. The Next task's sequence structure is destroyed by the encoder before any shared processing happens.

### 1.4 Parameter Distribution

```
Task-Specific:
  - category_encoder:  ~82K params (64→256→256)
  - next_encoder:      ~82K params (64→256→256)
  - category_poi head: ~131K params (3-path ensemble)
  - next_poi head:     ~1.1M params (Transformer encoder)
  Total task-specific: ~1.4M params

Shared:
  - shared_layers:     ~790K params (4 ResidualBlocks of size 256)
  - task_embedding:    512 params (2 × 256)
  - film:              ~131K params (gamma + beta linears)
  Total shared:        ~922K params

Ratio: ~60% task-specific / ~40% shared
```

---

## 2. Identified Bottlenecks & Weaknesses

### 2.1 Architecture Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| **Sequence collapse** | 🔴 High | Next task's temporal structure (9 steps) is flattened by encoder before shared layers. The shared backbone never processes sequential patterns. |
| **FiLM is too simple** | 🟡 Medium | FiLM only applies channel-wise affine transformation `γx + β`. It cannot selectively route features or create task-specific subnetworks. |
| **Rigid sharing** | 🟡 Medium | All features pass through the same shared backbone regardless of task. No mechanism to identify which features are task-specific vs shared. |
| **No cross-task interaction** | 🟡 Medium | Tasks don't exchange information during processing. Each passes independently through shared layers (just using same weights). |
| **Encoder bottleneck** | 🟠 Low-Med | Both encoders are simple MLPs. The category encoder is adequate for 1D embeddings, but the next encoder loses sequence information. |

### 2.2 Optimization Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| **NashMTL overhead** | 🔴 High | Solves a CVXPY optimization problem every `update_weights_every=4` steps. Uses ECOS solver with up to 100 iterations. Heavy per-batch cost. |
| **NashMTL solver fragility** | 🟡 Medium | Solver failures are silently caught (line 137), reverting to previous `alpha`. No logging of solver failures, potentially hiding instability. |
| **Shared-only gradient balancing** | 🟡 Medium | NashMTL only computes gradients w.r.t. shared parameters. Task-specific parameter conflicts are ignored. |
| **No Pareto awareness** | 🟠 Low-Med | Current optimization doesn't explicitly target Pareto-optimal solutions. May over-optimize one task at the expense of the other. |
| **Inconsistent gradient clipping** | 🟡 Medium | NashMTL clips gradients (`max_norm=2.2`), but PCGrad/GradNorm/Naive do not. |

### 2.3 Training Pipeline Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| **OR-logic early stopping** | 🟡 Medium | Saves model if *any* task improves, allowing degradation on the other task. |
| **No plateau detection** | 🟡 Medium | Only target cutoff and timeout-based stopping. No patience-based early stopping on F1 stagnation. |
| **Scheduler misalignment** | 🟠 Low | `OneCycleLR` uses `max(len(loader_next), len(loader_cat))` as `steps_per_epoch`, but actual steps may differ due to dataloader cycling. |

---

## 3. Architecture Improvements

### 3.1 MMoE (Multi-gate Mixture-of-Experts)

**Paper**: Ma et al., "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" (KDD 2018)

#### How It Works

Instead of a single shared backbone, MMoE maintains **multiple expert networks** and uses **task-specific gating networks** to softly combine expert outputs:

```
Input x
  │
  ├───► Expert_1(x) ──┐
  ├───► Expert_2(x) ──┼──► Gate_task_k(x) = softmax(W_k · x)
  ├───► Expert_3(x) ──┤
  └───► Expert_N(x) ──┘
                       │
                       ▼
              Σ gate_i · expert_i(x)  ──► Task_k Head
```

Each expert is a feed-forward network (like one of our current ResidualBlocks). Each task has its own gating network that learns which experts to attend to.

#### Proposed Integration with MTLnet

```
Category Input [B,1,64]        Next Input [B,9,64]
      │                              │
┌─────▼─────┐              ┌────────▼────────┐
│ Cat Encoder│              │  Next Encoder   │
│  (MLP)     │              │   (MLP)         │
└─────┬─────┘              └────────┬────────┘
      │ [B,256]                     │ [B,256]
      │                             │
┌─────▼─────────────────────────────▼──────┐
│              MMoE Layer                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │Expert 1 │  │Expert 2 │  │Expert N │  │
│  │(ResBlock)│  │(ResBlock)│  │(ResBlock)│  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │        │
│  ┌────▼────────────▼────────────▼────┐   │
│  │         Gate_cat(x)               │   │  → softmax weights for cat
│  │         Gate_next(x)              │   │  → softmax weights for next
│  └───────────────────────────────────┘   │
│                                          │
│  output_cat = Σ gate_cat_i · expert_i    │
│  output_next = Σ gate_next_i · expert_i  │
└──────┬────────────────────────┬──────────┘
       │                        │
 ┌─────▼─────┐          ┌──────▼──────┐
 │ Cat Head  │          │  Next Head  │
 └───────────┘          └─────────────┘
```

#### Analysis for Our Use Case

| Aspect | Assessment |
|--------|------------|
| **Task heterogeneity** | ✅ Our tasks are quite different (classification vs sequence prediction). MMoE excels when tasks have different feature needs. |
| **Seesaw problem** | ✅ MMoE addresses the seesaw problem (improving one task degrades another) by allowing task-specific expert selection. |
| **Overhead** | 🟡 N experts × params-per-expert. With N=4 experts of size 256, this is ~4× more shared params than current backbone. |
| **Gate collapse** | ⚠️ Known issue: gates may converge to using only 1-2 experts, effectively reducing to hard sharing. Need regularization. |
| **Sequence handling** | ❌ Standard MMoE doesn't address the sequence collapse issue. Would still need sequence-aware modifications. |

**Verdict**: ✅ **Worth trying**. Strong match for our heterogeneous task setup. Addresses rigid sharing and FiLM limitations. Well-established with extensive literature. But needs modifications to handle sequence data properly.

**Recommended variant**: MMoE with 4-6 experts, where expert_dim=256 (matching current shared_layer_size). Add entropy regularization to gates to prevent collapse.

---

### 3.2 PLE (Progressive Layered Extraction)

**Paper**: Tang et al., "Progressive Layered Extraction: A Novel Multi-Task Learning Model for Personalized Recommendations" (RecSys 2020, **Best Paper**)

#### How It Works

PLE extends MMoE by adding **task-specific expert networks** alongside shared experts, organized in **multiple extraction layers**:

```
Layer L:
  ├── Shared Experts:   {SE_1, SE_2, ..., SE_s}     ← Shared across tasks
  ├── Task_1 Experts:   {TE1_1, TE1_2, ..., TE1_t}  ← Exclusive to task 1
  ├── Task_2 Experts:   {TE2_1, TE2_2, ..., TE2_t}  ← Exclusive to task 2
  │
  ├── Gate_task1(input) = softmax(W1 · [all experts for task 1])
  ├── Gate_task2(input) = softmax(W2 · [all experts for task 2])
  └── Gate_shared(input) = softmax(Ws · [all experts])

Stacked over L layers progressively.
```

Each task's gate selects from both shared and its own task-specific experts. The shared gate selects from all experts and feeds into the next layer's shared experts.

#### Analysis for Our Use Case

| Aspect | Assessment |
|--------|------------|
| **Task isolation** | ✅ Task-specific experts ensure each task retains dedicated capacity even under sharing. |
| **Progressive refinement** | ✅ Multi-layer extraction progressively separates shared from task-specific features. |
| **Complexity** | 🔴 Significantly more parameters than MMoE. With 2 layers × (2 shared + 2 per-task) = 12 expert networks. |
| **Proven in RecSys** | ✅ PLE was specifically designed for recommendation systems with heterogeneous tasks, similar to our POI prediction. |
| **Over-engineering risk** | ⚠️ With only 2 tasks, PLE's progressive extraction may be overkill. |

**Verdict**: 🟡 **Consider as Phase 2 upgrade** if MMoE shows promise. PLE is the theoretically superior choice but significantly more complex. For 2 tasks, the simpler CGC variant (below) may be sufficient.

---

### 3.3 CGC (Customized Gate Control)

**Paper**: Same as PLE (Tang et al., RecSys 2020) — CGC is a single-layer version of PLE.

#### How It Works

CGC is essentially PLE with a single extraction layer. Each task has:
- Its own set of experts
- Access to shared experts
- A gating network that selects from both

```
Input x
  │
  ├───► Shared Expert 1 ──┐
  ├───► Shared Expert 2 ──┤
  ├───► Task1 Expert 1 ───┤──► Gate_1(x) → Task 1 output
  ├───► Task1 Expert 2 ───┘
  │
  ├───► Shared Expert 1 ──┐
  ├───► Shared Expert 2 ──┤
  ├───► Task2 Expert 1 ───┤──► Gate_2(x) → Task 2 output
  └───► Task2 Expert 2 ───┘
```

#### Analysis for Our Use Case

| Aspect | Assessment |
|--------|------------|
| **Simplicity** | ✅ Simpler than PLE, similar complexity to MMoE but with dedicated task experts. |
| **Negative transfer mitigation** | ✅ Task-specific experts protect against negative transfer. |
| **Parameter efficiency** | 🟡 More params than MMoE but well-contained. With 2 shared + 2 per-task = 6 experts total. |

**Verdict**: ✅ **Strong candidate**. Best balance of MMoE's flexibility with PLE's task isolation. Simpler to implement and debug than PLE.

---

### 3.4 Cross-Stitch Networks

**Paper**: Misra et al., "Cross-Stitch Networks for Multi-task Learning" (CVPR 2016)

#### How It Works

Maintains separate networks per task with learnable **cross-stitch units** that combine features at each layer:

```
[x_A^(l+1)]   [α_AA  α_AB] [x_A^(l)]
[x_B^(l+1)] = [α_BA  α_BB] [x_B^(l)]
```

The α values are learned, determining how much information flows between tasks.

#### Analysis for Our Use Case

| Aspect | Assessment |
|--------|------------|
| **Fine-grained mixing** | ✅ Learns optimal mixing ratio at each layer. |
| **Parameter cost** | 🔴 Doubles the network (separate backbone per task). |
| **Convergence** | ⚠️ α values may be hard to learn, especially with small datasets. |
| **Task asymmetry** | ❌ Assumes tasks have similar dimensionality at each layer, which our tasks don't (1D vs sequence). |

**Verdict**: ❌ **Not recommended** for our use case. The task asymmetry (single embedding vs sequence) makes cross-stitch units difficult to apply. Also doubles parameters.

---

### 3.5 MTAN (Multi-Task Attention Network)

**Paper**: Liu et al., "End-to-End Multi-Task Learning with Attention" (CVPR 2019)

#### How It Works

Uses a shared backbone with **task-specific attention modules** at each layer that selectively attend to shared features:

```
Shared Layer l → Shared features f_l
                    │
         ┌──────────┤──────────┐
         ▼                     ▼
  Attention_task1(f_l)   Attention_task2(f_l)
         │                     │
  Task 1 features       Task 2 features
```

The attention modules learn which spatial/channel features are important for each task.

#### Analysis for Our Use Case

| Aspect | Assessment |
|--------|------------|
| **Selective sharing** | ✅ Tasks can selectively attend to different features from the shared backbone. |
| **Efficiency** | ✅ Attention modules are lightweight; shared backbone is reused. |
| **Originally for vision** | ⚠️ Designed for spatial attention in CNNs. Needs adaptation for our MLP/Transformer setup. |

**Verdict**: 🟡 **Possible but needs significant adaptation**. The attention mechanism would need to be redesigned for our non-spatial feature space.

---

### 3.6 DSelect-k

**Paper**: Hazimeh et al., "DSelect-k: Differentiable Selection in the Mixture of Experts" (NeurIPS 2021)

#### How It Works

Replaces the soft gating in MMoE with **differentiable top-k selection**, activating only k out of N experts per input:

```
Gate(x) = DSelect-k(W · x)  → Sparse: only k experts activated
```

This provides:
- **Sparse expert usage**: More efficient inference
- **Harder expert specialization**: Experts learn more distinctive features
- **Smooth optimization**: Despite discrete selection, gradients flow through

#### Analysis for Our Use Case

| Aspect | Assessment |
|--------|------------|
| **Sparsity** | ✅ Reduces compute by only using k experts per input. |
| **Expert specialization** | ✅ Harder selection forces experts to specialize more. |
| **Complexity** | 🟡 More complex gating mechanism than standard MMoE. |

**Verdict**: 🟡 **Nice improvement over MMoE** but adds implementation complexity. Consider as Phase 2 after validating MMoE.

---

## 4. Architecture Comparison Matrix

| Architecture | Task Isolation | Negative Transfer | Params Overhead | Impl. Complexity | Sequence Support | RecSys Proven | Recommendation |
|-------------|---------------|-------------------|-----------------|-------------------|-----------------|---------------|----------------|
| **Current (FiLM)** | Low | Poor | Baseline | — | ❌ Lost | ❌ | — |
| **MMoE** | Medium | Good | ~2-4× shared | Low | Needs work | ✅ | **Phase 1** ✅ |
| **CGC** | High | Very Good | ~3-4× shared | Medium | Needs work | ✅ | **Phase 1** ✅ |
| **PLE** | Very High | Excellent | ~5-8× shared | High | Needs work | ✅ | Phase 2 |
| **Cross-Stitch** | High | Good | ~2× total | Medium | ❌ Bad fit | ❌ | ❌ Skip |
| **MTAN** | Medium | Good | Low | Medium | Possible | ❌ | 🟡 Maybe |
| **DSelect-k** | High | Very Good | ~2-4× shared | High | Needs work | ✅ | Phase 2 |

### Recommended Architecture Path

**Phase 1: MMoE or CGC** → Validate that mixture-of-experts improves over hard sharing with FiLM
**Phase 2: PLE or DSelect-k** → If Phase 1 shows improvement, try progressive extraction or sparse selection

---

## 5. Optimization Strategy Improvements

### 5.1 Current NashMTL Analysis

**How NashMTL works in our code** (`src/losses/nash_mtl.py`):

1. Computes per-task gradients w.r.t. shared parameters
2. Forms Gram matrix `G·G^T` (gradient inner products)
3. Solves a convex optimization problem via CVXPY/ECOS:
   - Finds Nash equilibrium weights `α` such that no task can improve without degrading another
4. Applies `weighted_loss = Σ α_i · loss_i`

**Current configuration**: `max_norm=2.2`, `update_weights_every=4`, `optim_niter=30`

**Observed issues**:
- **Computational cost**: CVXPY solver with ECOS backend is called every 4 steps with up to 100 max iterations
- **Solver fragility**: Exceptions are silently caught (falls back to previous α)
- **Limited effect**: Per the user's observation, NashMTL "doesn't seem to have a lot of effect" — this may be because:
  1. The FiLM + separate encoders already reduce gradient conflict (tasks are somewhat isolated)
  2. The Gram matrix of only 2 tasks is a 2×2 matrix — the optimization landscape is trivial
  3. With `update_weights_every=4`, many gradient steps use stale weights
  4. The 2 tasks may not have significant gradient conflicts in the shared layers

### 5.2 FAMO (Fast Adaptive Multitask Optimization)

**Paper**: Liu et al., "FAMO: Fast Adaptive Multitask Optimization" (NeurIPS 2023)

#### How It Works

FAMO optimizes task weights to minimize the **maximum per-task loss decrease**. It maintains a set of dynamic weights and updates them via a fast closed-form update rule.

Key idea: At each step, compute the direction that improves the worst-performing task the most.

```
w_t+1 = w_t · exp(-η · loss_t)   (per-task exponential weighting)
w_t+1 = w_t+1 / Σ w_t+1          (normalize)
loss = Σ w_t · loss_t             (weighted sum)
```

#### Advantages
- **O(1) overhead**: No gradient computation for weights, no solver
- **Theoretically grounded**: Provable convergence to Pareto-optimal solutions
- **Min-max fairness**: Naturally focuses on the harder task

| Aspect | Assessment |
|--------|------------|
| **Speed** | ✅ Orders of magnitude faster than NashMTL (no solver) |
| **Fairness** | ✅ Explicitly targets worst-case task performance |
| **Ease of implementation** | ✅ ~20 lines of code |
| **Hyperparameters** | Learning rate η for weight update |

**Verdict**: ✅ **Strong recommendation**. Best speed-to-performance ratio. Should be the first replacement to try.

---

### 5.3 CAGrad (Conflict-Averse Gradient Descent)

**Paper**: Liu et al., "Conflict-Averse Gradient Descent for Multi-task Learning" (NeurIPS 2021)

#### How It Works

CAGrad finds the gradient direction within a ball around the average gradient that **maximizes the minimum improvement** across all tasks:

```
g* = arg min_{g : ||g - g_avg|| ≤ c·||g_avg||}  max_i  <g, g_i>
```

It solves a simpler optimization problem than NashMTL: project the average gradient to avoid conflicts.

#### Advantages
- **Conflict-aware**: Explicitly avoids directions that hurt any task
- **Bounded deviation**: The constraint `||g - g_avg|| ≤ c·||g_avg||` prevents extreme weight swings
- **Moderate overhead**: Requires per-task gradients but uses a simpler optimization than NashMTL

| Aspect | Assessment |
|--------|------------|
| **Speed** | 🟡 Faster than NashMTL but still needs per-task gradient computation |
| **Conflict resolution** | ✅ Explicitly minimizes maximum task conflict |
| **Robustness** | ✅ Bounded deviation from average gradient prevents instability |
| **Hyperparameters** | c (constraint radius), typically 0.4 |

**Verdict**: ✅ **Good candidate**. More principled than PCGrad, faster than NashMTL. Good middle ground.

---

### 5.4 Aligned-MTL

**Paper**: Senushkin et al., "Independent Component Alignment for Multi-Task Learning" (CVPR 2023)

#### How It Works

Aligned-MTL decomposes the gradient matrix using SVD (Singular Value Decomposition) and aligns the gradient components to prevent interference:

1. Stack per-task gradients into matrix G
2. Compute SVD: `G = U·S·V^T`
3. Align gradients by rotating them to be orthogonal in the principal component space
4. Reconstruct the combined gradient

#### Advantages
- **No hyperparameters**: Fully automatic alignment
- **Principled**: Based on linear algebra (SVD decomposition)
- **Strong results**: State-of-the-art on NYUv2 and Cityscapes benchmarks

| Aspect | Assessment |
|--------|------------|
| **Speed** | 🟡 SVD computation on gradient matrix; moderate overhead |
| **Tuning** | ✅ No hyperparameters to tune |
| **Theoretical basis** | ✅ Clean linear algebra formulation |
| **2-task performance** | ⚠️ With only 2 tasks, SVD produces 2 components — benefit may be limited |

**Verdict**: 🟡 **Worth evaluating**. No hyperparameters is attractive, but the 2-task setting may limit its advantage over simpler methods.

---

### 5.5 MoCo (Mitigating Gradient Bias)

**Paper**: Fernando et al., "Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach" (ICLR 2023)

#### How It Works

MoCo identifies and corrects **gradient bias** — the tendency of gradient-based methods to favor tasks with larger gradient magnitudes:

1. Compute per-task gradients
2. Estimate the bias in the current gradient direction
3. Apply a momentum-corrected update that debiases the gradient

#### Advantages
- **Addresses root cause**: Instead of weighting losses or projecting gradients, it directly corrects the gradient bias
- **Provably convergent**: Theoretical convergence guarantees

| Aspect | Assessment |
|--------|------------|
| **Novelty** | ✅ Addresses gradient bias, which may explain why NashMTL has little effect |
| **Speed** | 🟡 Per-task gradient computation + bias correction |
| **Theory** | ✅ Strong convergence guarantees |

**Verdict**: 🟡 **Interesting approach** if gradient bias is the root cause of our NashMTL ineffectiveness.

---

### 5.6 DB-MTL (Dual-Balancing Multi-Task Learning)

**Paper**: Lin et al., "DB-MTL: Dual-Balancing Multi-Task Learning" (arXiv 2023)

#### How It Works

DB-MTL balances **both loss scales and gradient magnitudes** simultaneously:

1. **Loss balancing**: Normalize losses to similar scales using exponential moving averages
2. **Gradient balancing**: Apply gradient normalization after loss balancing

This dual approach addresses the issue that loss balancing alone doesn't prevent gradient magnitude imbalance.

#### Advantages
- **Dual balancing**: Addresses both loss scale and gradient magnitude disparities
- **Low overhead**: Uses EMA-based normalization, no solver needed
- **Complementary**: Can be combined with other methods

| Aspect | Assessment |
|--------|------------|
| **Speed** | ✅ Very fast — only EMA updates |
| **Simplicity** | ✅ Easy to implement |
| **Effectiveness** | ✅ Addresses a fundamental scaling issue |

**Verdict**: ✅ **Highly recommended**. Simple, fast, addresses a fundamental issue.

---

### 5.7 IMTL (Impartial Multi-Task Learning)

**Paper**: Liu et al., "Towards Impartial Multi-task Learning" (ICLR 2021)

#### How It Works

IMTL guarantees **equal treatment of all tasks** by finding a gradient direction with equal projections onto each task's gradient:

```
Find d such that: <d, g_i> = <d, g_j> for all task pairs (i,j)
```

This ensures no task is favored over another in the update direction.

| Aspect | Assessment |
|--------|------------|
| **Fairness** | ✅ Guarantees equal treatment |
| **Speed** | 🟡 Moderate — requires solving a system of equations |
| **2-task simplification** | ✅ With 2 tasks, simplifies to bisecting the angle between gradients |

**Verdict**: 🟡 **Solid choice** for ensuring neither task dominates.

---

### 5.8 Uncertainty Weighting (UW)

**Paper**: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)

#### How It Works

Learns task weights based on **homoscedastic uncertainty** (task-level noise):

```
L_total = (1/2σ²_1) · L_1 + (1/2σ²_2) · L_2 + log(σ_1) + log(σ_2)
```

Where σ_i are learnable parameters representing per-task uncertainty. Higher uncertainty → lower weight. The log(σ) terms prevent trivial solutions.

| Aspect | Assessment |
|--------|------------|
| **Speed** | ✅ Zero overhead — just 2 extra learnable parameters |
| **Simplicity** | ✅ Trivial to implement |
| **Effectiveness** | 🟡 Works well but may not address gradient conflicts |
| **Baseline** | ✅ Should be included as a baseline |

**Verdict**: ✅ **Must-have baseline**. Extremely simple and often competitive.

---

### 5.9 Random Loss Weighting (RLW)

**Paper**: Lin et al., "Reasonable Effectiveness of Random Weighting" (TMLR 2022)

#### How It Works

Samples random task weights from a **Dirichlet distribution** at each training step:

```
w_1, w_2 ~ Dirichlet(1, 1)
L_total = w_1 · L_1 + w_2 · L_2
```

Surprisingly effective as a regularizer — the stochasticity prevents overfitting to any particular task balance.

| Aspect | Assessment |
|--------|------------|
| **Speed** | ✅ Zero computational overhead |
| **Simplicity** | ✅ 2 lines of code |
| **Effectiveness** | 🟡 Competitive with expensive methods on many benchmarks |
| **Insight** | ⚠️ The fact that random weighting works well suggests many complex methods may be over-engineered |

**Verdict**: ✅ **Must-have baseline**. If RLW matches or beats NashMTL, it validates the intuition that the overhead isn't worth it.

---

## 6. Optimizer Comparison Matrix

| Method | Per-Batch Cost | Hyperparameters | Gradient Computation | Addresses | Implementation | Year |
|--------|---------------|-----------------|---------------------|-----------|----------------|------|
| **NashMTL** (current) | 🔴 Very High (CVXPY) | max_norm, optim_niter, update_freq | Per-task + solver | Conflicts | Complex | 2022 |
| **FAMO** | 🟢 Very Low | η (weight lr) | Standard only | Worst-case task | Simple | 2023 |
| **CAGrad** | 🟡 Medium | c (constraint radius) | Per-task | Conflicts | Moderate | 2021 |
| **Aligned-MTL** | 🟡 Medium | None | Per-task + SVD | Interference | Moderate | 2023 |
| **MoCo** | 🟡 Medium | Momentum coefficient | Per-task + correction | Gradient bias | Moderate | 2023 |
| **DB-MTL** | 🟢 Low | EMA decay | Standard + norms | Loss/grad scale | Simple | 2023 |
| **IMTL** | 🟡 Medium | None | Per-task | Equal treatment | Moderate | 2021 |
| **UW** | 🟢 Very Low | None (learned σ) | Standard only | Loss scale | Trivial | 2018 |
| **RLW** | 🟢 Very Low | Dirichlet α | Standard only | Regularization | Trivial | 2022 |
| **PCGrad** (available) | 🟡 Medium | None | Per-task + projection | Conflicts | Moderate | 2020 |
| **GradNorm** (available) | 🟡 Medium | α (rate exponent) | Per-task + norms | Gradient magnitude | Moderate | 2018 |
| **Equal Weighting** | 🟢 None | None | Standard only | — (baseline) | Trivial | — |

---

## 7. Critical Insight: Do MTL Optimizers Even Help?

**Paper to consider**: Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" (NeurIPS 2022)

This paper presents a provocative finding: on many benchmarks, **simple scalarization** (fixed equal weights) with proper tuning performs comparably to or better than complex MTL optimization methods.

### Key takeaways:
1. **Architecture matters more than optimization**: Changing the model architecture (MMoE, PLE) typically provides larger gains than changing the loss weighting strategy
2. **Equal weighting is a strong baseline**: With proper learning rate tuning, `L = L_1 + L_2` is hard to beat
3. **Grid search on static weights**: Simple `L = α·L_1 + (1-α)·L_2` with α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} often matches complex methods
4. **Gradient manipulation adds noise**: Per-task gradient computations can destabilize training, especially with small batch sizes

### Implications for our project:
- The user's observation that "NashMTL doesn't seem to have a lot of effect" aligns with this finding
- **Priority should be architecture changes (MMoE/CGC) over optimizer changes**
- However, replacing NashMTL with something cheaper (FAMO, UW, or even equal weighting) frees up compute budget for larger models or more experiments

---

## 8. Recommendations & Implementation Roadmap

### Phase 0: Baselines & Diagnostics (1-2 weeks)

**Goal**: Establish proper baselines and understand the current model's behavior.

1. **Equal Weighting baseline**: Run `L = L_cat + L_next` (no MTL optimizer) to quantify NashMTL's actual contribution
2. **Static weight search**: Try `L = α·L_cat + (1-α)·L_next` with α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
3. **Gradient conflict analysis**: Log gradient cosine similarity between tasks at the shared layers over training. If cosine > 0 consistently, there's minimal conflict and MTL optimizers add overhead for nothing
4. **Task dominance analysis**: Track per-task loss curves independently to identify if one task dominates training

### Phase 1A: Architecture — MMoE/CGC (2-3 weeks)

**Goal**: Replace the FiLM + shared backbone with a Mixture-of-Experts architecture.

**Option A: MMoE** (simpler, start here)
- Replace `shared_layers` with N=4 expert networks (each a ResidualBlock stack)
- Add 2 gating networks (one per task): `Gate_k(x) = softmax(W_k · x + b_k)` where x is the encoder output
- Keep the existing task-specific encoders and heads
- Add entropy regularization on gate outputs: `L_reg = -λ · Σ gate · log(gate)` to prevent expert collapse

**Option B: CGC** (adds task-specific experts)
- Same as MMoE but add 2 task-specific experts per task
- Each task's gate selects from shared + own experts
- More parameters but better task isolation

**Changes required**:
- New file: `src/models/mmoe.py` (or modify `mtlnet.py`)
- Update `registry.py` to register new model
- Update `ExperimentConfig` to support new model params (num_experts, expert_dim, etc.)
- Update `experiment.py` factory methods

### Phase 1B: Optimization — Quick Wins (1-2 weeks, parallel with 1A)

**Goal**: Replace NashMTL with faster, potentially better alternatives.

**Priority order**:
1. **UW (Uncertainty Weighting)**: Add 2 learnable σ parameters. 10 minutes to implement.
2. **FAMO**: Implement exponential weight updates. ~1 hour to implement.
3. **DB-MTL**: EMA-based dual balancing. ~2 hours to implement.
4. **RLW**: Random Dirichlet weights. 5 minutes to implement.

All should be registered in the existing `src/losses/registry.py` system.

### Phase 2: Advanced Architecture (2-4 weeks)

**Goal**: Address the sequence collapse issue and try PLE.

1. **Sequence-Aware MMoE**: Modify experts to process sequences `[B, seq_len, dim]` instead of `[B, dim]`. The category task would still collapse to 1D before experts, but next task preserves temporal structure.
2. **PLE**: Stack 2 extraction layers with progressive separation of shared and task-specific features.
3. **DSelect-k**: Replace soft gating with differentiable top-k selection.

### Phase 3: Training Pipeline Improvements (1 week)

1. **Pareto early stopping**: Save model only if it's on the Pareto frontier of (F1_cat, F1_next)
2. **Patience-based stopping**: Stop if neither task's val F1 improves for N epochs
3. **Gradient conflict logging**: Add periodic cosine similarity logging between task gradients
4. **Consistent gradient clipping**: Apply gradient clipping across all MTL methods

---

## 9. Experiment Plan

### 9.1 Experimental Matrix

| Experiment ID | Architecture | Optimizer | Purpose |
|--------------|-------------|-----------|---------|
| E0-baseline | Current MTLnet | NashMTL | Reference baseline |
| E1-equal | Current MTLnet | Equal Weighting | Quantify NashMTL value |
| E2-uw | Current MTLnet | UW | Simple loss weighting baseline |
| E3-famo | Current MTLnet | FAMO | Fast adaptive weighting |
| E4-rlw | Current MTLnet | RLW | Random weighting baseline |
| E5-dbmtl | Current MTLnet | DB-MTL | Dual balancing |
| E6-mmoe-nash | MMoE (4 experts) | NashMTL | Architecture improvement |
| E6-mmoe-famo | MMoE (4 experts) | FAMO | Architecture + optimizer |
| E6-mmoe-equal | MMoE (4 experts) | Equal Weighting | Architecture only |
| E7-cgc-famo | CGC (2s+2t experts) | FAMO | Best of both |
| E8-ple-famo | PLE (2 layers) | FAMO | Advanced architecture |

### 9.2 Evaluation Protocol

For each experiment:
- **5-fold cross-validation** (matching current setup)
- **Metrics**: Macro F1, accuracy, per-class F1 for both tasks
- **Report**: Mean ± std across folds
- **Training cost**: Wall-clock time per fold, FLOPs
- **Convergence**: Epochs to best validation F1

### 9.3 Success Criteria

| Metric | Current (Approximate) | Target |
|--------|----------------------|--------|
| Category macro F1 | ~0.65 | ≥ 0.70 |
| Next-POI macro F1 | ~0.45 | ≥ 0.50 |
| Training time/fold | ~X hours | ≤ 1.5X |
| Combined improvement | — | Both tasks improve or one improves significantly without degrading other |

---

## 10. References

### Architecture
1. Ma, J. et al. "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts." KDD 2018. **(MMoE)**
2. Tang, H. et al. "Progressive Layered Extraction: A Novel Multi-Task Learning Model for Personalized Recommendations." RecSys 2020. **(PLE/CGC)**
3. Misra, I. et al. "Cross-Stitch Networks for Multi-task Learning." CVPR 2016.
4. Liu, S. et al. "End-to-End Multi-Task Learning with Attention." CVPR 2019. **(MTAN)**
5. Hazimeh, H. et al. "DSelect-k: Differentiable Selection in the Mixture of Experts." NeurIPS 2021.

### Optimization
6. Navon, A. et al. "Multi-Task Learning as a Bargaining Game." ICML 2022. **(Nash-MTL)**
7. Liu, B. et al. "FAMO: Fast Adaptive Multitask Optimization." NeurIPS 2023.
8. Liu, B. et al. "Conflict-Averse Gradient Descent for Multi-task Learning." NeurIPS 2021. **(CAGrad)**
9. Senushkin, D. et al. "Independent Component Alignment for Multi-Task Learning." CVPR 2023. **(Aligned-MTL)**
10. Fernando, H. et al. "Mitigating Gradient Bias in Multi-objective Learning." ICLR 2023. **(MoCo)**
11. Lin, B. et al. "DB-MTL: Dual-Balancing Multi-Task Learning." arXiv 2023.
12. Liu, L. et al. "Towards Impartial Multi-task Learning." ICLR 2021. **(IMTL)**
13. Kendall, A. et al. "Multi-Task Learning Using Uncertainty to Weigh Losses." CVPR 2018. **(UW)**
14. Lin, B. et al. "Reasonable Effectiveness of Random Weighting." TMLR 2022. **(RLW)**
15. Chen, Z. et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing." ICML 2018.
16. Yu, T. et al. "Gradient Surgery for Multi-Task Learning." NeurIPS 2020. **(PCGrad)**

### Critical Analysis
17. Xin, D. et al. "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" NeurIPS 2022.

### Libraries
18. Lin, B. et al. "LibMTL: A PyTorch Library for Multi-Task Learning." JMLR 2023.
19. Lin, B. et al. "Comprehensive Survey on Gradient-based Multi-Objective Deep Learning." arXiv 2025.
