# MTL Backbone Architectures

All architectures follow the same pipeline:

```
Task Input → Task Encoder (per-task MLP) → Shared Backbone → Task Head → Output
```

The architectures differ only in the **shared backbone** — the component
responsible for multi-task knowledge transfer. Task encoders and heads
are shared across all architectures (pluggable via the model registry).

---

## 1. MTLnet (Baseline — FiLM + Shared Residual)

**Registry key:** `mtlnet`
**Source:** `src/models/mtl/mtlnet/model.py`

### Architecture

```
enc_cat ─→ FiLM(task_id=0) ─→ Shared ResidualBlocks ─→ category_head
enc_next ─→ FiLM(task_id=1) ─→ Shared ResidualBlocks ─→ next_head
```

- **Task embedding:** Learned embedding per task (2 × `shared_layer_size`)
- **FiLM layer:** Feature-wise Linear Modulation — `γ(task) × x + β(task)`.
  Learns per-feature scaling and bias conditioned on task identity.
- **Shared layers:** Linear → LeakyReLU → LayerNorm → Dropout, followed
  by `num_shared_layers - 1` ResidualBlocks (each: LayerNorm → Linear →
  LeakyReLU → Dropout → Linear → Dropout + skip).

### Limitation

FiLM provides **multiplicative gating only** — it scales and shifts
existing features but cannot learn task-specific basis vectors. This
limits the backbone's ability to produce genuinely different
representations for different tasks. Experimentally, FiLM-based MTLnet
is 16% worse than CGC on HGI (Finding 12 in PAPER_FINDINGS).

### Reference

Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
AAAI 2018. https://arxiv.org/abs/1709.07871

---

## 2. MTLnet-CGC (Customized Gate Control)

**Registry key:** `mtlnet_cgc`
**Source:** `src/models/mtl/mtlnet_cgc/model.py`
**Component:** `CGCLiteLayer` in `src/models/mtl/_components.py`

### Architecture

```
enc_cat ─→ ┌─ Shared Expert 1 ─┐     ┌─ Cat Expert 1 ─┐
            │  Shared Expert 2  │  +  │  Cat Expert 2  │  → gate(cat) → category_head
            └──────────────────┘     └────────────────┘

enc_next ─→ ┌─ Shared Expert 1 ─┐     ┌─ Next Expert 1 ─┐
             │  Shared Expert 2  │  +  │  Next Expert 2   │  → gate(next) → next_head
             └──────────────────┘     └─────────────────┘
```

- **Shared experts:** `num_shared_experts` independent expert networks
  (each: Linear → LeakyReLU → LayerNorm → Dropout + ResidualBlocks).
  Shared across both tasks.
- **Task-specific experts:** `num_task_experts` experts per task.
  Only used by their respective task.
- **Gating:** Softmax gate per task over all available experts
  (shared + task-specific). Gate input is the task's encoded features.
  Produces a weighted combination of expert outputs.

### Best Configuration

`num_shared_experts=2, num_task_experts=2` ("s2t2") — best on HGI
(joint=0.4855). More task-specific experts (t=2 vs t=1) consistently
improve results, suggesting the tasks benefit from dedicated capacity.

### Gate Diagnostics

Gate entropy and mean weights are logged per task for diagnosing
expert collapse (all weight on one expert) or degenerate routing.

### Reference

Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
Learning (MTL) Model for Personalized Recommendations", RecSys 2020.
CGC is the single-level building block of PLE.

---

## 3. MTLnet-MMoE (Multi-gate Mixture-of-Experts)

**Registry key:** `mtlnet_mmoe`
**Source:** `src/models/mtl/mtlnet_mmoe/model.py`
**Component:** `MMoELiteLayer` in `src/models/mtl/_components.py`

### Architecture

```
enc_cat ─→ ┌─ Expert 1 ─┐
            │  Expert 2  │  → gate_cat → category_head
            │  Expert 3  │
            │  Expert 4  │  → gate_next → next_head
enc_next ─→ └────────────┘
```

- **All experts are shared** (no task-specific experts).
- **Separate gates per task:** Each task has its own softmax gate
  over all experts.
- Simpler than CGC: same expert pool, only the routing differs.

### Difference from CGC

MMoE has only shared experts. CGC adds dedicated task-specific experts
that are not available to the other task. This gives CGC more capacity
for task-specific processing but more parameters.

### Reference

Ma et al., "Modeling Task Relationships in Multi-task Learning with
Multi-gate Mixture-of-Experts", KDD 2018.
https://dl.acm.org/doi/10.1145/3219819.3220007

---

## 4. MTLnet-DSelectK (Sparse Expert Selection)

**Registry key:** `mtlnet_dselectk`
**Source:** `src/models/mtl/mtlnet_dselectk/model.py`
**Component:** `DSelectKLiteLayer` in `src/models/mtl/_components.py`

### Architecture

Like MMoE but with **sparse, learnable selector-based routing**:

- `num_selectors` selector networks per task, each producing a softmax
  distribution over experts.
- Global selector mixture weights (per task) combine selector outputs.
- Temperature-controlled softmax for sharper/softer selection.

### Key Parameters

- `num_experts=4`: Number of shared expert networks
- `num_selectors=2`: Number of selector sub-networks per task
- `temperature=0.5`: Controls selection sharpness (lower = sparser)

### Behavior

DSelectK tends to win on weaker embeddings (best on DGI) where the
extra routing capacity compensates for weaker input signal. On stronger
embeddings (HGI), the overhead doesn't help and CGC's simpler gating
outperforms.

### Reference

Hazimeh et al., "DSelect-k: Differentiable Selection in the Mixture of
Experts with Applications to Multi-Task Learning", NeurIPS 2021.

---

## 5. MTLnet-PLE (Progressive Layered Extraction)

**Registry key:** `mtlnet_ple`
**Source:** `src/models/mtl/mtlnet_ple/model.py`
**Component:** `PLELiteLayer` in `src/models/mtl/_components.py`

### Architecture

Stacks multiple CGC layers:

```
Level 0:  enc_cat, enc_next → CGCLiteLayer → cat_h0, next_h0
Level 1:  cat_h0, next_h0   → CGCLiteLayer → cat_h1, next_h1
...
Level N:  cat_hN-1, next_hN-1 → CGCLiteLayer → output_cat, output_next
```

Each level's experts receive the gated outputs of the previous level,
enabling progressive refinement of shared and task-specific
representations.

### Key Parameter

- `num_levels=2` (default): Number of stacked CGC layers.

### Status

PLE was tested in Phase 4 but performed poorly (joint=0.235) when
combined with simultaneous head swaps. It has not been tested in
isolation with default heads — future work.

### Reference

Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
Learning (MTL) Model for Personalized Recommendations", RecSys 2020.
https://dl.acm.org/doi/10.1145/3383313.3412236

---

## Architecture Comparison

### Empirical Results (HGI, Alabama, 1-fold, 10 epochs)

| Architecture | Joint Score | Cat F1 | Next F1 |
|-------------|------------|--------|---------|
| **CGC (s=2, t=2)** | **0.430** | 0.553 | 0.307 |
| DSelectK (e=4, k=2) | 0.422 | 0.541 | 0.303 |
| CGC (s=2, t=1) | 0.408 | 0.526 | 0.290 |
| MMoE (e=4) | 0.395 | 0.507 | 0.283 |
| MTLnet (FiLM) | 0.371 | 0.487 | 0.255 |

### Parameter Sharing Spectrum

```
Hard sharing ←──────────────────────────────────→ Soft sharing
MTLnet(FiLM)    MMoE    CGC(s>0,t=0)    CGC(s,t)    DSelectK
   │              │          │              │           │
   All shared     Shared     Shared +       Shared +    Sparse
   via one set    experts,   task-specific  gated       selection
   of layers      task       experts        routing     routing
                  gates
```

---

## References

1. Perez et al., "FiLM: Visual Reasoning with a General Conditioning
   Layer", AAAI 2018.
2. Ma et al., "Modeling Task Relationships in Multi-task Learning with
   Multi-gate Mixture-of-Experts", KDD 2018.
3. Tang et al., "Progressive Layered Extraction (PLE)", RecSys 2020.
4. Hazimeh et al., "DSelect-k: Differentiable Selection in the Mixture
   of Experts", NeurIPS 2021.
5. Velickovic et al., "Deep Graph Infomax", ICLR 2019 (FiLM conditioning
   inspiration).
