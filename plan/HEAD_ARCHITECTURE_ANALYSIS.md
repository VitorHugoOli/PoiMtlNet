# Task Head Architecture Analysis (2026-04-13)

## Context

This document analyzes the current task heads in MTLnet, reviews recent
literature on classification/sequence heads for POI prediction, and
recommends changes grounded in both the ablation data and the research.

### Current Pipeline

```
Raw Input [B, D] or [B, S, D]
  -> Task Encoder (2-layer MLP, per-task)           28% of params
  -> Shared Backbone (FiLM/CGC/PLE/DSelectK)        10% of params
  -> Task Head (per-task classifier/sequence model)  62% of params
```

The model is **head-dominated**: the heads consume 62% of total parameters
while the shared backbone -- the part responsible for MTL knowledge
transfer -- is only 10%. This is architecturally inverted from best
practice, where the shared representation should be the heavyweight and
heads should be lightweight decoders.

---

## Part 1: Category Head Analysis

### Current Implementations (8 variants)

| Head | F1 (1ep) | Params | Approach |
|------|----------|--------|----------|
| cat_ensemble | 0.429 | ~250K | 3 parallel MLP paths, variable depth |
| cat_single | 0.426 | ~10K | 2-layer MLP |
| cat_residual | 0.427 | ~100K | Residual skip-connection MLP |
| cat_gated | 0.423 | ~263K | GLU-style gated layers |
| cat_attention | 0.414 | ~50K | Feature attention reweighting |
| cat_dcn | 0.420 | ~102K | Deep & Cross Network |
| cat_se | 0.309 | ~58K | Squeeze-and-Excitation |
| cat_transformer | 0.280 | ~263K | Synthetic tokenization + Transformer |

### Key Observation

**Head complexity is irrelevant for category classification.** The top 6
heads span only 0.009 F1 (0.420-0.429). The simplest head (cat_single,
~10K params) matches the most complex (cat_ensemble, ~250K params).

**Why this happens:** The category input after the shared backbone is a
flat 128-dimensional vector. There is no spatial structure, no sequence,
no multi-modal signal -- just a single embedding. For a 7-class
classification from a 128-dim input, a 2-layer MLP is the correct
inductive bias. Complex heads (transformer, ensemble, gated) add
parameters without adding useful structure.

### What the Literature Says

Recent MTL research confirms this pattern:

> "Task-specific heads are optimized, relying on a single fully connected
> layer. Most of the model's parameters and depth are thus in the feature
> encoding [...] computational resources can be efficiently shared to learn
> a more effective shared representation."
> -- (2024 MTL survey, Springer Nature)

For POI category classification specifically, the state-of-the-art
focuses on **improving the embedding** (hierarchy-enhanced representations,
GNN-based spatial encoding) rather than the classifier head. The head is
typically a single linear layer or shallow MLP.

### Recommendation for Category Head

**Use `category_single` with `hidden_dims=[128, 64]` as the default.**

This cuts category head parameters from ~263K (transformer) to ~10K,
freeing parameter budget for the shared backbone where it matters.
Performance is equivalent (0.426 vs 0.429 F1, within noise).

### New Category Head Candidates Worth Testing

Given the data, we do NOT need more category head variants. However,
one approach from the literature is worth noting for completeness:

1. **Linear probe** (no hidden layers, just `nn.Linear(128, 7)`).
   If this matches the MLP, it proves the shared backbone is doing all
   the work -- which is exactly what you want in MTL.

---

## Part 2: Next-Task Head Analysis

### Current Implementations (7 variants)

| Head | F1 (1ep) | Time | Params | Approach |
|------|----------|------|--------|----------|
| next_temporal_cnn | **0.242** | 9s | ~442K | Dilated causal Conv1d |
| next_transformer_opt | **0.237** | 14s | ~1.3M | Learned pos + temporal decay |
| next_hybrid | 0.057 | 282s | ~1.5M | GRU + cross-attention |
| next_single | 0.055 | 18s | ~1.3M | Learned pos + temporal decay bias |
| next_lstm | 0.043 | 285s | ~1.0M | Bidirectional LSTM |
| next_gru | 0.036 | 266s | ~770K | GRU |
| next_mtl (default) | **0.026** | 14s | ~1.3M | Sinusoidal pos + causal Transformer |

### Key Observations

1. **Temporal CNN dominates** (0.242 F1, 9 seconds, fewest params among
   competitive heads). It's 9x better than the default and 30x faster
   than LSTM/GRU.

2. **Transformer_opt is close** (0.237 F1) and differs from next_mtl
   mainly in using learned positional embeddings + temporal decay buffer
   instead of sinusoidal positional encoding.

3. **RNN-based heads (LSTM, GRU, hybrid) are both slow and worse.**
   For a fixed window of 9 steps, RNNs offer no advantage over parallel
   architectures and pay heavy sequential computation costs.

4. **next_mtl is the worst standalone** (0.026 F1). It uses sinusoidal
   positional encoding, higher dropout (0.35), and no temporal decay.
   Inside the MTL system it may benefit from the shared backbone, but
   the standalone signal is concerning.

### What the Literature Says

**Transformers vs TCN for short sequences:**

The TCN literature shows that for fixed-length, short sequences (like a
9-step check-in window), dilated causal convolutions have several
advantages over Transformers:
- Better parallelism (no sequential attention computation)
- Explicit control over receptive field via dilation rates
- More stable gradients through residual connections
- Fewer parameters for equivalent receptive field

Recent trajectory prediction papers (GETNext, STAN, TrajGraph) use
Transformers primarily because they handle **variable-length** sequences
and need **global** attention. For a fixed 9-step window, the global
attention is overkill -- a TCN with dilation [1, 2, 4, 8] already covers
the full window.

**Mamba / State Space Models:**

Mamba (Gu & Dao, 2023) offers O(n) sequence modeling with selective
state spaces. For a 9-step window this is not a meaningful efficiency
gain over Transformers (which are already fast at seq_len=9). Mamba
becomes relevant for much longer sequences (100+). Additionally, Mamba
requires the `mamba-ssm` CUDA kernel which doesn't work on MPS.
**Not recommended for this setup.**

**RetNet:**

Retentive Networks offer parallel training + O(1) recurrent inference.
Same reasoning as Mamba: the 9-step window doesn't benefit from linear
complexity. The standard Transformer is already efficient here.
**Not recommended.**

### Recommendation for Next Head

**Use `next_temporal_cnn` as the default.** It leads on F1, speed, and
parameter efficiency.

### New Next Head Candidates Worth Implementing

#### 1. TCN with Residual Blocks and Dilation Reset

**Motivation:** The current `next_temporal_cnn` uses simple stacked
Conv1d without residual connections or explicit dilation scheduling.
The TCN literature strongly recommends residual blocks with exponentially
increasing dilation rates.

**Architecture:**
```
for each block i:
    dilation = 2^i
    x -> Conv1d(dilation=d, kernel=3) -> BatchNorm -> GELU -> Dropout
      -> Conv1d(dilation=d, kernel=3) -> BatchNorm -> GELU -> Dropout
      + residual connection (with 1x1 conv if channels change)
```

With 4 blocks and dilation [1, 2, 4, 8], the receptive field covers
1 + 2*(3-1)*(1+2+4+8) = 61 positions -- far more than the 9-step window,
ensuring full coverage. This matches the canonical TCN design from
Bai et al. (2018).

**Why it fits:** Direct improvement over the current temporal_cnn.
Same inductive bias (causal convolution) but with skip connections
for better gradient flow in deeper configurations.

**Source:** Bai et al., "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling", 2018.
https://arxiv.org/abs/1803.01271

#### 2. Lightweight Transformer with Relative Position Bias

**Motivation:** The current transformer heads use either sinusoidal or
learned absolute positional encodings. For a fixed 9-step window,
relative position bias (like in T5 or ALiBi) is more natural: "how
far apart are two check-ins" matters more than "which position is this."

**Architecture:**
```
Standard TransformerEncoder but:
  - Replace positional encoding with learned relative position bias
    (9x9 learnable bias matrix added to attention logits)
  - Use pre-norm (norm_first=True)
  - 2 layers, 4 heads (smaller than current 4 layers, 8 heads)
  - Attention-weighted pooling for sequence reduction
```

**Why it fits:** Addresses the specific weakness of next_mtl (sinusoidal
pos encoding) while being smaller. Relative position captures "the
check-in 3 steps ago" regardless of absolute window position.

**Source:** Press et al., "Train Short, Test Long: Attention with Linear
Biases Enables Input Length Extrapolation" (ALiBi), ICLR 2022.
https://arxiv.org/abs/2108.12409
Shaw et al., "Self-Attention with Relative Position Representations",
NAACL 2018. https://arxiv.org/abs/1803.02155

#### 3. Conv-Attention Hybrid (TCN encoder + single attention pooling)

**Motivation:** Combine TCN's local feature extraction strength with
attention's ability to learn which timesteps matter for the final
prediction. This avoids full self-attention (O(n^2)) while keeping
the adaptive pooling benefit.

**Architecture:**
```
Input [B, 9, D]
  -> TCN encoder (2-3 causal conv blocks with residual connections)
  -> Single MultiheadAttention layer (queries = learned [CLS] token,
     keys/values = TCN output)
  -> Linear classifier
```

**Why it fits:** The temporal_cnn currently uses AdaptiveAvgPool1d for
sequence reduction, which weights all timesteps equally. For next-POI
prediction, recent check-ins should matter more. A single attention
layer over TCN features is cheaper than a full Transformer but more
expressive than average pooling.

**Source:** Inspired by the Conv-Transformer hybrid pattern used in
speech recognition (Conformer, Gulati et al., 2020) and recent
trajectory prediction (TrajGraph, 2024).

---

## Part 3: Candidates Considered and Rejected

### For Category Heads

**No new category heads recommended.** The data shows head complexity
doesn't help. The correct investment is in the embedding/backbone,
not the classifier.

Rejected candidates:
- **TabNet-style attentive feature selection**: Designed for tabular
  data with many sparse features. Our 128-dim dense embedding doesn't
  benefit from sparse selection.
- **Prototype networks**: Metric-learning approach for few-shot. We
  have sufficient training data per class.
- **Capsule networks**: Over-engineered for flat vector classification.

### For Next Heads

- **Mamba / S4**: O(n) sequence modeling. Irrelevant for seq_len=9.
  Requires CUDA kernels, doesn't work on MPS. **Rejected.**
- **RetNet**: Same reasoning as Mamba. Linear complexity doesn't
  matter at seq_len=9. **Rejected.**
- **Full GETNext-style architecture**: Requires a global trajectory
  flow map built from all users' trajectories, plus a graph neural
  network. This is an embedding-level change, not a head-level change.
  **Out of scope for head ablation.**
- **LSTM with temporal attention (ATST-LSTM)**: RNN-based methods are
  consistently slower and worse in our ablation. Adding attention to
  LSTM doesn't fix the fundamental sequential computation problem.
  **Rejected.**
- **NextLocMoE**: LLM-based with dual-level MoE inside Transformer
  layers. Requires a pretrained LLM backbone. Completely different
  paradigm from our embedding-based approach. **Rejected.**

---

## Part 4: Implementation Priority

| Priority | Name | Task | Rationale |
|----------|------|------|-----------|
| 1 | next_tcn_residual | next | Canonical TCN with residual blocks + dilation scheduling. Direct upgrade over current temporal_cnn. |
| 2 | next_conv_attn | next | TCN encoder + single attention pooling. Tests if adaptive pooling beats avg pooling. |
| 3 | next_transformer_relpos | next | Lightweight Transformer with relative position bias. Tests if relative > absolute position encoding. |
| 4 | category_linear | category | Single linear layer (no hidden). Diagnostic: proves backbone does all work. |

## Part 5: Structural Recommendation

Beyond specific head variants, the key architectural change is to
**rebalance the parameter budget**:

| Component | Current % | Recommended % |
|-----------|----------|---------------|
| Task encoders | 28% | 25-30% |
| Shared backbone | 10% | **30-40%** |
| Category head | 14% | **2-5%** |
| Next head | 48% | **25-35%** |

This can be achieved by:
1. Switching category head to simple MLP (14% -> 2%)
2. Switching next head to temporal CNN (48% -> ~25%)
3. Increasing `shared_layer_size` or `num_shared_layers` with the
   freed parameter budget

The shared backbone is where MTL actually works. Making it larger
while simplifying the heads should improve both tasks by giving the
sharing mechanism more capacity.

---

## References

1. Bai et al., "An Empirical Evaluation of Generic Convolutional and
   Recurrent Networks for Sequence Modeling", 2018.
   https://arxiv.org/abs/1803.01271
2. Press et al., "Train Short, Test Long: Attention with Linear Biases
   Enables Input Length Extrapolation" (ALiBi), ICLR 2022.
   https://arxiv.org/abs/2108.12409
3. Shaw et al., "Self-Attention with Relative Position Representations",
   NAACL 2018. https://arxiv.org/abs/1803.02155
4. Gulati et al., "Conformer: Convolution-augmented Transformer for
   Speech Recognition", Interspeech 2020.
   https://arxiv.org/abs/2005.08100
5. Yang et al., "GETNext: Trajectory Flow Map Enhanced Transformer for
   Next POI Recommendation", SIGIR 2022.
6. Luo et al., "STAN: Spatio-Temporal Attention Network for Next
   Location Recommendation", WWW 2021.
7. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
   Spaces", 2023. https://arxiv.org/abs/2312.00752 (rejected for this
   setup -- seq_len too short, no MPS support).
8. Wang et al., "Mixture-of-Experts for Personalized and Semantic-Aware
   Next Location Prediction" (NextLocMoE), 2025.
   https://arxiv.org/abs/2505.24597 (rejected -- requires LLM backbone).
9. Survey: "A survey on graph neural network-based next POI
   recommendation for smart cities", 2024.
   https://link.springer.com/article/10.1007/s40860-024-00233-z
10. Survey: "Deep multi-task learning: a review of concepts, methods,
    and cross-domain applications", 2025.
    https://link.springer.com/article/10.1007/s41060-025-00892-y
