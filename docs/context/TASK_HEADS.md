# Task Heads

Task heads are the final per-task decoder modules that receive the
shared backbone's output and produce class logits. They are pluggable
via the model registry.

---

## Category Heads

Input: `[B, D]` (flat vector from shared backbone, D=256 by default).
Output: `[B, 7]` (logits over 7 POI categories).

### CategoryHeadTransformer (default in MTLnet)

**Key:** `category_transformer` | ~263K params

Splits the D-dim input into `num_tokens` tokens of `token_dim` each,
adds learned positional embeddings, runs through a TransformerEncoder,
mean-pools, and classifies. Requires `num_tokens × token_dim = D`.

Despite being the worst standalone head (F1=0.402), it co-adapts well
with the shared backbone in MTL training (best results overall).

---

### DCNHead (Deep & Cross Network)

**Key:** `category_dcn` | ~102K params

Parallel deep (MLP) and cross (explicit feature cross products) paths:
- Cross path: iterative `x_l = x_0 × w_l^T × x_{l-1} + x_{l-1}`
- Deep path: standard MLP with dropout
- Concatenate + classify

Best standalone head on HGI (F1=0.728). Particularly relevant for
fusion inputs where cross-features between embedding sources may
capture useful interactions.

**Reference:** Wang et al., "Deep & Cross Network for Ad Click
Predictions", KDD 2017.

---

### CategoryHeadEnsemble

**Key:** `category_ensemble` | ~250K params

Multiple parallel MLP paths with varying depths (path i has depth i+2).
Concatenates all path outputs and classifies.

---

### CategoryHeadGated

**Key:** `category_gated` | ~263K params

GLU-style gating throughout: each layer has a transform branch and a
gate branch (sigmoid), multiplied element-wise.

---

### CategoryHeadResidual

**Key:** `category_residual` | ~100K params

Stacked residual blocks with LayerNorm + GELU + skip connections.

---

### CategoryHeadAttentionPooling

**Key:** `category_attention` | ~50K params

Learns attention scores over input features, applies element-wise
weighted representation, then classifies.

---

### SEHead (Squeeze-and-Excitation)

**Key:** `category_se` | ~58K params

Squeeze-and-Excitation channel gating: global → squeeze → excite →
per-feature scaling. Then MLP classifier.

**Reference:** Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.

---

### CategoryHeadSingle

**Key:** `category_single` | ~10K params

Simple multi-layer MLP with LayerNorm + GELU + Dropout.

---

### CategoryHeadLinear (diagnostic)

**Key:** `category_linear` | ~903 params

Single `nn.Linear(D, 7)`. Diagnostic probe: if this matches deeper
heads, the backbone is doing all the work.

---

## Next-POI Heads

Input: `[B, 9, D]` (sequence from shared backbone, 9 steps × D-dim).
Output: `[B, 7]` (logits over 7 next categories).

### NextHeadMTL (default in MTLnet)

**Key:** `next_mtl` | ~1.3M params

TransformerEncoder with sinusoidal positional encoding, causal mask,
and learned attention-weighted pooling. 4 layers, 8 heads,
dropout=0.35, norm_first.

Worst standalone head (F1=0.043) but best inside the MTL pipeline —
the canonical example of head-backbone co-adaptation.

---

### NextHeadTCNResidual

**Key:** `next_tcn_residual` | ~397K params

Canonical TCN with residual blocks and exponential dilation scheduling
[1, 2, 4, 8]. Each block: two dilated causal Conv1d layers with
BatchNorm + GELU + Dropout + skip connection. AdaptiveAvgPool1d for
sequence reduction.

Best standalone next head (F1=0.244). Covers the full 9-step receptive
field with dilation.

**Reference:** Bai et al., "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling", 2018.
https://arxiv.org/abs/1803.01271

---

### NextHeadTemporalCNN

**Key:** `next_temporal_cnn` | ~442K params

Simpler TCN without residual connections: stacked causal Conv1d layers
with BatchNorm + GELU + Dropout. AdaptiveAvgPool1d.

Second-best standalone (F1=0.200).

---

### NextHeadConvAttn

**Key:** `next_conv_attn` | ~216K params

Hybrid: TCN encoder followed by single cross-attention pooling layer.
A learned query token attends over TCN features to produce the
classification input. Tests adaptive pooling vs average pooling.

Inspired by the Conformer pattern (Gulati et al., Interspeech 2020).

---

### NextHeadTransformerOptimized

**Key:** `next_transformer_optimized` | ~1.3M params

TransformerEncoder with learned positional embeddings (instead of
sinusoidal) and temporal decay buffer for recency-weighted pooling
(`exp(-0.5 × t)`). Causal mask.

---

### NextHeadSingle

**Key:** `next_single` | ~1.3M params

TransformerEncoder with learned positional embeddings and learnable
temporal decay bias for attention. Returns optional attention weights
for interpretability.

---

### NextHeadTransformerRelPos

**Key:** `next_transformer_relpos` | ~398K params

Custom TransformerEncoder with ALiBi-style learned relative position
bias. The bias encodes "how far apart are two check-ins" rather than
absolute position. 2 layers, 4 heads (lighter than default).
Attention-weighted pooling.

**Reference:** Press et al., "ALiBi: Attention with Linear Biases",
ICLR 2022. https://arxiv.org/abs/2108.12409

---

### NextHeadLSTM

**Key:** `next_lstm` | ~1.0M params

Bidirectional LSTM with dropout. Extracts last valid timestep.
Slow (137s vs 11s for TCN) and worse (F1=0.189).

---

### NextHeadGRU

**Key:** `next_gru` | ~770K params

GRU with dropout. Extracts last valid timestep. Same issues as LSTM.

---

### NextHeadHybrid

**Key:** `next_hybrid` | ~1.5M params

GRU encoder + MultiheadAttention (self-attention on GRU output) with
residual connection. Combines recurrence with attention. Slow (141s).

---

## Empirical Rankings (HGI, Alabama, 10 epochs, standalone)

### Category Heads

| Rank | Head | F1 | Params |
|------|------|-----|--------|
| 1 | category_dcn | **0.728** | 102K |
| 2 | category_residual | 0.690 | 100K |
| 3 | category_gated | 0.679 | 263K |
| 4 | category_ensemble | 0.660 | 250K |
| 5 | category_single | 0.626 | 10K |
| 6 | category_se | 0.622 | 58K |
| 7 | category_attention | 0.581 | 50K |
| 8 | category_linear | 0.545 | 903 |
| 9 | category_transformer | 0.402 | 263K |

### Next Heads

| Rank | Head | F1 | Time | Params |
|------|------|-----|------|--------|
| 1 | next_tcn_residual | **0.244** | 11s | 397K |
| 2 | next_temporal_cnn | 0.200 | 9s | 442K |
| 3 | next_single | 0.198 | 11s | 1.3M |
| 4 | next_lstm | 0.189 | 138s | 1.0M |
| 5 | next_transformer_opt | 0.180 | 9s | 1.3M |
| 6 | next_gru | 0.163 | 141s | 770K |
| 7 | next_conv_attn | 0.161 | 9s | 216K |
| 8 | next_hybrid | 0.161 | 141s | 1.5M |
| 9 | next_mtl | 0.043 | 11s | 1.3M |
| 10 | next_transformer_relpos | 0.043 | 9s | 398K |

**Key insight:** Standalone rankings invert in the MTL pipeline.
The default heads (transformer for category, next_mtl for next) are the
worst standalone but the best inside MTL, due to co-adaptation with the
shared backbone.
