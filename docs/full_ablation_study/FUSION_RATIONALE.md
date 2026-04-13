# Fusion Embedding Rationale & Impact Assessment

## Current Fusion Configuration: `space_hgi_time`

| Task | Embeddings | Dim | Rationale |
|------|-----------|-----|-----------|
| **Category** | Sphere2Vec (64) + HGI (64) | 128 | Spatial location + graph structure |
| **Next** | HGI (64) + Time2Vec (64) | 128 | Graph structure + temporal context |

### Why This Combination Makes Sense

**Category task (POI classification):**
- **HGI** (Hierarchical Graph Infomax): Captures graph-structural relationships
  between POIs — co-visitation patterns, neighborhood structure, category
  clustering in the mobility graph. This is the core signal for category
  classification.
- **Sphere2Vec** (Spherical Position Encoding): Encodes geographic coordinates
  while preserving spherical distance. A restaurant in a downtown area vs
  a park in the suburbs have different spatial signatures. Sphere2Vec adds
  spatial context that HGI alone doesn't capture (HGI encodes graph
  topology, not raw geography).

The combination is sound: **"where is this POI" (Sphere2Vec) + "what is
this POI connected to" (HGI)**. These are complementary signals — a POI's
category correlates with both its location (restaurants cluster downtown)
and its graph neighbors (restaurants connect to other food/entertainment
POIs).

**Next task (next-category prediction):**
- **HGI** (64D, POI-level): Same structural embedding per POI, repeated
  across the 9-step window. Provides identity and neighborhood context
  for each visited POI.
- **Time2Vec** (64D, check-in-level): Temporal embedding per check-in
  event. The same POI visited at 8am vs 8pm gets different Time2Vec
  vectors. This transforms the window from "which POIs did the user
  visit" to "which POIs did the user visit and when."

The combination is the key insight: **the check-in sequence is no longer
just a POI sequence — it's a spatio-temporal trajectory.** A user visiting
{coffee shop, office, restaurant} tells you less than {coffee shop at 8am,
office at 9am, restaurant at 12pm}. Time2Vec encodes these temporal
patterns as learnable periodic + linear features.

### What This Changes for the Ablation

**For the category task:** The 128-dim input now has two semantically
distinct halves. The first 64 dimensions are spatial (Sphere2Vec), the
second 64 are graph-structural (HGI). This could affect which category
heads work best:
- **DCN (Deep & Cross)** might excel here because explicit cross-features
  between spatial and structural dimensions could capture interactions
  that concatenation alone misses.
- **Simple heads** (linear, single MLP) may struggle more than with HGI
  alone because the two embedding sources are on different scales
  (Sphere2Vec: [-0.5, 0.5], HGI: [-4, 14]).

**For the next task:** Time2Vec makes each position in the window unique
even for repeated POIs. This fundamentally changes what the sequence
models see:
- **Transformer heads** (which attend across positions) may benefit
  more because attention can now differentiate "same POI at different
  times."
- **TCN heads** (which apply local convolutions) may benefit less because
  the temporal signal is already encoded per-position — the convolution
  doesn't need to infer timing from position.

### Risk Assessment

**Main risk:** The two embedding sources have different scales and
distributions. If the model learns to ignore one source (e.g., the
smaller-scale Sphere2Vec gets dominated by HGI), the fusion provides
no benefit. The shared backbone's LayerNorm should mitigate this, but
it's worth monitoring per-source gradient magnitudes.

**Mitigation already in place:** The existing equal_weight and db_mtl
optimizers operate on task-level losses, not embedding-source-level.
If scale imbalance is a problem, it would manifest as the fusion model
underperforming HGI-only — which Stage 0 is designed to detect.

## Scale Imbalance Experiment (2026-04-13)

### Measured Scale Difference

| Task | Source 1 | L2 norm | Source 2 | L2 norm | Ratio |
|------|----------|---------|----------|---------|-------|
| Category | Sphere2Vec | 0.555 | HGI | 8.463 | **15.2x** |
| Next | Time2Vec | 1.000 | HGI | 8.702 | **8.7x** |

### Gradient Flow Analysis (10 steps, real data)

Category encoder first layer:
- Sphere2Vec gradient magnitude: ~0.0002
- HGI gradient magnitude: ~0.003
- **Ratio: ~12x** (consistent across all 10 steps)

Next encoder first layer:
- Time2Vec gradient magnitude: ~0.0005
- HGI gradient magnitude: ~0.004
- **Ratio: ~8x** (consistent across all 10 steps)

### Zero-Ablation (after 10 training steps)

Removing one source and measuring encoder output change:

| Task | Without Source 1 | Without Source 2 |
|------|-----------------|-----------------|
| Category | Sphere2Vec removed: **0.7% change** | HGI removed: **90.2% change** |
| Next | Time2Vec removed: **2.4% change** | HGI removed: **91.7% change** |

**The model almost entirely ignores the smaller-scale source.**

### Normalization Comparison (50 steps)

| Method | Loss | Cat Acc | Balance |
|--------|------|---------|---------|
| **RAW (no normalization)** | **2.725** | **0.606** | 0.01 |
| Pre-standardized (z-score) | 2.925 | 0.508 | 0.37 |
| Learnable LayerNorm | 2.927 | 0.504 | 0.45 |

### Conclusion

**Normalization hurts.** The model naturally down-weights the weaker
embedding source through gradient magnitude. HGI is the stronger signal
for both tasks. Forcing balanced contributions dilutes HGI with a weaker
source and degrades performance.

**Do NOT add per-source normalization to the pipeline.**

However, this raises a question: **is Sphere2Vec contributing anything
to category classification? Is Time2Vec contributing anything to next
prediction?** The zero-ablation suggests they may not be. A proper test
would be comparing fusion vs HGI-only at full training scale — which is
exactly what Stage 0 of the ablation study does.

**For the paper:** This is a publishable finding about multi-source
embedding fusion dynamics.

## Updated Study Design Implications

1. **Stage 0 must include HGI-only baseline** for direct comparison.
2. **Stage 1 optimizer selection should include uncertainty_weighting**
   (Kendall et al. 2018) — most-cited adaptive baseline, reviewers
   expect it.
3. **DWA moves to supplementary** — still tested but not in the main
   5-optimizer grid.
4. **Multi-state validation** after Stage 3 confirms generalization.
5. **Do NOT normalize fusion inputs** — experimentally validated that
   it hurts performance.
