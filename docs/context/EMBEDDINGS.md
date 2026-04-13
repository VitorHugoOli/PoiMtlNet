# Embedding Engines

Each engine produces a fixed-dimensional vector per POI (or per check-in)
that encodes different aspects of the POI's identity. These embeddings
serve as the raw input to MTLnet.

All engines produce 64-dimensional embeddings by default, except HMRM
(107-dim). In fusion mode, multiple engines are concatenated to form
128-dim inputs.

---

## Primary Engines

### HGI (Hierarchical Graph Infomax)

**Type:** POI-level (one embedding per unique POI)
**Dimension:** 64
**Directory:** `research/embeddings/hgi/`

**Method:** Learns POI embeddings by maximizing mutual information across
a 3-level spatial hierarchy: POI → Region → City.

1. **POI2Vec pre-training:** Learns embeddings at the functional class
   (fclass) level using skip-gram on POI co-visitation sequences. Multiple
   POIs with the same fclass share identical embeddings at this stage.
2. **GCN encoding:** A 2-layer Graph Convolutional Network refines POI
   embeddings using the Delaunay triangulation graph (spatial proximity).
3. **Hierarchical MI maximization:** Contrastive learning with bilinear
   discriminators at each level. POI embeddings are aggregated to region
   embeddings (via multi-head attention), then to city embeddings (via
   area-weighted sum). The model maximizes agreement between each level
   and its parent.

**What it captures:** Graph-structural relationships — co-visitation
patterns, spatial neighborhood, category clustering in the mobility graph.
HGI is the strongest single embedding engine in the project (best results
on Alabama).

**Reference:** Velickovic et al., "Deep Graph Infomax", ICLR 2019
(base method). The hierarchical extension is project-specific.

---

### Sphere2Vec (Spherical Position Encoding)

**Type:** POI-level
**Dimension:** 64
**Directory:** `research/embeddings/sphere2vec/`

**Method:** Learns a location embedding from raw GPS coordinates
(latitude, longitude) that preserves spherical surface distance between
points.

Two encoder variants exist:
- **`rbf` (default):** Custom multi-scale spherical-RBF encoder with 256
  random unit centroids × 32 log-spaced scales. Ported from a Colab
  notebook. Output: 8192 → FFN → 64.
- **`paper`:** Paper-faithful closed-form encoder (Eq. 8 from Mai et al.)
  using sin/cos of angular distance at 32 frequencies. Output: 256 → FFN → 64.

Training uses contrastive learning: positives are small Gaussian
perturbations of coordinates (σ=0.01°), negatives are random other
coordinates. BCE loss on cosine similarity.

**What it captures:** Pure geographic position. POIs that are
geographically close get similar embeddings regardless of their function.
Sphere2Vec preserves spherical distance, avoiding map projection distortion
that affects Euclidean position encoders.

**Note:** The `rbf` variant (used in production) is NOT the paper's
sphereM — it is a custom off-paper encoder from a mislabeled notebook.
See `research/embeddings/sphere2vec/README.md` for the discrepancy table.

**Reference:** Mai et al., "Sphere2Vec: A General-Purpose Location
Representation Learning over a Spherical Surface for Large-Scale
Geospatial Predictions", ISPRS 2023. https://arxiv.org/abs/2306.17624

---

### Time2Vec (Temporal Embedding)

**Type:** Check-in-level (one embedding per visit event)
**Dimension:** 64
**Directory:** `research/embeddings/time2vec/`

**Method:** Learns temporal embeddings from check-in timestamps using
contrastive learning over (hour-of-day, day-of-week) features.

Architecture:
- Input: 2D features (normalized hour ∈ [0,1], normalized day ∈ [0,1])
- Time2Vec layer: `t2v(t) = [ωt + φ, sin(ω₁t + φ₁), ..., sin(ωₖt + φₖ)]`
  (1 linear + 63 periodic components with learned frequencies and phases)
- Contrastive model: encoder projects to 64D, trained with BCE on cosine
  similarity between temporally close check-ins (positives) vs random
  (negatives)

**What it captures:** Temporal patterns of visitation. The same POI
visited at 8am and 8pm gets different Time2Vec embeddings. This encodes
routines (morning coffee, evening entertainment) as learnable periodic
features.

**Key property:** Check-in-level embeddings mean the 9-step next-task
window becomes a true spatio-temporal trajectory, not just a POI sequence.

**Reference:** Kazemi et al., "Time2Vec: Learning a Vector Representation
of Time", 2019. https://arxiv.org/abs/1907.05321

---

## Secondary Engines

### DGI (Deep Graph Infomax)

**Type:** POI-level
**Dimension:** 64
**Directory:** `research/embeddings/dgi/`

**Method:** Single-level graph infomax — maximizes mutual information
between node (POI) embeddings and a graph-level summary. Simpler than
HGI (no hierarchy). Uses a GCN encoder on the POI co-visitation graph.

**What it captures:** Graph neighborhood structure at a single scale.
Weaker than HGI because it lacks the region and city aggregation levels.

**Reference:** Velickovic et al., "Deep Graph Infomax", ICLR 2019.
https://arxiv.org/abs/1809.10341

---

### POI2HGI

**Type:** POI-level
**Dimension:** 64
**Directory:** `research/embeddings/poi2hgi/`

**Method:** HGI variant that uses temporal features (36-dim: hour
histograms, day-of-week histograms, cyclical encodings) instead of
category one-hot vectors as node features. Designed for downstream
category prediction where using category as input would leak the target.

**Architecture:** Same GCN + hierarchical MI maximization as HGI but
with temporal node features: 24 hour-bins + 7 day-bins + 4 cyclical
components + 1 visit count.

**What it captures:** POI identity through temporal visitation patterns
and spatial structure, without category information.

---

### Check2HGI

**Type:** Check-in-level
**Dimension:** 64
**Directory:** `research/embeddings/check2hgi/`

**Method:** Extends HGI from 3 to 4 levels: Check-in → POI → Region → City.
Each check-in event gets its own embedding based on the user's sequential
context (connected via user-sequence edges with temporal decay weights).

**4-level hierarchy:**
```
City (area-weighted region aggregation)
  └── Region (GCN on census tract adjacency)
       └── POI (GCN on Delaunay graph)
            └── Check-in (GCN on user-sequence graph)
```

**What it captures:** Individual visit context — the same POI visited by
different users or at different times gets different embeddings. Encodes
user mobility patterns, temporal context, and spatial relationships.

---

### Space2Vec

**Type:** POI-level
**Dimension:** 64
**Directory:** `research/embeddings/space2vec/`

**Method:** Multi-scale spatial representation learning using position
encoding in 2D Euclidean space. Predecessor to Sphere2Vec without
spherical distance preservation.

**Reference:** Mai et al., "Multi-Scale Representation Learning for
Spatial Feature Distributions using Grid Cells", ICLR 2020.
https://arxiv.org/abs/2003.00824

---

### HMRM (Heterogeneous Mobility Representation Model)

**Type:** POI-level
**Dimension:** 107
**Directory:** `research/embeddings/hmrm/`

**Method:** Learns mobility representations from heterogeneous check-in
data. Non-standard dimension (107 instead of 64) makes it incompatible
with the default model configuration without dimension adjustment.

---

## Embedding Level Summary

| Engine | Level | Dim | Signal | Used in Fusion |
|--------|-------|-----|--------|---------------|
| HGI | POI | 64 | Graph structure, co-visitation | Category + Next |
| Sphere2Vec | POI | 64 | Geographic position (spherical) | Category |
| Time2Vec | Check-in | 64 | Temporal patterns | Next |
| DGI | POI | 64 | Graph structure (single-level) | No |
| POI2HGI | POI | 64 | Temporal + spatial (no category) | No |
| Check2HGI | Check-in | 64 | User context + temporal + spatial | No |
| Space2Vec | POI | 64 | Geographic position (Euclidean) | No |
| HMRM | POI | 107 | Heterogeneous mobility | No |

---

## References

1. Velickovic et al., "Deep Graph Infomax", ICLR 2019.
   https://arxiv.org/abs/1809.10341
2. Mai et al., "Sphere2Vec: A General-Purpose Location Representation
   Learning over a Spherical Surface", ISPRS 2023.
   https://arxiv.org/abs/2306.17624
3. Mai et al., "Multi-Scale Representation Learning for Spatial Feature
   Distributions using Grid Cells", ICLR 2020.
   https://arxiv.org/abs/2003.00824
4. Kazemi et al., "Time2Vec: Learning a Vector Representation of Time",
   2019. https://arxiv.org/abs/1907.05321
