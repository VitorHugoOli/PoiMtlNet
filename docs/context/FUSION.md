# Multi-Embedding Fusion

## Motivation

Single embedding engines capture one aspect of a POI's identity:

- **HGI:** Who visits this POI and what other POIs are nearby in the
  mobility graph (structural signal)
- **Sphere2Vec:** Where this POI is on the earth's surface
  (spatial signal)
- **Time2Vec:** When users visit (temporal signal)

No single engine captures all three. Fusion concatenates complementary
embeddings per task to provide richer input representations.

## Design: Task-Specific Fusion

The key design decision is that **different tasks receive different
embedding combinations**. The active preset is `space_hgi_time`:

### Category Task: Sphere2Vec (64D) + HGI (64D) = 128D

```
Input vector: [sphere2vec_0, ..., sphere2vec_63, hgi_0, ..., hgi_63]
               ├── spatial signal ──────────────┤├── structural signal ─┤
```

**Rationale:** POI category correlates with both location and graph
neighborhood:
- Sphere2Vec encodes *where* the POI is — restaurants cluster in
  downtown areas, parks in suburbs, airports at city edges. Geographic
  zone is a strong category prior.
- HGI encodes *what the POI connects to* — restaurants neighbor other
  food/entertainment POIs in the co-visitation graph, regardless of
  absolute location.

The concatenation lets the classifier use both "geographic zone" and
"graph neighborhood" signals simultaneously.

### Next Task: HGI (64D) + Time2Vec (64D) = 128D

```
Per check-in: [hgi_0, ..., hgi_63, time2vec_0, ..., time2vec_63]
               ├── structural signal ──────────┤├── temporal signal ────┤
Window:       [step_0 (128D), step_1 (128D), ..., step_8 (128D)]
```

**Rationale:** Next-category prediction requires understanding both
*which* POIs the user visited and *when*:
- HGI identifies each POI in the sequence (structural identity).
- Time2Vec encodes the temporal context of each visit. The same coffee
  shop at 7am and 3pm gets different Time2Vec vectors, transforming the
  window from a POI sequence into a spatio-temporal trajectory.

This directly captures mobility routines: {coffee@7am, office@9am,
restaurant@12pm} contains temporal patterns that {coffee, office,
restaurant} alone does not.

### Why Not the Same Fusion for Both Tasks?

- **Sphere2Vec is irrelevant for next-category prediction.** The next
  task predicts *behavior* (what the user does next), not *location*.
  Geographic coordinates of past check-ins don't help predict the
  category of the next visit — a user leaving a restaurant could go
  anywhere regardless of the restaurant's coordinates.

- **Time2Vec is irrelevant for category classification.** Category is
  a static property of the POI. When a user visited doesn't change
  what the POI is. Adding temporal noise to the category embedding
  would dilute the signal.

## Scale Imbalance

The two embedding sources have very different scales:

| Task | Source 1 | L2 Norm | Source 2 | L2 Norm | Ratio |
|------|----------|---------|----------|---------|-------|
| Category | Sphere2Vec | 0.55 | HGI | 8.46 | **15.2x** |
| Next | Time2Vec | 1.00 | HGI | 8.70 | **8.7x** |

### Empirical Impact

Gradient flow analysis (10 steps on real Alabama data):
- Category encoder receives **12x larger gradients** from HGI features
  than from Sphere2Vec features.
- Zero-ablation: removing Sphere2Vec changes encoder output by only
  **0.7%**; removing HGI changes it by **90.2%**.
- The model essentially ignores Sphere2Vec.

### Normalization Was Tested and Rejected

| Strategy | Cat Accuracy | Source Balance |
|----------|-------------|---------------|
| **Raw (no normalization)** | **0.606** | 0.01 (HGI dominates) |
| Per-source z-score | 0.508 | 0.37 |
| Learnable per-source LayerNorm | 0.504 | 0.45 |

Forcing balanced source usage **degrades accuracy by 10 percentage
points**. The model performs better when it naturally ignores the
weaker source through gradient magnitude.

**Interpretation:** HGI is simply the stronger embedding for category
classification. Sphere2Vec's spatial signal is either redundant with
(or weaker than) what HGI already captures. The model acts as an
implicit feature selector — a useful property, not a bug.

**Open question:** Does fusion outperform HGI-only? If the model
ignores Sphere2Vec anyway, the 128-dim fusion may just be HGI + noise.
This is tested in Stage 0 of the full ablation study.

## Implementation

### Configuration

Defined in `src/configs/embedding_fusion.py`:

```python
FUSION_PRESETS = {
    "space_hgi_time": FusionConfig(
        category_embeddings=[
            EmbeddingSpec(EmbeddingEngine.SPHERE2VEC, EmbeddingLevel.POI, 64),
            EmbeddingSpec(EmbeddingEngine.HGI, EmbeddingLevel.POI, 64),
        ],
        next_embeddings=[
            EmbeddingSpec(EmbeddingEngine.HGI, EmbeddingLevel.POI, 64),
            EmbeddingSpec(EmbeddingEngine.TIME2VEC, EmbeddingLevel.CHECKIN, 64),
        ],
    ),
}
```

### Data Generation

Pipeline: `pipelines/fusion.pipe.py`

1. Loads each embedding source via `EmbeddingLoader`.
2. Aligns embeddings:
   - POI-level (Sphere2Vec, HGI): merged by `placeid`.
   - Check-in-level (Time2Vec): merged by `(userid, placeid, datetime)`.
3. Concatenates in source order.
4. Saves to `output/fusion/{state}/input/{category,next}.parquet`.

### Data Format

**Category:** `[placeid, category, 0, 1, ..., 127]` — 128 numeric
feature columns (Sphere2Vec cols 0-63, HGI cols 64-127).

**Next:** `[0, 1, ..., 1151, next_category, userid]` — 1152 numeric
columns (9 steps × 128D) plus metadata.

### Model Integration

The MTLnet model accepts `feature_size=128` via `--embedding-dim 128`.
Both task encoders receive 128-dim input and project to the shared
`shared_layer_size` (256D by default). The model's architecture is
otherwise unchanged — fusion is purely an input-level modification.

## Alternative Presets

Two other presets exist but are not the current default:

| Preset | Category | Next |
|--------|----------|------|
| `space_hgi_time` (active) | Sphere2Vec + HGI | HGI + Time2Vec |
| `hgi_time` | HGI + Sphere2Vec | HGI + Time2Vec |
| `space_time` | Sphere2Vec only (64D) | Sphere2Vec + Time2Vec |

The difference between `space_hgi_time` and `hgi_time` is only the
concatenation order for category (Sphere2Vec first vs HGI first).
`space_time` omits HGI entirely.

## References

1. Mai et al., "Sphere2Vec", ISPRS 2023 — spatial embedding.
2. Kazemi et al., "Time2Vec", 2019 — temporal embedding.
3. Velickovic et al., "Deep Graph Infomax", ICLR 2019 — HGI base.
4. See `docs/full_ablation_study/FUSION_RATIONALE.md` for scale
   imbalance experimental details.
