# Prediction Tasks

MTLnet jointly trains two tasks over POI (Point of Interest) check-in data.
Both tasks share a common backbone but have different inputs, outputs, and
inductive biases.

---

## Task 1: POI Category Classification

### Definition

Given a POI's embedding vector, predict its functional category.

### Input

A single embedding vector per POI: `[1, D]` where D is the embedding
dimension (64 for single-engine, 128 for fusion).

The embedding captures the POI's structural, spatial, or temporal
characteristics depending on the engine (HGI encodes graph topology,
Sphere2Vec encodes geographic coordinates, etc.).

### Output

7-class classification (softmax over logits):

| ID | Category | Description |
|----|----------|-------------|
| 0 | Community | Libraries, government buildings, religious institutions |
| 1 | Entertainment | Cinemas, arcades, concert venues, sports facilities |
| 2 | Food | Restaurants, cafes, bakeries, food trucks |
| 3 | Nightlife | Bars, nightclubs, lounges |
| 4 | Outdoors | Parks, beaches, hiking trails, gardens |
| 5 | Shopping | Malls, retail stores, markets |
| 6 | Travel | Hotels, airports, train stations, car rental |

An 8th category ("None") exists in the mapping but is not used in
the classification targets.

### Class Distribution (Alabama)

The distribution is imbalanced. Food and Shopping dominate:

| Category | Count | Percentage |
|----------|-------|------------|
| Food | 3,808 | 32.5% |
| Shopping | 3,666 | 31.3% |
| Community | 1,757 | 15.0% |
| Entertainment | 761 | 6.5% |
| Outdoors | 718 | 6.1% |
| Travel | 708 | 6.0% |
| Nightlife | 288 | 2.5% |

The training pipeline uses class-weighted CrossEntropyLoss to handle
this imbalance (weights computed via sklearn's `compute_class_weight`
with `class_weight='balanced'`).

### Why This Task Matters

POI category classification tests whether the embedding captures
functional semantics. A good embedding should produce distinct clusters
for different POI types. In the MTL setting, category classification
acts as an auxiliary task that regularizes the shared representation,
encouraging the backbone to learn features useful for both classification
and prediction.

### Evaluation Metric

**Macro F1** — the unweighted mean of per-class F1 scores. Chosen because
it treats all categories equally despite class imbalance, penalizing
models that only predict the majority class.

---

## Task 2: Next-Category Prediction

### Definition

Given a sequence of the user's last 9 check-ins (each represented by its
embedding), predict the category of the next POI the user will visit.

### Input

A sequence tensor: `[B, 9, D]` where B is batch size, 9 is the sliding
window length, and D is the embedding dimension.

Each position in the window corresponds to one check-in. The embeddings
can be:
- **POI-level** (same POI = same vector across all visits): used by
  HGI, DGI, Sphere2Vec, Space2Vec, POI2HGI.
- **Check-in-level** (same POI at different times = different vector):
  used by Time2Vec, Check2HGI.

### Sequence Generation

Sequences are created from each user's chronologically ordered check-in
history using a sliding window:

```
User check-in history: [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, ...]

Window 1: [c1, c2, c3, c4, c5, c6, c7, c8, c9]  → target: category(c10)
Window 2: [c10, c11, c12, c13, c14, c15, c16, c17, c18] → target: category(c19)
...
```

- **Window size:** 9 (configurable via `InputsConfig.SLIDE_WINDOW`)
- **Stride:** Non-overlapping (stride = window size)
- **Padding:** Sequences shorter than 9 are left-padded with
  `PAD_VALUE = -1` (converted to 0 in the model's forward pass)
- **Minimum user history:** 5 check-ins required to generate at least
  one sequence

### Output

Same 7-class classification as the category task. The target is the
**category** of the next POI, not the POI identity itself.

This is deliberately a simpler target than true next-POI prediction
(which would require ranking over thousands of POI candidates). The
plan documents a future `next_poi` upgrade path but the current system
uses `next_category` throughout.

### Class Distribution (Alabama, Next-Category)

| Category | Count | Percentage |
|----------|-------|------------|
| Food | 4,344 | 34.2% |
| Shopping | 3,300 | 26.0% |
| Community | 2,225 | 17.5% |
| Travel | 884 | 7.0% |
| Entertainment | 790 | 6.2% |
| Outdoors | 704 | 5.5% |
| Nightlife | 452 | 3.6% |

### Why This Task Matters

Next-category prediction captures user mobility patterns: people who
visit a coffee shop in the morning tend to go to an office next, then
a restaurant at lunch. In the MTL setting, it forces the shared
representation to encode sequential and temporal information alongside
the categorical features needed for Task 1.

The two tasks are complementary:
- **Category** asks "what IS this place?" (static property)
- **Next-category** asks "what will the user DO next?" (dynamic behavior)

### Evaluation Metric

**Macro F1**, same as the category task. The joint score used for model
selection is: `joint_score = 0.5 × next_macro_f1 + 0.5 × category_macro_f1`.

---

## Multi-Task Joint Training

### Mixed Batch Iteration

The two tasks have different dataset sizes (category: ~11.7K POIs,
next: ~12.7K sequences in Alabama). Training iterates over the longer
dataset, cycling the shorter one to match. Within each step, the model
receives one batch from each task.

### Joint Checkpoint

Model selection uses the joint score from the same checkpoint (not
per-task best epochs). The deployable model is one set of weights
evaluated on both tasks simultaneously.

### Cross-Validation

Stratified 5-fold CV with batch size 2048. Category data is split by
`placeid`, next data by `userid` to prevent information leakage
between users' training and validation sequences.

---

## References

- Foursquare check-in datasets: Yang et al., "Revisiting User Mobility
  and Social Relationships in LBSNs: A Hypergraph Embedding Approach",
  WWW 2019.
- POI category taxonomy: Foursquare venue categories (7 top-level groups).
- Next-POI prediction formulation: adapted from Chang et al., "Category-
  Aware Next Point-of-Interest Recommendation via Listwise Bayesian
  Personalized Ranking", IJCAI 2020.
