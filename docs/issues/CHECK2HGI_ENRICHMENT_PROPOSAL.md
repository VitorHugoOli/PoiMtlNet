# Check2HGI Enrichment Proposal (Time + Space + Extra Features)

## Goal

Improve Check2HGI representations by enriching:

1. temporal encoding (Time2Vec-like ideas),
2. spatial encoding (Sphere2Vec-like ideas),
3. graph connectivity and training objectives.

---

## 1. Temporal Enrichment (Time2Vec-inspired)

Current Check2HGI uses fixed temporal features (`sin/cos` of hour and day-of-week).  
Proposed upgrades:

- Replace fixed 4D temporal input with **learnable multi-frequency time embedding**.
- Add **time-gap** to previous check-in (`delta_t_prev`).
- Add **dwell-time proxy** (time between current and next/previous event where available).
- Add **recency decay feature** (e.g., exponential decay from sequence origin).

Suggested insertion point:

- `research/embeddings/check2hgi/preprocess.py` (`_build_node_features`)

---

## 2. Spatial Enrichment (Sphere2Vec-inspired)

Current Check2HGI mostly uses region assignment and graph edges; limited continuous geometry in node features.

Proposed upgrades:

- Add **continuous geospatial positional encoding** from `(lat, lon)` (spherical/geodesic basis).
- Add **distance-to-previous-POI** feature.
- Add **distance-to-user-centroid** (mobility anchor).
- Add **distance-to-region-centroid**.
- Optional: add Fourier/RBF spatial basis features.

Suggested insertion point:

- `research/embeddings/check2hgi/preprocess.py` (`_build_node_features`)

---

## 3. Graph Construction Upgrades

Current options: `user_sequence`, `same_poi`, `both`.

Proposed additional edge types:

- **KNN spatial edges** among nearby check-ins/POIs.
- **Temporal-window co-occurrence edges** (events within a short window).
- **Revisit-strength edges** (same POI with time-aware weighting).
- Keep weighted fusion of edge families with tunable coefficients.

Suggested insertion points:

- `research/embeddings/check2hgi/preprocess.py`
- new edge builders and a combined edge policy switch.

---

## 4. Objective/Loss Enhancements

Current loss: hierarchical MI across Check-in↔POI, POI↔Region, Region↔City.

Proposed additions:

- **Multi-view contrastive consistency**:
  - temporal view vs spatial view vs structural view.
- **Hard negatives**:
  - same region, close time slot, or similar mobility profile.
- **Auxiliary pretext tasks**:
  - next-region prediction,
  - next-time-gap prediction,
  - masked check-in feature reconstruction.

Suggested insertion points:

- `research/embeddings/check2hgi/model/Check2HGIModule.py`
- `research/embeddings/check2hgi/check2hgi.py` (training loop + weights).

---

## 5. Candidate Extra Features

High-value, low-complexity candidates:

- POI popularity count / log-frequency.
- User mobility dispersion (e.g., radius of gyration proxy).
- Transition entropy per user (routine vs exploration).
- Region context features:
  - region density,
  - category distribution entropy,
  - centrality in region adjacency graph.

---

## 6. Recommended Phased Rollout

### Phase 1 (low risk)

- Add new temporal + spatial node features only.
- Keep graph/loss unchanged.

### Phase 2

- Add one new edge family (spatial KNN).
- Run ablation vs baseline.

### Phase 3

- Add one auxiliary task (`next_region` recommended).
- Tune loss weights.

### Phase 4 (optional)

- Add multi-view contrastive consistency + hard negatives.

---

## 7. Evaluation Protocol Notes

To avoid inflated gains:

- Split before embedding training (inductive protocol where required).
- Keep a strict ablation table:
  - baseline,
  - +temporal features,
  - +spatial features,
  - +edges,
  - +auxiliary losses.
- Report both quality and cost (time/memory).

---

## 8. Immediate Next Step

Implement Phase 1 in `preprocess.py` with a config flag:

- `temporal_feature_mode = {basic, time2vec_like}`
- `spatial_feature_mode = {none, geo_basis}`

This keeps backward compatibility and enables clean ablations.
