# HGI Performance Improvement Plan

Date: 2026-04-13

## Goal

Improve the performance of the HGI embedding pipeline without hurting embedding
quality. The priority is to preserve the HGI algorithm and downstream metrics,
then remove avoidable implementation overhead.

## Context

HGI follows Huang et al. (2023), "Learning urban region representations with
POIs and hierarchical graph infomax":

1. Train POI/category initial features.
2. Build a POI graph and run POI-level graph convolution.
3. Aggregate POIs to regions with multi-head attention.
4. Run region-level graph convolution.
5. Aggregate regions to a city embedding.
6. Train by maximizing mutual information across the POI-region-city hierarchy.

The local implementation is in:

- `pipelines/embedding/hgi.pipe.py`
- `research/embeddings/hgi/preprocess.py`
- `research/embeddings/hgi/poi2vec.py`
- `research/embeddings/hgi/hgi.py`
- `research/embeddings/hgi/model/HGIModule.py`
- `research/embeddings/hgi/model/RegionEncoder.py`

The implementation already includes several quality-preserving performance
fixes relative to the original RightBank/HGI code:

- Cached hard-negative candidate sets.
- Vectorized negative pair construction.
- Vectorized POI-to-region PMA aggregation.
- CPU thread pinning for HGI training.
- CPU execution for HGI on Apple Silicon, avoiding MPS overhead.

## Current Validation

HGI tests passed locally:

```bash
python -m pytest tests/test_embeddings/test_hgi_perf_regression.py \
                 tests/test_embeddings/test_hgi.py -q
```

Result:

```text
37 passed
```

Phase 5 HGI training is already fast on Alabama:

```text
20 epochs, CPU
34.3 ms/epoch
forward:    13.68 ms
loss:        1.74 ms
backward:   17.82 ms
optimizer:   0.84 ms
```

The remaining major cost is POI2Vec and repeated preprocessing, not the HGI
training loop itself.

## Measured Bottlenecks

### 1. POI2Vec Walk Conversion

Current code converts generated POI walks to fclass walks with `pandas.iloc`
inside the innermost loop:

```python
fclass_walk = [int(self.pois.iloc[poi_idx]["fclass"]) for poi_idx in poi_walk]
```

Benchmark on identical Alabama walks:

```text
batches=20 walks=76800 values=384000
current pandas iloc conversion: 3.5511s
vectorized numpy conversion:   0.0081s
speedup: 437.6x
```

This is the strongest safe optimization because it changes only the lookup
method, not the walks, labels, loss, negative sampling, or model.

### 2. Repeated Graph Preprocessing

The pipeline runs `preprocess_hgi()` twice:

1. Before POI2Vec, to generate `edges.csv` and `pois.csv`.
2. After POI2Vec, to rebuild graph data with learned node features.

The second call recomputes spatial join, Delaunay graph, edge weights, encoders,
region features, and adjacency. Much of that could be cached after the first
call and reused exactly.

### 3. Dense Region Similarity Matrix

HGI only uses `coarse_region_similarity` to choose hard-negative candidates with
similarity in `(0.6, 0.8)`. Storing a full dense region-by-region matrix is more
expensive than storing exact per-region candidate lists.

### 4. Profiling Script Issues

The profiling scripts currently need `PYTHONPATH=src:research` when run from the
repo root. Their internal root calculation points to `experiments/`, not the repo
root.

`experiments/scripts/profile_poi2vec_alabama.py` also has a reporting units bug:
it stores per-batch times in seconds but names the variable `per_batch_ms` and
divides by `1000`.

## Recommended Plan

### Phase 1: Exact POI2Vec Walk Conversion Speedup

Implement a vectorized fclass lookup in `POI2Vec.generate_walks()`.

Current approach:

```python
for walk in pos_rw:
    poi_walk = walk.tolist()
    fclass_walk = [int(self.pois.iloc[poi_idx]["fclass"]) for poi_idx in poi_walk]
    self.fclass_walks.append(fclass_walk)
```

Recommended approach:

```python
fclass_values = self.pois["fclass"].to_numpy(dtype=np.int64)
for pos_rw, _ in loader:
    fclass_batch = fclass_values[pos_rw.cpu().numpy()]
    self.fclass_walks.extend(fclass_batch.tolist())
```

Also remove the unused `poi_walks` list.

Expected quality impact: none. The generated walks and fclass labels are
identical.

Validation:

```bash
python -m pytest tests/test_embeddings/test_hgi.py \
                 tests/test_embeddings/test_hgi_perf_regression.py -q
PYTHONPATH=src:research python experiments/scripts/profile_poi2vec_alabama.py
```

For stronger validation, run one full Alabama HGI regeneration and compare
downstream category and next-task metrics against the current artifact.

### Phase 2: Fix Profiling Scripts

Fix `_root` in both profiling scripts:

- `experiments/scripts/profile_hgi_alabama.py`
- `experiments/scripts/profile_poi2vec_alabama.py`

The root should be `Path(__file__).resolve().parents[2]`.

Fix the POI2Vec profile units:

```python
per_batch_s = sum(statistics.mean(v) for v in phase_times.values())
per_epoch_train_s = per_batch_s * n_batches
```

Expected quality impact: none. This only improves measurement reliability.

### Phase 3: Cache Phase 3a Graph Artifacts

Split preprocessing into two responsibilities:

1. Build and persist graph/static region artifacts.
2. Load POI2Vec embeddings into the existing graph and write `gowalla.pt`.

Cache key should include:

- Checkins parquet fingerprint.
- Shapefile or generated `boroughs_area.csv` fingerprint.
- `cross_region_weight`.
- Preprocessing code version or explicit schema version.

Expected quality impact: none if the cached graph is byte-identical to the
recomputed graph.

Validation:

```bash
PYTHONPATH=src:research python - <<'PY'
# Compare old recomputed graph_data and new cached graph_data:
# - edge_index equal
# - edge_weight allclose
# - region_id equal
# - region_area allclose
# - coarse_region_similarity allclose or candidate lists equivalent
# - y/place_id/category_classes/fclass_classes equal
PY
```

### Phase 4: Replace Dense Similarity Matrix With Candidate Lists

During preprocessing, compute and store:

- `hard_negative_candidates_per_region`
- `all_other_regions_per_region`, or enough metadata to sample all non-self
  regions efficiently.

Then adapt `HGIModule.forward()` to use these lists directly instead of building
them from `coarse_region_similarity` on first forward.

Expected quality impact: none if the candidate lists are exactly equivalent to:

```python
(similarity > 0.6) & (similarity < 0.8)
```

Validation:

- Add a unit test comparing candidate lists generated from the dense matrix and
  the new precomputed lists.
- Keep `test_hgi_reference_equivalence.py` passing where applicable, or add a
  separate equivalence test for the new storage format.

### Phase 5: Only After That, Test Training-Reduction Ideas

These can improve runtime but may affect quality, so they require downstream
metric validation:

- Reduce `poi2vec_epochs` from `100` to `75`, `50`, or `25`.
- Larger POI2Vec batch sizes.
- Alternative negative-sampling implementations.
- Early stopping on POI2Vec loss.

Suggested sweep:

```text
poi2vec_epochs: 25, 50, 75, 100
states: Alabama first, then one large state
metrics: downstream Cat F1, Cat Acc, Next F1, Next Acc
protocol: same folds, same MTL config, same random seed where possible
```

Do not change the production default until the quality delta is inside noise.

## Risks

### Low Risk

- Vectorized fclass lookup.
- Removing unused `poi_walks`.
- Profiling script fixes.
- Graph cache, if exact equality is tested.

### Medium Risk

- Precomputed hard-negative candidate lists, because it touches model input
  schema and reference-equivalence tests may need adjustment.

### High Risk

- Reducing POI2Vec epochs.
- Changing random-walk parameters.
- Changing negative-sampling behavior.
- Changing HGI epoch count or optimizer settings.

## Recommended Implementation Order

1. Vectorize POI2Vec fclass walk conversion.
2. Fix profiling scripts and rerun profiles.
3. Add graph-cache refactor with exact artifact equality tests.
4. Replace dense similarity storage with exact candidate lists.
5. Run POI2Vec epoch and batch-size sweeps only after the implementation-only
   speedups are complete.

## Success Criteria

Performance:

- Lower POI2Vec walk-conversion wall time.
- Lower full HGI regeneration time.
- HGI phase 5 remains under the existing perf-regression threshold.

Quality:

- HGI unit and perf tests pass.
- Generated graph artifacts are equal or numerically equivalent.
- Downstream Cat F1 and Next F1 do not regress beyond fold noise.

