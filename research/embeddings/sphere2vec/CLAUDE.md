# Sphere2Vec (sphereM)

POI-level location embedding ported faithfully from a Colab notebook
(`Location Encoders/...Sphere2Vec-sphereM.ipynb`).

## Architecture

- **Frozen** spherical-RBF position encoder (256 random unit centroids × 32 log-spaced scales).
- Trained: `Linear(8192→512)` → `MultiLayerFeedForwardNN(512→512→128)` → `Linear(128→64)` → L2-norm.
- Contrastive loss: BCE on cosine similarity, positives = `coord + N(0, 0.01°)`, negatives = random other coord.

## Entry point

`embeddings.sphere2vec.sphere2vec.create_embedding(state, args)` — loads
`IoPaths.get_city(state)`, trains on per-checkin coords (no dedup, faithful to
notebook), forward-passes all checkins, groups by `placeid` (mean) and writes
`output/sphere2vec/{state}/embeddings.parquet` with columns
`[placeid, category, "0"…"63"]`.

## Caveats

- Weak training signal (positives are noise perturbations). Faithful port only — do not "improve" without spec.
- Notebook had no random seeds; this package adds `args.seed` for reproducibility and equivalence testing.
- Equivalence to the notebook is verified by `tests/test_embeddings/test_sphere2vec.py` against the frozen `_sphere2vec_reference.py` snapshot.

## Integration

- Registered as `EmbeddingEngine.SPHERE2VEC` (value `"sphere2vec"`) in `src/configs/paths.py`.
- POI-level: validated via `src/configs/embedding_fusion.py`.
- Downstream input generation works automatically via `generate_category_input` and `generate_next_input_from_poi`.
