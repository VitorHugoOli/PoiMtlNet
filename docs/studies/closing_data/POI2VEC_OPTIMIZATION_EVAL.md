# POI2Vec builder — performance optimization eval (quality-first)

> From a multi-agent analysis of the POI2Vec baseline builder (2026-06-23), after the
> **phi vectorization** (`build_poi_routes`, committed `7a6f30e3`) already removed the ~1000×
> bottleneck (O(n_poi·n_leaf) Python loop → numpy, byte-identical). Bar: **speed up WITHOUT
> losing quality** — every item classified byte-identical / training-equivalent / quality-risk.

## Where time goes now (post phi-fix), per CA/TX cell
- **CBOW training — DOMINANT (~25–40 min):** 2.5M examples × 30 epochs, bs=1024, manual loop
  (`build_poi2vec_substrate.py:228`); per-batch `.to(device)` + scatter-heavy `forward_nll`.
- **Next-build (~20 min):** per-user streaming loop over ~37K users
  (`core.py:540` `convert_user_checkins_to_sequences`, called per user from `builders.py`).
- **Setup (~15 s, fold-INDEPENDENT, recomputed every cell):** `build_midpoint_tree` +
  `build_poi_routes` (now fast) + `_build_edge_index` (`model.py:295`, ~8.8M Python appends for CA).

## Quality-SAFE optimizations (byte-identical / training-equivalent)
| change | mechanism | est. speedup | quality | effort |
|---|---|---|---|---|
| **Next-build vectorize/parallelize** (`core.py:614` + the per-user outer loop in `builders.py`) | numpy gather over a `(n_seq,W)` index array; and/or multiprocess the per-user loop | ~20→~5 min/cell | byte-identical (deterministic windowing) | moderate–large; **shared core.py → all engines** |
| **Per-state cache of tree+phi+edge-index** | compute once per state (pre-build), inject into all cells (skip `model.py:295` rebuild) | ~15 s/cell | byte-identical (pure fn of poi_xy/bbox/theta/route_count; not on the exported `poi_embed` path) | moderate |
| DataLoader + pin_memory for CBOW (keep same `randperm` seed) | overlap H2D with compute | ~20–30% on training | byte-identical **iff** per-epoch permutation RNG + batch composition preserved | moderate |
| Vectorize state setup (`poi_xy` `.loc` loop; 64-col assign) | `reindex`/`searchsorted`; `assign` | 15–30× (small abs) | byte-identical | trivial |
| numpy edge-flatten; `scatter_reduce_` | drop-ins | 2–5% | byte-identical | trivial |
| `torch.compile(forward_nll)` | kernel fusion | 15–25% | byte-identical **only if verified** (compile may reorder reductions) | trivial+verify |

## NOT recommended (quality-risk — would change the exported `poi_embed`)
batch-size↑ (alters SGD trajectory over 30 epochs) · fp16/AMP (rounding drift) · fewer epochs
(under-trained) · pre-sampled/fixed negatives (different stochastic gradient) · context-window change
(hyperparameter) · KDTree nearest-leaf fallback (FP-tie routing differences) · grad-accumulation
(Adam eps/bias-correction + per-step negative RNG ≠ bit-identical).

## Decision (2026-06-23)
**For the reduced 10/state validation run (seeds {0,1} × 5 folds, CA+TX, workers=10): run with the
phi-fixed builder as-is — no further builder change.** Rationale: post-phi the cell is ~95%
training+next-build; the cache saves <1% per cell, and at **workers=10 the next-build is already
overlapped across cells**, so the remaining safe optimizations buy ~nothing in wall-clock while
risking a SHARED scientific builder (`core.py`, used by every engine + the already-built CTLE/b2b/Mac
cells). The phi fix was the decisive win.

**Defer to the FULL P3 board launch** (states × {0,1,7,100} × 5 folds ≈ hundreds of cells, where
next-build alone is ~tens of hours): implement, in priority order, (1) **next-build
vectorize/parallelize** (biggest board-scale lever), (2) **per-state tree/phi/edge cache**. Each
gated by the same byte-equivalence check: build one cheap AL cell old-vs-new and assert
`embeddings.parquet` identical at the same seed + cached structures `array_equal`. Keep new cells
consistent with already-built small-state POI2Vec (Mac lane) and the CA/TX cells.
