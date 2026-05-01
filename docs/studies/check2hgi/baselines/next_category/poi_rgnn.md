# POI-RGNN

## Source
- **Paper:** Capanema, de Oliveira, Silva, Silva, Loureiro. *Combining recurrent and Graph Neural Networks to predict the next place's category.* Ad Hoc Networks 138 (2023) 103016. [doi:10.1016/j.adhoc.2022.103016](https://doi.org/10.1016/j.adhoc.2022.103016).
- **Reference impl (Python port we cross-checked against):** `/Users/vitor/Desktop/mestrado/mtl_poi/src/{etl,models,train}/rgnn/`.
- **Original (TS):** `/Users/vitor/Desktop/mestrado/POI_Models/fork/poi_detection/`.
- **Architecture (paper §4):** Multi-modal embeddings (category + hour-of-week + distance + duration + dist×dur) → GRU(35) + 4-head Keras-style attention (key_dim=2) for the temporal branch; three parallel 2-layer Dense-GCN stacks over the 7×7 category transition graph (nodes weighted by median distance, median duration, dist×dur). Four logits heads (`y_sup`, `spatial`, `gnn`, `gnn_puro`) combined via entropy-weighted learnable scalars (`CombinePredictionsLayer`).

## Why this is a baseline (not our model)
External published-method reference for the next-category task. POI-RGNN is the canonical RNN+GNN combination for next-category prediction on Gowalla. We use it to:

1. Establish a **literature ceiling** for the 7-class next-category task using the paper architecture trained from raw inputs (no pre-trained substrate).
2. Anchor the floor / Markov / our STL substrate variants against a published method that combines temporal sequence modelling with a category-transition graph.

A future `stl_check2hgi` / `stl_hgi` POI-RGNN variant could replace the category-id input with a substrate embedding, but is **not implemented yet** (per the user's scoping decision: "don't need to create a stl head for now").

## What's faithful, what's adapted

### Faithful to paper
- Multi-modal embeddings: 7×7 category, 49×3 hour, 52×3 distance, 50×3 duration; weighted dist×dur multiplier (`MultiplyByWeight`, init 0.1).
- GRU(35) + Keras-style multi-head attention (4 heads, key_dim=2 — implemented via `_TFStyleMHA` to match the asymmetric Keras Q/K/V projection that PyTorch's `nn.MultiheadAttention` cannot express).
- Three parallel `_DenseGCN` branches (in→22→10) over the 7×7 graph; swish activations; no internal normalization (adjacency pre-normalized D^(-1/2)(A+I)D^(-1/2)).
- Four logits heads + `_CombinePredictions` entropy-weighted fusion with all 9 trainable scalars at the paper's initial values (entropy_w=0.5, base_w=1.0, spatial=−0.2, gnn=8.0, gnn_puro=8.0).
- Adam(lr=1e-3, betas=(0.8, 0.9), eps=1e-7), batch 400, 35 epochs, `ReduceLROnPlateau(patience=3, factor=0.5)` on val F1, early stop patience 10.
- Skip consecutive-duplicate POIs (`if placeid[i] == placeid[i-1]: continue` in ETL).
- Distance capped at 50 km, duration at 48 h.
- Hour token = `hour + (24 if weekend else 0)`.

### Adapted because our task / data differ
- **Cross-validation protocol.** Reference uses per-user `KFold(shuffle=False)` (warm-user, in-user temporal). We use `StratifiedGroupKFold(5, seed=42)` on userid stratified by target_category — matches the rest of our baselines (cold-user; train and val users are disjoint). This makes results comparable to STAN, ReHDM, GETNext, etc.
- **Window strategy.** Non-overlapping windows of size 9 + 1-step target. Reference uses overlapping prefix-expansion. We match our in-house pipeline so cross-method comparisons stay apples-to-apples.
- **Per-fold global graph matrices.** `adj`, `cat_dist`, `cat_dur` are computed **once per fold on training rows only** and passed as `[7, 7]` constants. Reference computes per-user matrices and tiles them per-window. With 7 categories, per-user graphs are sparse and unstable; population-level transition statistics retain the paper's signal while avoiding leakage and complexity. Documented in code as `Adapted`.
- **Loss.** `CrossEntropyLoss` over 7-class logits (closed-set classification, no class weights — reference uses optional CLASS_WEIGHTS, we leave them off for parity with other baselines).
- **PAD index handling.** Hour / distance / duration token tables are sized +1 over the paper to reserve index 0 as PAD; real values shift by +1.
- **Model variants.** Only `faithful` for now — no `stl_check2hgi` / `stl_hgi` head yet (out of scope per current request).

## Variants we run

| Variant | Inputs | Output | Where |
|---|---|---|---|
| `faithful` | raw category id + hour-of-week + distance/duration buckets | linear → 7-class logits via entropy-weighted fusion | `research/baselines/poi_rgnn/` |

## Reproduction commands

```bash
PY=/Users/vitor/Desktop/mestrado/ingred/.venv/bin/python
ENV='PYTHONPATH=src DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data
     OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output
     PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1'

# ETL — one pass per state
caffeinate -i env $ENV "$PY" -m research.baselines.poi_rgnn.etl --state alabama

# Train — 5f x 35ep
caffeinate -i env $ENV "$PY" -m research.baselines.poi_rgnn.train \
    --state alabama --folds 5 --epochs 35 \
    --tag FAITHFUL_POIRGNN_al_5f35ep
```

Wall time: AL 5f x 35ep ~70 s on M4 Pro MPS (early-stop kicks in around epoch 20–25).

## Source JSONs

| Variant | State | JSON |
|---|---|---|
| `faithful` | AL | `docs/studies/check2hgi/results/baselines/faithful_poi_rgnn_alabama_5f_35ep_FAITHFUL_POIRGNN_al_5f35ep.json` |

## Cross-references

- Aggregated metrics by state: `results/{alabama,...}.json`.
- Reference implementation we cross-checked against: `/Users/vitor/Desktop/mestrado/mtl_poi/src/{etl,models,train}/rgnn/`.
