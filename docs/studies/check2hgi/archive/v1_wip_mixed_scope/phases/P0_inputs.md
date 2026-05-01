# Phase P0 — next_region label derivation pipeline

**Gates:** P-1 complete (`checkin_graph.pt` exists per state, uncorrupted).
**Exit gate:** `output/check2hgi/{state}/input/next_region.parquet` exists for FL and AL, unmapped-placeid count is zero (or exhaustively documented), region cardinality logged to `HANDOFF.md`.

## Purpose

Turn the check-in-level next-POI sequence parquet (produced by the embedding pipeline) into a next-region-labeled parquet consumed by the downstream `FoldCreator` path.

## Pipeline

```
sequences_next.parquet (rows: [x_0..x_N, next_placeid, userid])
  ⊕ checkin_graph.pt :: poi_to_region   (placeid → region_idx)
  ⊕ placeid_map (from preprocessing)
  → next_region.parquet (rows: [x_0..x_N, region_idx, userid])
```

## Implementation

- New file: `src/data/inputs/next_region.py` with `load_next_region_data(state, engine)` function.
- New pipeline: `pipelines/create_inputs_check2hgi.pipe.py` with `generate_next_region_input(state)`.
- New path helper: `IoPaths.get_next_region(state, engine)` in `src/configs/paths.py`.

## Label derivation code sketch

```python
import pandas as pd
import torch

graph = torch.load(checkin_graph_path, weights_only=False)
poi_to_region = graph['poi_to_region']  # tensor[n_pois]
placeid_map   = graph['placeid_map']    # dict placeid → poi_idx

seq = pd.read_parquet(sequences_next_path)
seq['poi_idx']    = seq['next_placeid'].map(placeid_map)

# Fail loud on unmapped placeids — do not silently drop.
missing = seq['poi_idx'].isna().sum()
if missing > 0:
    sample = seq[seq['poi_idx'].isna()]['next_placeid'].head(20).tolist()
    raise ValueError(f"{missing} unmapped placeids; sample: {sample}")

seq['region_idx'] = poi_to_region[seq['poi_idx'].astype(int).values].numpy()
seq = seq.drop(columns=['poi_idx', 'next_placeid'])
seq.to_parquet(output_path)
```

## Integrity checks

1. `len(next_region.parquet) == len(sequences_next.parquet)`.
2. `0 <= region_idx < n_regions` for all rows.
3. No NaN in `region_idx`.
4. Row-level `userid` preserved (so the StratifiedGroupKFold user-partition works).
5. `x_0..x_N` feature columns byte-identical to source — the X tensor is shared across next_category and next_region tasks.

## Claims touched

Sets up the data for CH04 (next-region meaningfulness) — cardinality number goes into the claim statement.

## Failure modes & recovery

- **Unmapped placeids:** indicates placeid_map mismatch between sequences_next and checkin_graph. Rerun P-1 preprocessing to regenerate both consistently.
- **Region cardinality < 7** for a state: region shapefile is too coarse; not expected with census tracts but worth an assert.
