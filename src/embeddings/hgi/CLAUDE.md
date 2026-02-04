# HGI Module - Agent Reference

## What This Module Does

Generates **POI and region embeddings** from spatial check-in data using Hierarchical Graph Infomax. Three-level hierarchy: POI → Region → City.

## Entry Points

| Function | File | Purpose |
|----------|------|---------|
| `create_embedding(state, args)` | `hgi.py` | Full pipeline (preprocess + POI2Vec + train) |
| `train_hgi(city, args)` | `hgi.py` | Train HGI only (requires existing `graph_data.pkl`) |
| `train_poi2vec(city, ...)` | `poi2vec.py` | Generate POI embeddings only |
| `preprocess_hgi(city, shapefile, poi_emb_path)` | `preprocess.py` | Build graph structure |

## Critical Concept: fclass-Level Embeddings

**POI2Vec learns embeddings at the fclass (functional class) level, NOT at the POI level.**

```
POI2Vec output: fclass_embeddings[vocab_size, dim]

Reconstruction (Phase 3d):
  poi_embedding[i] = fclass_embeddings[poi.fclass[i]]

Result: Multiple POIs with same fclass share IDENTICAL embeddings
```

## Pipeline Phases

```
Phase 3a: preprocess_hgi(poi_emb_path=None)
          → edges.csv, pois.csv (Delaunay graph)

Phase 3b-3d: train_poi2vec()
             → poi_embeddings.pt (fclass-level!)

Phase 4: preprocess_hgi(poi_emb_path=<path>)
         → graph_data.pkl (complete data dict)

Phase 5: train_hgi()
         → embeddings.parquet, region_embeddings.parquet
```

**Important**: `preprocess_hgi` is called TWICE - first to build graph, second to add embeddings.

## File Map

```
src/embeddings/hgi/
├── hgi.py           # create_embedding(), train_hgi()
├── poi2vec.py       # train_poi2vec(), POI2Vec, EmbeddingModel
├── preprocess.py    # preprocess_hgi(), HGIPreprocess
├── utils.py         # haversine, bbox diagonal
└── model/
    ├── HGIModule.py     # HierarchicalGraphInfomax, corruption()
    ├── POIEncoder.py    # GCN-based POI encoding
    ├── RegionEncoder.py # PMA + Region GCN (POI2Region)
    └── SetTransformer.py # MAB, SAB, PMA
```

## Common Tasks

### Run full HGI pipeline
```python
from embeddings.hgi.hgi import create_embedding
from argparse import Namespace

args = Namespace(
    dim=64, epoch=2000, poi2vec_epochs=100,
    alpha=0.5, attention_head=4, lr=0.001,
    gamma=1.0, max_norm=0.9, device='cpu',
    shapefile='/path/to/tracts.shp', force_preprocess=True
)
create_embedding(state="Texas", args=args)
```

### Train HGI only (preprocessing already done)
```python
from embeddings.hgi.hgi import train_hgi
train_hgi(city="Texas", args=args)
```

### Generate POI2Vec embeddings only
```python
from embeddings.hgi.poi2vec import train_poi2vec
poi_emb_path = train_poi2vec(city="Texas", epochs=100, embedding_dim=64)
```

## Key Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `dim` | 64 | Embedding dimension (match downstream model) |
| `epoch` | 2000 | HGI training epochs (reduce for testing) |
| `poi2vec_epochs` | 100 | POI2Vec epochs (reduce for testing) |
| `alpha` | 0.5 | Loss balance: α×L_poi2region + (1-α)×L_region2city |
| `attention_head` | 4 | PMA attention heads |
| `force_preprocess` | True | Regenerate graph even if exists |

## Output Files

| File | Location | Content |
|------|----------|---------|
| `edges.csv` | `temp/` | Delaunay graph edges |
| `pois.csv` | `temp/` | POI metadata (placeid, category, fclass) |
| `graph_data.pkl` | `temp/` | Complete data dict for training |
| `embeddings.parquet` | `output/` | Final POI embeddings [N_pois, dim] |
| `region_embeddings.parquet` | `output/` | Final region embeddings [N_regions, dim] |

## Gotchas

1. **POI2Vec is fclass-level**: Don't expect unique embeddings per POI
2. **Two preprocess calls**: First without embeddings, second with
3. **Shapefile requirement**: Must have `GEOID` column for census tracts
4. **Edge weights**: Combine spatial distance AND regional penalties (same region = 1.0, different = 0.5)
5. **Hard negatives in HGI**: 25% of negatives are "hard" (similarity 0.6-0.8)

## Data Dict Structure (`graph_data.pkl`)

```python
{
    'node_features': [N_pois, dim],      # POI2Vec embeddings
    'edge_index': [2, N_edges],          # COO format
    'edge_weight': [N_edges],            # Normalized [0,1]
    'region_id': [N_pois],               # POI → region mapping
    'region_area': [N_regions],          # km²
    'region_adjacency': [2, N_adj],      # Adjacent region pairs
    'coarse_region_similarity': [N_regions, N_regions],  # Cosine sim
    'y': [N_pois],                       # Category labels
    'place_id': [N_pois],                # Original placeids
    'category_classes': list,            # Decoder
    'fclass_classes': list,              # Decoder
}
```
