# HGI (Hierarchical Graph Infomax) Embedding Module

This module implements the complete HGI pipeline for generating POI and region embeddings from spatial check-in data.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Flow](#pipeline-flow)
3. [Intermediate Files](#intermediate-files)
4. [Model Architectures](#model-architectures)
5. [Theory & Key Insights](#theory--key-insights)
6. [Usage](#usage)
7. [Configuration Reference](#configuration-reference)
8. [File Structure](#file-structure)

---

## Overview

### What is HGI?

**Hierarchical Graph Infomax (HGI)** is a self-supervised learning method that generates embeddings at multiple spatial granularities:

- **POI Level**: Individual points of interest (restaurants, shops, etc.)
- **Region Level**: Census tracts or administrative areas
- **City Level**: Global summary of the entire area

### What Problem Does It Solve?

HGI learns spatial representations that capture:
- **Local patterns**: Which POIs are spatially close and functionally similar
- **Regional patterns**: What characterizes each census tract
- **Global patterns**: City-wide functional organization

These embeddings are used as input features for downstream tasks like:
- Next-POI prediction
- POI category classification
- Region similarity analysis

### Three-Level Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                        CITY LEVEL                           │
│     Area-weighted aggregation of all region embeddings      │
│                    city_emb = sigmoid(sum(region * area))   │
└─────────────────────────────────────────────────────────────┘
                              ^
┌─────────────────────────────────────────────────────────────┐
│                       REGION LEVEL                          │
│      Attention-pooled POIs + Region GCN refinement          │
│            region_emb = PMA(POIs) + GCN(regions)            │
└─────────────────────────────────────────────────────────────┘
                              ^
┌─────────────────────────────────────────────────────────────┐
│                        POI LEVEL                            │
│         GCN encoding over Delaunay spatial graph            │
│              poi_emb = GCN(features, edges)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

The HGI pipeline consists of 5 phases that run sequentially:

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3a: Spatial Graph Construction                       │
│  preprocess_hgi(poi_emb_path=None)                          │
│  ├── Load POI data + census tract shapefile                 │
│  ├── Delaunay triangulation (connect nearby POIs)           │
│  ├── Compute edge weights (spatial x regional)              │
│  └── Output: edges.csv, pois.csv, encodings.json            │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3b-3d: POI2Vec Pre-training                          │
│  train_poi2vec()                                            │
│  ├── 3b: Generate random walks on POI graph                 │
│  ├── 3c: Train FCLASS embeddings (skip-gram + hierarchy)    │
│  ├── 3d: Reconstruct POI embeddings from fclass lookup      │
│  └── Output: poi_embeddings.pt                              │
│                                                             │
│  *** CRITICAL INSIGHT ***                                   │
│  POI2Vec learns at FCLASS level, NOT POI level!             │
│  Multiple POIs with same fclass share identical embeddings  │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: Graph Enrichment                                  │
│  preprocess_hgi(poi_emb_path=poi_embeddings.pt)             │
│  ├── Reload graph structure                                 │
│  ├── Load POI2Vec embeddings as node features               │
│  ├── Compute region features (area, adjacency, similarity)  │
│  └── Output: graph_data.pkl                                 │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: HGI Training                                      │
│  train_hgi()                                                │
│  ├── Initialize POIEncoder, POI2Region, region2city         │
│  ├── Train with mutual information maximization loss        │
│  ├── Track best embeddings (lowest loss)                    │
│  └── Output: embeddings.parquet, region_embeddings.parquet  │
└─────────────────────────────────────────────────────────────┘
```

---

## Intermediate Files

### Temporary Files (`output/hgi/{state}/temp/`)

| File | Columns/Keys | Shape | Purpose |
|------|--------------|-------|---------|
| `edges.csv` | source, target, weight | [N_edges, 3] | Delaunay graph for Node2Vec walks |
| `pois.csv` | placeid, category, fclass | [N_pois, 3] | POI metadata for fclass lookup |
| `encodings.json` | category: {name->code}, fclass: {name->code} | dict | Reverse mapping for decoding |
| `graph_data.pkl` | node_features, edge_index, edge_weight, region_id, region_area, coarse_region_similarity, region_adjacency, y, place_id, category_classes, fclass_classes | dict | Complete graph state for HGI training |

### Output Files (`output/hgi/{state}/`)

| File | Columns | Shape | Purpose |
|------|---------|-------|---------|
| `embeddings.parquet` | placeid, 0, 1, ..., dim-1, category | [N_pois, dim+2] | **Final POI embeddings** |
| `region_embeddings.parquet` | region_id, reg_0, ..., reg_(dim-1) | [N_regions, dim+1] | **Final region embeddings** |
| `poi_embeddings.pt` | in_embed.weight, placeids | tensor + list | POI2Vec output (fclass->POI mapped) |

### Key File: `graph_data.pkl`

This pickle contains the complete graph representation:

```python
{
    'node_features': np.ndarray,           # [N_pois, feature_dim] - POI2Vec embeddings
    'edge_index': np.ndarray,              # [2, N_edges] - COO sparse format
    'edge_weight': np.ndarray,             # [N_edges] - normalized to [0,1]
    'region_id': np.ndarray,               # [N_pois] - maps POI -> region index
    'region_area': np.ndarray,             # [N_regions] - area in km^2
    'region_adjacency': np.ndarray,        # [2, N_adj] - which regions touch
    'coarse_region_similarity': np.ndarray,# [N_regions, N_regions] - cosine similarity
    'y': np.ndarray,                       # [N_pois] - category labels (encoded)
    'place_id': np.ndarray,                # [N_pois] - original placeid values
    'category_classes': list,              # Category names for decoding
    'fclass_classes': list,                # Fclass names for decoding
    'number_pois': int,
    'number_regions': int,
}
```

---

## Model Architectures

### 1. POI2Vec (`poi2vec.py`)

**Purpose**: Pre-train POI embeddings using spatial random walks.

#### Critical Insight: fclass-Level Learning

**POI2Vec learns embeddings at the fclass (functional class) level, NOT at the POI level.** This is the most important concept to understand:

```
┌─────────────────────────────────────────────────────────────┐
│                    POI2Vec OUTPUT                           │
│                                                             │
│   What you might expect:                                    │
│   POI_1 -> embedding_1                                      │
│   POI_2 -> embedding_2                                      │
│   POI_3 -> embedding_3                                      │
│                                                             │
│   What actually happens:                                    │
│   fclass "Coffee Shop" -> embedding_A                       │
│   fclass "Restaurant"  -> embedding_B                       │
│   fclass "Gym"         -> embedding_C                       │
│                                                             │
│   Then reconstruction:                                      │
│   POI_1 (fclass="Coffee Shop") -> embedding_A               │
│   POI_2 (fclass="Coffee Shop") -> embedding_A  (SAME!)      │
│   POI_3 (fclass="Restaurant")  -> embedding_B               │
└─────────────────────────────────────────────────────────────┘
```

**Why fclass-level?**
- **Stability**: More training data per fclass (many POIs per fclass)
- **Generalization**: Prevents overfitting to individual POI quirks
- **Semantic structure**: Hierarchy loss enforces category-fclass relationships

#### POI2Vec Phases

```
PHASE 3b: Random Walk Generation
┌─────────────────────────────────────────────────────────────┐
│  Delaunay Graph --> Node2Vec Walks --> fclass Sequences     │
│                                                             │
│  POI walk:    [POI_1, POI_3, POI_7, POI_2, ...]            │
│                   |       |       |       |                 │
│                   v       v       v       v                 │
│  fclass walk: [cafe, restaurant, bar, cafe, ...]           │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
PHASE 3c: fclass Embedding Training
┌─────────────────────────────────────────────────────────────┐
│  EmbeddingModel:                                            │
│    in_embed:  [vocab_size, embed_dim]  # vocab = num fclass │
│    out_embed: [vocab_size, embed_dim]                       │
│                                                             │
│  Loss = Skip-gram + lambda * Hierarchy                      │
│                                                             │
│  Skip-gram:  maximize P(context | center)                   │
│              -log sigma(center . context)                   │
│              -log sigma(-center . negative)                 │
│                                                             │
│  Hierarchy:  L2(category_emb - fclass_emb)                  │
│              Forces related fclasses to cluster by category │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
PHASE 3d: POI-Level Reconstruction
┌─────────────────────────────────────────────────────────────┐
│  For each POI i:                                            │
│      poi_embedding[i] = fclass_embeddings[poi.fclass[i]]    │
│                                                             │
│  This is a LOOKUP TABLE operation!                          │
│  NO learning happens here - just index mapping.             │
│                                                             │
│  Result: POIs with same fclass share IDENTICAL embeddings   │
└─────────────────────────────────────────────────────────────┘
```

#### Hard Negative Sampling

Instead of random negatives, POI2Vec samples fclasses that **never** co-occur with the center fclass in any walk:

```python
# For center fclass "Coffee Shop":
positive_context = {"Restaurant", "Bar", "Gym"}  # Seen together in walks
negative_candidates = ALL_FCLASSES - positive_context - {"Coffee Shop"}
# These are fclasses that NEVER appeared near Coffee Shop
# Learning to distinguish them is harder but more meaningful
```

---

### 2. POIEncoder (`model/POIEncoder.py`)

**Purpose**: Encode POIs using graph convolution over the Delaunay spatial graph.

**Architecture**:
```
Input: x [N_pois, in_channels], edge_index [2, E], edge_weight [E]
                              |
                              v
                    GCNConv(in -> hidden)
                              |
                              v
                         PReLU activation
                              |
                              v
Output: [N_pois, hidden_channels]
```

**Why GCN?**
- Respects spatial structure (nearby POIs influence each other)
- Edge weights encode both distance and region boundaries
- Single layer captures 1-hop neighborhood information

**Code**:
```python
class POIEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        self.conv = GCNConv(in_channels, hidden_channels)
        self.prelu = nn.PReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x
```

---

### 3. POI2Region (`model/RegionEncoder.py`)

**Purpose**: Aggregate POI embeddings into region embeddings using attention, then refine with region-level GCN.

**Two-Stage Architecture**:

```
Stage 1: POI-to-Region Aggregation (PMA)
┌─────────────────────────────────────────────────────────────┐
│  For each region r:                                         │
│    1. Extract POIs: pois_r = x[region_id == r]              │
│    2. Learnable seed query: S [1, dim]                      │
│    3. Multihead Attention: region_r = Attention(S, pois_r)  │
│                                                             │
│  PMA learns WHICH POIs matter for summarizing each region   │
│                                                             │
│  Attention formula:                                         │
│    score = softmax(Q . K^T / sqrt(d))                       │
│    region_emb = score . V                                   │
│                                                             │
│  Segmented softmax: normalizes WITHIN each region           │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
Stage 2: Region-Level GCN
┌─────────────────────────────────────────────────────────────┐
│  Input: region_emb [N_regions, dim]                         │
│         region_adjacency [2, N_adj]                         │
│                                                             │
│  GCNConv + PReLU                                            │
│                                                             │
│  Region GCN captures WHICH REGIONS are similar              │
│  Adjacent regions influence each other's embeddings         │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
Output: [N_regions, hidden_channels]
```

**Why Two Stages?**
- **PMA**: Learns attention weights to identify important POIs per region
- **Region GCN**: Incorporates regional context (adjacent regions influence each other)

---

### 4. HGI Module (`model/HGIModule.py`)

**Purpose**: Learn embeddings by maximizing mutual information across the three-level hierarchy.

**Forward Pass**:
```
Input: Graph data with POI features, edges, region assignments
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  1. POI Encoding                                            │
│     pos_poi = POIEncoder(x, edges)        # Real features   │
│     neg_poi = POIEncoder(corrupt(x), edges) # Shuffled      │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  2. Region Aggregation                                      │
│     region_emb = POI2Region(pos_poi)                        │
│     neg_region = POI2Region(neg_poi)                        │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  3. City Encoding                                           │
│     city_emb = sigmoid(sum(region_emb * region_area))       │
└─────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────┐
│  4. Hard Negative Sampling for Loss                         │
│     25%: regions with similarity 0.6-0.8 (hard negatives)   │
│     75%: random regions (easy negatives)                    │
└─────────────────────────────────────────────────────────────┘
```

**Loss Function** (Mutual Information Maximization):
```
L_poi2region = -log sigma(poi @ W @ same_region)      # Positive
             - log(1 - sigma(poi @ W @ other_region)) # Negative

L_region2city = -log sigma(region @ W @ city)         # Positive
              - log(1 - sigma(neg_region @ W @ city)) # Negative

Total Loss = alpha * L_poi2region + (1-alpha) * L_region2city
```

**Corruption Function**: Random permutation of POI features breaks the correspondence between features and graph structure, creating negative samples.

```python
def corruption(x):
    """Shuffle rows to break feature-structure correspondence"""
    return x[torch.randperm(x.size(0))]
```

---

## Theory & Key Insights

### 1. Why fclass-Level Embeddings?

POI2Vec learns at the **fclass** (functional class) level rather than individual POIs:

| Approach | Pros | Cons |
|----------|------|------|
| POI-level | Unique per POI | Sparse data, overfitting |
| fclass-level | More data, generalizes | POIs with same fclass are identical |

The fclass approach is chosen because:
- **Data efficiency**: Each fclass has many POI examples
- **Semantic coherence**: fclasses are meaningful functional categories
- **Hierarchy enforcement**: Category-fclass L2 loss creates semantic structure

```
fclass embeddings cluster by category:

     "Food" category
     ┌─────────────────┐
     │  o Coffee Shop  │
     │  o Restaurant   │
     │     o Bar       │
     └─────────────────┘

     "Retail" category
     ┌─────────────────┐
     │  o Grocery      │
     │    o Mall       │
     │  o Boutique     │
     └─────────────────┘
```

### 2. Graph Infomax Principle

The core idea is to **maximize mutual information** between local (POI/region) and global (region/city) representations:

```
I(local; global) = H(local) - H(local | global)
```

By maximizing this, we ensure:
- Local embeddings contain information predictive of global context
- Global embeddings summarize local patterns effectively

### 3. Contrastive Learning Framework

HGI uses contrastive learning with positive and negative pairs:

| Pair Type | Example | Goal |
|-----------|---------|------|
| Positive | (POI, its region) | High score |
| Negative | (POI, random region) | Low score |
| Positive | (Region, city) | High score |
| Negative | (Corrupted region, city) | Low score |

The model learns to score positive pairs high and negative pairs low using bilinear discriminators:
```
score = sigmoid(z1^T @ W @ z2)
```

### 4. Hard Negative Mining

Easy negatives (random regions) provide weak learning signal. HGI uses **hard negatives**:

```
Region similarity matrix:
      R0    R1    R2    R3
R0   1.0   0.3   0.7   0.1
R1   0.3   1.0   0.2   0.8
R2   0.7   0.2   1.0   0.4
R3   0.1   0.8   0.4   1.0

For R0:
  Easy negatives: R3 (sim=0.1) - obviously different
  Hard negatives: R2 (sim=0.7) - confusingly similar!

Hard negatives force the model to learn subtle distinctions.
```

### 5. Edge Weight Formula

Delaunay edges are weighted by both spatial distance and regional boundaries:

```
w_spatial = log((1 + D^1.5) / (1 + dist^1.5))

    D = bounding box diagonal (normalizer)
    dist = haversine distance in meters

w_regional = 1.0 if same census tract
           = 0.5 if different census tracts

final_weight = normalize(w_spatial * w_regional)  -> [0, 1]
```

This encourages:
- Nearby POIs to have strong connections
- Within-region connections to be stronger than cross-region

---

## Usage

### Running the Full Pipeline

```python
from embeddings.hgi.hgi import create_embedding
from argparse import Namespace

args = Namespace(
    dim=64,
    epoch=2000,
    poi2vec_epochs=100,
    alpha=0.5,
    attention_head=4,
    lr=0.001,
    gamma=1.0,
    max_norm=0.9,
    device='cpu',
    shapefile='/path/to/census_tracts.shp',
    force_preprocess=True,
)

create_embedding(state="Texas", args=args)
```

### Using the Pipeline Script

```bash
cd /path/to/ingred
python pipelines/embedding/hgi.pipe.py
```

Configure states in the script:
```python
STATES = {
    'Texas': Resources.TL_TX,
    'California': Resources.TL_CA,
}
```

### Loading Generated Embeddings

```python
import pandas as pd

# POI embeddings
poi_emb = pd.read_parquet("output/hgi/Texas/embeddings.parquet")
# Columns: placeid, 0, 1, ..., 63, category

# Region embeddings
region_emb = pd.read_parquet("output/hgi/Texas/region_embeddings.parquet")
# Columns: region_id, reg_0, reg_1, ..., reg_63
```

---

## Configuration Reference

### HGI Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 64 | Embedding dimension for POI and region embeddings |
| `epoch` | 2000 | Number of HGI training epochs |
| `alpha` | 0.5 | Loss balance: alpha * L_poi2region + (1-alpha) * L_region2city |
| `attention_head` | 4 | Number of attention heads in PMA |
| `lr` | 0.001 | Adam learning rate |
| `gamma` | 1.0 | StepLR decay factor (1.0 = no decay) |
| `max_norm` | 0.9 | Gradient clipping max norm |
| `device` | 'cpu' | Training device ('cpu' or 'cuda') |

### POI2Vec Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poi2vec_epochs` | 100 | fclass embedding training epochs |
| `walk_length` | 10 | Steps per random walk |
| `walks_per_node` | 5 | Walks generated per POI |
| `context_size` | 5 | Skip-gram context window |
| `p`, `q` | 0.5 | Node2Vec return/in-out parameters |
| `k` | 5 | Negative samples per positive |
| `batch_size` | 2048 | Training batch size |
| `lr` | 0.05 | Adam learning rate |

### Preprocessing Parameters

| Parameter | Description |
|-----------|-------------|
| `force_preprocess` | If True, regenerate graph even if pickle exists |
| `shapefile` | Path to TIGER/Line census tract shapefile |
| `cta_file` | Optional: Pre-computed boroughs CSV |

---

## File Structure

```
src/embeddings/hgi/
├── README.md                 # This documentation
├── __init__.py               # Module exports
├── hgi.py                    # Pipeline orchestrator (create_embedding, train_hgi)
├── poi2vec.py                # POI2Vec pre-training (POI2Vec, EmbeddingModel, POISet)
├── preprocess.py             # Graph construction (HGIPreprocess, preprocess_hgi)
├── utils.py                  # Spatial utilities (haversine, bbox diagonal)
└── model/
    ├── __init__.py           # Model exports
    ├── HGIModule.py          # Core HGI model (HierarchicalGraphInfomax, corruption)
    ├── POIEncoder.py         # GCN-based POI encoding
    ├── RegionEncoder.py      # Attention pooling + region GCN (POI2Region)
    └── SetTransformer.py     # Attention components (MAB, SAB, PMA)
```

---

## Summary: What Each Component Does

| Component | Input | Output | Key Operation |
|-----------|-------|--------|---------------|
| `preprocess.py` | POI data + shapefile | edges.csv, pois.csv | Delaunay triangulation |
| `poi2vec.py` | edges.csv, pois.csv | poi_embeddings.pt | fclass-level skip-gram |
| `POIEncoder` | POI features + graph | POI embeddings | 1-layer GCN |
| `POI2Region` | POI embeddings | Region embeddings | PMA + Region GCN |
| `HGIModule` | All of above | Final embeddings | Mutual info maximization |

---

## References

- **Deep Graph Infomax**: Velickovic et al., 2019 - The foundational graph representation learning method
- **Node2Vec**: Grover & Leskovec, 2016 - Random walk-based graph embedding
- **Set Transformer**: Lee et al., 2019 - Attention-based set functions (PMA, SAB)
- **Region Embedding Benchmark**: Original implementation this module follows
