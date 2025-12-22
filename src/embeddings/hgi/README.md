# HGI - Hierarchical Graph Infomax

## What is HGI?

HGI learns **spatial embeddings** for Points of Interest (POIs) by understanding their relationships at three levels:

```
                    ┌─────────────┐
                    │    CITY     │  ← Entire geographic area
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
      ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
      │  REGION   │  │  REGION   │  │  REGION   │  ← Census tracts
      └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
            │              │              │
         ┌──┴──┐        ┌──┴──┐        ┌──┴──┐
         │POI│POI│      │POI│POI│      │POI│POI│     ← Individual locations
         └────────┘      └────────┘      └────────┘
```

**Purpose:** Generate embeddings that capture:
- **Spatial relationships** - POIs near each other have similar embeddings
- **Functional similarity** - POIs with similar contexts are similar
- **Regional patterns** - POIs in the same region share characteristics

---

## Quick Start

```bash
cd /Users/vitor/Desktop/mestrado/ingred

# Run full pipeline (with POI2Vec pre-training)
PYTHONPATH=src python src/embeddings/hgi/hgi.py \
    --city Texas \
    --shapefile resources/shapefiles/tl_2022_48_tract.shp \
    --dim 64 \
    --epoch 2000

# Or without POI2Vec (faster, uses one-hot encoding)
PYTHONPATH=src python src/embeddings/hgi/hgi.py \
    --city Texas \
    --shapefile resources/shapefiles/tl_2022_48_tract.shp \
    --no_poi2vec
```

**Output:**
- `output/hgi/texas/embeddings.parquet` - POI embeddings
- `output/hgi/texas/region_embeddings.parquet` - Region embeddings

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FULL PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  INPUT                    STEP 1                    STEP 2
  ─────                    ──────                    ──────

┌──────────────┐      ┌──────────────┐         ┌──────────────┐
│ POI Data     │      │ Spatial      │         │ POI2Vec      │
│ (parquet)    │─────▶│ Graph        │────────▶│ Embeddings   │
│              │      │              │         │ (optional)   │
└──────────────┘      └──────────────┘         └──────────────┘
                             │                        │
┌──────────────┐             │                        │
│ Shapefile    │─────────────┤                        │
│ (regions)    │             │                        │
└──────────────┘             ▼                        ▼
                      ┌──────────────┐         ┌──────────────┐
                      │ Graph Data   │         │ Node Features│
                      │ (pickle)     │────────▶│ (enriched)   │
                      └──────────────┘         └──────────────┘
                                                      │
                                                      ▼
                      STEP 3                   ┌──────────────┐
                      ──────                   │ HGI Training │
                                               │              │
                                               └──────┬───────┘
                                                      │
                             ┌────────────────────────┼────────────────────────┐
                             ▼                        ▼
                      ┌──────────────┐         ┌──────────────┐
                      │ POI          │         │ Region       │
                      │ Embeddings   │         │ Embeddings   │
                      │ (parquet)    │         │ (parquet)    │
                      └──────────────┘         └──────────────┘
```

---

## Step-by-Step Explanation

### Step 1: Preprocessing (`preprocess.py`)

**Purpose:** Transform raw POI data into a graph structure.

**Input:**
- POI data (parquet) with columns: `placeid`, `category`, `latitude`, `longitude`
- Shapefile with region boundaries (census tracts with `GEOID`)

**What happens:**

```
1. LOAD POI DATA
   ├── Read parquet file
   ├── Validate required columns (placeid, category, lat/lon)
   └── Handle duplicate placeids (aggregate by mode)

2. SPATIAL JOIN
   ├── Create Point geometries from coordinates
   └── Assign each POI to a region (GEOID) via spatial join

3. BUILD GRAPH (Delaunay Triangulation)
   ├── Connect nearby POIs based on Delaunay triangulation
   ├── Calculate edge weights:
   │   ├── Spatial weight: log((1 + D^1.5) / (1 + dist^1.5))
   │   │   where D = bounding box diagonal, dist = haversine distance
   │   └── Region weight: 1.0 (same region) or 0.4 (different region)
   └── Normalize weights to [0, 1]

4. COMPUTE REGION FEATURES
   ├── Region areas (from shapefile geometries)
   ├── Region adjacency matrix (which regions touch)
   └── Region similarity matrix (based on shared edges)

5. SAVE
   └── Pickle file with all graph data
```

**Output:** `output/hgi/{city}/temp/gowalla.pt`

```python
{
    'node_features': array[N_pois, dim],     # POI features (one-hot or POI2Vec)
    'edge_index': array[2, N_edges],         # Graph edges
    'edge_weight': array[N_edges],           # Edge weights
    'region_id': array[N_pois],              # Which region each POI belongs to
    'region_area': array[N_regions],         # Area of each region
    'region_adjacency': array[2, N_adj],     # Adjacent region pairs
    'coarse_region_similarity': array[N_regions, N_regions],
    'y': array[N_pois],                      # Category labels (encoded)
    'place_id': array[N_pois],               # Original POI IDs
}
```

---

### Step 2: POI2Vec (`poi2vec.py`) - Optional

**Purpose:** Pre-train POI embeddings using the graph structure.

**Why use it?**
- One-hot encoding: Only knows POI category (sparse, no spatial info)
- POI2Vec: Learns from graph structure (dense, captures spatial patterns)

**What happens:**

```
1. LOAD GRAPH
   └── Read edges.csv from preprocessing

2. RUN NODE2VEC
   ├── Generate random walks on the graph
   │   ├── Walk length: 10 steps
   │   ├── Walks per node: 5
   │   └── Parameters p=0.5, q=0.5 (balanced BFS/DFS)
   │
   └── Train Skip-gram model
       ├── Predict context POIs from target POI
       ├── Context window: 5
       └── Negative samples: 2

3. SAVE EMBEDDINGS
   └── poi-encoder.tensor with shape [N_pois, embedding_dim]
```

**Output:** `output/hgi/{city}/temp/poi-encoder.tensor`

---

### Step 3: HGI Training (`hgi.py`)

**Purpose:** Learn hierarchical embeddings through mutual information maximization.

**What happens:**

```
1. LOAD DATA
   ├── Load preprocessed graph data
   └── Create PyTorch Geometric Data object

2. BUILD MODEL (3 components)
   │
   ├── POI ENCODER (GCN)
   │   ├── Input: POI features + graph structure
   │   ├── Graph convolution: aggregates neighbor information
   │   └── Output: POI embeddings [N_pois, dim]
   │
   ├── POI2REGION (Attention Pooling)
   │   ├── Input: POI embeddings grouped by region
   │   ├── Pooling by Multihead Attention (PMA)
   │   │   └── Learns which POIs are most important per region
   │   ├── Region GCN on adjacency graph
   │   └── Output: Region embeddings [N_regions, dim]
   │
   └── REGION2CITY (Area-weighted Aggregation)
       ├── Input: Region embeddings + region areas
       ├── Sigmoid area-weighted sum
       └── Output: City embedding [dim]

3. TRAIN WITH MUTUAL INFORMATION LOSS
   │
   ├── POSITIVE SAMPLES
   │   ├── POI ↔ its region (should be similar)
   │   └── Region ↔ city (should be similar)
   │
   ├── NEGATIVE SAMPLES (corruption)
   │   ├── Shuffle POI-region assignments
   │   └── Compare with wrong pairs
   │
   └── LOSS FUNCTION
       Loss = α × L_poi-region + (1-α) × L_region-city

       where each L uses binary cross-entropy:
       - Maximize: log(σ(pos · summary))
       - Minimize: log(1 - σ(neg · summary))

4. SAVE EMBEDDINGS
   ├── POI embeddings → embeddings.parquet
   └── Region embeddings → region_embeddings.parquet
```

---

## File Reference

```
src/embeddings/hgi/
├── hgi.py              # Main pipeline orchestrator
├── preprocess.py       # Graph construction from raw data
├── poi2vec.py          # Node2Vec pre-training
├── utils.py            # Spatial utilities (haversine, etc.)
├── __init__.py         # Module exports
└── model/
    ├── HGIModule.py    # Main HGI model + loss function
    ├── POIEncoder.py   # GCN for POI-level encoding
    ├── RegionEncoder.py# Attention pooling + Region GCN
    └── SetTransformer.py # PMA, MAB, SAB implementations
```

---

## Technical Deep Dive

### Graph Construction (Delaunay Triangulation)

We connect POIs using Delaunay triangulation, which creates edges between nearby POIs without crossing:

```
    POI1 ●────────● POI2
         │╲      ╱│
         │ ╲    ╱ │
         │  ╲  ╱  │
         │   ╲╱   │
         │   ╱╲   │
         │  ╱  ╲  │
         │ ╱    ╲ │
    POI4 ●────────● POI3
```

**Edge Weight Formula:**
```
weight = w_spatial × w_region

w_spatial = log((1 + D^1.5) / (1 + dist^1.5))
            where D = bounding box diagonal
                  dist = haversine distance in meters

w_region = 1.0  if same region
         = 0.4  if different regions
```

### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         HGI MODEL                           │
└─────────────────────────────────────────────────────────────┘

INPUT: x [N_pois, in_features], edge_index, edge_weight

        ┌─────────────────────────────────────────────┐
        │               POI ENCODER                    │
        │  GCNConv(in_features → hidden) + PReLU      │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
                    poi_emb [N_pois, hidden]
                              │
        ┌─────────────────────┴───────────────────────┐
        │              POI2REGION                      │
        │                                              │
        │  For each region r:                          │
        │    pois_in_r = poi_emb[region_id == r]      │
        │    region_emb[r] = PMA(pois_in_r)           │
        │                                              │
        │  Then: region GCN on adjacency graph        │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
                   region_emb [N_regions, hidden]
                              │
        ┌─────────────────────┴───────────────────────┐
        │             REGION2CITY                      │
        │                                              │
        │  city = sigmoid(Σ region_emb × area)        │
        └─────────────────────┬───────────────────────┘
                              │
                              ▼
                    city_emb [hidden]

OUTPUT: poi_emb, region_emb, city_emb
```

### Loss Function (Mutual Information Maximization)

The model learns by maximizing agreement between:
1. POIs and their regions (local)
2. Regions and the city (global)

```python
# Positive: Real POI-region pairs should agree
pos_loss = -log(sigmoid(poi_emb · region_emb))

# Negative: Shuffled pairs should disagree
neg_loss = -log(1 - sigmoid(corrupted_poi · region_emb))

# Combined loss
loss = α * (pos_loss + neg_loss)_poi-region
     + (1-α) * (pos_loss + neg_loss)_region-city
```

---

## Parameters

### Preprocessing (`preprocess.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--city` | Texas | City/state name |
| `--shapefile` | - | Path to census tract shapefile |
| `--poi_emb` | None | Path to pre-trained POI embeddings |

### POI2Vec (`poi2vec.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--dim` | 64 | Embedding dimension |
| `--batch_size` | 128 | Training batch size |

### HGI Training (`hgi.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dim` | 64 | Embedding dimension |
| `--epoch` | 2000 | Training epochs |
| `--lr` | 0.001 | Learning rate |
| `--alpha` | 0.5 | POI-region vs region-city loss balance |
| `--attention_head` | 4 | Attention heads in PMA |
| `--no_poi2vec` | False | Skip POI2Vec, use one-hot encoding |
| `--force_preprocess` | False | Re-run preprocessing even if data exists |

---

## Output Format

### POI Embeddings (`embeddings.parquet`)
```
| placeid | 0     | 1     | 2     | ... | 63    |
|---------|-------|-------|-------|-----|-------|
| abc123  | 0.123 | -0.45 | 0.789 | ... | 0.234 |
| def456  | 0.567 | 0.123 | -0.34 | ... | 0.891 |
```

### Region Embeddings (`region_embeddings.parquet`)
```
| region_id | reg_0 | reg_1 | reg_2 | ... | reg_63 |
|-----------|-------|-------|-------|-----|--------|
| 0         | 0.234 | 0.567 | -0.12 | ... | 0.345  |
| 1         | 0.891 | -0.23 | 0.456 | ... | 0.678  |
```

---

## Understanding HGI Embeddings

### What Do HGI Embeddings Capture?

HGI creates **spatial embeddings** that combine three types of information:

```
┌─────────────────────────────────────────────────────────────┐
│              WHAT HGI LEARNS FROM                           │
└─────────────────────────────────────────────────────────────┘

1. 🗺️  SPATIAL STRUCTURE (Primary Signal)
   ├── POI geographic locations (lat/lon)
   ├── Which POIs are neighbors (Delaunay graph)
   ├── Distance between POIs (edge weights)
   └── Result: Nearby POIs → similar embeddings

2. 🏷️  CATEGORICAL INFORMATION (Initial Features)
   ├── POI category (e.g., "Restaurant", "Shop")
   ├── Encoded as one-hot OR POI2Vec embeddings
   └── Result: POIs in similar contexts → similar embeddings

3. 🏘️  REGIONAL CONTEXT
   ├── Census tract membership
   ├── Region characteristics (size, density)
   ├── Regional POI composition
   └── Result: POIs in similar regions → share patterns
```

### Key Insight: Spatial-First, Not Category-First

**HGI is fundamentally a spatial embedding method.**

The graph is built from **coordinates only**, not categories:

```python
# Graph construction
edges = DelaunayTriangulation(poi_locations)  # Based on lat/lon
                                               # NOT based on categories!

# Two POIs are connected if:
✅ They are geographically close
❌ NOT if they have the same category
❌ NOT if they have the same name
❌ NOT if they share users
```

### What Does "Similar Embeddings" Mean?

Two POIs have similar embeddings if they share:

#### 1. **Spatial Proximity**
```
Example:
POI_A: Starbucks at (30.2672, -97.7431)
POI_B: Whole Foods at (30.2680, -97.7445)  ← 2 blocks away

→ Similar embeddings (they're neighbors in the graph)
```

#### 2. **Functional Context**
```
Example:
POI_A: McDonald's in shopping mall
        neighbors: Target, Gap, Starbucks, parking lots

POI_B: McDonald's in residential area
        neighbors: gas station, small shops, houses

→ DIFFERENT embeddings (same category, different contexts)
```

#### 3. **Regional Characteristics**
```
Example:
All POIs in downtown census tract share:
- High density
- Mixed commercial use
- Similar POI composition

→ Share some "downtown-ness" signal in embeddings
```

#### 4. **Category** (Weaker Signal)
```
Category matters, but ONLY in combination with spatial context:

Coffee shops near universities     → Cluster A
Coffee shops in office buildings   → Cluster B
Coffee shops in airports          → Cluster C

(All are "Coffee Shop" category, but different spatial contexts)
```

### Common Misconceptions

```
❌ WRONG: "HGI groups restaurants with restaurants"
✅ RIGHT: "HGI groups POIs in similar spatial contexts"

❌ WRONG: "Two McDonald's will always be similar"
✅ RIGHT: "Two McDonald's are similar ONLY if they have similar neighborhoods"

❌ WRONG: "HGI learns from user check-ins"
✅ RIGHT: "HGI learns from spatial structure (no check-in data needed)"
```

---

## Using HGI Embeddings

### Use Case 1: Find Similar POIs

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
df = pd.read_parquet('output/hgi/texas/embeddings.parquet')

# Get embedding for a target POI
target_poi_id = 'abc123'
target_row = df[df['placeid'] == target_poi_id]
target_emb = target_row.iloc[0, 1:].values  # Skip placeid column

# Compute similarities
all_embs = df.iloc[:, 1:].values  # All embeddings
similarities = cosine_similarity([target_emb], all_embs)[0]

# Find top 10 most similar POIs
df['similarity'] = similarities
similar_pois = df.nlargest(10, 'similarity')

print(similar_pois[['placeid', 'similarity']])
```

**What you'll get:**
- POIs geographically near the target
- POIs in similar neighborhoods
- POIs with similar functional contexts

### Use Case 2: Spatial Clustering

```python
from sklearn.cluster import KMeans

# Cluster POIs by spatial patterns
embeddings = df.iloc[:, 1:].values
kmeans = KMeans(n_clusters=20, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings)

# Analyze clusters
for cluster_id in range(20):
    cluster_pois = df[df['cluster'] == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(f"  Size: {len(cluster_pois)}")
    print(f"  Top categories: {cluster_pois['category'].value_counts().head(3)}")
    print(f"  Avg lat: {cluster_pois['latitude'].mean()}")  # If you have coords
```

**What clusters represent:**
- Groups of POIs with similar spatial contexts
- Functional zones (commercial, residential, mixed-use)
- Neighborhood types

### Use Case 3: Region Characterization

```python
# Load region embeddings
regions = pd.read_parquet('output/hgi/texas/region_embeddings.parquet')

# Compare regions
region_embs = regions.iloc[:, 1:].values
region_similarity = cosine_similarity(region_embs)

# Find similar census tracts
target_region = 0
similar_regions = region_similarity[target_region].argsort()[-5:]
print(f"Regions similar to {target_region}: {similar_regions}")
```

**What you'll discover:**
- Regions with similar POI distributions
- Areas with similar urban character
- Functional similarity across space

### Use Case 4: Downstream Tasks

HGI embeddings are useful for:

```python
# 1. Next POI prediction
# POIs with similar embeddings → likely next destinations

# 2. POI recommendation
# "Users who visited X might like Y" (similar embeddings)

# 3. Urban zone classification
# Cluster regions into: commercial, residential, industrial, mixed

# 4. POI attribute prediction
# Predict missing attributes (price, popularity) from neighbors

# 5. Anomaly detection
# POIs with unusual embeddings for their category → interesting outliers
```

---

## When to Use HGI vs. Alternatives

```
Use HGI when:
├── ✅ You care about spatial context
├── ✅ You want to capture neighborhood effects
├── ✅ You need region-level embeddings too
├── ✅ You have lat/lon + census tract data
└── ✅ Examples: POI recommendation, urban analysis, spatial clustering

Use Category-Only (e.g., one-hot) when:
├── ✅ You only care about POI type
├── ✅ Spatial context is irrelevant
├── ✅ You need simple, interpretable features
└── ✅ Examples: "Find all restaurants", category classification

Use User-Based Embeddings (e.g., Word2Vec on check-ins) when:
├── ✅ You have rich check-in sequences
├── ✅ You care about behavioral patterns
├── ✅ Spatial structure is less important
└── ✅ Examples: Sequential recommendation, user profiling
```

---

## References

- [Hierarchical Graph Infomax (Zhang et al., 2020)](https://dl.acm.org/doi/10.1145/3397536.3422213)
- [Node2Vec (Grover & Leskovec, 2016)](https://arxiv.org/abs/1607.00653)
- [Deep Graph Infomax (Velickovic et al., 2019)](https://arxiv.org/abs/1809.10341)
