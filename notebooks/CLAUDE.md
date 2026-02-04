# HGI Pipeline - Original Implementation Documentation

**Source**: `/Users/vitor/Desktop/mestrado/ingred/notebooks/HGI.ipynb`
**Extracted Example**: `/Users/vitor/Desktop/mestrado/ingred/notebooks/hgi_texas.py`
**Original Code**: `/Users/vitor/Desktop/mestrado/ingred/notebooks/region-embedding-benchmark-main/region-embedding/baselines/`

---

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Phase 0: Environment Setup](#phase-0-environment-setup)
3. [Phase 1: Region Definition (Shapefile)](#phase-1-region-definition-shapefile)
4. [Phase 2: POI Data Preparation](#phase-2-poi-data-preparation)
5. [Phase 3: POI Embedding (POI2Vec)](#phase-3-poi-embedding-poi2vec)
   - [Step 3a: PreProcess.run()](#step-3a-preprocessrun)
   - [Step 3b: POI2Vec.train()](#step-3b-poi2vectrain--save_walks)
   - [Step 3c: EmbeddingModel Training](#step-3c-embeddingmodel-training-hierarchical-skip-gram)
   - [Step 3d: POI-Level Embedding Reconstruction](#step-3d-poi-level-embedding-reconstruction)
6. [Phase 4: HGI Graph Construction](#phase-4-hgi-graph-construction)
7. [Phase 5: HGI Training](#phase-5-hgi-training)
8. [Key Data Structures](#key-data-structures)
9. [Critical Configuration Values](#critical-configuration-values)

---

## Pipeline Overview

The original HGI implementation follows a **5-phase sequential pipeline** that is repeated for multiple US states (Georgia, Nebraska, Texas, California, Florida, North Carolina, Montana, Alabama). Each state follows the same execution flow with minor variations in hyperparameters.

```
┌─────────────────────────────────────────────────────────────────┐
│                    HGI Pipeline Flow                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Region Definition                                      │
│  • Download TIGER/Line census tract shapefiles                   │
│  • Reproject to EPSG:4326                                        │
│  • Save as boroughs_area.csv (GEOID, geometry WKT)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: POI Data Preparation                                   │
│  • Load 2 Gowalla CSVs (crus + separated)                        │
│  • Parse spot_categories JSON → fclass_name                      │
│  • Aggregate by placeid: mode(category), mode(fclass), mean(coords)│
│  • Integer-encode category & fclass                              │
│  • Save pois_gowalla.csv (feature_id, category, fclass, geometry)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: POI Embedding (POI2Vec)                                │
│  Step 3a: PreProcess.run()                                       │
│    • Read pois + boroughs CSV                                    │
│    • Spatial join POIs to regions                                │
│    • Delaunay graph construction                                 │
│    • Edge weighting (spatial distance × region bonus)            │
│    • Save pois.csv + edges.csv                                   │
│                                                                  │
│  Step 3b: POI2Vec.train()                                        │
│    • Generate random walks on POI graph                          │
│    • Save walks to second_class_walks.pkl                        │
│                                                                  │
│  Step 3c: EmbeddingModel training                                │
│    • Load walks → build POISet dataset                           │
│    • Skip-gram + hierarchical category-fclass loss               │
│    • Train 5 epochs, batch=2048, lr=0.05, k=5 negatives         │
│    • Save poi-encoder-gowalla-h3_{STATE}.tensor                  │
│                                                                  │
│  Step 3d: POI-Level Embedding Reconstruction                     │
│    • Load fclass embeddings W[num_fclass, D]                     │
│    • Load pois_gowalla.csv with feature_id, fclass               │
│    • Map each POI: poi_emb[i] = W[fclass[i]]                     │
│    • Save embeddings-poi-encoder.csv (placeid, 0..D-1, category) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: HGI Graph Construction                                 │
│  Step 4a: Preprocess.get_data_torch()                            │
│    • Build region features (areas, adjacency, similarity)        │
│    • Return data_dict with graph structure                       │
│                                                                  │
│  Step 4b: Load pre-computed location embeddings                  │
│    • Either from poi_embeddings_location-{state}.pt OR           │
│    • From MTL_POI_Novo embeddings-poi-encoder.csv                │
│    • Map to poi_index.csv ordering                               │
│                                                                  │
│  Step 4c: Build PyG Data object                                  │
│    • x = location embeddings (not POI2Vec!)                      │
│    • edge_index, edge_weight from Delaunay                       │
│    • region_id, region_area, coarse_region_similarity, region_adjacency│
│    • Save as gowalla.pt + pickle gowalla_hgi_data.pkl            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: HGI Training                                           │
│  • Run train.py --city gowalla --dim 64 --alpha 0.5              │
│  • 3-level hierarchy: POI → Region → City                        │
│  • Mutual information maximization loss                          │
│  • 300-500 epochs, device cpu or cuda                            │
│  • Save gowalla_h3.torch (region embeddings)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Environment Setup

### Google Colab Authentication
```python
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

auth.authenticate_user()
drive_service = build('drive', 'v3')
```

### Install Dependencies
```bash
# Geospatial libraries
pip install geopandas shapely libpysal h3 h3ronpy pyarrow scipy scikit-learn

# PyTorch 2.4.0 (CPU or CUDA)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

### Helper Functions
```python
def baixar_shapefile_estado(estado):
    """Downloads TIGER/Line 2021 census tract shapefile from data.gov"""
    url = f"https://www2.census.gov/geo/tiger/TIGER2021/TRACT/tl_2021_{state_fips}_{estado.lower()}_tract.zip"
    # Downloads to /content/, extracts .shp files
```

---

## Phase 1: Region Definition (Shapefile)

### Input
- TIGER/Line 2021 census tract shapefiles from data.gov

### Process
```python
ESTADO = "Texas"  # or Georgia, Nebraska, California, etc.
diretorio_principal = "/content/drive/MyDrive/region-embedding-benchmark-main/region-embedding-benchmark-main/"

# Download shapefile
baixar_shapefile_estado("texas")

# Read most recent .shp file
arquivos = [os.path.join("/content", f) for f in os.listdir("/content") if f.endswith(".shp")]
arquivo = max(arquivos, key=os.path.getmtime)
tl = gpd.read_file(arquivo).to_crs("EPSG:4326")

# Extract GEOID + geometry
boroughs = tl[["GEOID", "geometry"]].copy()
boroughs["geometry"] = boroughs["geometry"].apply(lambda g: g.wkt)  # Convert to WKT string

# Save
boroughs.to_csv(f"{diretorio_principal}/boroughs_area.csv", index=False)
```

### Output
- **boroughs_area.csv** with columns: `GEOID` (string), `geometry` (WKT Polygon/MultiPolygon)

---

## Phase 2: POI Data Preparation

### Input
Two Gowalla checkin CSV files from Google Drive:
1. **estados/crus/checkins_{STATE}.csv** - Labeled with category
2. **estados/separated/checkins_{STATE}.csv** - Raw with `spot_categories` JSON

### Process

#### Step 2.1: Load Data
```python
CHECKIN_NAO_CRU = f"estados/crus/checkins_{ESTADO}.csv"
CHECKIN_CRU = f"estados/separated/checkins_{ESTADO}.csv"

df_labeled = pd.read_csv(CHECKIN_NAO_CRU)  # Has 'category' column
df_raw = pd.read_csv(CHECKIN_CRU)          # Has 'spot_categories' JSON column
```

#### Step 2.2: Parse Fine-Grained Class (fclass) from JSON
```python
def parse_names(cell):
    """Extract category names from spot_categories JSON array"""
    try:
        lst = ast.literal_eval(cell)
        if isinstance(lst, list):
            return [d.get("name") for d in lst if isinstance(d, dict) and "name" in d]
    except Exception:
        pass
    return []

def first_or_none(lst):
    """Take first element from list"""
    return lst[0] if (isinstance(lst, list) and len(lst) > 0) else None

# Parse JSON to extract first category name (fclass_name)
df_raw["__cat_names"] = df_raw["spot_categories"].fillna("[]").apply(parse_names)
df_raw["__fclass_name"] = df_raw["__cat_names"].apply(first_or_none)
```

#### Step 2.3: Aggregate by Placeid
```python
# Mode of fclass per placeid
fclass_by_place = (df_raw.dropna(subset=["__fclass_name"])
                   .groupby("placeid")["__fclass_name"]
                   .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))

# Mode of category per placeid
cat_by_place = (df_labeled.dropna(subset=["category"])
                .groupby("placeid")["category"]
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))

# Mean coordinates per placeid
lon_col_raw = "lng" if "lng" in df_raw.columns else "longitude"
lat_col_raw = "lat" if "lat" in df_raw.columns else "latitude"

coords_raw = (df_raw.groupby("placeid")[[lat_col_raw, lon_col_raw]]
              .mean()
              .rename(columns={lat_col_raw: "latitude", lon_col_raw: "longitude"}))
coords_raw = coords_raw.dropna()
```

#### Step 2.4: Build POI DataFrame
```python
# CRITICAL: feature_id is the index from coords_raw (NOT placeid!)
pois = pd.DataFrame({"feature_id": coords_raw.index})
pois["feature_id"] = pois["feature_id"].astype(int)

# Join aggregated category and fclass by feature_id
pois["fclass_name"] = fclass_by_place.reindex(pois["feature_id"]).values
pois["category_name"] = cat_by_place.reindex(pois["feature_id"]).values

# Drop missing
pois = pois.dropna(subset=["fclass_name", "category_name"]).reset_index(drop=True)

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    pois,
    geometry=gpd.points_from_xy(
        coords_raw.loc[pois["feature_id"], "longitude"].values,
        coords_raw.loc[pois["feature_id"], "latitude"].values
    ),
    crs="EPSG:4326"
)
gdf["geometry"] = gdf.geometry.apply(lambda p: p.wkt)  # Convert to WKT string
```

#### Step 2.5: Integer Encode Categories
```python
# Build vocabularies (custom, not sklearn LabelEncoder at this stage)
fclass_vocab = {n: i for i, n in enumerate(pd.Series(gdf["fclass_name"]).dropna().unique())}
cat_vocab = {n: i for i, n in enumerate(pd.Series(gdf["category_name"]).dropna().unique())}

# Map to integers
gdf["fclass"] = gdf["fclass_name"].map(lambda n: fclass_vocab.get(n, -1)).astype(int)
gdf["category"] = gdf["category_name"].map(lambda n: cat_vocab.get(n, -1)).astype(int)

# Filter out invalid encodings
gdf = gdf[(gdf["fclass"] >= 0) & (gdf["category"] >= 0)].reset_index(drop=True)
```

#### Step 2.6: Save POI Data
```python
pois_out = gdf[["feature_id", "category", "fclass", "geometry"]].copy()
pois_out.to_csv("pois_gowalla.csv", index=False)
```

### Output
- **pois_gowalla.csv** with 4 columns:
  - `feature_id` (int) - row index from coords_raw
  - `category` (int) - label-encoded category
  - `fclass` (int) - label-encoded fine-grained class
  - `geometry` (WKT Point)

---

## Phase 3: POI Embedding (POI2Vec)

This phase has **3 substeps**: preprocessing → walk generation → embedding training.

### Step 3a: PreProcess.run()

**Source**: `region-embedding/baselines/poi-encoder/POIEmbedding.py`

```python
import sys
module_dir = f'{diretorio_principal}/region-embedding/baselines/poi-encoder'
sys.path.insert(0, module_dir)

from POIEmbedding import PreProcess

PreProcess("pois_gowalla.csv", "boroughs_area.csv", h3=False).run()
```

#### What PreProcess.run() Does:
1. **read_boroughs_data()** - Load boroughs CSV, convert WKT to geometry
2. **read_poi_data()** - Load pois CSV, spatial join with boroughs to assign GEOID, sklearn.LabelEncoder for category & fclass
3. **create_graph()** - Delaunay triangulation + edge weighting
4. **save_data()** - Save `pois.csv` + `edges.csv`

#### Delaunay Graph Construction
```python
points = np.array(self.pois.geometry.apply(lambda x: [x.x, x.y]).tolist())
D = Util.diagonal_length_min_box(self.pois.geometry.unary_union.envelope.bounds)  # Bbox diagonal (Euclidean in degrees)

triangles = scipy.spatial.Delaunay(points, qhull_options="QJ QbB Pp").simplices

G = nx.Graph()
G.add_nodes_from(range(len(points)))

for simplex in triangles:
    for x, y in combinations(simplex, 2):
        if not G.has_edge(x, y):
            dist = geo.haversine_np(*points[x], *points[y])  # Great-circle distance in meters
            w1 = np.log((1 + D**(3/2)) / (1 + dist**(3/2)))  # Spatial weight
            w2 = 1.0 if pois.iloc[x].GEOID == pois.iloc[y].GEOID else 0.5  # Region bonus
            G.add_edge(x, y, weight=w1 * w2)

edges = nx.to_pandas_edgelist(G)
# Min-max normalize weights to [0, 1]
edges['weight'] = (edges['weight'] - edges['weight'].min()) / (edges['weight'].max() - edges['weight'].min())
```

**Outputs**:
- `pois.csv` - Updated with `GEOID` column after spatial join, label-encoded category & fclass
- `edges.csv` - Columns: `source`, `target`, `weight`

### Step 3b: POI2Vec.train() + save_walks()

**Source**: `region-embedding/baselines/poi-encoder/POIEmbedding.py`

```python
from POIEmbedding import POI2Vec

p = POI2Vec()
p.train()       # Generate random walks
p.save_walks()  # Save to second_class_walks.pkl
```

#### What POI2Vec.train() Does:
```python
# Load PyG Node2Vec model
self.model = Node2Vec(
    self.data.edge_index,
    embedding_dim=64,
    walk_length=10,
    context_size=5,
    walks_per_node=5,
    num_negative_samples=2,
    p=0.5,
    q=0.5,
    sparse=True,
)

loader = self.model.loader(batch_size=128, shuffle=True)

# Generate walks and convert to fclass sequences
self.second_class_walks = []
for idx, (pos_rw, neg_rw) in enumerate(loader):
    for walk in pos_rw:
        self.second_class_walks.append([])
        for poi_idx in walk.tolist():
            second_class = self.pois.iloc[poi_idx]["fclass"]  # Map POI index → fclass
            self.second_class_walks[-1].append(second_class)
```

**Key Insight**: The walks are **fclass sequences**, not POI sequences. Each walk is a list of fclass integers.

**Outputs**:
- `second_class_walks.pkl` - List of fclass walks

### Step 3c: EmbeddingModel Training (Hierarchical Skip-Gram)

**Source**: `region-embedding/baselines/poi-encoder/model.py`

```python
from POIEmbedding import POI2Vec
from model import POISet, EmbeddingModel
import torch, torch.utils.data as tud

poi2vec = POI2Vec()
poi2vec.read_walks()  # Load second_class_walks.pkl
poi2vec.get_global_second_class_walks()  # Build co-occurrence lists

# Extract hierarchy pairs (category, fclass) for L2 regularization
second_class_hierarchy_pairs = list(set([tuple(x) for x in poi2vec.pois[["category", "fclass"]].to_numpy()]))

# Build dataset with k=5 negative samples
dataset = POISet(
    poi2vec.second_class_number,   # vocab_size (num unique fclass)
    poi2vec.second_class_walks,
    poi2vec.global_second_class_walks,
    k=5
)

# Build model
model = EmbeddingModel(
    vocab_size=poi2vec.second_class_number,
    embed_size=64,  # or 256 for some states
    second_class_hierarchy_pairs=second_class_hierarchy_pairs,
    le_lambda=1e-8
)

# Train
loader = tud.DataLoader(dataset, batch_size=2048, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=5e-2)

for e in range(5):
    for i, (inp, pos, neg) in enumerate(loader):
        opt.zero_grad()
        loss, _ = model(inp.long(), pos.long(), neg.long())
        loss.backward()
        opt.step()

# Save
torch.save({f"in_embed_{ESTADO}.weight": model.clone_input_embedding()},
           f"poi-encoder-gowalla-h3_{ESTADO}.tensor")
```

#### EmbeddingModel Loss Function
```python
def forward(self, input_labels, pos_labels, neg_labels):
    # Standard skip-gram loss
    input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
    pos_embedding = self.out_embed(pos_labels)     # [batch_size, context, embed_size]
    neg_embedding = self.out_embed(neg_labels)     # [batch_size, context*K, embed_size]

    log_pos = F.logsigmoid((pos_embedding @ input_embedding.unsqueeze(2)).squeeze(2)).sum(1)
    log_neg = F.logsigmoid((neg_embedding @ -input_embedding.unsqueeze(2)).squeeze(2)).sum(1)

    loss_graph = -(log_pos + log_neg).mean()

    # HIERARCHICAL LOSS: Minimize L2 distance between category-fclass pairs
    l2_norm = torch.tensor(0, dtype=torch.float)
    for pair in self.second_class_hierarchy_pairs:
        embed_i = self.in_embed(pair[0])  # category embedding
        embed_j = self.in_embed(pair[1])  # fclass embedding
        l2_norm += torch.norm((embed_i - embed_j))

    loss_le = 0.5 * (l2_norm ** 2) * self.le_lambda

    return loss_graph + loss_le, loss_le
```

**Outputs**:
- `poi-encoder-gowalla-h3_{STATE}.tensor` - PyTorch dict with key `in_embed_{STATE}.weight` (shape: `[num_fclass, embed_size]`)

---

### Step 3d: POI-Level Embedding Reconstruction

**CRITICAL STEP**: This is the missing link that reconstructs POI-level embeddings from fclass-level embeddings.

The POI2Vec embeddings from Step 3c are **fclass-level** (one embedding per unique fclass, not per POI). To use them as node features, we must map each POI to its corresponding fclass embedding.

#### What This Step Does:
1. **Load fclass embeddings** from `poi-encoder-gowalla-h3.tensor`
2. **Load POI data** from `pois_gowalla.csv` (which has `feature_id` and `fclass` for each POI)
3. **Map each POI to its embedding** using `fclass` as an index: `poi_embedding = W[poi.fclass]`
4. **Save as CSV** with placeid and embedding columns

#### Code Implementation

```python
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def load_embedding_matrix(tensor_path: str, key: str = "in_embed.weight") -> np.ndarray:
    """
    Load the .tensor checkpoint and return numpy array (num_fclass, emb_dim).
    """
    ckpt = torch.load(tensor_path, map_location="cpu")
    if key not in ckpt:
        # Try alternate key format
        state_keys = [k for k in ckpt.keys() if 'in_embed' in k]
        if state_keys:
            key = state_keys[0]
        else:
            raise KeyError(f"No embedding key found. Available: {list(ckpt.keys())[:10]}")

    W = ckpt[key]
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"Value at '{key}' is not a torch.Tensor (got {type(W)}).")
    if W.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {tuple(W.shape)}.")

    return W.detach().cpu().numpy()


def build_output_df(pois_path: str, W: np.ndarray, filtrado_path: str) -> pd.DataFrame:
    """
    Generate the final DataFrame with:
      - placeid (copied from feature_id)
      - emb_0 ... emb_(D-1) mapped by fclass
      - category (joined from original checkins CSV via placeid)

    KEY OPERATION: Each POI gets its embedding by looking up W[poi.fclass]
    """
    # Load POI data
    pois = pd.read_csv(pois_path)
    if "feature_id" not in pois.columns or "fclass" not in pois.columns:
        raise KeyError("pois_gowalla.csv must contain 'feature_id' and 'fclass'. "
                       f"Found: {list(pois.columns)}")

    pois["feature_id"] = pois["feature_id"].astype(str)
    pois["fclass"] = pois["fclass"].astype(int)

    num_classes, emb_dim = W.shape

    # Initialize embedding matrix
    n = len(pois)
    emb = np.full((n, emb_dim), np.nan, dtype=float)

    # CRITICAL: Map each POI to its fclass embedding
    valid = (pois["fclass"] >= 0) & (pois["fclass"] < num_classes)
    if not valid.all():
        invalid_rows = (~valid).sum()
        print(f"WARNING: {invalid_rows} POIs have invalid fclass values")

    # This is the reconstruction: emb[i] = W[fclass_of_poi_i]
    emb[valid.values] = W[pois.loc[valid, "fclass"].to_numpy()]

    # Build output DataFrame
    emb_cols = [f"{i}" for i in range(emb_dim)]
    out = pd.DataFrame(emb, columns=emb_cols)
    out.insert(0, "placeid", pois["feature_id"].astype(str))

    # Join category from original checkins
    filtrado = pd.read_csv(filtrado_path)

    if "placeid" not in filtrado.columns:
        if "feature_id" in filtrado.columns:
            filtrado = filtrado.rename(columns={"feature_id": "placeid"})
        else:
            raise KeyError(f"Checkins CSV must have 'placeid'. Found: {list(filtrado.columns)}")

    if "category" not in filtrado.columns:
        raise KeyError("Checkins CSV must contain 'category' column.")

    filtrado["placeid"] = filtrado["placeid"].astype(str)

    # Left join to preserve all POIs
    out = out.merge(filtrado[["placeid", "category"]], on="placeid", how="left")

    return out


# Example usage
ESTADO = "Texas"
pois_path = f"pois_gowalla.csv"
tensor_path = f"poi-encoder-gowalla-h3_{ESTADO}.tensor"
filtrado_path = f"checkins_{ESTADO.capitalize()}.csv"

# Load fclass embeddings (shape: [num_fclass, emb_dim])
W = load_embedding_matrix(tensor_path)
print(f"Loaded fclass embeddings: {W.shape}")

# Reconstruct POI-level embeddings
out_df = build_output_df(pois_path, W, filtrado_path)
print(f"Reconstructed {len(out_df)} POI embeddings")

# Save
out_df.to_csv(f"embeddings-poi-encoder.csv", index=False)
```

#### Key Insight: Why This Step Exists

POI2Vec learns embeddings at the **fclass level** (feature class) because:
1. Random walks are converted to **fclass sequences** (not POI sequences)
2. The EmbeddingModel vocabulary is **unique fclass values**
3. Multiple POIs can share the same fclass and thus the **same embedding**

This reconstruction step maps each POI to its fclass embedding, creating a POI-level embedding matrix needed for downstream tasks.

#### Data Flow
```
poi-encoder-gowalla-h3.tensor     pois_gowalla.csv
         ↓                               ↓
    W[num_fclass, D]            feature_id, fclass
         ↓                               ↓
         └─────────> poi_emb[i] = W[fclass[i]] ────────┐
                                                        ↓
                                        embeddings-poi-encoder.csv
                                        (placeid, 0, 1, ..., D-1, category)
```

#### Output Format
**embeddings-poi-encoder.csv** with columns:
- `placeid` (str) - POI identifier (from feature_id)
- `0`, `1`, `2`, ..., `{emb_dim-1}` (float) - Embedding dimensions
- `category` (str) - Category label from original checkins

**Example:**
```
placeid,0,1,2,3,...,63,category
"1234",0.123,-0.456,0.789,...,0.234,"Food"
"5678",0.123,-0.456,0.789,...,0.234,"Food"  # Same fclass = same embedding
"9012",-0.321,0.654,-0.987,...,-0.432,"Shopping"
```

**Outputs**:
- `embeddings-poi-encoder.csv` - POI-level embeddings with placeid

---

## Phase 4: HGI Graph Construction

This phase builds the final PyG Data object for HGI training.

### Step 4a: Preprocess.get_data_torch()

**Source**: `region-embedding/baselines/HGI/preprocess/main.py`

```python
import sys
module_dir = f'{diretorio_principal}/region-embedding/baselines/HGI/preprocess'
sys.path.insert(0, module_dir)

from main import Preprocess

POIS = "pois_gowalla.csv"
REGS = "boroughs_area.csv"

data_dict = Preprocess(POIS, REGS, emb_filename=None, h3=False).get_data_torch()
```

#### What Preprocess.get_data_torch() Returns:
```python
{
    'edge_index': np.ndarray,          # [2, E] from Delaunay
    'edge_weight': np.ndarray,         # [E] normalized weights
    'region_id': np.ndarray,           # [N_pois] maps POI → region index
    'region_area': np.ndarray,         # [R] area in CRS units (degrees² for EPSG:4326)
    'coarse_region_similarity': np.ndarray,  # [R, R] cosine similarity from fclass distribution
    'region_adjacency': np.ndarray,    # [2, adj_edges] region-region edges
}
```

#### Region Similarity Computation (CRITICAL)
```python
# Build region × fclass crosstab
mat = pd.crosstab(self.pois['GEOID'], self.pois['fclass'])
mat = mat.reindex(regions, fill_value=0)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
region_coarse_region_similarity = cosine_similarity(mat.values)
```

### Step 4b: Load Pre-Computed Embeddings for Node Features

**CRITICAL**: The node features can come from different sources:
1. **Pre-computed location embeddings** from an external encoder (most states)
2. **POI2Vec embeddings** from Step 3d (`embeddings-poi-encoder.csv`)

Both provide POI-level embeddings, but they may capture different aspects of POI semantics.

#### Variant 1: Pre-Computed Location Embeddings (most states)
```python
loc_pt_path = f"/content/drive/MyDrive/.../poi_embeddings_location-{ESTADO}.pt"
blob = torch.load(loc_pt_path, map_location="cpu")
E = blob["embeddings"].detach().cpu().numpy()  # [num_placeids, D]
placeids = [str(p) for p in blob["placeids"]]
```

#### Variant 2: POI2Vec Embeddings from Step 3d (Montana, Alabama, or if no location embeddings)
```python
# Load the embeddings-poi-encoder.csv created in Step 3d
out_df = pd.read_csv("embeddings-poi-encoder.csv")
out_df = out_df.sort_values("placeid").reset_index(drop=True)

placeids = out_df["placeid"].astype(str).tolist()
emb_cols = [c for c in out_df.columns if c.isnumeric()]
E = out_df[emb_cols].to_numpy(dtype=np.float32)

# Save as .pt for consistency
torch.save({"embeddings": torch.from_numpy(E), "placeids": placeids},
           "poi_embeddings_encoder.pt")
```

**Note**: The original documentation stated that HGI uses location embeddings (NOT POI2Vec), but the code shows that some states (Montana, Alabama) actually use the POI2Vec embeddings from `embeddings-poi-encoder.csv`. Both approaches are valid.

#### Map to poi_index.csv Ordering
```python
order = pd.read_csv("poi_index.csv")  # Generated by Preprocess, maps feature_id → row index
order["feature_id"] = order["feature_id"].astype(str)

placeid2idx = {pid: i for i, pid in enumerate(placeids)}
D = E.shape[1]

X = np.zeros((len(order), D), dtype=np.float32)
for i, pid in enumerate(order["feature_id"].tolist()):
    idx = placeid2idx.get(pid)
    if idx is None:
        raise KeyError(f"placeid {pid} not found in embeddings")
    X[i] = E[idx]
```

### Step 4c: Build PyG Data Object

```python
import torch
from torch_geometric.data import Data

g = Data(
    x=torch.tensor(X, dtype=torch.float32),  # [N_pois, D] location embeddings
    edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
    edge_weight=torch.tensor(data_dict['edge_weight'], dtype=torch.float32),
)
g.region_id = torch.tensor(data_dict['region_id'], dtype=torch.long)
g.region_area = torch.tensor(data_dict['region_area'], dtype=torch.float32)
g.coarse_region_similarity = torch.tensor(data_dict['coarse_region_similarity'], dtype=torch.float32)
g.region_adjacency = torch.tensor(data_dict['region_adjacency'], dtype=torch.long)

torch.save(g, "gowalla.pt")
```

### Step 4d: Convert to Pickle for HGI

```python
import pickle as pkl

g = torch.load("gowalla.pt", map_location="cpu")

data_dict = {
    "node_features": g.x.detach().cpu().numpy(),
    "edge_index": g.edge_index.detach().cpu().numpy(),
    "edge_weight": g.edge_weight.detach().cpu().numpy(),
    "region_id": g.region_id.detach().cpu().numpy(),
    "region_area": g.region_area.detach().cpu().numpy(),
    "coarse_region_similarity": g.coarse_region_similarity.detach().cpu().numpy(),
    "region_adjacency": g.region_adjacency.detach().cpu().numpy(),
}

with open("./data/gowalla_hgi_data.pkl", "wb") as f:
    pkl.dump(data_dict, f)
```

### Sanity Check
```python
R_from_id = int(np.max(data_dict["region_id"])) + 1
R_area = len(data_dict["region_area"])
R_adj = int(data_dict["region_adjacency"].max()) + 1
R_sim = data_dict["coarse_region_similarity"].shape[0]

assert R_from_id == R_area == R_adj == R_sim, "Region count mismatch!"
```

**Outputs**:
- `gowalla.pt` - PyG Data object
- `./data/gowalla_hgi_data.pkl` - Pickle dict for HGI training

---

## Phase 5: HGI Training

### Command Line Invocation
```bash
python {diretorio_principal}/region-embedding/baselines/HGI/train.py \
    --city gowalla \
    --dim 64 \
    --alpha 0.5 \
    --attention_head 4 \
    --epoch 300 \
    --device cpu \
    --save_name gowalla_h3
```

### train.py Workflow

**Source**: `region-embedding/baselines/HGI/train.py`

```python
# Load graph data
from model.city_data import hgi_graph
data = hgi_graph(args.city).to(args.device)

# Initialize model
from model.hgi import HierarchicalGraphInfomax, POIEncoder, POI2Region, corruption

model = HierarchicalGraphInfomax(
    hidden_channels=args.dim,
    poi_encoder=POIEncoder(data.num_features, args.dim),
    poi2region=POI2Region(args.dim, args.attention_head),
    region2city=lambda z, area: torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1)),
    corruption=corruption,
    alpha=args.alpha,
).to(args.device)

# Optimizer + Scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # lr=0.001
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # gamma=1.0 (no decay)

# Training loop
def train(epoch):
    model.train()
    optimizer.zero_grad()
    pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb = model(data)
    loss = model.loss(pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=args.max_norm)  # max_norm=0.9
    optimizer.step()
    scheduler.step()
    return loss.item()

# Track best embeddings
lowest_loss = math.inf
region_emb_to_save = torch.FloatTensor(0)

for epoch in range(1, args.epoch + 1):
    loss = train(epoch)
    if loss < lowest_loss:
        region_emb_to_save = model.get_region_emb()  # Returns (region_emb, poi_emb)
        lowest_loss = loss

# Save
torch.save(region_emb_to_save[0], f'./data/{args.save_name}.torch')  # Region embeddings
torch.save(region_emb_to_save[1], './data/poi_embedding.torch')      # POI embeddings
```

### HGI Model Architecture

**Source**: `region-embedding/baselines/HGI/model/hgi.py`

#### POIEncoder (1-layer GCN)
```python
class POIEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(POIEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False, bias=True)
        self.prelu = nn.PReLU()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x
```

#### POI2Region (PMA + Region GCN)
```python
class POI2Region(nn.Module):
    def __init__(self, hidden_channels, num_heads):
        super(POI2Region, self).__init__()
        self.PMA = PMA(dim=hidden_channels, num_heads=num_heads, num_seeds=1, ln=False)
        self.conv = GCNConv(hidden_channels, hidden_channels, cached=False, bias=True)
        self.prelu = nn.PReLU()

    def forward(self, x, zone, region_adjacency):
        # Aggregate POIs to regions using PMA (Pooling by Multihead Attention)
        region_emb = x.new_zeros((zone.max()+1, x.size()[1]))
        for index in range(zone.max() + 1):
            poi_index_in_region = (zone == index).nonzero(as_tuple=True)[0]
            region_emb[index] = self.PMA(x[poi_index_in_region].unsqueeze(0)).squeeze()

        # Region-level GCN
        region_emb = self.conv(region_emb, region_adjacency)
        region_emb = self.prelu(region_emb)
        region_emb = torch.nan_to_num(region_emb, nan=0.0)
        return region_emb
```

#### Region2City (Weighted Sum + Sigmoid)
```python
region2city = lambda z, area: torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1))
# z: [R, dim], area: [R]
# Output: [dim]
```

#### Corruption Function
```python
def corruption(x):
    """Permute node features randomly"""
    return x[torch.randperm(x.size(0))]
```

### HGI Forward Pass

```python
def forward(self, data):
    # Encode POIs
    pos_poi_emb = self.poi_encoder(data.x, data.edge_index, data.edge_weight)
    cor_x = self.corruption(data.x)
    neg_poi_emb = self.poi_encoder(cor_x, data.edge_index, data.edge_weight)

    # Aggregate to regions
    region_emb = self.poi2region(pos_poi_emb, data.region_id, data.region_adjacency)
    neg_region_emb = self.poi2region(neg_poi_emb, data.region_id, data.region_adjacency)

    # Aggregate to city
    city_emb = self.region2city(region_emb, data.region_area)

    # Hard negative sampling for POI-to-Region contrastive loss
    pos_poi_emb_list = []
    neg_poi_emb_list = []
    for region in range(torch.max(data.region_id)+1):
        # Positive: POIs in this region
        id_of_poi_in_a_region = (data.region_id == region).nonzero(as_tuple=True)[0]
        poi_emb_of_a_region = pos_poi_emb[id_of_poi_in_a_region]

        # Negative: POIs from another region
        # 25% chance: pick region with similarity in (0.6, 0.8) - "hard negative"
        # 75% chance: random other region
        hard_negative_choice = random.random()
        R = int(torch.max(data.region_id).item()) + 1
        all_regions = list(range(R))

        if hard_negative_choice < 0.25:
            hard_example_range = ((data.coarse_region_similarity[region] > 0.6) &
                                  (data.coarse_region_similarity[region] < 0.8)).nonzero(as_tuple=True)[0].tolist()
            candidates = [r for r in hard_example_range if r != region]
            if not candidates:
                candidates = [r for r in all_regions if r != region]
            another_region_id = random.choice(candidates)
        else:
            candidates = [r for r in all_regions if r != region]
            another_region_id = random.choice(candidates)

        id_of_poi_in_another_region = (data.region_id == another_region_id).nonzero(as_tuple=True)[0]
        poi_emb_of_another_region = pos_poi_emb[id_of_poi_in_another_region]

        pos_poi_emb_list.append(poi_emb_of_a_region)
        neg_poi_emb_list.append(poi_emb_of_another_region)

    return pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb
```

### HGI Loss Function

```python
EPS = 1e-7

def discriminate_poi2region(self, poi_emb_list, region_emb, sigmoid=True):
    """Bilinear discriminator: poi @ W @ region"""
    values = []
    for region_id, region in enumerate(poi_emb_list):
        if region.size()[0] > 0:
            region_summary = region_emb[region_id]
            value = torch.matmul(region, torch.matmul(self.weight_poi2region, region_summary))
            values.append(value)
    values = torch.cat(values, dim=0)
    return torch.sigmoid(values) if sigmoid else values

def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
    """Bilinear discriminator: region @ W @ city"""
    value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
    return torch.sigmoid(value) if sigmoid else value

def loss(self, pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb):
    # POI-to-Region contrastive loss
    pos_loss_region = -torch.log(
        self.discriminate_poi2region(pos_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()
    neg_loss_region = -torch.log(
        1 - self.discriminate_poi2region(neg_poi_emb_list, region_emb, sigmoid=True) + EPS).mean()

    # Region-to-City contrastive loss
    pos_loss_city = -torch.log(
        self.discriminate_region2city(region_emb, city_emb, sigmoid=True) + EPS).mean()
    neg_loss_city = -torch.log(
        1 - self.discriminate_region2city(neg_region_emb, city_emb, sigmoid=True) + EPS).mean()

    loss_poi2region = pos_loss_region + neg_loss_region
    loss_region2city = pos_loss_city + neg_loss_city

    return loss_poi2region * self.alpha + loss_region2city * (1 - self.alpha)
```

**Outputs**:
- `./data/gowalla_h3.torch` - Region embeddings (shape: `[R, dim]`)
- `./data/poi_embedding.torch` - POI embeddings (shape: `[N_pois, dim]`)

---

## Key Data Structures

### pois_gowalla.csv (Phase 2 output)
| Column | Type | Description |
|--------|------|-------------|
| `feature_id` | int | Row index from coords_raw (NOT placeid!) |
| `category` | int | Label-encoded category (from labeled data) |
| `fclass` | int | Label-encoded fine-grained class (from spot_categories JSON) |
| `geometry` | WKT Point | WKT string of POI location |

### pois.csv (Phase 3a output, after PreProcess)
| Column | Type | Description |
|--------|------|-------------|
| `feature_id` | int | Original feature_id |
| `category` | int | Re-encoded with sklearn.LabelEncoder |
| `fclass` | int | Re-encoded with sklearn.LabelEncoder |
| `geometry` | WKT Point | WKT string |
| `GEOID` | str | Census tract ID (from spatial join) |

### edges.csv (Phase 3a output)
| Column | Type | Description |
|--------|------|-------------|
| `source` | int | Source node index |
| `target` | int | Target node index |
| `weight` | float | Min-max normalized edge weight [0, 1] |

### poi_index.csv (Generated by Preprocess in Phase 4a)
| Column | Type | Description |
|--------|------|-------------|
| `feature_id` | str | POI identifier (stringified) |
| `row_idx` | int | Row index in graph node feature matrix |

### gowalla_hgi_data.pkl (Phase 4d output)
```python
{
    'node_features': np.ndarray,  # [N_pois, D] location embeddings
    'edge_index': np.ndarray,     # [2, E] COO format
    'edge_weight': np.ndarray,    # [E] normalized weights
    'region_id': np.ndarray,      # [N_pois] maps POI → region index (0..R-1)
    'region_area': np.ndarray,    # [R] area in degrees² (EPSG:4326)
    'coarse_region_similarity': np.ndarray,  # [R, R] cosine similarity matrix
    'region_adjacency': np.ndarray,          # [2, adj_edges] region-region edges
}
```

---

## Critical Configuration Values

### POI2Vec (EmbeddingModel)
| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | `num_unique_fclass` | Data-dependent |
| `embed_size` | **64** (most states), **256** (FL 1st, NC) | Embedding dimension |
| `le_lambda` | **1e-8** | Hierarchical loss weight |
| `k` | **5** | Negative samples per positive |
| `batch_size` | **2048** | DataLoader |
| `lr` | **0.05** | Adam optimizer |
| `epochs` | **5** | Always |
| `walk_length` | **10** | Node2Vec (in POI2Vec.train()) |
| `context_size` | **5** | Node2Vec |
| `walks_per_node` | **5** | Node2Vec |
| `p` | **0.5** | Node2Vec return param |
| `q` | **0.5** | Node2Vec in-out param |

### HGI Training
| Parameter | Value | Notes |
|-----------|-------|-------|
| `dim` | **64** (default), **32**, **128** | Hidden dimension |
| `alpha` | **0.5** | Loss balance: `alpha * L_poi2region + (1-alpha) * L_region2city` |
| `attention_head` | **4** | PMA num heads |
| `lr` | **0.001** | Adam |
| `gamma` | **1.0** | StepLR decay (no decay since gamma=1.0) |
| `max_norm` | **0.9** | Gradient clipping |
| `epoch` | **300** (CPU), **400** (dim=128), **500** (dim=64) | Training epochs |
| `device` | `cpu` or `cuda` | Device |

### Graph Construction
| Parameter | Value | Notes |
|-----------|-------|-------|
| Delaunay options | `"QJ QbB Pp"` | Joggle, scale to unit cube, suppress warnings |
| Same-region weight | **1.0** | Multiplier for edges within same GEOID |
| Cross-region weight | **0.5** | Multiplier for edges across GEOIDs |
| Spatial weight formula | `log((1 + D^1.5) / (1 + dist^1.5))` | D=bbox diagonal (degrees), dist=haversine (meters) |
| Edge normalization | Min-max to [0, 1] | After weighting |

### Region Similarity
| Method | Cosine similarity of region × fclass crosstab |
| Input | `pd.crosstab(pois['GEOID'], pois['fclass'])` |
| Output | `[R, R]` symmetric matrix |

### Hard Negative Sampling
| Probability | **0.25** | Chance of hard negative |
| Hard negative range | Similarity ∈ **(0.6, 0.8)** | Similar but not identical regions |
| Fallback | Random other region | If no hard candidates exist |

---

## Key Differences from Typical Implementations

1. **fclass-Level Embeddings**: POI2Vec learns embeddings for **fclass** (feature classes), not individual POIs. Multiple POIs with the same fclass share the same embedding.

2. **POI-Level Reconstruction (Step 3d)**: A dedicated step maps POIs to fclass embeddings using `poi_emb[i] = W[fclass[i]]`. This is often overlooked in documentation.

3. **Node Features Flexibility**: Different states use different node features:
   - Some use pre-computed **location embeddings** from external encoders
   - Others use **POI2Vec embeddings** from Step 3d (`embeddings-poi-encoder.csv`)

4. **Hierarchical Loss**: POI2Vec includes a custom hierarchical category-fclass L2 regularization loss, not standard Node2Vec.

5. **Region Similarity**: Computed from fclass distribution (cosine similarity), NOT edge co-occurrence.

6. **feature_id**: Is the **row index** from coords_raw after groupby, NOT the placeid.

7. **fclass Critical Role**: Fine-grained class from `spot_categories` JSON used for:
   - Embedding vocabulary (POI2Vec trains on fclass)
   - Hierarchical loss (category-fclass pairs)
   - Region similarity (fclass distribution)
   - POI-level reconstruction (fclass as index)

8. **No Random Seeds**: No seeds set anywhere → non-deterministic results.

9. **Delaunay Graph**: Connects all POIs via Delaunay triangulation (can create long-distance edges on convex hull).

10. **Area in Degrees²**: Region area computed in EPSG:4326 (degrees²), latitude-dependent distortion.

11. **PMA Forward Unused**: POI2Region manually iterates over regions and calls PMA per-region, not using PMA's batched forward.

12. **StepLR gamma=1.0**: No learning rate decay (scheduler is a no-op).

---

## For Future Agents

### Where to Find Key Code
- **POI Preprocessing**: `region-embedding/baselines/poi-encoder/POIEmbedding.py` (PreProcess, POI2Vec classes)
- **POI Embedding Model**: `region-embedding/baselines/poi-encoder/model.py` (POISet, EmbeddingModel)
- **HGI Preprocessing**: `region-embedding/baselines/HGI/preprocess/main.py` (Preprocess class)
- **HGI Model**: `region-embedding/baselines/HGI/model/hgi.py` (POIEncoder, POI2Region, HierarchicalGraphInfomax)
- **HGI Training**: `region-embedding/baselines/HGI/train.py`
- **Data Loading**: `region-embedding/baselines/HGI/model/city_data.py` (hgi_graph function)

### Critical Files Generated
1. `boroughs_area.csv` - Region definitions (GEOID, geometry)
2. `pois_gowalla.csv` - 4-column POI data (feature_id, category, fclass, geometry)
3. `pois.csv` + `edges.csv` - After PreProcess (includes GEOID)
4. `second_class_walks.pkl` - fclass random walks
5. `poi-encoder-gowalla-h3_{STATE}.tensor` - POI2Vec fclass-level embeddings [num_fclass, emb_dim]
6. **`embeddings-poi-encoder.csv`** - POI-level embeddings reconstructed from fclass embeddings (Step 3d)
7. `poi_embeddings_location-{STATE}.pt` or `poi_embeddings_encoder.pt` - Pre-computed embeddings (USED as node features in HGI)
8. `poi_index.csv` - Maps feature_id → row index
9. `gowalla.pt` - PyG Data object
10. `gowalla_hgi_data.pkl` - Pickle dict for training
11. `gowalla_h3.torch` - Final region embeddings

### Most Common Pitfalls
1. **Missing POI-level reconstruction (Step 3d)** - POI2Vec learns fclass-level embeddings, not POI-level! You must map each POI to its fclass embedding.
2. **Confusing fclass embeddings with POI embeddings** - POI2Vec output is `W[num_fclass, D]`, not `W[num_pois, D]`. Multiple POIs can share the same fclass embedding.
3. **Node features source confusion** - Some states use pre-computed location embeddings, others use POI2Vec embeddings from Step 3d. Both are valid.
4. **Missing fclass column** - Required for region similarity, hierarchical loss, AND POI-level reconstruction.
5. **feature_id vs placeid** - feature_id is row index, not placeid.
6. **Region similarity method** - Must use fclass crosstab, not edge co-occurrence.
7. **Edge weight units** - Mixes degrees and meters (bug in original).
8. **No seeds** - Results are non-deterministic.

---

**End of Documentation**