#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGI Pipeline for Texas State - Organized from Original Notebook

This script implements the complete HGI (Hierarchical Graph Infomax) pipeline
for generating region embeddings from Gowalla checkin data.

Pipeline Phases:
1. Region Definition (Census tracts)
2. POI Data Preparation (merge category + fclass)
3. POI Embedding (POI2Vec with hierarchical loss)
4. HGI Graph Construction
5. HGI Training

Reference: /Users/vitor/Desktop/mestrado/ingred/notebooks/HGI.ipynb
Documentation: /Users/vitor/Desktop/mestrado/ingred/notebooks/CLAUDE.md
"""

import os
import sys
import ast
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.utils.data as tud
from shapely import wkt
from shapely.geometry import Point, box
from torch_geometric.data import Data

# ============================================================================
# CONFIGURATION
# ============================================================================

ESTADO = "Texas"
WORKING_DIR = Path(".")  # Change this to your working directory
BENCHMARK_DIR = Path(__file__).parent / "region-embedding-benchmark-main" / "region-embedding-benchmark-main"

# Input paths (adjust these to your actual data locations)
SHAPEFILE_PATH = WORKING_DIR / "tl_2021_48_tract.shp"  # Texas TIGER/Line shapefile
CHECKIN_LABELED = WORKING_DIR / "estados" / "crus" / f"checkins_{ESTADO}.csv"
CHECKIN_RAW = WORKING_DIR / "estados" / "separated" / f"checkins_{ESTADO}.csv"

# Optional: Pre-computed location embeddings
LOCATION_EMBEDDINGS_PATH = None  # Set to .pt file path if available
# Alternative: MTL POI encoder embeddings CSV
MTL_EMBEDDINGS_CSV = None  # Set to CSV path if available

# Output paths
OUTPUT_DIR = WORKING_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BOROUGHS_CSV = OUTPUT_DIR / "boroughs_area.csv"
POIS_GOWALLA_CSV = OUTPUT_DIR / "pois_gowalla.csv"
POIS_CSV = OUTPUT_DIR / "pois.csv"
EDGES_CSV = OUTPUT_DIR / "edges.csv"
WALKS_PKL = OUTPUT_DIR / "second_class_walks.pkl"
POI2VEC_TENSOR = OUTPUT_DIR / f"poi-encoder-gowalla-h3_{ESTADO}.tensor"
POI_EMBEDDINGS_PT = OUTPUT_DIR / "poi_embeddings_encoder.pt"
POI_INDEX_CSV = OUTPUT_DIR / "poi_index.csv"
GRAPH_PT = OUTPUT_DIR / "gowalla.pt"
GRAPH_PKL = OUTPUT_DIR / "data" / "gowalla_hgi_data.pkl"

# Add original code to path
sys.path.insert(0, str(BENCHMARK_DIR / "region-embedding" / "baselines" / "poi-encoder"))
sys.path.insert(0, str(BENCHMARK_DIR / "region-embedding" / "baselines" / "HGI" / "preprocess"))

# ============================================================================
# PHASE 1: REGION DEFINITION (CENSUS TRACTS)
# ============================================================================

def phase1_prepare_boroughs():
    """
    Load TIGER/Line census tract shapefile and save as boroughs_area.csv.

    Output: boroughs_area.csv with columns [GEOID, geometry (WKT)]
    """
    print("=" * 80)
    print("PHASE 1: Region Definition")
    print("=" * 80)

    if not SHAPEFILE_PATH.exists():
        print(f"ERROR: Shapefile not found at {SHAPEFILE_PATH}")
        print("Download from: https://www.census.gov/cgi-bin/geo/shapefiles/index.php")
        return False

    print(f"Reading shapefile: {SHAPEFILE_PATH}")
    tl = gpd.read_file(SHAPEFILE_PATH).to_crs("EPSG:4326")

    print(f"Loaded {len(tl)} census tracts")

    # Extract GEOID + geometry, convert to WKT
    boroughs = tl[["GEOID", "geometry"]].copy()
    boroughs["geometry"] = boroughs["geometry"].apply(lambda g: g.wkt)

    boroughs.to_csv(BOROUGHS_CSV, index=False)
    print(f"Saved: {BOROUGHS_CSV}")
    print(f"Columns: {boroughs.columns.tolist()}")
    print()
    return True


# ============================================================================
# PHASE 2: POI DATA PREPARATION
# ============================================================================

def parse_spot_categories(cell):
    """Parse spot_categories JSON to extract category names."""
    try:
        lst = ast.literal_eval(cell)
        if isinstance(lst, list):
            return [d.get("name") for d in lst if isinstance(d, dict) and "name" in d]
    except Exception:
        pass
    return []


def first_or_none(lst):
    """Get first element from list or None."""
    return lst[0] if (isinstance(lst, list) and len(lst) > 0) else None


def phase2_prepare_pois():
    """
    Merge labeled and raw Gowalla data to create POI dataset with 4 columns:
    - feature_id: Row index from coords_raw
    - category: Label-encoded category (from labeled data)
    - fclass: Label-encoded fine-grained class (from spot_categories JSON)
    - geometry: WKT Point

    Output: pois_gowalla.csv
    """
    print("=" * 80)
    print("PHASE 2: POI Data Preparation")
    print("=" * 80)

    if not CHECKIN_LABELED.exists() or not CHECKIN_RAW.exists():
        print(f"ERROR: Checkin files not found:")
        print(f"  Labeled: {CHECKIN_LABELED}")
        print(f"  Raw: {CHECKIN_RAW}")
        return False

    print(f"Reading labeled data: {CHECKIN_LABELED}")
    df_labeled = pd.read_csv(CHECKIN_LABELED)

    print(f"Reading raw data: {CHECKIN_RAW}")
    df_raw = pd.read_csv(CHECKIN_RAW)

    # Detect coordinate column names
    lon_col_raw = "lng" if "lng" in df_raw.columns else "longitude"
    lat_col_raw = "lat" if "lat" in df_raw.columns else "latitude"
    lon_col_lab = "lng" if "lng" in df_labeled.columns else ("longitude" if "longitude" in df_labeled.columns else None)
    lat_col_lab = "lat" if "lat" in df_labeled.columns else ("latitude" if "latitude" in df_labeled.columns else None)

    print(f"Coordinate columns: {lat_col_raw}, {lon_col_raw}")

    # Parse fclass (fine-grained class) from spot_categories JSON
    print("Parsing spot_categories JSON...")
    df_raw["__cat_names"] = df_raw["spot_categories"].fillna("[]").apply(parse_spot_categories)
    df_raw["__fclass_name"] = df_raw["__cat_names"].apply(first_or_none)

    # Aggregate by placeid: mode of fclass
    print("Aggregating fclass by placeid (mode)...")
    fclass_by_place = (df_raw.dropna(subset=["__fclass_name"])
                       .groupby("placeid")["__fclass_name"]
                       .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))

    # Aggregate by placeid: mode of category
    print("Aggregating category by placeid (mode)...")
    cat_by_place = (df_labeled.dropna(subset=["category"])
                    .groupby("placeid")["category"]
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]))

    # Aggregate coordinates by placeid: mean
    print("Aggregating coordinates by placeid (mean)...")
    coords_raw = (df_raw.groupby("placeid")[[lat_col_raw, lon_col_raw]]
                  .mean()
                  .rename(columns={lat_col_raw: "latitude", lon_col_raw: "longitude"}))

    # Fallback to labeled data if raw coords are empty
    if coords_raw.empty and lon_col_lab and lat_col_lab:
        print("Raw coords empty, falling back to labeled data...")
        coords_raw = (df_labeled.groupby("placeid")[[lat_col_lab, lon_col_lab]]
                      .mean()
                      .rename(columns={lat_col_lab: "latitude", lon_col_lab: "longitude"}))

    coords_raw = coords_raw.dropna()
    print(f"Aggregated {len(coords_raw)} unique placeids")

    # Build POI dataframe with feature_id as row index
    print("Building POI dataframe...")
    pois = pd.DataFrame({"feature_id": coords_raw.index})
    pois["feature_id"] = pois["feature_id"].astype(int)

    # Join category and fclass by feature_id (placeid)
    pois["fclass_name"] = fclass_by_place.reindex(pois["feature_id"]).values
    pois["category_name"] = cat_by_place.reindex(pois["feature_id"]).values

    # Drop missing
    pois = pois.dropna(subset=["fclass_name", "category_name"]).reset_index(drop=True)
    print(f"After filtering: {len(pois)} POIs")

    # Create GeoDataFrame with Point geometries
    print("Creating GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        pois,
        geometry=gpd.points_from_xy(
            coords_raw.loc[pois["feature_id"], "longitude"].values,
            coords_raw.loc[pois["feature_id"], "latitude"].values
        ),
        crs="EPSG:4326"
    )
    gdf["geometry"] = gdf.geometry.apply(lambda p: p.wkt)

    # Integer-encode categories and fclass
    print("Integer-encoding category and fclass...")
    fclass_vocab = {n: i for i, n in enumerate(pd.Series(gdf["fclass_name"]).dropna().unique())}
    cat_vocab = {n: i for i, n in enumerate(pd.Series(gdf["category_name"]).dropna().unique())}

    gdf["fclass"] = gdf["fclass_name"].map(lambda n: fclass_vocab.get(n, -1)).astype(int)
    gdf["category"] = gdf["category_name"].map(lambda n: cat_vocab.get(n, -1)).astype(int)

    # Filter out invalid encodings
    gdf = gdf[(gdf["fclass"] >= 0) & (gdf["category"] >= 0)].reset_index(drop=True)
    print(f"After encoding: {len(gdf)} POIs")
    print(f"Unique categories: {gdf['category'].nunique()}")
    print(f"Unique fclass: {gdf['fclass'].nunique()}")

    # Save with 4 required columns
    pois_out = gdf[["feature_id", "category", "fclass", "geometry"]].copy()
    pois_out.to_csv(POIS_GOWALLA_CSV, index=False)
    print(f"Saved: {POIS_GOWALLA_CSV}")
    print(f"Columns: {pois_out.columns.tolist()}")
    print()
    return True


# ============================================================================
# PHASE 3: POI EMBEDDING (POI2VEC)
# ============================================================================

def phase3a_preprocess():
    """
    Run PreProcess.run() from original code.
    - Spatial join POIs with boroughs
    - Build Delaunay graph
    - Save pois.csv + edges.csv
    """
    print("=" * 80)
    print("PHASE 3a: PreProcess (Graph Construction)")
    print("=" * 80)

    try:
        from POIEmbedding import PreProcess
    except ImportError:
        print("ERROR: Could not import PreProcess from POIEmbedding")
        print("Make sure the benchmark code is at the correct path")
        return False

    print(f"Running PreProcess on {POIS_GOWALLA_CSV} and {BOROUGHS_CSV}...")

    # Change to output directory so files are saved there
    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    try:
        pre = PreProcess(
            str(POIS_GOWALLA_CSV.name),
            str(BOROUGHS_CSV.name),
            h3=False
        )
        pre.run()
        print(f"Saved: {POIS_CSV}")
        print(f"Saved: {EDGES_CSV}")
    finally:
        os.chdir(original_dir)

    print()
    return True


def phase3b_poi2vec_walks():
    """
    Generate random walks on POI graph and convert to fclass sequences.
    Output: second_class_walks.pkl
    """
    print("=" * 80)
    print("PHASE 3b: POI2Vec Walk Generation")
    print("=" * 80)

    try:
        from POIEmbedding import POI2Vec
    except ImportError:
        print("ERROR: Could not import POI2Vec from POIEmbedding")
        return False

    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    try:
        print("Initializing POI2Vec...")
        p = POI2Vec()

        print("Generating random walks (this may take a while)...")
        p.train()

        print(f"Saving walks to {WALKS_PKL.name}...")
        p.save_walks()

        print(f"Generated {len(p.second_class_walks)} walks")
    finally:
        os.chdir(original_dir)

    print()
    return True


def phase3c_poi2vec_train():
    """
    Train EmbeddingModel with hierarchical category-fclass loss.
    Output: poi-encoder-gowalla-h3_{STATE}.tensor
    """
    print("=" * 80)
    print("PHASE 3c: POI2Vec Embedding Training")
    print("=" * 80)

    try:
        from POIEmbedding import POI2Vec
        from model import POISet, EmbeddingModel
    except ImportError:
        print("ERROR: Could not import POI2Vec or model classes")
        return False

    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    try:
        print("Loading POI2Vec...")
        poi2vec = POI2Vec()
        poi2vec.read_walks()
        poi2vec.get_global_second_class_walks()

        print(f"Vocabulary size (fclass): {poi2vec.second_class_number}")

        # Extract hierarchical pairs for L2 loss
        print("Extracting category-fclass hierarchy pairs...")
        second_class_hierarchy_pairs = list(set([
            tuple(x) for x in poi2vec.pois[["category", "fclass"]].to_numpy()
        ]))
        print(f"Found {len(second_class_hierarchy_pairs)} unique hierarchy pairs")

        # Build dataset with k=5 negative samples
        print("Building POISet dataset...")
        dataset = POISet(
            poi2vec.second_class_number,
            poi2vec.second_class_walks,
            poi2vec.global_second_class_walks,
            k=5
        )

        # Build model
        print("Initializing EmbeddingModel...")
        model = EmbeddingModel(
            vocab_size=poi2vec.second_class_number,
            embed_size=64,
            second_class_hierarchy_pairs=second_class_hierarchy_pairs,
            le_lambda=1e-8
        )

        # Train
        print("Training for 5 epochs (batch=2048, lr=0.05)...")
        loader = tud.DataLoader(dataset, batch_size=2048, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)

        for e in range(5):
            epoch_loss = 0.0
            for i, (inp, pos, neg) in enumerate(loader):
                opt.zero_grad()
                loss, _ = model(inp.long(), pos.long(), neg.long())
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            print(f"  Epoch {e+1}/5: Loss = {epoch_loss/len(loader):.4f}")

        # Save
        save_path = POI2VEC_TENSOR.name
        torch.save(
            {f"in_embed_{ESTADO}.weight": model.clone_input_embedding()},
            save_path
        )
        print(f"Saved: {save_path}")
    finally:
        os.chdir(original_dir)

    print()
    return True


# ============================================================================
# PHASE 4: HGI GRAPH CONSTRUCTION
# ============================================================================

def phase4a_load_location_embeddings():
    """
    Load pre-computed location embeddings (or create from MTL CSV if available).
    Output: poi_embeddings_encoder.pt
    """
    print("=" * 80)
    print("PHASE 4a: Load Location Embeddings")
    print("=" * 80)

    # Option 1: Use existing .pt file
    if LOCATION_EMBEDDINGS_PATH and Path(LOCATION_EMBEDDINGS_PATH).exists():
        print(f"Using existing embeddings: {LOCATION_EMBEDDINGS_PATH}")
        # Copy to output directory
        import shutil
        shutil.copy(LOCATION_EMBEDDINGS_PATH, POI_EMBEDDINGS_PT)
        print(f"Copied to: {POI_EMBEDDINGS_PT}")
        print()
        return True

    # Option 2: Convert from MTL CSV
    if MTL_EMBEDDINGS_CSV and Path(MTL_EMBEDDINGS_CSV).exists():
        print(f"Converting MTL embeddings from: {MTL_EMBEDDINGS_CSV}")
        out_df = pd.read_csv(MTL_EMBEDDINGS_CSV)
        out_df = out_df.sort_values("placeid").reset_index(drop=True)

        placeids = out_df["placeid"].astype(str).tolist()
        emb_cols = [c for c in out_df.columns if c.isnumeric()]
        E = out_df[emb_cols].to_numpy(dtype=np.float32)

        torch.save({
            "embeddings": torch.from_numpy(E),
            "placeids": placeids
        }, POI_EMBEDDINGS_PT)

        print(f"Saved: {POI_EMBEDDINGS_PT}")
        print(f"Shape: {E.shape}")
        print()
        return True

    # Option 3: No embeddings available
    print("WARNING: No pre-computed location embeddings found!")
    print("Please provide either:")
    print("  - LOCATION_EMBEDDINGS_PATH: .pt file with location embeddings")
    print("  - MTL_EMBEDDINGS_CSV: CSV file from MTL POI encoder")
    print()
    print("HGI requires location embeddings (not POI2Vec embeddings) as node features.")
    print("See CLAUDE.md for details.")
    return False


def phase4b_hgi_preprocess():
    """
    Run HGI Preprocess to build region features.
    Output: data_dict with edge_index, region_id, region_area, etc.
    """
    print("=" * 80)
    print("PHASE 4b: HGI Preprocessing")
    print("=" * 80)

    try:
        from main import Preprocess
    except ImportError:
        print("ERROR: Could not import Preprocess from HGI/preprocess/main")
        return None

    # Change to output directory
    original_dir = os.getcwd()
    os.chdir(OUTPUT_DIR)

    try:
        print("Running HGI Preprocess...")
        data_dict = Preprocess(
            POIS_GOWALLA_CSV.name,
            BOROUGHS_CSV.name,
            emb_filename=None,
            h3=False
        ).get_data_torch()

        print(f"Graph stats:")
        print(f"  POIs: {data_dict.get('number_pois', 'N/A')}")
        print(f"  Regions: {data_dict.get('number_regions', 'N/A')}")
        print(f"  Edges: {len(data_dict['edge_weight'])}")

    finally:
        os.chdir(original_dir)

    print()
    return data_dict


def phase4c_build_graph(data_dict):
    """
    Build PyG Data object with location embeddings as node features.
    Output: gowalla.pt, gowalla_hgi_data.pkl
    """
    print("=" * 80)
    print("PHASE 4c: Build PyG Graph")
    print("=" * 80)

    if not POI_EMBEDDINGS_PT.exists():
        print(f"ERROR: Location embeddings not found at {POI_EMBEDDINGS_PT}")
        return False

    print(f"Loading location embeddings from {POI_EMBEDDINGS_PT}...")
    blob = torch.load(POI_EMBEDDINGS_PT, map_location="cpu")
    E = blob["embeddings"].detach().cpu().numpy()
    placeids = [str(p) for p in blob["placeids"]]
    placeid2idx = {pid: i for i, pid in enumerate(placeids)}
    D = E.shape[1]

    print(f"Embedding dimension: {D}")
    print(f"Number of placeids: {len(placeids)}")

    # Load poi_index.csv (maps feature_id → row index)
    if not POI_INDEX_CSV.exists():
        print(f"ERROR: poi_index.csv not found at {POI_INDEX_CSV}")
        return False

    print(f"Loading POI index from {POI_INDEX_CSV}...")
    order = pd.read_csv(POI_INDEX_CSV)
    order["feature_id"] = order["feature_id"].astype(str)

    # Build node feature matrix
    print("Mapping placeids to node features...")
    X = np.zeros((len(order), D), dtype=np.float32)
    for i, pid in enumerate(order["feature_id"].tolist()):
        idx = placeid2idx.get(pid)
        if idx is None:
            print(f"WARNING: placeid {pid} not found in embeddings, using zeros")
        else:
            X[i] = E[idx]

    # Sanity check: edge_index should not exceed X.shape[0]
    ei = np.asarray(data_dict['edge_index'])
    assert ei.max() < X.shape[0], f"edge_index references node {ei.max()} >= {X.shape[0]}"
    print(f"Edge index range: [0, {ei.max()}]")

    # Build PyG Data object
    print("Building PyG Data object...")
    g = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(data_dict['edge_index'], dtype=torch.long),
        edge_weight=torch.tensor(data_dict['edge_weight'], dtype=torch.float32),
    )
    g.region_id = torch.tensor(data_dict['region_id'], dtype=torch.long)
    g.region_area = torch.tensor(data_dict['region_area'], dtype=torch.float32)
    g.coarse_region_similarity = torch.tensor(data_dict['coarse_region_similarity'], dtype=torch.float32)
    g.region_adjacency = torch.tensor(data_dict['region_adjacency'], dtype=torch.long)

    # Save PyG Data
    torch.save(g, GRAPH_PT)
    print(f"Saved: {GRAPH_PT}")

    # Convert to pickle for HGI training
    print("Converting to pickle format...")
    GRAPH_PKL.parent.mkdir(exist_ok=True)

    data_dict_final = {
        "node_features": g.x.detach().cpu().numpy(),
        "edge_index": g.edge_index.detach().cpu().numpy(),
        "edge_weight": g.edge_weight.detach().cpu().numpy(),
        "region_id": g.region_id.detach().cpu().numpy(),
        "region_area": g.region_area.detach().cpu().numpy(),
        "coarse_region_similarity": g.coarse_region_similarity.detach().cpu().numpy(),
        "region_adjacency": g.region_adjacency.detach().cpu().numpy(),
    }

    with open(GRAPH_PKL, "wb") as f:
        pkl.dump(data_dict_final, f)

    print(f"Saved: {GRAPH_PKL}")

    # Sanity check: all region counts should match
    R_from_id = int(np.max(data_dict_final["region_id"])) + 1
    R_area = len(data_dict_final["region_area"])
    R_adj = int(data_dict_final["region_adjacency"].max()) + 1
    R_sim = data_dict_final["coarse_region_similarity"].shape[0]

    print(f"\nRegion count sanity check:")
    print(f"  From region_id: {R_from_id}")
    print(f"  From region_area: {R_area}")
    print(f"  From region_adjacency: {R_adj}")
    print(f"  From region_similarity: {R_sim}")

    assert R_from_id == R_area == R_adj == R_sim, "Region count mismatch!"
    print("✓ All region counts match")
    print()
    return True


# ============================================================================
# PHASE 5: HGI TRAINING
# ============================================================================

def phase5_train_hgi():
    """
    Train HGI model using train.py from the original code.
    """
    print("=" * 80)
    print("PHASE 5: HGI Training")
    print("=" * 80)

    train_script = BENCHMARK_DIR / "region-embedding" / "baselines" / "HGI" / "train.py"

    if not train_script.exists():
        print(f"ERROR: train.py not found at {train_script}")
        return False

    print(f"To train HGI, run the following command:")
    print()
    print(f"python {train_script} \\")
    print(f"    --city gowalla \\")
    print(f"    --dim 64 \\")
    print(f"    --alpha 0.5 \\")
    print(f"    --attention_head 4 \\")
    print(f"    --epoch 300 \\")
    print(f"    --device cuda \\")  # or cpu
    print(f"    --save_name gowalla_h3")
    print()
    print("Note: The train.py script expects gowalla_hgi_data.pkl in ./data/")
    print(f"Make sure to copy {GRAPH_PKL} to the train.py working directory")
    print()
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete HGI pipeline."""
    print("\n" + "=" * 80)
    print("HGI PIPELINE FOR TEXAS")
    print("=" * 80)
    print(f"Working directory: {WORKING_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"Benchmark directory: {BENCHMARK_DIR.absolute()}")
    print()

    # Phase 1: Region Definition
    if not phase1_prepare_boroughs():
        print("ERROR in Phase 1. Exiting.")
        return

    # Phase 2: POI Data Preparation
    if not phase2_prepare_pois():
        print("ERROR in Phase 2. Exiting.")
        return

    # Phase 3a: Preprocessing (Graph Construction)
    if not phase3a_preprocess():
        print("ERROR in Phase 3a. Exiting.")
        return

    # Phase 3b: POI2Vec Walk Generation
    if not phase3b_poi2vec_walks():
        print("ERROR in Phase 3b. Exiting.")
        return

    # Phase 3c: POI2Vec Training
    if not phase3c_poi2vec_train():
        print("ERROR in Phase 3c. Exiting.")
        return

    # Phase 4a: Load Location Embeddings
    if not phase4a_load_location_embeddings():
        print("ERROR in Phase 4a. Exiting.")
        return

    # Phase 4b: HGI Preprocessing
    data_dict = phase4b_hgi_preprocess()
    if data_dict is None:
        print("ERROR in Phase 4b. Exiting.")
        return

    # Phase 4c: Build Graph
    if not phase4c_build_graph(data_dict):
        print("ERROR in Phase 4c. Exiting.")
        return

    # Phase 5: HGI Training (instructions only)
    phase5_train_hgi()

    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nKey outputs:")
    print(f"  - Boroughs: {BOROUGHS_CSV}")
    print(f"  - POIs: {POIS_GOWALLA_CSV}")
    print(f"  - Graph: {GRAPH_PT}")
    print(f"  - Graph (pickle): {GRAPH_PKL}")
    print()


if __name__ == "__main__":
    main()