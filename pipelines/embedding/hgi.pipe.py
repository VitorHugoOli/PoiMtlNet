"""
HGI Pipeline - Train HGI embeddings for multiple states.

Reference flow (from hgi_texas.py / CLAUDE.md):
  Phase 3a: preprocess_hgi(no embeddings) → Delaunay graph, edges.csv, pois.csv
  Phase 3b-d: train_poi2vec() → fclass walks, train, reconstruct POI embeddings
  Phase 4: preprocess_hgi(with embeddings) → full data dict, save pickle
  Phase 5: train_hgi() → train HGI model, save POI + region embeddings
  + create_input() → generate downstream task inputs

Usage: python pipelines/embedding/hgi.pipe.py
"""

import logging
import pickle as pkl
from argparse import Namespace
from datetime import datetime

import torch

from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine, IoPaths
from configs.model import InputsConfig
from embeddings.hgi.hgi import train_hgi
from embeddings.hgi.preprocess import preprocess_hgi
from embeddings.hgi.poi2vec import train_poi2vec
from etl.mtl_input.builders import generate_category_input, generate_next_input_from_poi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATES = {
    # Local
    'Alabama': Resources.TL_AL,
    'Arizona': Resources.TL_AZ,
    'Georgia': Resources.TL_GA,
    # Articles
    'Florida': Resources.TL_FL,
    'California': Resources.TL_CA,
    'Texas': Resources.TL_TX,
}

HGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    attention_head=4,
    alpha=0.5,
    lr=0.001,
    gamma=1.0,
    max_norm=0.9,
    epoch=2000,
    poi2vec_epochs=6,
    force_preprocess=True,

    device='cpu',
    shapefile=None  # Will be set per state
)


# =============================================================================
# PIPELINE
# =============================================================================

def process_state(name: str, shapefile, cta_file=None) -> bool:
    """Run full HGI pipeline for a single state.

    Steps:
        1. Preprocess (graph only) → edges.csv, pois.csv for POI2Vec
        2. Train POI2Vec → fclass embeddings → POI embeddings
        3. Preprocess (with embeddings) → full data dict → save pickle
        4. Train HGI → POI + region embeddings
        5. Generate downstream inputs
    """
    try:
        HGI_CONFIG.shapefile = shapefile

        # 1. First pass: build Delaunay graph → edges.csv + pois.csv (needed by POI2Vec)
        logger.info(f"[1/5] Building graph (Delaunay + edges.csv + pois.csv): {name}")
        preprocess_hgi(city=name, city_shapefile=str(shapefile), poi_emb_path=None, cta_file=cta_file)

        # 2. Train POI2Vec (phases 3b-3d: walks → fclass embeddings → POI embeddings)
        logger.info(f"[2/5] Training POI2Vec: {name}")
        poi_emb_path = train_poi2vec(
            city=name, epochs=HGI_CONFIG.poi2vec_epochs,
            embedding_dim=HGI_CONFIG.dim,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 3. Second pass: preprocess with learned embeddings → save pickle
        logger.info(f"[3/5] Preprocessing with embeddings: {name}")
        data = preprocess_hgi(
            city=name, city_shapefile=str(shapefile),
            poi_emb_path=str(poi_emb_path),
            cta_file=cta_file
        )

        graph_data_file = IoPaths.HGI.get_graph_data_file(name)
        graph_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_data_file, "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Saved graph data: {graph_data_file}")

        # 4. Train HGI model (loads pickle, trains, saves embeddings)
        logger.info(f"[4/5] Training HGI: {name}")
        train_hgi(name, HGI_CONFIG)

        # 5. Generate downstream inputs
        logger.info(f"[5/5] Generating inputs: {name}")
        generate_category_input(name, EmbeddingEngine.HGI)
        generate_next_input_from_poi(name, EmbeddingEngine.HGI)

        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(f"HGI Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={HGI_CONFIG.dim}")

    start = datetime.now()
    results = {name: process_state(name, shp) for name, shp in STATES.items()}
    duration = (datetime.now() - start).total_seconds()

    # Summary
    success = sum(results.values())
    logger.info(f"Completed: {success}/{len(STATES)} succeeded in {duration / 60:.1f}min")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")

    return results


if __name__ == '__main__':
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)
