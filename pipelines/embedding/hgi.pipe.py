"""
HGI Pipeline - Train HGI embeddings for multiple states.

Stages: preprocess(graph) -> poi2vec -> preprocess(final) -> create embeddings -> generate inputs

Usage: python pipelines/embedding/hgi.pipe.py
"""

import logging
from argparse import Namespace
from datetime import datetime

from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.hgi.hgi import create_embedding
from embeddings.hgi.preprocess import preprocess_hgi
from embeddings.hgi.poi2vec import train_poi2vec
from etl.create_input import create_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATES = {
    # Local
    'Alabama': Resources.TL_AL,
    # 'Arizona': Resources.TL_AZ,
    # 'Georgia': Resources.TL_GA,
    # Articles
    # 'Florida': Resources.TL_FL,
    # 'California': Resources.TL_CA,
    # 'Texas': Resources.TL_TX,
}

HGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    attention_head=4,
    alpha=0.5,
    lr=0.001,
    gamma=1.0,
    max_norm=0.9,
    epoch=2000,
    poi2vec_epochs=100,
    force_preprocess=True,
    no_poi2vec=False,
    device='cpu',
    shapefile=None  # Will be set per state
)


# =============================================================================
# PIPELINE
# =============================================================================

def process_state(name: str, shapefile, cta_file=None) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        HGI_CONFIG.shapefile = shapefile

        # 1. First pass preprocessing to generate graph files (edges.csv, pois.csv) for POI2Vec
        # We need this because train_poi2vec requires these files to exist.
        logger.info(f"[1/5] Pre-preprocessing (Graph Generation): {name}")
        preprocess_hgi(city=name, city_shapefile=str(shapefile), poi_emb_path=None, cta_file=cta_file)

        # 2. Train POI2Vec
        logger.info(f"[2/5] Training POI2Vec: {name}")
        poi_emb_path = None
        if not HGI_CONFIG.no_poi2vec:
            poi_emb_path = train_poi2vec(city=name, epochs=HGI_CONFIG.poi2vec_epochs, embedding_dim=HGI_CONFIG.dim,
                                         device=HGI_CONFIG.device)

        # 3. Second pass preprocessing with learned embeddings
        # This regenerates the graph data pickle including the learned POI embeddings
        logger.info(f"[3/5] Preprocessing with Embeddings: {name}")
        preprocess_hgi(city=name, city_shapefile=str(shapefile),
                       poi_emb_path=str(poi_emb_path) if poi_emb_path else None, cta_file=cta_file)

        # 4. Train HGI
        # HGI_CONFIG.force_preprocess is False, so it will use the data we just generated
        logger.info(f"[4/5] Creating embeddings: {name}")
        create_embedding(state=name, args=HGI_CONFIG)

        # 5. Generate Inputs
        logger.info(f"[5/5] Generating inputs: {name}")
        create_input(state=name, embedding_engine=EmbeddingEngine.HGI)

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
