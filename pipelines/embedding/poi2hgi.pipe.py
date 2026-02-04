"""
POI2HGI Pipeline - Train POI2HGI embeddings for multiple states.

Uses temporal patterns (not category) as node features for POI embeddings.

Stages: preprocess(temporal features) -> train poi2hgi -> generate inputs

Usage: python pipelines/embedding/poi2hgi.pipe.py
"""

import logging
from argparse import Namespace
from datetime import datetime

from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.poi2hgi.poi2hgi import create_embedding
from embeddings.poi2hgi.preprocess import preprocess_poi2hgi
from etl.mtl_input.builders import generate_category_input, generate_next_input_from_poi

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

POI2HGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    attention_head=4,
    alpha=0.5,
    lr=0.001,
    gamma=1.0,
    max_norm=0.9,
    epoch=2000,
    force_preprocess=True,
    device="cpu",
    shapefile=None  # Will be set per state
)


# =============================================================================
# PIPELINE
# =============================================================================

def process_state(name: str, shapefile, cta_file=None) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        POI2HGI_CONFIG.shapefile = shapefile

        # 1. Preprocess with temporal features
        logger.info(f"[1/3] Preprocessing with temporal features: {name}")
        preprocess_poi2hgi(city=name, city_shapefile=str(shapefile), cta_file=cta_file)

        # 2. Train POI2HGI
        logger.info(f"[2/3] Creating embeddings: {name}")
        POI2HGI_CONFIG.force_preprocess = False  # Use preprocessed data
        create_embedding(state=name, args=POI2HGI_CONFIG)

        # 3. Generate Inputs
        logger.info(f"[3/3] Generating inputs: {name}")
        generate_category_input(name, EmbeddingEngine.POI2HGI)
        generate_next_input_from_poi(name, EmbeddingEngine.POI2HGI)

        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(f"POI2HGI Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={POI2HGI_CONFIG.dim}")

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
