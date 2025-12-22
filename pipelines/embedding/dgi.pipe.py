"""
DGI Pipeline - Train DGI embeddings for multiple states.

Stages: preprocess -> create embeddings -> generate inputs

Usage: python pipelines/embedding/dgi.pipe.py
"""

import logging
from argparse import Namespace
from datetime import datetime

from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.dgi.dgi import create_embedding
from embeddings.dgi.preprocess import preprocess_dgi
from etl.create_input import create_input

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

DGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    lr=0.01,
    gamma=1.0,
    epoch=70,
    max_norm=0.9,
    device=DEVICE,
)

# =============================================================================
# PIPELINE
# =============================================================================

def process_state(name: str, shapefile, cta_file=None) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        logger.info(f"[1/3] Preprocessing: {name}")
        preprocess_dgi(city=name, city_shapefile=str(shapefile), cta_file=cta_file)

        logger.info(f"[2/3] Creating embeddings: {name}")
        create_embedding(city=name, args=DGI_CONFIG)

        logger.info(f"[3/3] Generating inputs: {name}")
        create_input(state=name, embedd_eng=EmbeddingEngine.DGI)

        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(f"DGI Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={DGI_CONFIG.dim}")

    start = datetime.now()
    results = {name: process_state(name, shp) for name, shp in STATES.items()}
    duration = (datetime.now() - start).total_seconds()

    # Summary
    success = sum(results.values())
    logger.info(f"Completed: {success}/{len(STATES)} succeeded in {duration/60:.1f}min")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")

    return results


if __name__ == '__main__':
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)