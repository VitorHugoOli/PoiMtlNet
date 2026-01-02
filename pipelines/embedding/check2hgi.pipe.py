"""
Check2HGI Pipeline - Train Check2HGI embeddings for multiple states.

Stages: preprocess -> train

Usage: python pipelines/embedding/check2hgi.pipe.py
"""

import logging
from argparse import Namespace
from datetime import datetime

from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.check2hgi.check2hgi import create_embedding
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
    'Texas': Resources.TL_TX
}

CHECK2HGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    num_layers=2,
    attention_head=4,
    alpha_c2p=0.4,
    alpha_p2r=0.3,
    alpha_r2c=0.3,
    lr=0.001,
    gamma=1.0,
    max_norm=0.9,
    epoch=500,
    mini_batch_threshold=5_000_000,
    batch_size=2**13,
    num_neighbors=10,
    device=str("cpu"),
    shapefile=None,  # Will be set per state
    force_preprocess=True,
    edge_type='user_sequence',
    temporal_decay=3600.0,
    use_compile=True,
    use_amp=False,
)


# =============================================================================
# PIPELINE
# =============================================================================

def process_state(name: str, shapefile) -> bool:
    """Run pipeline for a single state."""
    try:
        CHECK2HGI_CONFIG.shapefile = shapefile

        logger.info(f"[1/2] Creating embeddings (Preprocess + Train): {name}")
        create_embedding(state=name, args=CHECK2HGI_CONFIG)

        logger.info(f"[2/2] Creating inputs for HGI: {name}")
        create_input(state=name, embedding_engine=EmbeddingEngine.CHECK2HGI,use_checkin_embeddings=True)

        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(f"Check2HGI Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={CHECK2HGI_CONFIG.dim}")

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
