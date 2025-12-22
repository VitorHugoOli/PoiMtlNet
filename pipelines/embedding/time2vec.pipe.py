"""
Time2Vec Pipeline - Train Time2Vec embeddings for multiple states.
Stages: create embeddings -> generate inputs
Usage: python pipelines/embedding/time2vec.pipe.py
"""
import logging
from argparse import Namespace
from datetime import datetime

import torch
from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.time2vec.time2vec import create_embedding
from etl.create_input import create_input

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATES = [
    # Local
    'Alabama',
    'Arizona',
    'Georgia',
    # Articles
    'Florida',
    'California',
    'Texas',
]

# Default configuration matching time2vec.py defaults
TIME2VEC_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    out_features=64,
    activation='sin',
    lr=1e-3,
    epoch=100,
    batch_size=256,
    r_pos_hours=1.0,
    r_neg_hours=24.0,
    max_pairs=2_000_000,
    k_neg_per_i=5,
    max_pos_per_i=20,
    seed=42,
    tau=0.3,
    device=DEVICE,
)

# Ensure device is correct type
if isinstance(TIME2VEC_CONFIG.device, str):
    TIME2VEC_CONFIG.device = torch.device(TIME2VEC_CONFIG.device)


# =============================================================================
# PIPELINE
# =============================================================================
def process_state(name: str) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        logger.info(f"[1/2] Creating embeddings: {name}")
        create_embedding(state=name, args=TIME2VEC_CONFIG)
        
        # logger.info(f"[2/2] Generating inputs: {name}")
        # create_input(state=name, embedd_eng=EmbeddingEngine.TIME2VEC)
        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(f"Time2Vec Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={TIME2VEC_CONFIG.dim}")
    
    start = datetime.now()
    results = {name: process_state(name) for name in STATES}
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
