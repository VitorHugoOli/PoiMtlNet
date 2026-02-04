"""
Space2Vec Pipeline - Train Space2Vec embeddings for multiple states.
Stages: create embeddings -> generate inputs
Usage: python pipelines/embedding/space2vec.pipe.py
"""
import logging
from argparse import Namespace
from datetime import datetime

import torch
from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.space2vec.space2vec import create_embedding
from etl.mtl_input.builders import generate_category_input, generate_next_input_from_poi

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

# Default configuration matching space2vec.py defaults
SPACE2VEC_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    spa_embed_dim=128,
    freq_num=16,
    max_radius=50,
    min_radius=0.02,
    epoch=40,
    batch_size=256,
    lr=3e-4,
    r_pos_km=10.0,
    r_neg_km=70.0,
    k_pos_per_i=8,
    k_neg_per_i=16,
    hard_neg_ratio=0.0,
    tau=0.15,
    max_grad_norm=1.0,
    seed=42,
    device=DEVICE,
    force_pairs=True,
    use_torch_encoder=True,
    no_torch_encoder=False,
    loss_type="bce",
    deduplicate=True,
    compile=True,
    max_pairs=None,
    block_size=500_000,
    amp=True,
)

# Ensure device is correct type
if isinstance(SPACE2VEC_CONFIG.device, str):
    SPACE2VEC_CONFIG.device = torch.device(SPACE2VEC_CONFIG.device)


# =============================================================================
# PIPELINE
# =============================================================================
def process_state(name: str) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        # NOTE: space2vec.create_embedding handles preprocessing (pair generation) internally
        logger.info(f"[1/2] Creating embeddings: {name}")
        create_embedding(state=name, args=SPACE2VEC_CONFIG)
        # logger.info(f"[2/2] Generating inputs: {name}")
        # generate_category_input(name, EmbeddingEngine.SPACE2VEC)
        # generate_next_input_from_poi(name, EmbeddingEngine.SPACE2VEC)
        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(f"Space2Vec Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={SPACE2VEC_CONFIG.dim}")

    start = datetime.now()
    results = {}

    for name in STATES:
        results[name] = process_state(name)

    duration = (datetime.now() - start).total_seconds()

    # Summary
    success = sum(results.values())
    logger.info(f"Completed: {success}/{len(STATES)} succeeded in {duration / 60:.1f}min")
    for name in STATES:
        ok = results.get(name, False)
        logger.info(f"  {'✓' if ok else '✗'} {name}")
    return results


if __name__ == '__main__':
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)
