"""
Sphere2Vec Pipeline - Train Sphere2Vec embeddings for multiple states.
Stages: create embeddings -> generate inputs (category + next-from-poi)
Usage: python pipelines/embedding/sphere2vec.pipe.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
_src = str(_root / "src")
_research = str(_root / "research")
if _src not in sys.path:
    sys.path.insert(0, _src)
if _research not in sys.path:
    sys.path.insert(0, _research)

import logging
from argparse import Namespace
from datetime import datetime

import torch
from configs.globals import DEVICE
from configs.model import InputsConfig
from configs.paths import EmbeddingEngine
from data.inputs.builders import (
    generate_category_input,
    generate_next_input_from_poi,
)
from embeddings.sphere2vec.sphere2vec import create_embedding

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

# Default configuration matching the source notebook
SPHERE2VEC_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    spa_embed_dim=128,
    num_scales=32,
    min_scale=10,
    max_scale=1e7,
    num_centroids=256,
    ffn_hidden_dim=512,
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_act="relu",
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    epoch=50,
    batch_size=64,
    lr=1e-3,
    tau=0.15,
    pos_radius=0.01,
    seed=42,
    num_workers=2,
    eval_batch_size=10000,
    device=DEVICE,
)

# Ensure device is correct type
if isinstance(SPHERE2VEC_CONFIG.device, str):
    SPHERE2VEC_CONFIG.device = torch.device(SPHERE2VEC_CONFIG.device)


# =============================================================================
# PIPELINE
# =============================================================================
def process_state(name: str) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        logger.info(f"[1/3] Creating embeddings: {name}")
        create_embedding(state=name, args=SPHERE2VEC_CONFIG)
        logger.info(f"[2/3] Generating category input: {name}")
        generate_category_input(name, EmbeddingEngine.SPHERE2VEC)
        logger.info(f"[3/3] Generating next-POI input: {name}")
        generate_next_input_from_poi(name, EmbeddingEngine.SPHERE2VEC)
        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    logger.info(
        f"Sphere2Vec Pipeline - {len(STATES)} state(s) | device={DEVICE} | dim={SPHERE2VEC_CONFIG.dim}"
    )

    start = datetime.now()
    results = {}

    for name in STATES:
        results[name] = process_state(name)

    duration = (datetime.now() - start).total_seconds()

    success = sum(results.values())
    logger.info(f"Completed: {success}/{len(STATES)} succeeded in {duration / 60:.1f}min")
    for name in STATES:
        ok = results.get(name, False)
        status = 'OK' if ok else 'FAIL'
        logger.info(f"  [{status}] {name}")
    return results


if __name__ == '__main__':
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)
