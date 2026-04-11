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

# Default configuration.
#
# Encoder variant: 'paper' (SphereMixScale, Eq.8 + sphereC terms from the
# official gengchenmai/sphere2vec repo). This is the paper-faithful variant
# and was adopted as the pipe default on 2026-04-11 after a 5-fold × 50-epoch
# Alabama ablation showed:
#     - rbf  cat F1 = 14.15% ± 1.26   (higher mean, higher std)
#     - paper cat F1 = 13.35% ± 0.65   (~½ std, ~35% faster training)
# The gap is within 1σ of rbf's std, so the variants are statistically tied.
# The paper variant wins on (a) stability across folds and (b) honest
# citation of Mai et al. 2023. See research/embeddings/sphere2vec/README.md
# for the full ablation table.
#
# To revert to the notebook's rbf variant, change encoder_variant='rbf' and
# optionally drop min_radius/max_radius (they are paper-variant-only).
#
# Architecture / loss / pos_radius are kept exactly as the notebook source.
# Batch size + dataset are tuned for speed: bs=4096 with the vectorized
# FastContrastiveSpatialDataset gives ~9× faster epoch times on MPS than
# the notebook's bs=64 + per-item dataset, with no observed quality loss
# on Alabama (validated against the notebook-mode 50-epoch baseline).
#
# To reproduce the canonical notebook training exactly, set
#     encoder_variant='rbf', batch_size=64, legacy_dataset=True
SPHERE2VEC_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    spa_embed_dim=128,
    num_scales=32,
    # Scale params are used by both variants but with different semantics:
    # - rbf:   min_scale/max_scale are the RBF kernel scales (dimensionless)
    # - paper: min_radius/max_radius are the geometric frequency range
    min_scale=10,
    max_scale=1e7,
    num_centroids=256,      # ignored by paper variant
    # Paper-variant-specific radii (upstream defaults from the
    # official SphereMixScaleSpatialRelationEncoder class):
    min_radius=10.0,
    max_radius=10000.0,
    ffn_hidden_dim=512,
    ffn_num_hidden_layers=1,
    ffn_dropout_rate=0.5,
    ffn_act="relu",
    ffn_use_layernormalize=True,
    ffn_skip_connection=True,
    epoch=50,
    batch_size=4096,        # was 64 (notebook); 9× faster on MPS at this size
    lr=1e-3,
    tau=0.15,
    pos_radius=0.01,
    seed=42,
    num_workers=2,
    eval_batch_size=10000,
    device=DEVICE,
    legacy_dataset=False,   # use FastContrastiveSpatialDataset
    encoder_variant="paper",  # paper-faithful Eq.8 SphereMixScale (default)
    eval_inference=False,
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
