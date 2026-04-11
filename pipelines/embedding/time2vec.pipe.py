"""
Time2Vec Pipeline - Train Time2Vec embeddings for multiple states.
Stages: create embeddings -> generate inputs
Usage: python pipelines/embedding/time2vec.pipe.py
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
from configs.paths import EmbeddingEngine
from configs.model import InputsConfig
from embeddings.time2vec.time2vec import create_embedding
from data.inputs.builders import generate_category_input, generate_next_input_from_checkins

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATES = [
    # Local
    # 'Alabama',
    # 'Arizona',
    # 'Georgia',
    # Articles
    # 'Florida',
    'California',
    # 'Texas',
]

# Default configuration matching time2vec.py defaults
TIME2VEC_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    out_features=64,
    activation='sin',
    lr=1e-3,
    epoch=100,
    batch_size=2048,  # ~3.5x faster than 256, identical final loss on Alabama
    r_pos_hours=1.0,
    r_neg_hours=24.0,
    max_pairs=2_000_000,
    k_neg_per_i=5,
    max_pos_per_i=20,
    seed=42,
    tau=0.3,
    device=DEVICE,  # MPS on Apple Silicon is 2.4x faster than CPU with the
                    # new bs=2048 + compile path (was the opposite for the
                    # old bs=256 + DataLoader path — kernel-launch overhead
                    # dominated there). Loss stays bit-identical. Falls back
                    # to CPU automatically via DEVICE auto-detect.
    compile=True,  # ~10% extra speedup via torch.compile, bit-identical loss
    # Paper-faithfulness fix (Rec 1, 2026-04-11): wrap-aware feature-space
    # pair sampling. Delivers +0.81 ± 0.19pp F1 on Alabama next-task vs the
    # legacy absolute-time sampler. See research/embeddings/time2vec/README.md.
    sampling_mode="feat_space",
    r_pos_feat=0.03,
    r_neg_feat=0.30,
    no_train=False,
)


# =============================================================================
# PIPELINE
# =============================================================================
def process_state(name: str) -> bool:
    """Run all pipeline stages for a single state."""
    try:
        logger.info(f"[1/2] Creating embeddings: {name}")
        create_embedding(state=name, args=TIME2VEC_CONFIG)

        logger.info(f"[2/2] Generating inputs: {name}")
        generate_category_input(name, EmbeddingEngine.TIME2VEC)
        generate_next_input_from_checkins(name, EmbeddingEngine.TIME2VEC)
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
