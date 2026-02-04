#!/usr/bin/env python3
"""
Multi-Embedding Fusion Pipeline

Generates fused inputs for MTLnet by concatenating multiple embeddings:
- Category task: POI-level embeddings (e.g., Space2Vec + HGI) → 128 dimensions
- Next-POI task: POI-level + check-in-level (e.g., Space2Vec + Time2Vec) → 128 dimensions

This pipeline orchestrates the fusion process by:
  1. Loading individual embeddings (HGI, Space2Vec, Time2Vec, etc.)
  2. Aligning embeddings by POI ID or check-in ID
  3. Concatenating aligned embeddings
  4. Generating category and next-POI inputs

Usage:
    python pipelines/fusion.pipe.py

Configuration:
    - STATES: List of states to process (e.g., ['alabama', 'florida'])
    - FUSION_PRESET: Preset fusion configuration to use
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from configs.embedding_fusion import get_preset
from configs.paths import IoPaths
from etl.mtl_input.fusion import MultiEmbeddingInputGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

STATES: List[str] = [
    # "alabama",    # Start with smaller state for testing
    # "arizona",
    # "georgia",
    "florida",
    # "california",
    # "texas",
]

FUSION_PRESET = "space_hgi_time"  # Options: space_hgi_time, hgi_time, space_time
MAX_WORKERS = 2  # Adjust based on available memory (fusion is memory-intensive)


# ============================================================================
# Orchestration Functions
# ============================================================================

def process_state(state: str, fusion_preset: str) -> bool:
    """
    Generate fusion inputs for a single state.

    Pure orchestration - calls MultiEmbeddingInputGenerator only.

    Args:
        state: State name (e.g., 'alabama')
        fusion_preset: Name of fusion preset to use

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"[{state}] Starting fusion pipeline...")

        # Load fusion configuration
        fusion_config = get_preset(fusion_preset)
        logger.info(f"[{state}] Using preset '{fusion_preset}':")
        logger.info(f"  Category: {fusion_config.get_category_dim()} dims")
        logger.info(f"  Next-POI: {fusion_config.get_next_dim()} dims")

        # Initialize generator
        generator = MultiEmbeddingInputGenerator(state, fusion_config)

        # Get output paths
        category_path = IoPaths.get_fusion_category(state)
        sequences_path = IoPaths.get_fusion_seq_next(state)
        next_path = IoPaths.get_fusion_next(state)

        # Create directories
        category_path.parent.mkdir(parents=True, exist_ok=True)
        sequences_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate category input
        logger.info(f"[{state}] [1/2] Generating category input...")
        generator.generate_category_input(str(category_path))
        logger.info(f"  Saved to: {category_path}")

        # Generate next-POI input
        logger.info(f"[{state}] [2/2] Generating next-POI input...")
        generator.generate_next_input(str(sequences_path), str(next_path))
        logger.info(f"  Saved to: {next_path}")

        # Clean up cache to free memory
        generator.loader.clear_cache()

        logger.info(f"[{state}] ✓ Complete")
        return True

    except Exception as e:
        logger.error(f"[{state}] ✗ Failed: {e}", exc_info=True)
        return False


def run_pipeline() -> dict:
    """
    Process all configured states.

    Returns:
        Dictionary mapping state keys to success status
    """
    logger.info("=" * 80)
    logger.info(f"Multi-Embedding Fusion Pipeline - {len(STATES)} state(s)")
    logger.info(f"Preset: {FUSION_PRESET}")
    logger.info("=" * 80)

    start = datetime.now()
    results = {}

    # Process states (sequential for memory safety, or parallel with low MAX_WORKERS)
    if MAX_WORKERS == 1:
        # Sequential processing
        for state in STATES:
            results[state] = process_state(state, FUSION_PRESET)
    else:
        # Parallel processing with limited workers
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_state = {
                executor.submit(process_state, state, FUSION_PRESET): state
                for state in STATES
            }

            for future in as_completed(future_to_state):
                state = future_to_state[future]
                try:
                    results[state] = future.result()
                except Exception as e:
                    logger.error(f"[{state}] Exception in worker: {e}")
                    results[state] = False

    # Summary
    duration = (datetime.now() - start).total_seconds()
    success = sum(results.values())

    logger.info("=" * 80)
    logger.info(f"Pipeline completed: {success}/{len(STATES)} successful in {duration / 60:.1f}min")
    logger.info("=" * 80)

    for state, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {state}")

    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)
