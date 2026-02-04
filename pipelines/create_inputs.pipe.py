#!/usr/bin/env python3
"""
MTL Input Generation Pipeline

Orchestrates input generation for all configured states and embedding engines.
Pure orchestration - business logic in src/etl/mtl_input/builders.py

This pipeline coordinates the execution of input generation functions from the
mtl_input module. It handles state selection, parallel processing, and logging.

Usage:
    python pipelines/create_inputs.pipe.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from configs.paths import EmbeddingEngine
from etl.mtl_input.builders import (
    generate_category_input,
    generate_next_input_from_poi,
    generate_next_input_from_checkins,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

STATE_CONFIGS: List[Tuple[str, EmbeddingEngine, bool]] = [
    ("alabama", EmbeddingEngine.HGI, False),
    # ("arizona", EmbeddingEngine.HGI, False),
    # ("georgia", EmbeddingEngine.HGI, False),
    # ("florida", EmbeddingEngine.HGI, False),
    # ("california", EmbeddingEngine.HGI, False),
    # ("texas", EmbeddingEngine.HGI, False),

    # Time2Vec uses check-in-level embeddings
    # ("alabama", EmbeddingEngine.TIME2VEC, True),
    # ("arizona", EmbeddingEngine.TIME2VEC, True),
    # ("georgia", EmbeddingEngine.TIME2VEC, True),
    # ("florida", EmbeddingEngine.TIME2VEC, True),
    # ("california", EmbeddingEngine.TIME2VEC, True),
    # ("texas", EmbeddingEngine.TIME2VEC, True),
]

MAX_WORKERS = 4  # Adjust based on available memory

# ============================================================================
# Orchestration
# ============================================================================

def process_state(state: str, engine: EmbeddingEngine, use_checkin_embeddings: bool = False) -> bool:
    """
    Generate category and next-POI inputs for a single state/engine combination.

    Pure orchestration - calls mtl_input module functions only.

    Args:
        state: State name (e.g., 'alabama')
        engine: Embedding engine
        use_checkin_embeddings: If True, use check-in-level embeddings (Time2Vec);
                                 if False, use POI-level embeddings (HGI, DGI, etc.)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"[{state}/{engine.value}] Starting input generation...")

        logger.info(f"  [1/2] Generating category input")
        generate_category_input(state, engine)

        logger.info(f"  [2/2] Generating next-POI input")
        if use_checkin_embeddings:
            generate_next_input_from_checkins(state, engine)
        else:
            generate_next_input_from_poi(state, engine)

        logger.info(f"[{state}/{engine.value}] ✓ Complete")
        return True

    except Exception as e:
        logger.error(f"[{state}/{engine.value}] ✗ Failed: {e}", exc_info=True)
        return False


def run_pipeline() -> dict:
    """
    Process all configured states.

    Returns:
        Dictionary mapping state/engine keys to success status
    """
    logger.info("=" * 80)
    logger.info(f"MTL Input Pipeline - {len(STATE_CONFIGS)} configuration(s)")
    logger.info("=" * 80)

    start = datetime.now()
    results = {}

    # Process in parallel
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(process_state, state, engine, use_checkin): (state, engine)
            for state, engine, use_checkin in STATE_CONFIGS
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            state, engine = future_to_config[future]
            key = f"{state}/{engine.value}"
            try:
                results[key] = future.result()
            except Exception as e:
                logger.error(f"[{key}] Exception in worker: {e}")
                results[key] = False

    # Summary
    duration = (datetime.now() - start).total_seconds()
    success = sum(results.values())

    logger.info("=" * 80)
    logger.info(f"Pipeline completed: {success}/{len(STATE_CONFIGS)} successful in {duration / 60:.1f}min")
    logger.info("=" * 80)

    for key, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {key}")

    return results


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)
