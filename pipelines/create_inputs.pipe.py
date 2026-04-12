#!/usr/bin/env python3
"""Input generation pipeline — generate category + next-POI inputs. Usage: python pipelines/create_inputs.pipe.py"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from configs.paths import EmbeddingEngine
from data.inputs.builders import (
    generate_category_input,
    generate_next_input_from_poi,
    generate_next_input_from_checkins,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SETTINGS
# =============================================================================

MAX_WORKERS = 4

# =============================================================================
# STATES
# Ordered dict — execution follows insertion order, MAX_WORKERS at a time.
# Keys: 'engine' (EmbeddingEngine), 'use_checkin' (bool, default False).
# NOTE: FUSION engine is NOT supported here — use pipelines/fusion.pipe.py instead.
# =============================================================================

STATES = {
    'Alabama': {'engine': EmbeddingEngine.HGI, 'use_checkin': False},
    # 'Arizona': {'engine': EmbeddingEngine.HGI, 'use_checkin': False},
    # 'Georgia': {'engine': EmbeddingEngine.HGI, 'use_checkin': False},
    # 'Florida': {'engine': EmbeddingEngine.HGI, 'use_checkin': False},
    # 'California': {'engine': EmbeddingEngine.HGI, 'use_checkin': False},
    # 'Texas': {'engine': EmbeddingEngine.HGI, 'use_checkin': False},

    # Time2Vec uses check-in-level embeddings
    # 'Alabama_t2v': {'engine': EmbeddingEngine.TIME2VEC, 'use_checkin': True},
}

# =============================================================================
# PIPELINE
# =============================================================================


def process_state(name: str, state_cfg: dict) -> bool:
    """Generate category and next-POI inputs for a single state/engine."""
    try:
        engine = state_cfg['engine']
        use_checkin = state_cfg.get('use_checkin', False)

        if engine == EmbeddingEngine.FUSION:
            raise ValueError("FUSION not supported here — use pipelines/fusion.pipe.py")

        logger.info(f"[{name}/{engine.value}] Starting input generation...")

        logger.info(f"  [1/2] Generating category input")
        generate_category_input(name, engine)

        logger.info(f"  [2/2] Generating next-POI input")
        if use_checkin:
            generate_next_input_from_checkins(name, engine)
        else:
            generate_next_input_from_poi(name, engine)

        logger.info(f"[{name}/{engine.value}] Complete")
        return True

    except Exception as e:
        logger.error(f"[{name}] Failed: {e}", exc_info=True)
        return False


def run_pipeline() -> dict:
    """Process all configured states in order, MAX_WORKERS at a time."""
    logger.info(f"Input Pipeline - {len(STATES)} configuration(s)")

    start = datetime.now()
    results = {}
    states = list(STATES.items())

    for i in range(0, len(states), MAX_WORKERS):
        chunk = states[i:i + MAX_WORKERS]
        if MAX_WORKERS == 1:
            for name, cfg in chunk:
                results[name] = process_state(name, cfg)
        else:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_state, name, dict(cfg)): name
                    for name, cfg in chunk
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()

    duration = (datetime.now() - start).total_seconds()
    success = sum(results.values())
    logger.info(f"Completed: {success}/{len(STATES)} succeeded in {duration / 60:.1f}min")
    for name, ok in results.items():
        logger.info(f"  {'✓' if ok else '✗'} {name}")

    return results


if __name__ == '__main__':
    results = run_pipeline()
    exit(0 if all(results.values()) else 1)
