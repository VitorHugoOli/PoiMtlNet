"""HGI Pipeline — train HGI embeddings (5-stage), generate inputs. Usage: python pipelines/embedding/hgi.pipe.py"""

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
import pickle as pkl
from argparse import Namespace
from copy import copy
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from configs.globals import DEVICE
from configs.paths import Resources, EmbeddingEngine, IoPaths
from configs.model import InputsConfig
from embeddings.hgi.hgi import train_hgi
from embeddings.hgi.preprocess import preprocess_hgi
from embeddings.hgi.poi2vec import train_poi2vec
from data.inputs.builders import generate_category_input, generate_next_input_from_poi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SETTINGS
# =============================================================================

MAX_WORKERS = 1

# =============================================================================
# CONFIG
# =============================================================================

# Mirrors RightBank/HGI/train.py (Huang et al., ISPRS 2023).
# lr=0.006 requires the 40-epoch LinearWarmup in train_hgi — don't change one without the other.
# cross_region_weight (w_r, Eq. 2): scales inversely with POI density.
# Only Alabama has been swept; other values are extrapolated from density. See hgi/README.md §5.
CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    alpha=0.5,
    attention_head=4,
    lr=0.006,
    gamma=1.0,
    max_norm=0.9,
    epoch=2000,
    warmup_period=40,
    poi2vec_epochs=100,
    force_preprocess=True,
    cross_region_weight=0.7,
    device='cpu',  # CPU pinned: HGI small-ops workload is ~176x slower on MPS
    shapefile=None,
)

# =============================================================================
# STATES
# Ordered dict — execution follows insertion order, MAX_WORKERS at a time.
# Each entry: 'StateName': {'shapefile': <Resource>, ...overrides, 'config': <Namespace>}
# When 'config' key is absent, CONFIG (default) is used.
# =============================================================================

STATES = {
    # 'Alabama':    {'shapefile': Resources.TL_AL, 'cross_region_weight': 0.7},   # swept, confirmed
    # 'Arizona':    {'shapefile': Resources.TL_AZ, 'cross_region_weight': 0.7},   # extrapolated (sparse)
    # 'Georgia':    {'shapefile': Resources.TL_GA},
    'Florida':    {'shapefile': Resources.TL_FL, 'cross_region_weight': 0.7},   # extrapolated (dense)
    # 'California': {'shapefile': Resources.TL_CA, 'cross_region_weight': 0.6},   # extrapolated (medium)
    # 'Texas':      {'shapefile': Resources.TL_TX, 'cross_region_weight': 0.7},   # extrapolated (sparse)
}

# =============================================================================
# PIPELINE
# =============================================================================


def process_state(name: str, state_cfg: dict) -> bool:
    """Run full HGI pipeline for a single state (5 stages)."""
    try:
        state_cfg = dict(state_cfg)
        base = state_cfg.pop('config', CONFIG)
        config = copy(base)
        config.shapefile = state_cfg.pop('shapefile')
        for k, v in state_cfg.items():
            setattr(config, k, v)

        w_r = config.cross_region_weight
        logger.info(f"[setup] {name}: cross_region_weight w_r={w_r}")

        # 1. Build Delaunay graph -> edges.csv + pois.csv (needed by POI2Vec)
        logger.info(f"[1/5] Building graph: {name}")
        preprocess_hgi(
            city=name, city_shapefile=str(config.shapefile), poi_emb_path=None,
            cta_file=None, cross_region_weight=w_r,
        )

        # 2. Train POI2Vec (walks -> fclass embeddings -> POI embeddings)
        logger.info(f"[2/5] Training POI2Vec: {name}")
        poi_emb_path = train_poi2vec(
            city=name, epochs=config.poi2vec_epochs,
            embedding_dim=config.dim,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 3. Preprocess with learned embeddings -> save pickle
        logger.info(f"[3/5] Preprocessing with embeddings: {name}")
        data = preprocess_hgi(
            city=name, city_shapefile=str(config.shapefile),
            poi_emb_path=str(poi_emb_path),
            cta_file=None, cross_region_weight=w_r,
        )

        graph_data_file = IoPaths.HGI.get_graph_data_file(name)
        graph_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_data_file, "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Saved graph data: {graph_data_file}")

        # 4. Train HGI model
        logger.info(f"[4/5] Training HGI: {name}")
        train_hgi(name, config)

        # 5. Generate downstream inputs
        logger.info(f"[5/5] Generating inputs: {name}")
        generate_category_input(name, EmbeddingEngine.HGI)
        generate_next_input_from_poi(name, EmbeddingEngine.HGI)

        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states in order, MAX_WORKERS at a time."""
    logger.info(
        f"HGI Pipeline - {len(STATES)} state(s) | "
        f"hgi_device={CONFIG.device} | global_device={DEVICE} | dim={CONFIG.dim}"
    )

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
