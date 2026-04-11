"""
HGI Pipeline - Train HGI embeddings for multiple states.

Reference flow (from hgi_texas.py / CLAUDE.md):
  Phase 3a: preprocess_hgi(no embeddings) → Delaunay graph, edges.csv, pois.csv
  Phase 3b-d: train_poi2vec() → fclass walks, train, reconstruct POI embeddings
  Phase 4: preprocess_hgi(with embeddings) → full data dict, save pickle
  Phase 5: train_hgi() → train HGI model, save POI + region embeddings
  + create_input() → generate downstream task inputs

Usage: python pipelines/embedding/hgi.pipe.py
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
import pickle as pkl
from argparse import Namespace
from datetime import datetime

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

STATES = {
    # Local
    'Alabama': Resources.TL_AL,
    'Arizona': Resources.TL_AZ,
    'Georgia': Resources.TL_GA,
    # Articles
    'Florida': Resources.TL_FL,
    'California': Resources.TL_CA,
    'Texas': Resources.TL_TX,
}

# Per-state override for the cross-region edge weight `w_r` (Eq. 2 of Huang et
# al., ISPRS 2023). Leave a state out to use HGI_CONFIG.cross_region_weight.
#
# The optimum is dataset-specific and appears to scale INVERSELY with POI
# density. Anchors: Xiamen ~26 POI/km² and Shenzhen ~150 POI/km² both use
# w_r = 0.4 (paper); Alabama at 0.089 POI/km² (~290× sparser) has an Alabama-
# swept optimum at w_r = 0.7. On our Alabama sweep Cat F1 rose monotonically
# 0.74 → 0.82 from 0.4 to 0.7 (see `research/embeddings/hgi/README.md` §5).
#
# The values below are best-effort starting points extrapolated from POI
# density vs. the paper's anchors — ONLY Alabama has been swept. Re-sweep
# per state when tuning ({0.4, 0.7, 1.0} brackets the optimum in ~75 min).
CROSS_REGION_WEIGHT_PER_STATE = {
    'Alabama':    0.7,  # swept 2026-04-11 (density 0.089 POI/km², confirmed optimum)
    'Arizona':    0.7,  # density 0.070 POI/km² — sparsest; matches Alabama regime
    'Texas':      0.7,  # density 0.229 POI/km² — still firmly in sparse regime
    'California': 0.6,  # density 0.411 POI/km² — medium, interpolated
    'Florida':    0.6,  # density 0.536 POI/km² — densest of the five, interpolated
    # 'Georgia': 0.7,   # not yet measured; set when available
}

# Hyperparameters mirror RightBank/HGI/train.py (the canonical reference for
# "Learning urban region representations with POIs and hierarchical graph
# infomax", ISPRS J. Photogramm. Remote Sens., 2023). lr=0.006 is only safe
# in combination with the 40-epoch LinearWarmup wired into train_hgi —
# do not bump one without the other.
HGI_CONFIG = Namespace(
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

    # Cross-region edge weight. Paper: 0.4. Third-party repro: 0.5. Our Alabama
    # best: 0.7 (Cat F1 +8 pp vs paper). Override per state via the dict above.
    cross_region_weight=0.7,

    device='cpu',
    shapefile=None  # Will be set per state
)


# =============================================================================
# PIPELINE
# =============================================================================

def process_state(name: str, shapefile, cta_file=None) -> bool:
    """Run full HGI pipeline for a single state.

    Steps:
        1. Preprocess (graph only) → edges.csv, pois.csv for POI2Vec
        2. Train POI2Vec → fclass embeddings → POI embeddings
        3. Preprocess (with embeddings) → full data dict → save pickle
        4. Train HGI → POI + region embeddings
        5. Generate downstream inputs
    """
    try:
        HGI_CONFIG.shapefile = shapefile
        # Per-state override (falls back to the global default in HGI_CONFIG).
        w_r = CROSS_REGION_WEIGHT_PER_STATE.get(name, HGI_CONFIG.cross_region_weight)
        HGI_CONFIG.cross_region_weight = w_r
        logger.info(f"[setup] {name}: cross_region_weight w_r={w_r}")

        # 1. First pass: build Delaunay graph → edges.csv + pois.csv (needed by POI2Vec)
        logger.info(f"[1/5] Building graph (Delaunay + edges.csv + pois.csv): {name}")
        preprocess_hgi(
            city=name, city_shapefile=str(shapefile), poi_emb_path=None,
            cta_file=cta_file, cross_region_weight=w_r,
        )

        # 2. Train POI2Vec (phases 3b-3d: walks → fclass embeddings → POI embeddings)
        logger.info(f"[2/5] Training POI2Vec: {name}")
        poi_emb_path = train_poi2vec(
            city=name, epochs=HGI_CONFIG.poi2vec_epochs,
            embedding_dim=HGI_CONFIG.dim,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 3. Second pass: preprocess with learned embeddings → save pickle
        logger.info(f"[3/5] Preprocessing with embeddings: {name}")
        data = preprocess_hgi(
            city=name, city_shapefile=str(shapefile),
            poi_emb_path=str(poi_emb_path),
            cta_file=cta_file,
            cross_region_weight=w_r,
        )

        graph_data_file = IoPaths.HGI.get_graph_data_file(name)
        graph_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_data_file, "wb") as f:
            pkl.dump(data, f)
        logger.info(f"Saved graph data: {graph_data_file}")

        # 4. Train HGI model (loads pickle, trains, saves embeddings)
        logger.info(f"[4/5] Training HGI: {name}")
        train_hgi(name, HGI_CONFIG)

        # 5. Generate downstream inputs
        logger.info(f"[5/5] Generating inputs: {name}")
        generate_category_input(name, EmbeddingEngine.HGI)
        generate_next_input_from_poi(name, EmbeddingEngine.HGI)

        return True
    except Exception as e:
        logger.error(f"Failed processing {name}: {e}", exc_info=True)
        return False


def run_pipeline():
    """Process all configured states."""
    # NOTE: HGI and POI2Vec are pinned to CPU in HGI_CONFIG/process_state because
    # the global DEVICE (MPS on Apple Silicon) is ~176x slower than CPU for HGI's
    # many-small-ops workload. We log both so the discrepancy is visible.
    logger.info(
        f"HGI Pipeline - {len(STATES)} state(s) | "
        f"hgi_device={HGI_CONFIG.device} | global_device={DEVICE} | dim={HGI_CONFIG.dim}"
    )

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
