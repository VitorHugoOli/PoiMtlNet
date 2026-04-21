"""Compute region-transition matrix for a state — GETNext-style.

Reads the check2HGI preprocessing artefacts and aggregates region→region
transition counts observed at training windows. Output is a log-probability
matrix that can be used as an additive bias to next-region logits per
GETNext (Yang et al., SIGIR 2022).

For each row in ``sequences_next.parquet`` we observe one transition:

    region(poi_8)  →  region(target_poi)

We count these across all rows, add Laplace smoothing (epsilon), row-normalize
to a conditional distribution, and take log. The result ``log_T`` has shape
``[n_regions, n_regions]``; ``log_T[i]`` is the log-probability distribution
over next regions conditional on the last-observed region being ``i``.

Usage::

    python scripts/compute_region_transition.py --state alabama
    python scripts/compute_region_transition.py --state arizona --state florida

Output::

    ${OUTPUT_DIR}/check2hgi/<state>/region_transition_log.pt
"""

from __future__ import annotations

import argparse
import logging
import pickle as pkl
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pandas as pd
import torch

from configs.paths import IoPaths, EmbeddingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_graph_maps(state: str):
    path = IoPaths.CHECK2HGI.get_graph_data_file(state)
    with open(path, "rb") as f:
        graph = pkl.load(f)
    placeid_to_idx = graph["placeid_to_idx"]
    poi_to_region = graph["poi_to_region"]
    if hasattr(poi_to_region, "cpu"):
        poi_to_region = poi_to_region.cpu().numpy()
    return placeid_to_idx, np.asarray(poi_to_region, dtype=np.int64)


def build_transition_matrix(state: str, smoothing_eps: float = 0.01) -> tuple[np.ndarray, int]:
    seq_path = IoPaths.CHECK2HGI.get_temp_dir(state) / "sequences_next.parquet"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}; run the check2HGI pipeline first")
    seq_df = pd.read_parquet(seq_path)

    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    n_regions = int(poi_to_region.max()) + 1

    last_placeids = seq_df["poi_8"].astype(np.int64).to_numpy()
    target_placeids = seq_df["target_poi"].astype(np.int64).to_numpy()

    valid_mask = (last_placeids != -1)
    unmapped = ~np.isin(last_placeids[valid_mask], list(placeid_to_idx.keys()))
    if unmapped.any():
        bad = last_placeids[valid_mask][unmapped][:5].tolist()
        raise ValueError(f"{unmapped.sum()} last-step placeids unmapped: {bad}")
    unmapped_t = ~np.isin(target_placeids, list(placeid_to_idx.keys()))
    if unmapped_t.any():
        bad = target_placeids[unmapped_t][:5].tolist()
        raise ValueError(f"{unmapped_t.sum()} target placeids unmapped: {bad}")

    last_poi_idx = pd.Series(last_placeids[valid_mask]).map(placeid_to_idx).to_numpy(dtype=np.int64)
    target_poi_idx = pd.Series(target_placeids[valid_mask]).map(placeid_to_idx).to_numpy(dtype=np.int64)
    last_region = poi_to_region[last_poi_idx]
    target_region = poi_to_region[target_poi_idx]

    counts = np.full((n_regions, n_regions), smoothing_eps, dtype=np.float64)
    np.add.at(counts, (last_region, target_region), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    log_probs = np.log(probs).astype(np.float32)

    n_observed = int(valid_mask.sum())
    logger.info(
        "[%s] Built transition matrix: n_regions=%d, n_observed_transitions=%d, "
        "avg non-zero per row=%.1f",
        state, n_regions, n_observed,
        float((counts > smoothing_eps + 0.5).sum()) / n_regions,
    )
    return log_probs, n_regions


def save(state: str, log_probs: np.ndarray, smoothing_eps: float) -> Path:
    out_dir = IoPaths.CHECK2HGI.get_state_dir(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "region_transition_log.pt"
    tensor = torch.from_numpy(log_probs)
    payload = {
        "log_transition": tensor,
        "smoothing_eps": smoothing_eps,
        "n_regions": tensor.shape[0],
    }
    torch.save(payload, out_path)
    logger.info("[%s] Saved %s (shape=%s)", state, out_path, tuple(tensor.shape))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", required=True)
    parser.add_argument("--smoothing-eps", type=float, default=0.01)
    args = parser.parse_args()

    for state in args.state:
        log_probs, n_regions = build_transition_matrix(state, smoothing_eps=args.smoothing_eps)
        save(state, log_probs, args.smoothing_eps)


if __name__ == "__main__":
    main()
