"""Region-embedding sequence construction for Check2HGI MTL input modality.

For each row of ``sequences_next.parquet`` (which already has the 9-window
of past placeids), look up each placeid's region index and emit the
corresponding region embedding. Padded positions (placeid == -1) become
zero vectors so the downstream ``x.abs().sum(dim=-1) == 0`` padding-mask
logic in the heads keeps working.

This is the central helper for the P4 per-task input modality variants:

- ``"checkin"``: load ``next.parquet``'s 9-window of check-in embeddings.
- ``"region"``: build [N, 9, D] of region embeddings via this helper.
- ``"concat"``: concatenate the two along the feature axis -> [N, 9, 2D].

The MTL fold creator calls the "region" or "concat" branch when the
user picks a non-default ``task_a_input_type`` / ``task_b_input_type``.
The P1 region-head ablation script uses it directly.
"""

from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from configs.model import InputsConfig
from configs.paths import EmbeddingEngine, IoPaths


def _load_region_embeddings(state: str) -> np.ndarray:
    """Load region_embeddings.parquet as a [n_regions, D] float32 array,
    sorted by region_id so row index == region_idx.
    """
    path = IoPaths.CHECK2HGI.get_state_dir(state) / "region_embeddings.parquet"
    df = pd.read_parquet(path).sort_values("region_id").reset_index(drop=True)
    emb_cols = [c for c in df.columns if c.startswith("reg_")]
    return df[emb_cols].to_numpy(dtype=np.float32)


def _load_graph_maps(state: str) -> Tuple[dict, np.ndarray]:
    graph_path = IoPaths.CHECK2HGI.get_graph_data_file(state)
    with open(graph_path, "rb") as f:
        graph = pkl.load(f)
    placeid_to_idx = graph["placeid_to_idx"]
    poi_to_region = graph["poi_to_region"]
    if hasattr(poi_to_region, "cpu"):
        poi_to_region = poi_to_region.cpu().numpy()
    return placeid_to_idx, np.asarray(poi_to_region, dtype=np.int64)


def build_region_sequence_tensor(state: str) -> torch.Tensor:
    """Build the [N, 9, D] region-embedding sequence for Check2HGI MTL.

    Row i position k holds the region embedding of the region the user
    was in at step k of their 9-window (placeid i,k → poi_idx → region_idx
    → region_emb). Padded steps (placeid == -1) are zero vectors so the
    head-level padding mask ``x.abs().sum(dim=-1) == 0`` picks them up.

    Engine is fixed to Check2HGI — region embeddings only exist there.

    Row alignment: identical ordering to ``next.parquet`` /
    ``next_region.parquet`` because all three are produced from
    ``sequences_next.parquet`` in one pass.
    """
    seq_path = IoPaths.CHECK2HGI.get_temp_dir(state) / "sequences_next.parquet"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"sequences_next.parquet missing at {seq_path}. "
            f"Run pipelines/embedding/check2hgi.pipe.py for {state} first."
        )
    seq_df = pd.read_parquet(seq_path)

    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    region_emb = _load_region_embeddings(state)
    emb_dim = region_emb.shape[1]
    slide_window = InputsConfig.SLIDE_WINDOW

    n = len(seq_df)
    out = np.zeros((n, slide_window, emb_dim), dtype=np.float32)

    for i in range(slide_window):
        col = f"poi_{i}"
        placeids = seq_df[col].astype(np.int64).to_numpy()
        mask = placeids != -1
        valid = placeids[mask]
        # Vectorised placeid → poi_idx lookup.
        poi_idx = pd.Series(valid).map(placeid_to_idx).to_numpy(dtype=np.int64)
        unmapped = np.isnan(pd.Series(poi_idx).to_numpy(dtype=np.float64))
        if unmapped.any():
            # Defensive: should never happen because the upstream preprocessing
            # emits only in-vocabulary placeids. Fail loud rather than silently
            # produce zero vectors and confuse the downstream padding mask.
            raise ValueError(
                f"{int(unmapped.sum())} placeids in poi_{i} column missing "
                f"from placeid_to_idx for state={state}. Regenerate inputs."
            )
        region_idx = poi_to_region[poi_idx]
        out[np.where(mask)[0], i, :] = region_emb[region_idx]

    return torch.from_numpy(out)


def build_concat_sequence_tensor(
    state: str,
    checkin_tensor: torch.Tensor,
) -> torch.Tensor:
    """Build [N, 9, 2D] by concatenating check-in + region along feature dim.

    ``checkin_tensor`` must be the [N, 9, D] tensor already produced by
    ``_convert_to_tensors`` for ``next.parquet``. We load the region
    sequence separately and concatenate on ``dim=-1``.
    """
    region_tensor = build_region_sequence_tensor(state)
    if region_tensor.shape[0] != checkin_tensor.shape[0]:
        raise RuntimeError(
            f"Region tensor has {region_tensor.shape[0]} rows, check-in "
            f"tensor has {checkin_tensor.shape[0]} — inputs not aligned."
        )
    if region_tensor.shape[1] != checkin_tensor.shape[1]:
        raise RuntimeError(
            f"Sequence length mismatch: region={region_tensor.shape[1]} "
            f"vs checkin={checkin_tensor.shape[1]}."
        )
    return torch.cat([checkin_tensor, region_tensor], dim=-1)


__all__ = [
    "build_region_sequence_tensor",
    "build_concat_sequence_tensor",
]
