"""Next-POI label derivation for the check2HGI MTL track.

Mirrors ``next_region.py`` but stops one join earlier. The label is the
POI index itself (not the region the POI belongs to).

Builds ``output/check2hgi/<state>/input/next_poi.parquet`` from:

1. ``sequences_next.parquet`` produced by ``Check2HGIPreprocess`` — per-row
   ``target_poi`` is the raw placeid (int) of the next check-in.
2. ``checkin_graph.pt`` produced by the same preprocessing — supplies
   ``placeid_to_idx`` to map raw placeids to dense POI indices in
   ``[0, n_pois)``.
3. ``next.parquet`` — shares X columns and userid row-order with
   ``sequences_next.parquet``; we reuse its embedding columns so
   next_poi and next_region use identical feature tensors at the row
   level.

Output schema::

    col 0..575    — flattened 9-window of check2HGI check-in embeddings
                    (identical to next.parquet columns '0'..'575')
    poi_idx       — int64 POI index in ``[0, n_pois)``
    userid        — int64 user id (for StratifiedGroupKFold)

Fails loud on any target_poi missing from ``placeid_to_idx``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from configs.paths import EmbeddingEngine, IoPaths
from data.inputs.next_region import _load_graph_maps, _load_sequences


def build_next_poi_frame(state: str) -> Tuple[pd.DataFrame, int]:
    """Build the next-POI input DataFrame for ``state``.

    Returns ``(df, n_pois)``. ``df`` has the same row count as the
    check2HGI ``next.parquet`` and shares the X columns row-for-row.
    """
    next_df = IoPaths.load_next(state, EmbeddingEngine.CHECK2HGI)
    seq_df = _load_sequences(state)

    if len(next_df) != len(seq_df):
        raise ValueError(
            f"next.parquet rows ({len(next_df)}) and sequences_next.parquet "
            f"rows ({len(seq_df)}) disagree for {state}. Regenerate both "
            f"via the check2HGI pipeline in one pass."
        )
    next_uid = next_df["userid"].astype(str).reset_index(drop=True)
    seq_uid = seq_df["userid"].astype(str).reset_index(drop=True)
    if not (next_uid == seq_uid).all():
        raise ValueError(
            f"userid columns of next.parquet and sequences_next.parquet "
            f"disagree for {state} — the two files are not row-aligned."
        )

    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    n_pois = int(len(placeid_to_idx))
    # Sanity: placeid_to_idx values must cover [0, n_pois).
    max_idx = max(placeid_to_idx.values())
    if max_idx >= n_pois:
        raise ValueError(
            f"placeid_to_idx for {state} has max value {max_idx} but only "
            f"{n_pois} entries — non-dense POI index range."
        )
    # poi_to_region length must match n_pois; otherwise the graph artefact
    # is internally inconsistent and we want to fail before writing.
    if len(poi_to_region) != n_pois:
        raise ValueError(
            f"poi_to_region length ({len(poi_to_region)}) disagrees with "
            f"placeid_to_idx size ({n_pois}) for {state} — graph artefact "
            f"is internally inconsistent."
        )

    target_placeid = seq_df["target_poi"].astype(np.int64).to_numpy()
    unmapped_mask = ~np.isin(target_placeid, list(placeid_to_idx.keys()))
    if unmapped_mask.any():
        sample = target_placeid[unmapped_mask][:10].tolist()
        raise ValueError(
            f"{int(unmapped_mask.sum())} target_poi values are not in "
            f"placeid_to_idx for {state}. Sample unmapped placeids: {sample}."
        )
    poi_idx = pd.Series(target_placeid).map(placeid_to_idx).to_numpy(dtype=np.int64)

    if (poi_idx < 0).any() or (poi_idx >= n_pois).any():
        raise ValueError(
            f"poi_idx out of range [0, {n_pois}) for {state}."
        )

    out = next_df.drop(columns=["next_category"]).copy()
    out["poi_idx"] = poi_idx
    return out, n_pois


def load_next_poi_data(state: str, engine: EmbeddingEngine) -> pd.DataFrame:
    """Read a pre-built next_poi.parquet. Fails if missing."""
    if engine != EmbeddingEngine.CHECK2HGI:
        raise ValueError(
            f"next_poi labels are only defined on CHECK2HGI (got {engine})."
        )
    return IoPaths.load_next_poi(state, engine)


__all__ = [
    "build_next_poi_frame",
    "load_next_poi_data",
]
