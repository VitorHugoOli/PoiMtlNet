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

Default mode (``--state STATE``) builds log_T from ALL rows. This was the
original behaviour through F50 — and is the source of audit finding C4
(`F50_T3_AUDIT_FINDINGS.md`): every fold's val rows leak into log_T because
the transitions on those rows are counted into the prior the trainer
consumes.

Per-fold mode (``--state STATE --per-fold``) reproduces the trainer's
``StratifiedGroupKFold(groups=userid)`` split (same seed) and writes one
log_T per fold using ONLY that fold's train userids. Output:

    ${OUTPUT_DIR}/check2hgi/<state>/region_transition_log_fold1.pt
    ...
    ${OUTPUT_DIR}/check2hgi/<state>/region_transition_log_fold5.pt

The trainer activates per-fold mode via ``--per-fold-transition-dir``
(see ``scripts/train.py``); without that flag the legacy single-file
behaviour is preserved.

Usage::

    # legacy (full-data prior — has C4 leakage)
    python scripts/compute_region_transition.py --state alabama

    # per-fold (clean — no val leakage)
    python scripts/compute_region_transition.py --state florida --per-fold
"""

from __future__ import annotations

import argparse
import logging
import pickle as pkl
import sys
from pathlib import Path
from typing import Iterable, Optional

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


def _log_probs_from_rows(
    last_placeids: np.ndarray,
    target_placeids: np.ndarray,
    placeid_to_idx: dict,
    poi_to_region: np.ndarray,
    n_regions: int,
    smoothing_eps: float,
) -> np.ndarray:
    """Pure helper: count (last_region → target_region) transitions and
    return ``log P(target | last)`` with Laplace smoothing.

    Pulled out of ``build_transition_matrix`` so ``build_*_from_userids``
    (per-fold) can reuse the same code path, guaranteeing identical
    output when called with all rows.
    """
    valid_mask = (last_placeids != -1)
    last_valid = last_placeids[valid_mask]
    target_valid = target_placeids[valid_mask]

    placeid_keys = list(placeid_to_idx.keys())
    unmapped = ~np.isin(last_valid, placeid_keys)
    if unmapped.any():
        bad = last_valid[unmapped][:5].tolist()
        raise ValueError(f"{unmapped.sum()} last-step placeids unmapped: {bad}")
    unmapped_t = ~np.isin(target_valid, placeid_keys)
    if unmapped_t.any():
        bad = target_valid[unmapped_t][:5].tolist()
        raise ValueError(f"{unmapped_t.sum()} target placeids unmapped: {bad}")

    last_poi_idx = pd.Series(last_valid).map(placeid_to_idx).to_numpy(dtype=np.int64)
    target_poi_idx = pd.Series(target_valid).map(placeid_to_idx).to_numpy(dtype=np.int64)
    last_region = poi_to_region[last_poi_idx]
    target_region = poi_to_region[target_poi_idx]

    counts = np.full((n_regions, n_regions), smoothing_eps, dtype=np.float64)
    np.add.at(counts, (last_region, target_region), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums
    return np.log(probs).astype(np.float32)


def _load_sequences(state: str) -> pd.DataFrame:
    seq_path = IoPaths.CHECK2HGI.get_temp_dir(state) / "sequences_next.parquet"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}; run the check2HGI pipeline first")
    return pd.read_parquet(seq_path)


def build_transition_matrix(state: str, smoothing_eps: float = 0.01) -> tuple[np.ndarray, int]:
    """Legacy: build log_T from ALL rows of ``sequences_next.parquet``.

    Has C4 leakage: val rows of every fold contribute to the prior.
    Use ``build_transition_matrix_from_userids`` for per-fold builds.
    """
    seq_df = _load_sequences(state)
    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    n_regions = int(poi_to_region.max()) + 1

    last_placeids = seq_df["poi_8"].astype(np.int64).to_numpy()
    target_placeids = seq_df["target_poi"].astype(np.int64).to_numpy()

    log_probs = _log_probs_from_rows(
        last_placeids, target_placeids,
        placeid_to_idx, poi_to_region, n_regions, smoothing_eps,
    )
    n_observed = int((last_placeids != -1).sum())
    logger.info(
        "[%s] Built transition matrix (full): n_regions=%d, n_observed_transitions=%d",
        state, n_regions, n_observed,
    )
    return log_probs, n_regions


def build_transition_matrix_from_userids(
    state: str,
    train_userids: Iterable[int],
    smoothing_eps: float = 0.01,
    seq_df: Optional[pd.DataFrame] = None,
) -> tuple[np.ndarray, int]:
    """C4-clean: build log_T from rows whose userid is in ``train_userids``.

    Parameters
    ----------
    state:
        State name (alabama, florida, ...).
    train_userids:
        Iterable of integer userids that are in the FOLD'S TRAIN SET. Rows
        whose userid is not in this set are excluded — eliminating the
        val→train leakage in the prior.
    smoothing_eps:
        Laplace smoothing for unobserved (last_region, target_region)
        pairs. Same default as the legacy function.
    seq_df:
        Optional preloaded DataFrame to avoid re-reading the parquet
        across folds. Must have columns ``poi_8``, ``target_poi``,
        ``userid``.
    """
    if seq_df is None:
        seq_df = _load_sequences(state)
    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    n_regions = int(poi_to_region.max()) + 1

    train_set = set(int(u) for u in train_userids)
    in_train = seq_df["userid"].astype(np.int64).isin(train_set).to_numpy()
    sub = seq_df.loc[in_train]
    if len(sub) == 0:
        raise ValueError(
            f"No rows match the supplied train_userids set "
            f"(state={state}, n_train_users={len(train_set)})"
        )

    last_placeids = sub["poi_8"].astype(np.int64).to_numpy()
    target_placeids = sub["target_poi"].astype(np.int64).to_numpy()

    log_probs = _log_probs_from_rows(
        last_placeids, target_placeids,
        placeid_to_idx, poi_to_region, n_regions, smoothing_eps,
    )
    n_observed = int((last_placeids != -1).sum())
    logger.info(
        "[%s] Built transition matrix (per-fold): n_regions=%d, "
        "n_train_users=%d, n_train_rows=%d, n_observed_transitions=%d",
        state, n_regions, len(train_set), len(sub), n_observed,
    )
    return log_probs, n_regions


def save(state: str, log_probs: np.ndarray, smoothing_eps: float,
         filename: str = "region_transition_log.pt") -> Path:
    out_dir = IoPaths.CHECK2HGI.get_state_dir(state)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    tensor = torch.from_numpy(log_probs)
    payload = {
        "log_transition": tensor,
        "smoothing_eps": smoothing_eps,
        "n_regions": tensor.shape[0],
    }
    torch.save(payload, out_path)
    logger.info("[%s] Saved %s (shape=%s)", state, out_path, tuple(tensor.shape))
    return out_path


def _build_per_fold(state: str, smoothing_eps: float, n_splits: int, seed: int):
    """Reproduce the trainer's StratifiedGroupKFold split and write
    one log_T per fold from train userids only.

    The split here MUST match ``FoldCreator._create_mtl_folds_with_isolation``
    in ``src/data/folds.py``: same algorithm (StratifiedGroupKFold), same
    groups (userid), same y (next_category), same seed. We reuse the
    project's data loaders to guarantee bit-equality.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from data.folds import load_next_data

    X_next, y_next, next_userids, _ = load_next_data(state, EmbeddingEngine.CHECK2HGI)
    seq_df = _load_sequences(state)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    paths = []
    for fold_idx, (train_idx, _val_idx) in enumerate(
        sgkf.split(X_next, y_next, groups=next_userids)
    ):
        train_userids = set(int(u) for u in next_userids[train_idx])
        log_probs, _ = build_transition_matrix_from_userids(
            state,
            train_userids=train_userids,
            smoothing_eps=smoothing_eps,
            seq_df=seq_df,
        )
        # Filename encodes seed so a trainer running at --seed N cannot
        # silently load a per-fold log_T built for a different seed.
        # Pre-2026-04-30 builds wrote the unseeded
        # ``region_transition_log_fold{N}.pt`` form; those files were
        # always seed=42 and have been migrated to the seeded form.
        out = save(
            state, log_probs, smoothing_eps,
            filename=f"region_transition_log_seed{seed}_fold{fold_idx + 1}.pt",
        )
        paths.append(out)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", required=True)
    parser.add_argument("--smoothing-eps", type=float, default=0.01)
    parser.add_argument(
        "--per-fold", action="store_true",
        help="Build one log_T per fold from train userids only (fixes C4 leakage)",
    )
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of CV folds (must match trainer; default 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fold seed (must match trainer; default 42 = FoldCreator default)")
    args = parser.parse_args()

    for state in args.state:
        if args.per_fold:
            _build_per_fold(state, args.smoothing_eps, args.n_splits, args.seed)
        else:
            log_probs, _ = build_transition_matrix(state, smoothing_eps=args.smoothing_eps)
            save(state, log_probs, args.smoothing_eps)


if __name__ == "__main__":
    main()
