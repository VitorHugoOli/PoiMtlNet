"""Compute region×category co-location matrix P(region|cat) for a state.

R1 (`mtl_frontier`) — the ESMM-style probability-chain analog of log_T. Where
``compute_region_transition.py`` builds P(target_region | last_region) (a spatial
Markov-1 prior), this builds **P(target_region | target_category)** — a
co-location prior tying the two MTL heads in label space:

    prior(reg) = Σ_c P(reg | c) · P̂(c)        (ESMM pCTCVR = pCTR·pCVR analog)

where P̂(c) is the live cat-head posterior at train time. The matrix here is the
fixed, train-only P(reg|c) factor; the marginalization against P̂(c) happens in
the trainer's log_C-KD branch.

Construction MIRRORS ``compute_region_transition.py`` for leak hygiene:
- per-fold, per-seed, **train-userids only** (same StratifiedGroupKFold split as
  the trainer — engine-invariant: userids/labels/row-order are identical across
  the check2hgi base and the v14 design_k substrate, verified 2026-06-15);
- region index = ``poi_to_region[placeid_to_idx[target_poi]]`` (the SAME mapping
  log_T and the reg head's label space use);
- category index = ``inv(CATEGORIES_MAP)[category_lookup[target_poi]]`` (the SAME
  mapping the cat task's label uses — `create_category_lookup` on the city
  check-ins, exactly as `core.py` builds `next_category`);
- targets whose category is 'None'/missing (not in 0..n_cats-1) are EXCLUDED
  (they are not valid cat-task labels — the cat head has no column for them).

Normalization: **column-normalized** so each category column is a distribution
over regions, P(reg|cat) (Σ_reg = 1). Stored as log-probabilities with Laplace
smoothing, payload key ``log_colocation``, shape ``[n_regions, n_cats]``.

The save directory is the engine's substrate dir (so it sits beside the per-fold
log_T), e.g. ``output/check2hgi_design_k_resln_mae_l0_1/<state>/``.

Usage::

    python scripts/compute_region_colocation.py --state alabama --per-fold --seed 0 \
        --engine check2hgi_design_k_resln_mae_l0_1
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
from configs.globals import CATEGORIES_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cat-task label space: indices 0..N_CATS-1 (CATEGORIES_MAP 0..6; 'None'=7 excluded).
N_CATS = 7
_INV_CATEGORIES = {v: k for k, v in CATEGORIES_MAP.items()}  # label str -> int idx


def _load_graph_maps(state: str):
    path = IoPaths.CHECK2HGI.get_graph_data_file(state)
    with open(path, "rb") as f:
        graph = pkl.load(f)
    placeid_to_idx = graph["placeid_to_idx"]
    poi_to_region = graph["poi_to_region"]
    if hasattr(poi_to_region, "cpu"):
        poi_to_region = poi_to_region.cpu().numpy()
    return placeid_to_idx, np.asarray(poi_to_region, dtype=np.int64)


def _load_category_idx_lookup(state: str) -> dict:
    """placeid -> category INDEX (0..N_CATS-1), via the SAME source the cat
    task uses: create_category_lookup(city check-ins) then inv(CATEGORIES_MAP).
    Placeids whose category is 'None'/missing/unmapped get index -1 (excluded)."""
    from data.inputs.core import create_category_lookup
    checkins_df = IoPaths.load_city(state)
    label_lookup = create_category_lookup(checkins_df)  # placeid -> label str
    idx_lookup = {}
    for pid, label in label_lookup.items():
        idx = _INV_CATEGORIES.get(label, -1)
        idx_lookup[pid] = idx if (idx is not None and 0 <= idx < N_CATS) else -1
    return idx_lookup


def _load_sequences(state: str) -> pd.DataFrame:
    seq_path = IoPaths.CHECK2HGI.get_temp_dir(state) / "sequences_next.parquet"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}; run the check2HGI pipeline first")
    return pd.read_parquet(seq_path)


def _log_colocation_from_rows(
    target_placeids: np.ndarray,
    placeid_to_idx: dict,
    poi_to_region: np.ndarray,
    cat_idx_lookup: dict,
    n_regions: int,
    smoothing_eps: float,
) -> tuple[np.ndarray, int, int]:
    """Count (target_region, target_cat) co-occurrences → column-normalized
    log P(region | cat), shape [n_regions, N_CATS]. Returns (log_probs,
    n_used, n_excluded_cat)."""
    valid_mask = (target_placeids != -1)
    target_valid = target_placeids[valid_mask]

    placeid_keys = list(placeid_to_idx.keys())
    unmapped = ~np.isin(target_valid, placeid_keys)
    if unmapped.any():
        bad = target_valid[unmapped][:5].tolist()
        raise ValueError(f"{unmapped.sum()} target placeids unmapped: {bad}")

    target_poi_idx = pd.Series(target_valid).map(placeid_to_idx).to_numpy(dtype=np.int64)
    target_region = poi_to_region[target_poi_idx]
    target_cat = pd.Series(target_valid).map(cat_idx_lookup).fillna(-1).to_numpy(dtype=np.int64)

    cat_ok = (target_cat >= 0) & (target_cat < N_CATS)
    n_excluded = int((~cat_ok).sum())
    r = target_region[cat_ok]
    c = target_cat[cat_ok]

    counts = np.full((n_regions, N_CATS), smoothing_eps, dtype=np.float64)
    np.add.at(counts, (r, c), 1.0)

    # Column-normalize: P(region | cat) — each category column sums to 1.
    col_sums = counts.sum(axis=0, keepdims=True)
    probs = counts / col_sums
    return np.log(probs).astype(np.float32), int(cat_ok.sum()), n_excluded


def build_colocation_from_userids(
    state: str,
    train_userids: Iterable[int],
    smoothing_eps: float = 0.01,
    seq_df: Optional[pd.DataFrame] = None,
    cat_idx_lookup: Optional[dict] = None,
) -> tuple[np.ndarray, int]:
    """Train-only P(region|cat) from rows whose userid ∈ train_userids."""
    if seq_df is None:
        seq_df = _load_sequences(state)
    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    if cat_idx_lookup is None:
        cat_idx_lookup = _load_category_idx_lookup(state)
    n_regions = int(poi_to_region.max()) + 1

    train_set = set(int(u) for u in train_userids)
    in_train = seq_df["userid"].astype(np.int64).isin(train_set).to_numpy()
    sub = seq_df.loc[in_train]
    if len(sub) == 0:
        raise ValueError(f"No rows match train_userids (state={state})")

    target_placeids = sub["target_poi"].astype(np.int64).to_numpy()
    log_probs, n_used, n_excl = _log_colocation_from_rows(
        target_placeids, placeid_to_idx, poi_to_region, cat_idx_lookup,
        n_regions, smoothing_eps,
    )
    logger.info(
        "[%s] co-location (per-fold): n_regions=%d n_cats=%d n_train_rows=%d "
        "n_used=%d n_excluded_cat(None/missing)=%d",
        state, n_regions, N_CATS, len(sub), n_used, n_excl,
    )
    return log_probs, n_regions


def save(
    state: str,
    engine: EmbeddingEngine,
    log_probs: np.ndarray,
    smoothing_eps: float,
    filename: str,
    n_splits: Optional[int] = None,
    seed: Optional[int] = None,
) -> Path:
    """Persist to output/<engine>/<state>/ (beside the per-fold log_T)."""
    out_dir = _root / "output" / engine.value / state
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    tensor = torch.from_numpy(log_probs)
    payload = {
        "log_colocation": tensor,
        "smoothing_eps": smoothing_eps,
        "n_regions": int(tensor.shape[0]),
        "n_cats": int(tensor.shape[1]),
    }
    if n_splits is not None:
        payload["n_splits"] = int(n_splits)
    if seed is not None:
        payload["seed"] = int(seed)
    torch.save(payload, out_path)
    logger.info("[%s] Saved %s (shape=%s, n_splits=%s, seed=%s)",
                state, out_path, tuple(tensor.shape), n_splits, seed)
    return out_path


def _build_per_fold(state: str, engine: EmbeddingEngine, smoothing_eps: float,
                    n_splits: int, seed: int):
    """Reproduce the trainer's StratifiedGroupKFold split (engine-invariant —
    we use the check2hgi base for the split inputs, verified identical to v14)
    and write one P(region|cat) per fold from train userids only."""
    from sklearn.model_selection import StratifiedGroupKFold
    from data.folds import load_next_data

    X_next, y_next, next_userids, _ = load_next_data(state, EmbeddingEngine.CHECK2HGI)
    seq_df = _load_sequences(state)
    cat_idx_lookup = _load_category_idx_lookup(state)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    paths = []
    for fold_idx, (train_idx, _val_idx) in enumerate(
        sgkf.split(X_next, y_next, groups=next_userids)
    ):
        train_userids = set(int(u) for u in next_userids[train_idx])
        log_probs, _ = build_colocation_from_userids(
            state, train_userids=train_userids, smoothing_eps=smoothing_eps,
            seq_df=seq_df, cat_idx_lookup=cat_idx_lookup,
        )
        out = save(
            state, engine, log_probs, smoothing_eps,
            filename=f"region_colocation_log_seed{seed}_fold{fold_idx + 1}.pt",
            n_splits=n_splits, seed=seed,
        )
        paths.append(out)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="append", required=True)
    parser.add_argument("--engine", default="check2hgi_design_k_resln_mae_l0_1",
                        help="Substrate engine — controls the SAVE dir only "
                             "(split/maps are engine-invariant base check2hgi).")
    parser.add_argument("--smoothing-eps", type=float, default=0.01)
    parser.add_argument("--per-fold", action="store_true",
                        help="Build one P(region|cat) per fold from train userids only")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    engine = EmbeddingEngine(args.engine)
    for state in args.state:
        if not args.per_fold:
            raise SystemExit("R1 requires --per-fold (leak hygiene). Full-data mode disabled.")
        _build_per_fold(state, engine, args.smoothing_eps, args.n_splits, args.seed)


if __name__ == "__main__":
    main()
