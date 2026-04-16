"""Tests for src/data/inputs/next_poi.py.

Mirrors test_next_region_loader.py but targets the next_poi label path:

  sequences_next.parquet (target_poi = raw placeid int)
      → placeid_to_idx (from checkin_graph.pt)
      → next_poi.parquet (poi_idx column)

Synthetic graph + sequences simulate the schema; the real loader is
called end-to-end.
"""

from __future__ import annotations

import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from configs.paths import EmbeddingEngine, IoPaths


@pytest.fixture
def synthetic_check2hgi_state(tmp_path, monkeypatch):
    """Materialise a small fake check2HGI output tree for a synthetic state."""
    import configs.paths as _paths_mod

    state = "Synthtown"
    state_l = state.lower()

    # Re-fetch the live class reference for _Check2HGIIoPath. Another
    # test (test_paths::test_data_root_env_var) does importlib.reload on
    # configs.paths, which recreates all class objects.  If we capture
    # the reference before that reload (e.g. from the top-level import),
    # monkeypatch's teardown restores the wrong object and the next test
    # sees stale paths.
    monkeypatch.setattr(
        _paths_mod.IoPaths.CHECK2HGI, "_check2hgi_dir", tmp_path / "output" / "check2hgi",
    )
    monkeypatch.setattr(_paths_mod, "OUTPUT_DIR", tmp_path / "output")

    n_seq = 20
    emb_dim = 64
    seq_len = 9
    n_pois = 10
    n_regions = 5

    placeid_to_idx = {100 + i: i for i in range(n_pois)}
    poi_to_region = np.array([i % n_regions for i in range(n_pois)], dtype=np.int64)

    temp_dir = tmp_path / "output" / "check2hgi" / state_l / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    graph = {
        "placeid_to_idx": placeid_to_idx,
        "poi_to_region": poi_to_region,
        "num_pois": n_pois,
        "num_regions": n_regions,
        "num_checkins": n_seq,
    }
    with open(temp_dir / "checkin_graph.pt", "wb") as f:
        pkl.dump(graph, f)

    rng = np.random.default_rng(0)
    placeids = list(placeid_to_idx.keys())
    seq_cols = {f"poi_{i}": rng.choice(placeids, size=n_seq) for i in range(seq_len)}
    seq_cols["target_poi"] = rng.choice(placeids, size=n_seq)
    seq_cols["userid"] = rng.integers(0, 5, size=n_seq)
    seq_df = pd.DataFrame(seq_cols)
    seq_df.to_parquet(temp_dir / "sequences_next.parquet")

    input_dir = tmp_path / "output" / "check2hgi" / state_l / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    emb_cols = {str(i): rng.standard_normal(n_seq).astype(np.float32) for i in range(seq_len * emb_dim)}
    next_df = pd.DataFrame(emb_cols)
    next_df["next_category"] = rng.choice(["Food", "Shopping", "Travel"], size=n_seq)
    next_df["userid"] = seq_df["userid"].astype(str).values
    next_df.to_parquet(input_dir / "next.parquet")

    return state, n_seq, n_pois, placeid_to_idx


def test_build_next_poi_frame_happy_path(synthetic_check2hgi_state):
    from data.inputs.next_poi import build_next_poi_frame

    state, n_seq, n_pois, placeid_to_idx = synthetic_check2hgi_state
    df, n_pois_out = build_next_poi_frame(state)

    assert len(df) == n_seq
    assert n_pois_out == n_pois
    assert "poi_idx" in df.columns
    assert "next_category" not in df.columns
    assert df["poi_idx"].between(0, n_pois - 1).all()
    assert "userid" in df.columns
    assert all(str(i) in df.columns for i in range(0, 576, 64))


def test_build_next_poi_frame_fails_on_unmapped_placeid(synthetic_check2hgi_state, tmp_path):
    from data.inputs.next_poi import build_next_poi_frame

    state, _, _, _ = synthetic_check2hgi_state
    state_l = state.lower()
    seq_path = tmp_path / "output" / "check2hgi" / state_l / "temp" / "sequences_next.parquet"
    df = pd.read_parquet(seq_path)
    df.loc[0, "target_poi"] = 999_999_999
    df.to_parquet(seq_path)

    with pytest.raises(ValueError, match="not in placeid_to_idx"):
        build_next_poi_frame(state)


def test_build_next_poi_frame_fails_on_row_count_mismatch(synthetic_check2hgi_state, tmp_path):
    from data.inputs.next_poi import build_next_poi_frame

    state, _, _, _ = synthetic_check2hgi_state
    state_l = state.lower()
    next_path = tmp_path / "output" / "check2hgi" / state_l / "input" / "next.parquet"
    df = pd.read_parquet(next_path)
    df.iloc[:10].to_parquet(next_path)

    with pytest.raises(ValueError, match="disagree"):
        build_next_poi_frame(state)


def test_build_next_poi_poi_idx_matches_placeid_to_idx(synthetic_check2hgi_state, tmp_path):
    """Verify the label round-trip: poi_idx should map back to the
    correct placeid via the inverse of placeid_to_idx."""
    from data.inputs.next_poi import build_next_poi_frame

    state, _, _, placeid_to_idx = synthetic_check2hgi_state
    state_l = state.lower()

    df, _ = build_next_poi_frame(state)
    idx_to_placeid = {v: k for k, v in placeid_to_idx.items()}

    seq_path = tmp_path / "output" / "check2hgi" / state_l / "temp" / "sequences_next.parquet"
    seq_df = pd.read_parquet(seq_path)
    target_placeids = seq_df["target_poi"].astype(np.int64).to_numpy()

    for i in range(len(df)):
        poi_idx = int(df.iloc[i]["poi_idx"])
        recovered = idx_to_placeid[poi_idx]
        assert recovered == target_placeids[i], (
            f"Row {i}: poi_idx={poi_idx} → placeid={recovered}, "
            f"expected {target_placeids[i]}"
        )


def test_load_next_poi_data_rejects_non_check2hgi():
    from data.inputs.next_poi import load_next_poi_data

    with pytest.raises(ValueError, match="CHECK2HGI"):
        load_next_poi_data("alabama", EmbeddingEngine.HGI)
