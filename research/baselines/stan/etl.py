"""Faithful STAN baseline ETL.

Self-contained: depends only on the canonical Gowalla check-ins
(``data/checkins/<State>.parquet``) and the TIGER/Line 2022 census-tract
shapefile for the state (``data/miscellaneous/tl_2022_*_tract_*/``). It
does NOT consume any artifact from the Check2HGI / HGI embedding
pipelines, so the baseline cannot inherit substrate-side filtering or
encoding decisions.

Region target derivation matches what the in-house pipelines do, but as
an independent step:

    1. Group check-ins by ``placeid`` and take the first (lat, lon) as
       the POI's representative location.
    2. Spatial-join POIs to tracts via ``geopandas.sjoin(predicate='intersects')``.
    3. Drop POIs outside every tract (~few %, matches Check2HGI behaviour).
    4. Assign deterministic ``placeid_to_idx`` and ``region_to_idx``
       (order = first appearance in the post-join POI frame).

Then slide 9+1 non-overlapping windows over each user's check-ins
(after sorting by datetime) — same window strategy as
``src/data/inputs/core.py::generate_sequences`` — and emit one row per
window:

    poi_idx_0..8        int64    POI index in [0, n_pois) (PAD=-1)
    lat_0..8            float32  latitude of the check-in at position k
    lon_0..8            float32  longitude of the check-in at position k
    t_minutes_0..8      int64    minutes since the window's first check-in
    target_region_idx   int64    region index in [0, n_regions)
    target_category     str      target check-in's raw category (for stratification)
    userid              int64    user id (for StratifiedGroupKFold)

Padded positions get ``poi_idx=-1`` and ``lat=lon=t_minutes=0``; the
model masks those positions in attention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[3]
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.paths import DATA_ROOT, OUTPUT_DIR, Resources  # noqa: E402

# Audit fix #4 (2026-06-26): STAN's NATIVE sequence construction is full-trajectory
# PREFIX-EXPANSION (reference load.py truncates each user to the last max_len, train.py's
# expanding causal mask supervises EVERY position t from its causal prefix -> L-1 targets/user),
# NOT our stride-9 single-target windows. CONTEXT_LEN = STAN's max_len (paper 100; 50 here as a
# memory-feasible value — covers AL/AZ p90 and most of FL, and the matching layer's [B, n, R]
# tensors stay GPU-bounded at large region counts). Tunable via --context-len.
CONTEXT_LEN = 50
PAD = -1
MIN_HISTORY = 5  # skip users with fewer than this many check-ins (too short to learn from)

_STATE_TO_SHAPEFILE = {
    "alabama": Resources.TL_AL,
    "arizona": Resources.TL_AZ,
    "florida": Resources.TL_FL,
    "california": Resources.TL_CA,
    "texas": Resources.TL_TX,
    "georgia": Resources.TL_GA,
    "newyork": Resources.TL_NY,
}


def _checkins_path(state: str) -> Path:
    return DATA_ROOT / "checkins" / f"{state.capitalize()}.parquet"


def _shapefile_path(state: str) -> Path:
    key = state.lower().replace(" ", "")
    if key not in _STATE_TO_SHAPEFILE:
        raise KeyError(f"No TIGER shapefile registered for state={state}")
    return _STATE_TO_SHAPEFILE[key]


def _assign_regions_from_shapefile(checkins: pd.DataFrame, shapefile: Path
                                   ) -> tuple[pd.DataFrame, dict, dict]:
    """Spatial-join unique POIs to tracts; return (poi_df, placeid_to_idx, region_to_idx).

    ``poi_df`` has columns: ``placeid, latitude, longitude, GEOID, region_idx, poi_idx``.
    POIs outside every tract are dropped.
    """
    pois = (
        checkins.groupby("placeid", as_index=False)
        .agg(latitude=("latitude", "first"), longitude=("longitude", "first"))
    )
    pois_geom = gpd.GeoDataFrame(
        pois,
        geometry=gpd.points_from_xy(pois["longitude"], pois["latitude"]),
        crs="EPSG:4326",
    )
    tracts = gpd.read_file(shapefile).to_crs("EPSG:4326")[["GEOID", "geometry"]]

    pois_geom = pois_geom.sjoin(tracts, how="left", predicate="intersects")
    pois_geom = pois_geom.dropna(subset=["GEOID"]).reset_index(drop=True)
    # Some POIs may sit on a tract boundary and join multiple — keep first.
    pois_geom = pois_geom.drop_duplicates(subset=["placeid"], keep="first").reset_index(drop=True)

    region_to_idx = {g: i for i, g in enumerate(pois_geom["GEOID"].unique())}
    pois_geom["region_idx"] = pois_geom["GEOID"].map(region_to_idx).astype(np.int64)

    placeid_to_idx = {pid: i for i, pid in enumerate(pois_geom["placeid"].tolist())}
    pois_geom["poi_idx"] = pois_geom["placeid"].map(placeid_to_idx).astype(np.int64)

    # Region centroid in EPSG:4326 (lat/lon). Used by STAN's matching-layer
    # candidate-side Δd bias (paper §4.1.2 / Eq. 4 — interpolated distance
    # embedding from each candidate to each trajectory position).
    tracts_proj = tracts.to_crs(epsg=3857)  # planar for accurate centroid
    centroids = tracts_proj.set_index("GEOID").geometry.centroid.to_crs(epsg=4326)
    geoid_to_centroid = {g: (pt.y, pt.x) for g, pt in centroids.items()}
    centroid_rows = [
        {"region_idx": region_to_idx[g],
         "GEOID": g,
         "centroid_lat": geoid_to_centroid[g][0],
         "centroid_lon": geoid_to_centroid[g][1]}
        for g in region_to_idx
    ]
    centroid_df = pd.DataFrame(centroid_rows).sort_values("region_idx").reset_index(drop=True)

    return (
        pd.DataFrame(pois_geom.drop(columns="geometry")),
        placeid_to_idx,
        region_to_idx,
        centroid_df,
    )


def build_windows(state: str, context_len: int = CONTEXT_LEN) -> tuple[pd.DataFrame, int, int]:
    """Build the per-window input frame for ``state`` via STAN-native prefix-expansion.

    Returns ``(df, n_pois, n_regions, centroid_df)``. ``context_len`` = STAN max_len.
    """
    raw = pd.read_parquet(
        _checkins_path(state),
        columns=["userid", "placeid", "datetime", "latitude", "longitude", "category"],
    )
    raw = raw.dropna(subset=["placeid", "datetime", "latitude", "longitude", "category"])
    raw["placeid"] = raw["placeid"].astype(np.int64)

    poi_df, placeid_to_idx, region_to_idx, centroid_df = _assign_regions_from_shapefile(
        raw, _shapefile_path(state),
    )
    n_pois = len(placeid_to_idx)
    n_regions = len(region_to_idx)

    raw = raw[raw["placeid"].isin(placeid_to_idx)].copy()
    raw["poi_idx"] = raw["placeid"].map(placeid_to_idx).astype(np.int64)
    raw["region_idx"] = raw["poi_idx"].map(
        dict(zip(poi_df["poi_idx"], poi_df["region_idx"]))
    ).astype(np.int64)
    raw = raw.sort_values(["userid", "datetime"]).reset_index(drop=True)
    raw["t_minutes"] = (
        raw["datetime"].astype("int64") // (60 * 1_000_000_000)
    ).astype(np.int64)
    # STAN's hour-of-week token: 1..168 (paper §4.1.1; reference layers.py:104
    # `(t-1) % hours + 1` with hours=24*7). Distinct from the Δt scalar that
    # the pairwise bias consumes.
    raw["hour_of_week"] = (
        raw["datetime"].dt.dayofweek * 24 + raw["datetime"].dt.hour
    ).astype(np.int64) + 1  # 1..168 (0 reserved for pad)

    rows = []
    for uid, grp in raw.groupby("userid", sort=False):
        n = len(grp)
        if n < MIN_HISTORY:
            continue
        poi_seq = grp["poi_idx"].to_numpy()
        reg_seq = grp["region_idx"].to_numpy()
        cat_seq = grp["category"].to_numpy()
        lat_seq = grp["latitude"].to_numpy(dtype=np.float32)
        lon_seq = grp["longitude"].to_numpy(dtype=np.float32)
        t_seq = grp["t_minutes"].to_numpy()
        hour_seq = grp["hour_of_week"].to_numpy()

        # STAN-native prefix-expansion: predict EVERY position t (1..n-1) from its
        # causal prefix = the last CONTEXT_LEN check-ins before t (left-aligned,
        # right-padded). Mirrors load.py (truncate to last max_len) + train.py's
        # expanding causal mask. Yields n-1 targets per user (vs ceil(n/9) before).
        for t in range(1, n):
            ctx_start = max(0, t - context_len)
            hist_idx = range(ctx_start, t)               # length 1..context_len
            target_region = int(reg_seq[t])
            target_category = str(cat_seq[t])

            poi_pad = [PAD] * context_len
            lat_pad = [0.0] * context_len
            lon_pad = [0.0] * context_len
            t_pad = [0] * context_len
            hour_pad = [0] * context_len  # 0 = pad token; real values 1..168
            for k, idx in enumerate(hist_idx):
                poi_pad[k] = int(poi_seq[idx])
                lat_pad[k] = float(lat_seq[idx])
                lon_pad[k] = float(lon_seq[idx])
                t_pad[k] = int(t_seq[idx])
                hour_pad[k] = int(hour_seq[idx])
            t0 = next((tt for tt in t_pad if tt > 0), 0)
            t_rel = [int(max(0, tt - t0)) if tt > 0 else 0 for tt in t_pad]

            row = {
                "userid": int(uid),
                "target_region_idx": target_region,
                "target_category": target_category,
            }
            for k in range(context_len):
                row[f"poi_idx_{k}"] = poi_pad[k]
                row[f"lat_{k}"] = lat_pad[k]
                row[f"lon_{k}"] = lon_pad[k]
                row[f"t_minutes_{k}"] = t_rel[k]
                row[f"hour_of_week_{k}"] = hour_pad[k]
            rows.append(row)

    return pd.DataFrame(rows), n_pois, n_regions, centroid_df


def out_path(state: str) -> Path:
    return OUTPUT_DIR / "baselines" / "stan" / state / "inputs.parquet"


def centroids_path(state: str) -> Path:
    return OUTPUT_DIR / "baselines" / "stan" / state / "region_centroids.parquet"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN,
                   help="STAN max trajectory length (prefix-expansion context).")
    args = p.parse_args()

    df, n_pois, n_regions, centroid_df = build_windows(args.state, context_len=args.context_len)
    out = out_path(args.state)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    centroid_df.to_parquet(centroids_path(args.state), index=False)
    print(
        f"[{args.state}] wrote {len(df):,} windows  n_pois={n_pois}  "
        f"n_regions={n_regions}  → {out}\n"
        f"          centroids ({len(centroid_df)} regions) → {centroids_path(args.state)}"
    )


if __name__ == "__main__":
    main()
