#!/usr/bin/env python3
"""Random-pair inter-centroid distance floor for the near-miss metric.

The geographic near-miss numbers (near_miss_RESULTS.md: median in-distribution
miss 3.16-8.13 km across Istanbul/AL/AZ/FL) need a reference point per the
paper's own rule (GLOSSARY.md par.4: never give a number without its reference
point). This script computes that floor: the distance between the centroids of
two RANDOMLY DRAWN, DISTINCT candidate regions of the same map, restricted to
the model's exact region vocabulary (the classes it can actually predict, from
checkin_graph.pt's region_to_idx -- NOT the full boroughs_area.csv, which holds
more polygons than the model's candidate set, e.g. 1,437 vs 1,109 at Alabama).

This is a bare geometric scale reference ("how far apart are two random
candidate regions"), not a model baseline: it answers "if a wrong prediction
landed on a random candidate region, how far would its centroid typically be
from the true one, under a uniform draw?" The true-region distribution is not
uniform, so this is a scale anchor, not a simulated random predictor.

Method: load region_to_idx from output/check2hgi/<state>/temp/checkin_graph.pt
(a pickle despite the .pt extension), zero-pad numeric GEOIDs to 11 chars, join
to boroughs_area.csv WKT polygons, take shapely centroids, sample 200,000
uniformly random ordered pairs (seed 0), keep the distinct ones, and report
P50/P90/mean haversine distance in km.

Results (2026-07-08, all vocabulary GEOIDs matched a polygon at every state):

  state      vocab  matched  P50 (km)  P90 (km)  mean (km)
  alabama     1109     1109    170.67    377.39     198.46
  arizona     1547     1547    120.32    286.93     134.74
  florida     4703     4703    241.22    507.89     262.28
  istanbul     520      520     20.45     59.45      26.38

Against the model's in-distribution miss medians (8.13 / 8.05 / 7.04 / 3.16 km
per near_miss_RESULTS.md), model misses land roughly 21x / 15x / 34x / 6x
closer than the random-pair floor.

Run:  .venv/bin/python "articles/[mobiwac]/analysis/near_miss_floor.py"
(CPU-only, seconds per state; needs output/check2hgi/<state>/temp/ locally.)
"""
import os
import pickle

import numpy as np
import pandas as pd
from shapely import wkt

HERE = os.path.dirname(os.path.abspath(__file__))
# Same portable repo-root resolution as near_miss_distance.py (three levels up
# from articles/[mobiwac]/analysis/), overridable with POIMTLNET_REPO.
REPO = os.environ.get("POIMTLNET_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(HERE))
)
EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius -- matches near_miss_distance.py
N_PAIRS = 200_000
STATES = ["alabama", "arizona", "florida", "istanbul"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def geoid_str(g):
    # Gowalla GEOIDs are 11-char census-tract ids stored as int64 with the
    # leading zero dropped (Alabama 1073012918 == 01073012918); Istanbul uses
    # non-numeric OSM relation ids, passed through as-is.
    s = str(g)
    return s.zfill(11) if s.isdigit() else s


def main():
    rng = np.random.default_rng(0)
    for state in STATES:
        with open(f"{REPO}/output/check2hgi/{state}/temp/checkin_graph.pt", "rb") as f:
            graph = pickle.load(f)
        vocab = {geoid_str(k) for k in graph["region_to_idx"].keys()}

        df = pd.read_csv(f"{REPO}/output/check2hgi/{state}/temp/boroughs_area.csv")
        geom_col = next(
            c for c in df.columns
            if df[c].astype(str).str.startswith(("POLYGON", "MULTIPOLYGON")).any()
        )
        id_col = next(
            c for c in df.columns
            if df[c].astype(str).apply(lambda s: geoid_str(s) in vocab).mean() > 0.5
        )
        df["_gid"] = df[id_col].apply(geoid_str)
        sub = df[df["_gid"].isin(vocab)].drop_duplicates("_gid")

        cents = sub[geom_col].apply(lambda s: wkt.loads(s).centroid)
        lats = np.array([c.y for c in cents])
        lons = np.array([c.x for c in cents])
        n = len(lats)

        i = rng.integers(0, n, N_PAIRS)
        j = rng.integers(0, n, N_PAIRS)
        m = i != j
        d = haversine_km(lats[i[m]], lons[i[m]], lats[j[m]], lons[j[m]])
        print(
            f"{state:10s} vocab={len(vocab):5d} matched={n:5d}  "
            f"P50={np.percentile(d, 50):7.2f} km  "
            f"P90={np.percentile(d, 90):7.2f} km  mean={d.mean():7.2f} km"
        )


if __name__ == "__main__":
    main()
