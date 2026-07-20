#!/usr/bin/env python3
"""Matched random comparator for the shortlist-compactness sentence (section 7).

WHY THIS EXISTS (2026-07-20, readability-review follow-up): the section-7
sentence compared the model's TEN-region shortlist spread ("a median of 3 to 8
km from the shortlist's centroid", shortlist_compactness_RESULTS.md) against
the TWO-region random-pair inter-centroid floor ("20 to 241 km",
near_miss_floor.py). A reviewer flagged the asymmetry: ten predicted regions
vs two random ones. Author ruling: recompute the comparator as TEN regions
drawn at random, so both sides of the comparison are the same quantity (the
spread of a ten-region set around its own centroid). This script does NOT
touch the shortlist side (whose raw inputs, the MTL_DUMP_VAL_PREDS parquet
dumps, live on the A40); it only replaces the random comparator.

MATCHED QUANTITY (identical to the shortlist side, reusing its code verbatim):
for each draw of 10 regions, ``centroid_spread_mean_km`` exactly as
``shortlist_compactness._shortlist_stats`` computes it for a visit's top-10
shortlist -- the MEAN haversine distance (km, IUGG radius 6371.0088) from each
of the 10 region centroids to their spherical unit-vector geographic mean.
Over N_DRAWS draws we report the same bare percentile grid the shortlist side
uses (``_grid_summary``: np.percentile, linear interpolation); the headline is
the P50 (median) with the P25-P75 IQR. NOTE the published sentence's gloss
("the regions lie a median of X km from the centroid") compresses the actual
pipeline statistic (median over visits of the per-visit MEAN distance to the
centroid); the matched comparator keeps the pipeline statistic, and the JSON
also carries a per-draw-MEDIAN sensitivity variant for transparency.

POOL (identical to the original random-pair floor, near_miss_floor.py): the
model's exact candidate-region vocabulary -- ``region_to_idx`` from
``output/check2hgi/<state>/temp/checkin_graph.pt`` (a pickle), GEOIDs
zero-padded to 11 chars, joined to ``boroughs_area.csv`` WKT polygons, shapely
centroids, first polygon kept per duplicated GEOID, uniform draw. Same pool,
same loader logic, same haversine.

DRAW SCHEME (the one place the original comparator's scheme cannot be copied
verbatim, documented choice): the pair floor drew 200,000 ORDERED pairs WITH
replacement and dropped i == j (equivalent to uniform distinct pairs). Here
each draw is 10 DISTINCT regions (``rng.choice(n, 10, replace=False)``),
matching the shortlist side, whose top-10 predicted region indices are always
distinct. N_DRAWS = 10,000 draws per state, fresh ``np.random.default_rng(0)``
PER STATE (the pair floor shared one seed-0 stream across states in file
order; a per-state stream makes each state independently reproducible and is
statistically immaterial at 10,000 draws).

REPRODUCE-FIRST GATE (mandatory, runs before the matched computation): this
script first recomputes the PUBLISHED two-region pair floor bit-compatibly
(same single seed-0 stream across states in near_miss_floor.STATES order, same
200,000 ordered pairs, same haversine) and hard-fails unless P50/P90/mean
match the published values to their printed 2-decimal rounding. It also
verifies the published shortlist-side pooled in-distribution spread P50s
against the recorded ``shortlist_compactness_<state>.json`` grids (the raw
parquet dumps are A40-only, so the shortlist side is verified against its
artifact of record, not recomputed).

Run (CPU-only, seconds):
  .venv/bin/python "articles/[mobiwac]/analysis/shortlist_compactness_matched.py"

Writes (next to this script):
  shortlist_compactness_matched.json
and prints the summary table to paste into shortlist_compactness_RESULTS.md.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd
from shapely import wkt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("POIMTLNET_REPO") or os.path.dirname(
    os.path.dirname(os.path.dirname(HERE))
)
sys.path.insert(0, HERE)
# Reuse the exact published code paths: the pair floor's haversine + GEOID
# normalization, and the shortlist side's spread statistic + percentile grid.
from near_miss_floor import (  # noqa: E402
    EARTH_RADIUS_KM,
    N_PAIRS,
    STATES,
    geoid_str,
    haversine_km,
)
from shortlist_compactness import _grid_summary, _shortlist_stats  # noqa: E402

N_DRAWS = 10_000
SHORTLIST_SIZE = 10
SEED = 0

# Published values this script must reproduce before computing anything new.
# Source: near_miss_floor.py docstring / near_miss_RESULTS.md §Floor (2026-07-08).
PUBLISHED_PAIR_FLOOR = {
    "alabama": {"p50": 170.67, "p90": 377.39, "mean": 198.46, "vocab": 1109},
    "arizona": {"p50": 120.32, "p90": 286.93, "mean": 134.74, "vocab": 1547},
    "florida": {"p50": 241.22, "p90": 507.89, "mean": 262.28, "vocab": 4703},
    "istanbul": {"p50": 20.45, "p90": 59.45, "mean": 26.38, "vocab": 520},
}
# Source: shortlist_compactness_RESULTS.md summary table (pooled in-dist spread P50).
PUBLISHED_SHORTLIST_P50 = {
    "alabama": 6.24, "arizona": 6.09, "florida": 7.53, "istanbul": 2.86,
}


def load_pool(state: str):
    """(lats, lons) of the model's candidate-region centroids, in the SAME
    construction and ROW ORDER as near_miss_floor.py (CSV order, filtered to
    the vocabulary, first polygon per duplicated GEOID) -- so the pair-floor
    gate below consumes an identical index space."""
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
    return lats, lons, len(vocab)


def gate_pair_floor(pools: dict) -> dict:
    """Recompute the published two-region floor bit-compatibly; hard-fail on
    any mismatch beyond the published 2-decimal rounding."""
    rng = np.random.default_rng(SEED)  # ONE stream across states, file order
    out = {}
    for state in STATES:
        lats, lons, n_vocab = pools[state]
        n = len(lats)
        i = rng.integers(0, n, N_PAIRS)
        j = rng.integers(0, n, N_PAIRS)
        m = i != j
        d = haversine_km(lats[i[m]], lons[i[m]], lats[j[m]], lons[j[m]])
        got = {"p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90)),
               "mean": float(d.mean()), "vocab": n_vocab, "matched": n}
        pub = PUBLISHED_PAIR_FLOOR[state]
        for k in ("p50", "p90", "mean"):
            if abs(got[k] - pub[k]) > 0.005:
                raise SystemExit(
                    f"REPRODUCE GATE FAILED [{state}] pair-floor {k}: "
                    f"recomputed {got[k]:.4f} vs published {pub[k]:.2f}"
                )
        if got["vocab"] != pub["vocab"] or got["matched"] != pub["vocab"]:
            raise SystemExit(
                f"REPRODUCE GATE FAILED [{state}] pool size: vocab={got['vocab']} "
                f"matched={got['matched']} vs published {pub['vocab']}"
            )
        print(f"[gate] {state:9s} pair floor reproduced: "
              f"P50={got['p50']:7.2f} P90={got['p90']:7.2f} mean={got['mean']:7.2f} "
              f"(published {pub['p50']:.2f}/{pub['p90']:.2f}/{pub['mean']:.2f}) OK")
        out[state] = got
    return out


def gate_shortlist_side() -> dict:
    """Verify the published shortlist P50s against the recorded per-state JSON
    grids (artifact of record; raw parquet dumps are A40-only)."""
    out = {}
    for state, pub in PUBLISHED_SHORTLIST_P50.items():
        path = os.path.join(HERE, f"shortlist_compactness_{state}.json")
        with open(path) as fh:
            rec = json.load(fh)
        got = rec["pooled"]["in_distribution"]["centroid_spread_mean_km"]["p50_km"]
        if abs(got - pub) > 0.005:
            raise SystemExit(
                f"REPRODUCE GATE FAILED [{state}] shortlist P50: recorded {got:.4f} "
                f"vs published {pub:.2f}"
            )
        print(f"[gate] {state:9s} shortlist pooled in-dist spread P50 verified: "
              f"{got:.4f} (published {pub:.2f}) OK")
        out[state] = {"p50_recorded": got, "rundir": rec["rundir"]}
    return out


def _per_draw_median_spread(draws, lats, lons):
    """Per-draw MEDIAN distance to the spherical mean (sensitivity variant;
    the matched headline uses the pipeline's MEAN, via _shortlist_stats).
    Same spherical unit-vector mean and haversine as _shortlist_stats."""
    la = lats[draws]  # (D, 10)
    lo = lons[draws]
    la_r, lo_r = np.radians(la), np.radians(lo)
    x = np.cos(la_r) * np.cos(lo_r)
    y = np.cos(la_r) * np.sin(lo_r)
    z = np.sin(la_r)
    xm, ym, zm = x.mean(axis=1), y.mean(axis=1), z.mean(axis=1)
    lat_m = np.degrees(np.arctan2(zm, np.hypot(xm, ym)))
    lon_m = np.degrees(np.arctan2(ym, xm))
    d = haversine_km(la, lo, lat_m[:, None], lon_m[:, None])
    return np.median(d, axis=1)


def matched_comparator(pools: dict) -> dict:
    out = {}
    for state in STATES:
        lats, lons, _ = pools[state]
        n = len(lats)
        rng = np.random.default_rng(SEED)  # fresh per state (documented)
        draws = np.stack([rng.choice(n, SHORTLIST_SIZE, replace=False)
                          for _ in range(N_DRAWS)])
        # EXACT shortlist-side statistic, reused verbatim.
        spread_mean, spread_max, bbox_diag, n_empty = _shortlist_stats(
            draws.astype(np.int64), lats, lons
        )
        assert n_empty == 0 and not np.isnan(spread_mean).any()
        med_variant = _per_draw_median_spread(draws, lats, lons)
        g = _grid_summary(spread_mean)
        out[state] = {
            "pool_n_regions": n,
            "n_draws": N_DRAWS,
            "shortlist_size": SHORTLIST_SIZE,
            "seed": SEED,
            "draw_scheme": "uniform without replacement, fresh default_rng(0) per state",
            "centroid_spread_mean_km": g,
            "centroid_spread_max_km": _grid_summary(spread_max),
            "bbox_diagonal_km": _grid_summary(bbox_diag),
            "sensitivity_per_draw_median_km": _grid_summary(med_variant),
        }
        print(f"[matched] {state:9s} 10-random-region spread: "
              f"P50={g['p50_km']:7.2f} IQR={g['p25_km']:7.2f}-{g['p75_km']:7.2f} "
              f"mean={g['mean_km']:7.2f}  (shortlist P50 "
              f"{PUBLISHED_SHORTLIST_P50[state]:.2f} -> "
              f"{g['p50_km']/PUBLISHED_SHORTLIST_P50[state]:.1f}x tighter)")
    return out


def main():
    pools = {s: load_pool(s) for s in STATES}

    print("== Reproduce-first gate ==")
    gate_pair = gate_pair_floor(pools)
    gate_short = gate_shortlist_side()

    print("== Matched comparator (10 regions drawn at random) ==")
    matched = matched_comparator(pools)

    payload = {
        "generated": date.today().isoformat(),
        "script": "shortlist_compactness_matched.py",
        "purpose": (
            "matched random comparator (TEN regions drawn at random) for the "
            "section-7 shortlist-compactness sentence; replaces the two-region "
            "pair floor as the comparator, same pool, same spread statistic"
        ),
        "metric": (
            "per-draw MEAN haversine km from each of the 10 region centroids to "
            "their spherical unit-vector geographic mean (identical to "
            "shortlist_compactness._shortlist_stats centroid_spread_mean_km); "
            "grid over draws via the same _grid_summary (np.percentile, linear)"
        ),
        "pool": (
            "model candidate-region vocabulary (checkin_graph.pt region_to_idx "
            "joined to boroughs_area.csv WKT polygon shapely centroids), identical "
            "to near_miss_floor.py"
        ),
        "earth_radius_km": EARTH_RADIUS_KM,
        "reproduce_gate": {
            "pair_floor_recomputed": gate_pair,
            "pair_floor_published": PUBLISHED_PAIR_FLOOR,
            "shortlist_p50_verified": gate_short,
            "shortlist_p50_published": PUBLISHED_SHORTLIST_P50,
            "note": (
                "pair floor recomputed bit-compatibly from local "
                "output/check2hgi/<state>/temp inputs; shortlist side verified "
                "against the recorded shortlist_compactness_<state>.json grids "
                "(raw fold{N}_reg_val_preds.parquet dumps are A40-only)"
            ),
        },
        "matched": matched,
    }
    out_json = os.path.join(HERE, "shortlist_compactness_matched.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
