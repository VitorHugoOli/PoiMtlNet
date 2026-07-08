#!/usr/bin/env python3
"""
Shortlist spatial compactness for next-region top-10 predictions (A40-6, P7 support).

Question (MOBILITY_SCIENCE_BRIDGE_PLAN.md §12.7, v17_completion/A40.md · A40-6):
the paper's P7 usage sketch argues a top-10 region shortlist is actionable via
its enrichment over chance (~78-547x). Enrichment is a *relative* claim; the
sharpest residual reviewer counter is "is the shortlist also spatially compact,
or is it scattered across a whole state?" This script answers that from data
that already exists on this machine (the near-miss dumps), no retraining.

For each validation visit it takes the model's top-10 predicted regions'
centroids (dropping the -1 padding a head with <10 classes would emit) and
measures how tightly they cluster:

  (i)  centroid spread -- the mean and the max great-circle distance (km) from
       each of the (up to 10) centroids to their geographic mean (the spherical
       unit-vector mean of the set, so it is robust to longitude wraparound);
  (ii) bounding-box diagonal (km) -- the great-circle distance between the
       (min lat, min lon) and (max lat, max lon) corners of the shortlist.

Per fold it reports P50/P90 (plus a bare percentile grid) of each per-visit
statistic over the fold's visits, IN-DISTRIBUTION and OUT-OF-DISTRIBUTION (OOD,
true region absent from that fold's training vocabulary) strictly SEPARATE,
never pooled together -- the same split near_miss_distance.py uses.

Guardrails (settled venue-bridge ruling, motivation-only -- MOBILITY_PLAN.md
§0/§3.1; A40-6 spec): BARE percentiles / a bare distribution grid and nothing
else. It does NOT overlay any service radius or coverage threshold (no
"coverage curve" in disguise) and does not read the numbers as a measured
service result -- it only describes the geometry of the model's own top-10
shortlists. Unlike the near-miss metric it is computed over EVERY visit (the
spec says "per validation visit", not "per miss"): the shortlist is what a
downstream system would act on for every visit, whether or not the top-1 is
right.

Data sources (both read-only; shared with near_miss_distance.py, whose loaders
this script reuses verbatim):
  1. Per-sample validation predictions -- ``<rundir>/metrics/
     fold{N}_reg_val_preds.parquet`` (the opt-in ``MTL_DUMP_VAL_PREDS=1`` dump).
     Columns: ``true_region_idx`` (int), ``is_ood`` (bool),
     ``pred_region_idx_top1..10`` (int, -1 padded if the head has <10 classes).
  2. Region geometry -- ``output/check2hgi/<state>/temp/{checkin_graph.pt (a
     pickle), boroughs_area.csv}`` via ``load_region_centroids`` (region_idx ->
     (lat, lon) centroid; 11-char GEOID zero-pad handled there).

Deliberate deviation from "full per-visit distributions in the JSON"
(A40-6 spec) -- documented, not an oversight: a raw per-visit array for all
visits x 3 statistics is ~1.3M rows/state at Florida (~100 MB of JSON), far
past the near-miss dumps' already-heavy 17 MB. Instead each JSON stores a
DENSE BARE PERCENTILE GRID (P1..P99 + mean/max/std/n) per statistic, per
subset, per fold and pooled -- which fully characterizes the distribution shape
for any later plot, is literally "bare percentiles only" (guardrail-safe), and
keeps the artifact at KB scale. The .md mirrors the near_miss_distance_*.md
layout.

Run (one state + one rundir per invocation; CPU-only, no training dependency):
  PYTHONPATH=src python \\
      "articles/[mobiwac]/analysis/shortlist_compactness.py" \\
      --state alabama --rundir /path/to/results/check2hgi_dk_ovl/alabama/<run_folder>

Writes (next to this script, one pair per --state):
  shortlist_compactness_<state>.json  -- per-fold + pooled percentile grids
  shortlist_compactness_<state>.md    -- P50/P90 summary table + short write-up

If a state has no dumped parquets, the script says so and writes nothing (no
placeholders), exactly like near_miss_distance.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the near-miss loaders verbatim (A40-6 spec: "reuse the near-miss
# script's loaders"): region centroids, val-preds, haversine, GEOID zero-pad.
sys.path.insert(0, HERE)
from near_miss_distance import (  # noqa: E402
    load_region_centroids,
    load_val_preds,
    haversine_km,
)

PRED_COLS = [f"pred_region_idx_top{k}" for k in range(1, 11)]
# The bare percentile grid stored per statistic (guardrail-safe: percentiles,
# not a service-radius overlay). P50/P90 are the paper's headline; the rest
# just characterize the distribution shape for any later plot.
PCTL_GRID = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def _centroid_arrays(state: str):
    """(lat[idx], lon[idx]) as NaN-padded arrays indexed by region_idx.

    NaN marks a region_idx with no centroid (should not happen -- near-miss
    reported 0 dropped centroids at every state -- but is handled, never
    silently zero-filled)."""
    idx_to_centroid = load_region_centroids(state)  # {idx: (lat, lon)}
    max_idx = max(idx_to_centroid) if idx_to_centroid else -1
    lat = np.full(max_idx + 1, np.nan)
    lon = np.full(max_idx + 1, np.nan)
    for idx, (la, lo) in idx_to_centroid.items():
        lat[idx] = la
        lon[idx] = lo
    return lat, lon


def _shortlist_stats(preds: np.ndarray, cent_lat: np.ndarray, cent_lon: np.ndarray):
    """Per-visit shortlist compactness statistics, fully vectorized.

    preds: (N, 10) int region indices (-1 = padding). Returns three (N,) float
    arrays -- centroid_spread_mean_km, centroid_spread_max_km, bbox_diagonal_km
    -- with NaN for visits that have no usable centroid (dropped downstream)."""
    n_regions = cent_lat.shape[0]
    valid = (preds >= 0) & (preds < n_regions)
    safe = np.where(valid, preds, 0)  # clamp so fancy-indexing never goes OOB
    lat = np.where(valid, cent_lat[safe], np.nan)  # (N, 10)
    lon = np.where(valid, cent_lon[safe], np.nan)
    valid &= ~np.isnan(lat)  # a present index whose centroid is NaN is not usable

    # Spherical (unit-vector) geographic mean of each row's usable centroids.
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    cnt = valid.sum(axis=1)  # (N,)
    safe_cnt = np.where(cnt == 0, 1, cnt)
    xm = np.where(valid, x, 0.0).sum(axis=1) / safe_cnt
    ym = np.where(valid, y, 0.0).sum(axis=1) / safe_cnt
    zm = np.where(valid, z, 0.0).sum(axis=1) / safe_cnt
    lat_m = np.degrees(np.arctan2(zm, np.hypot(xm, ym)))  # (N,)
    lon_m = np.degrees(np.arctan2(ym, xm))

    # (i) spread: distance of each usable centroid to its row's geographic mean.
    d = haversine_km(lat, lon, lat_m[:, None], lon_m[:, None])  # (N, 10), NaN where invalid
    with np.errstate(invalid="ignore"):
        spread_mean = np.nanmean(d, axis=1)
        spread_max = np.nanmax(d, axis=1)
        # (ii) bounding-box diagonal over each row's usable centroids.
        min_lat = np.nanmin(lat, axis=1)
        max_lat = np.nanmax(lat, axis=1)
        min_lon = np.nanmin(lon, axis=1)
        max_lon = np.nanmax(lon, axis=1)
    bbox_diag = haversine_km(min_lat, min_lon, max_lat, max_lon)

    empty = cnt == 0
    spread_mean[empty] = np.nan
    spread_max[empty] = np.nan
    bbox_diag[empty] = np.nan
    return spread_mean, spread_max, bbox_diag, int(empty.sum())


def _grid_summary(a: np.ndarray) -> dict:
    """Bare percentile grid + moments for a 1-D array (NaNs already removed)."""
    if a.size == 0:
        out = {"n": 0, "mean_km": None, "max_km": None, "std_km": None}
        out.update({f"p{p}_km": None for p in PCTL_GRID})
        return out
    pct = np.percentile(a, PCTL_GRID)
    out = {
        "n": int(a.size),
        "mean_km": float(a.mean()),
        "max_km": float(a.max()),
        "std_km": float(a.std()),
    }
    out.update({f"p{p}_km": float(v) for p, v in zip(PCTL_GRID, pct)})
    return out


def _subset_block(sm, sx, bd) -> dict:
    return {
        "centroid_spread_mean_km": _grid_summary(sm[~np.isnan(sm)]),
        "centroid_spread_max_km": _grid_summary(sx[~np.isnan(sx)]),
        "bbox_diagonal_km": _grid_summary(bd[~np.isnan(bd)]),
    }


def compactness_for_fold(df, cent_lat, cent_lon) -> dict:
    preds = df[PRED_COLS].to_numpy()
    is_ood = df["is_ood"].to_numpy().astype(bool)
    sm, sx, bd, n_no_centroid = _shortlist_stats(preds, cent_lat, cent_lon)

    ind = ~is_ood
    ood = is_ood
    return {
        "n_total": int(len(df)),
        "n_ood_total": int(is_ood.sum()),
        "n_dropped_no_usable_centroid": n_no_centroid,
        "in_distribution": _subset_block(sm[ind], sx[ind], bd[ind]),
        "ood": _subset_block(sm[ood], sx[ood], bd[ood]),
        # keep the raw per-visit arrays alive only within the process, for pooling
        "_raw": {
            "sm_ind": sm[ind], "sx_ind": sx[ind], "bd_ind": bd[ind],
            "sm_ood": sm[ood], "sx_ood": sx[ood], "bd_ood": bd[ood],
        },
    }


def _pool(per_fold: dict) -> dict:
    def cat(key):
        return np.concatenate([r["_raw"][key] for r in per_fold.values()]) if per_fold else np.array([])
    sm_ind, sx_ind, bd_ind = cat("sm_ind"), cat("sx_ind"), cat("bd_ind")
    sm_ood, sx_ood, bd_ood = cat("sm_ood"), cat("sx_ood"), cat("bd_ood")
    return {
        "n_total": int(sum(r["n_total"] for r in per_fold.values())),
        "in_distribution": _subset_block(sm_ind, sx_ind, bd_ind),
        "ood": _subset_block(sm_ood, sx_ood, bd_ood),
    }


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def write_markdown(state, per_fold, pooled, out_path):
    today = date.today().isoformat()
    L = []
    a = L.append
    a(f"# Shortlist spatial compactness -- {state}")
    a("")
    a(
        f"> Generated by `analysis/shortlist_compactness.py` on {today}. For each "
        "validation visit, how tightly the model's top-10 predicted regions cluster: "
        "**centroid spread** (mean great-circle km from each shortlist centroid to their "
        "geographic mean) and **bounding-box diagonal** (km). In-distribution and "
        "out-of-distribution (OOD, true region absent from the training vocabulary) visits "
        "are reported SEPARATELY, never pooled. Bare percentiles only, no service-radius / "
        "coverage-threshold overlay (venue-bridge guardrail, MOBILITY_PLAN.md §0/§3.1). This "
        "describes the geometry of the model's own shortlists, not a measured service result "
        "(A40-6 / MOBILITY_SCIENCE_BRIDGE_PLAN.md §12.7)."
    )
    a("")
    a("## Per-fold summary")
    a("")
    a(
        "| Fold | n visits | spread P50 (in-dist) | P90 | bbox-diag P50 (in-dist) | P90 | "
        "n OOD | spread P50 (OOD) | P90 |"
    )
    a("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def row(label, n_total, n_ood, ind, ood):
        s, b = ind["centroid_spread_mean_km"], ind["bbox_diagonal_km"]
        os_ = ood["centroid_spread_mean_km"]
        a(
            f"| {label} | {n_total} | {_fmt(s['p50_km'])} | {_fmt(s['p90_km'])} | "
            f"{_fmt(b['p50_km'])} | {_fmt(b['p90_km'])} | {n_ood} | "
            f"{_fmt(os_['p50_km'])} | {_fmt(os_['p90_km'])} |"
        )

    for fn in sorted(per_fold):
        r = per_fold[fn]
        row(str(fn), r["n_total"], r["n_ood_total"], r["in_distribution"], r["ood"])
    row("**all folds (pooled)**", pooled["n_total"],
        sum(per_fold[f]["n_ood_total"] for f in per_fold),
        pooled["in_distribution"], pooled["ood"])
    a("")
    a(
        "_spread = median / 90th-percentile of the per-visit mean distance (km) from each of "
        "the 10 shortlist centroids to their geographic mean; bbox-diag = the shortlist's "
        "bounding-box diagonal (km). in-dist = the true region was in that fold's training "
        "vocabulary; OOD = it was not (reported separately, MOBILITY_PLAN.md §3). The pooled "
        "row concatenates every fold's visits (informational). The per-visit max spread and "
        "the full P1..P99 grid are in the .json. Bare percentiles only, no coverage overlay "
        "(venue-bridge guardrail)._"
    )
    a("")
    a("## Provenance")
    a("")
    a(
        "Per-sample predictions: `fold{N}_reg_val_preds.parquet` under the training rundir's "
        "`metrics/` directory (`MTL_DUMP_VAL_PREDS=1`). Region geometry: "
        f"`output/check2hgi/{state.lower()}/temp/{{checkin_graph.pt,boroughs_area.csv}}` (same "
        "`load_region_centroids` as `near_miss_distance.py`). Computed over EVERY validation "
        "visit (the shortlist is deployed per visit, not only on top-1 misses)."
    )
    a("")
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def _strip_raw(per_fold: dict) -> dict:
    return {fn: {k: v for k, v in r.items() if k != "_raw"} for fn, r in per_fold.items()}


def main():
    ap = argparse.ArgumentParser(
        description="Shortlist spatial compactness for next-region top-10 predictions "
                    "(A40-6 / MOBILITY_SCIENCE_BRIDGE_PLAN.md §12.7).",
    )
    ap.add_argument("--state", required=True,
                    help="State/city matching output/check2hgi/<state>/ (alabama, arizona, "
                         "florida, istanbul, ...).")
    ap.add_argument("--rundir", required=True,
                    help="Training rundir with metrics/fold*_reg_val_preds.parquet, or that "
                         "metrics/ dir directly.")
    ap.add_argument("--out-prefix", default=None,
                    help="Output prefix (default: shortlist_compactness_<state> next to this script).")
    args = ap.parse_args()

    state = args.state
    out_prefix = args.out_prefix or os.path.join(HERE, f"shortlist_compactness_{state.lower()}")

    val_preds = load_val_preds(args.rundir)
    if not val_preds:
        print(f"[{state}] no fold{{N}}_reg_val_preds.parquet under {args.rundir!r} -- "
              "was this run launched with MTL_DUMP_VAL_PREDS=1? Writing nothing.")
        return

    cent_lat, cent_lon = _centroid_arrays(state)

    per_fold = {}
    for fn, df in sorted(val_preds.items()):
        per_fold[fn] = compactness_for_fold(df, cent_lat, cent_lon)
        ind = per_fold[fn]["in_distribution"]
        s, b = ind["centroid_spread_mean_km"], ind["bbox_diagonal_km"]
        print(f"[{state}] fold {fn}: n={per_fold[fn]['n_total']} "
              f"spread P50={_fmt(s['p50_km'])} P90={_fmt(s['p90_km'])} | "
              f"bbox-diag P50={_fmt(b['p50_km'])} P90={_fmt(b['p90_km'])}"
              + (f" | dropped(no-centroid)={per_fold[fn]['n_dropped_no_usable_centroid']}"
                 if per_fold[fn]["n_dropped_no_usable_centroid"] else ""))

    pooled = _pool(per_fold)

    out_json = f"{out_prefix}.json"
    out_md = f"{out_prefix}.md"
    with open(out_json, "w") as fh:
        json.dump(
            {"state": state, "rundir": args.rundir, "metric": "shortlist_compactness",
             "note": "bare percentile grid per statistic (see script docstring); "
                     "raw per-visit arrays deliberately not stored (~100 MB at FL)",
             "per_fold": _strip_raw(per_fold), "pooled": pooled},
            fh, indent=2,
        )
    write_markdown(state, _strip_raw(per_fold), pooled, out_md)
    print(f"Wrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
