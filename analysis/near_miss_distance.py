#!/usr/bin/env python3
"""
Geographic near-miss distance for next-region predictions.

Question (MOBILITY_PLAN.md §3, BRIDGING_METRICS.md "Geographic near-miss --
deferred item 4"): a rank metric like Acc@10 over thousands of census-tract
regions says nothing about how far a miss lands. For each validation visit
where the model's top-1 predicted region differs from the true region, this
script measures the haversine distance (km) between the predicted region's
centroid and the true region's centroid, and reports the distribution per
fold -- separately for in-distribution and out-of-distribution (OOD) visits,
since an OOD visit's true region was never in the training vocabulary and has
no in-vocabulary correct answer (the same OOD share the Acc@10 "full" metric
already discounts).

Guardrails (settled venue-bridge ruling, motivation-only -- see
MOBILITY_PLAN.md §0/§3.1): this script reports BARE percentiles / the bare
distribution and nothing else. It does NOT compute or plot a distance CDF
against any service radius or coverage threshold (that would be the banned
"coverage curve" in disguise), and it does not interpret the numbers as a
measured service result -- it only measures the model's own predictions.

Data sources (both read-only; this script never writes into either):

  1. Per-sample validation predictions -- the ``fold{N}_reg_val_preds.parquet``
     files produced by the opt-in, default-OFF ``MTL_DUMP_VAL_PREDS=1``
     training-time dump (``src/training/runners/mtl_eval.py`` + ``mtl_cv.py``;
     see the header comments there for the exact trigger: it fires whenever the
     reg/next_region task's diagnostic-best metric improves, the same event the
     per-task BestModelTracker uses -- ``top10_acc_indist`` for the check2HGI
     next_region preset). Columns (one row per validation sample):

       true_region_idx           int32  -- ground-truth region class index
       pred_region_idx_top1..10  int32  -- top-10 predicted region class
                                           indices, descending confidence
                                           (padded -1 if the reg head has
                                           fewer than 10 classes)
       is_ood                    bool   -- True if true_region_idx's region was
                                           absent from this fold's training
                                           vocabulary (same definition
                                           top10_acc_indist already scores
                                           against)

     These live at ``<rundir>/metrics/fold{N}_reg_val_preds.parquet``, right
     alongside the existing ``fold{N}_{task}_val.csv`` files (the same rundir
     convention ``scripts/aggregate_folds.py`` already globs by real fold id).
     Note: this codebase's protocol scores the 5-fold cross-validation's
     validation split as the reported "test" numbers (see RESULTS_BOARD.md's
     convention) -- there is no separate held-out test set, so "validation
     visit" here is exactly what the paper calls a test visit.

  2. Region geometry -- ``output/check2hgi/<state>/temp/checkin_graph.pt`` (a
     PICKLE despite the ``.pt`` extension -- ``pickle.load``, NOT
     ``torch.load``; verified against ``research/embeddings/check2hgi/
     check2hgi.py``'s own loader, which reads it with ``pkl.load``) holds
     ``region_to_idx``: GEOID -> region_idx (the SAME index the model
     predicts). ``output/check2hgi/<state>/temp/boroughs_area.csv`` holds
     GEOID + WKT polygon (EPSG:4326, lon/lat coordinate order) for every
     candidate region.

     GEOID gotcha (verified 2026-07-06 against the real committed artifacts,
     MOBILITY_PLAN.md §3.2): for the five Gowalla states, GEOID is a US
     census-tract id stored as int64 with the leading zero silently dropped
     (e.g. Alabama's ``1073012918`` is really ``01073012918``); zero-pad
     numeric GEOIDs to 11 characters before using them as a join key or
     displaying them (``_geoid_str`` below). Istanbul's GEOID is a non-numeric
     OSM relation-id string (``"relation/1275322"``) and is used as-is.

Run (one state + one training rundir per invocation -- no training-time
dependency, does not need to run on the training machine):

  PYTHONPATH=src python \\
      "analysis/near_miss_distance.py" \\
      --state alabama --rundir /path/to/results/check2hgi_dk_ovl/alabama/<run_folder>

  Loop over states/seeds from the shell as needed, e.g.:
      for s in alabama arizona florida california texas istanbul; do
          python "analysis/near_miss_distance.py" --state "$s" --rundir "$RUNDIR_FOR_$s"
      done

Writes (next to this script, one pair per --state):
  near_miss_distance_<state>.json  -- full per-fold distributions + percentiles
  near_miss_distance_<state>.md    -- P50/P90 summary table + short write-up

If a state has no dumped parquets yet (MTL_DUMP_VAL_PREDS was never turned on
for that run), the script says so plainly and writes nothing -- it never
fabricates or interpolates numbers (folder convention: no placeholders).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
from datetime import date

import numpy as np
import pandas as pd
from shapely import wkt

HERE = os.path.dirname(os.path.abspath(__file__))
# Repo root that holds output/check2hgi/<state>/temp/. Derived from this file's
# location (analysis/ -> one level up == repo root) so it
# resolves on any checkout, worktree, or machine (Mac or the A40 box); override
# with REPO_ROOT if the layout ever differs.
REPO = os.environ.get("REPO_ROOT") or os.path.dirname(HERE)

EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius -- matches common haversine/geopy defaults


def _geoid_str(x) -> str:
    """Normalize a GEOID to its canonical string form.

    US census-tract GEOIDs are stored upstream as int64 with the leading zero
    dropped (MOBILITY_PLAN.md §3.2; verified against the real artifact: Alabama
    ``1073012918`` == ``01073012918``). Zero-pad purely-numeric GEOIDs to 11
    characters; non-numeric ids (e.g. Istanbul's OSM ``"relation/<id>"``
    mahalle keys) pass through unchanged.
    """
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    s = str(x).strip()
    if s.isdigit() and len(s) < 11:
        return s.zfill(11)
    return s


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between paired (lat, lon) arrays (degrees)."""
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def load_region_centroids(state: str) -> dict[int, tuple[float, float]]:
    """region_idx -> (lat, lon) centroid, for every region the model can predict.

    Sourced from ``output/check2hgi/<state>/temp/{checkin_graph.pt,
    boroughs_area.csv}`` (see the module docstring for the exact format and
    the GEOID gotcha).
    """
    state_dir = os.path.join(REPO, "output", "check2hgi", state.lower(), "temp")
    graph_path = os.path.join(state_dir, "checkin_graph.pt")
    boroughs_path = os.path.join(state_dir, "boroughs_area.csv")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"{graph_path} not found -- is the Check2HGI substrate built for {state!r}?"
        )
    if not os.path.exists(boroughs_path):
        raise FileNotFoundError(
            f"{boroughs_path} not found -- is the Check2HGI substrate built for {state!r}?"
        )

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)  # NOT torch.load -- checkin_graph.pt is a plain pickle
    region_to_idx: dict = graph["region_to_idx"]  # GEOID -> region_idx
    idx_to_geoid_str = {idx: _geoid_str(geoid) for geoid, idx in region_to_idx.items()}

    boroughs = pd.read_csv(boroughs_path)
    centroid_by_geoid: dict[str, tuple[float, float]] = {}
    for geoid, geometry_wkt in zip(boroughs["GEOID"], boroughs["geometry"]):
        geoid_s = _geoid_str(geoid)
        if geoid_s in centroid_by_geoid:
            continue  # keep the first polygon for a duplicated GEOID (rare)
        c = wkt.loads(geometry_wkt).centroid
        centroid_by_geoid[geoid_s] = (c.y, c.x)  # (lat, lon) -- WKT coords are (lon, lat)

    idx_to_centroid: dict[int, tuple[float, float]] = {}
    missing = []
    for idx, geoid_s in idx_to_geoid_str.items():
        c = centroid_by_geoid.get(geoid_s)
        if c is None:
            missing.append(geoid_s)
        else:
            idx_to_centroid[idx] = c
    if missing:
        print(
            f"[warn] {state}: {len(missing)}/{len(idx_to_geoid_str)} region_idx have no "
            f"matching centroid in boroughs_area.csv (e.g. {missing[:5]})"
        )
    return idx_to_centroid


def load_val_preds(rundir: str) -> dict[int, pd.DataFrame]:
    """fold label (1-indexed, real fold id) -> the dumped val-preds DataFrame.

    Accepts either the run's top-level rundir (containing
    ``metrics/fold*_reg_val_preds.parquet``, the standard HistoryStorage
    layout) or that ``metrics/`` directory directly.
    """
    candidates = [
        os.path.join(rundir, "metrics", "fold*_reg_val_preds.parquet"),
        os.path.join(rundir, "fold*_reg_val_preds.parquet"),
    ]
    paths: list[str] = []
    for pattern in candidates:
        paths = sorted(glob.glob(pattern))
        if paths:
            break
    out: dict[int, pd.DataFrame] = {}
    for p in paths:
        m = re.search(r"fold(\d+)_reg_val_preds\.parquet$", os.path.basename(p))
        fold_n = int(m.group(1)) if m else len(out) + 1
        out[fold_n] = pd.read_parquet(p)
    return out


def _distances_for_mask(
    true_idx: np.ndarray, pred_idx: np.ndarray, mask: np.ndarray,
    idx_to_centroid: dict[int, tuple[float, float]],
) -> tuple[np.ndarray, int]:
    """Haversine distance (km) for the rows selected by *mask*.

    Rows whose true or predicted region has no centroid (should be rare, and
    would indicate a substrate/geometry mismatch if it happens at all) are
    dropped and counted separately -- never silently zero-filled or ignored.
    """
    sub_true, sub_pred = true_idx[mask], pred_idx[mask]
    lat_t, lon_t, lat_p, lon_p = [], [], [], []
    dropped = 0
    for t, p in zip(sub_true.tolist(), sub_pred.tolist()):
        ct = idx_to_centroid.get(int(t))
        cp = idx_to_centroid.get(int(p))
        if ct is None or cp is None:
            dropped += 1
            continue
        lat_t.append(ct[0]); lon_t.append(ct[1])
        lat_p.append(cp[0]); lon_p.append(cp[1])
    if not lat_t:
        return np.array([], dtype=float), dropped
    d = haversine_km(np.array(lat_p), np.array(lon_p), np.array(lat_t), np.array(lon_t))
    return d, dropped


def _percentile_summary(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"n": 0, "p50_km": None, "p90_km": None, "mean_km": None, "max_km": None}
    return {
        "n": int(d.size),
        "p50_km": float(np.percentile(d, 50)),
        "p90_km": float(np.percentile(d, 90)),
        "mean_km": float(d.mean()),
        "max_km": float(d.max()),
    }


def near_miss_for_fold(df: pd.DataFrame, idx_to_centroid: dict[int, tuple[float, float]]) -> dict:
    """Per-fold near-miss summary: bare percentiles + the full distribution,
    split in-distribution vs OOD misses (never mixed -- MOBILITY_PLAN.md §3)."""
    true_idx = df["true_region_idx"].to_numpy()
    pred_idx = df["pred_region_idx_top1"].to_numpy()
    is_ood = df["is_ood"].to_numpy().astype(bool)
    is_miss = pred_idx != true_idx

    indist_mask = is_miss & ~is_ood
    ood_mask = is_miss & is_ood

    d_indist, dropped_indist = _distances_for_mask(true_idx, pred_idx, indist_mask, idx_to_centroid)
    d_ood, dropped_ood = _distances_for_mask(true_idx, pred_idx, ood_mask, idx_to_centroid)

    return {
        "n_total": int(len(df)),
        "n_ood_total": int(is_ood.sum()),
        "n_miss_total": int(is_miss.sum()),
        "n_miss_in_distribution": int(indist_mask.sum()),
        "n_miss_ood": int(ood_mask.sum()),
        "in_distribution_misses": _percentile_summary(d_indist),
        "ood_misses": _percentile_summary(d_ood),
        "dropped_missing_centroid_indist": dropped_indist,
        "dropped_missing_centroid_ood": dropped_ood,
        "distances_km_in_distribution": d_indist.tolist(),  # the "full distribution"
        "distances_km_ood": d_ood.tolist(),
    }


def _fmt_km(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def write_markdown(state: str, per_fold: dict[int, dict], pooled: dict, out_path: str) -> None:
    today = date.today().isoformat()
    lines: list[str] = []
    L = lines.append
    L(f"# Geographic near-miss distance -- {state}")
    L("")
    L(
        f"> Generated by `analysis/near_miss_distance.py` on {today}. For each validation "
        "visit whose top-1 predicted region is wrong, the haversine distance (km) between "
        "the predicted and true region's centroid. In-distribution and out-of-distribution "
        "(OOD -- true region absent from the training vocabulary) misses are reported "
        "SEPARATELY; they are never pooled together. This is a property of the model's own "
        "predictions on the validation folds, not a measured service result "
        "(MOBILITY_PLAN.md §3)."
    )
    L("")
    L("## Per-fold summary")
    L("")
    L(
        "| Fold | n visits | n misses (in-dist) | P50 km (in-dist) | P90 km (in-dist) | "
        "n misses (OOD) | P50 km (OOD) | P90 km (OOD) |"
    )
    L("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fold_n in sorted(per_fold):
        r = per_fold[fold_n]
        idm, odm = r["in_distribution_misses"], r["ood_misses"]
        L(
            f"| {fold_n} | {r['n_total']} | {idm['n']} | {_fmt_km(idm['p50_km'])} | "
            f"{_fmt_km(idm['p90_km'])} | {odm['n']} | {_fmt_km(odm['p50_km'])} | "
            f"{_fmt_km(odm['p90_km'])} |"
        )
    pidm, podm = pooled["in_distribution_misses"], pooled["ood_misses"]
    L(
        f"| **all folds (pooled)** | {pooled['n_total']} | {pidm['n']} | {_fmt_km(pidm['p50_km'])} | "
        f"{_fmt_km(pidm['p90_km'])} | {podm['n']} | {_fmt_km(podm['p50_km'])} | "
        f"{_fmt_km(podm['p90_km'])} |"
    )
    L("")
    L(
        "_P50/P90 = median / 90th-percentile great-circle distance (km) between the predicted "
        "and true region centroid, computed only on visits where the top-1 prediction is "
        "wrong. in-dist = the true region was present in that fold's training vocabulary; "
        "OOD = it was not (no in-vocabulary correct answer exists for these visits, so they "
        "are reported separately, per MOBILITY_PLAN.md §3). The pooled row simply concatenates "
        "every fold's misses (informational, not a replacement for the per-fold row). Bare "
        "percentiles only -- no service-radius or coverage-threshold overlay is computed here "
        "(venue-bridge guardrail, MOBILITY_PLAN.md §0/§3.1)._"
    )
    L("")
    L("## Provenance")
    L("")
    L(
        "Per-sample predictions: `fold{N}_reg_val_preds.parquet` under the training rundir's "
        "`metrics/` directory (`MTL_DUMP_VAL_PREDS=1`, `src/training/runners/mtl_eval.py` + "
        "`mtl_cv.py`, dumped on every reg diagnostic-best improvement). Region geometry: "
        f"`output/check2hgi/{state.lower()}/temp/{{checkin_graph.pt,boroughs_area.csv}}`."
    )
    L("")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _pool_folds(per_fold: dict[int, dict]) -> dict:
    """Informational cross-fold pool (simple concatenation of every fold's
    per-visit misses) -- NOT a substitute for the per-fold report, which is
    the one MOBILITY_PLAN.md §3 asks for; kept separate and clearly labelled."""
    d_indist = np.concatenate(
        [np.asarray(r["distances_km_in_distribution"]) for r in per_fold.values()]
    ) if per_fold else np.array([])
    d_ood = np.concatenate(
        [np.asarray(r["distances_km_ood"]) for r in per_fold.values()]
    ) if per_fold else np.array([])
    return {
        "n_total": int(sum(r["n_total"] for r in per_fold.values())),
        "in_distribution_misses": _percentile_summary(d_indist),
        "ood_misses": _percentile_summary(d_ood),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Geographic near-miss distance for next-region predictions "
                    "(MOBILITY_PLAN.md §3).",
    )
    ap.add_argument(
        "--state", required=True,
        help="State/city name matching output/check2hgi/<state>/ (e.g. alabama, "
             "arizona, florida, california, texas, istanbul).",
    )
    ap.add_argument(
        "--rundir", required=True,
        help="Training rundir containing metrics/fold*_reg_val_preds.parquet "
             "(the MTL_DUMP_VAL_PREDS=1 output), or that metrics/ dir directly.",
    )
    ap.add_argument(
        "--out-prefix", default=None,
        help="Output file prefix (default: near_miss_distance_<state> next to this script).",
    )
    args = ap.parse_args()

    state = args.state
    out_prefix = args.out_prefix or os.path.join(HERE, f"near_miss_distance_{state.lower()}")

    val_preds = load_val_preds(args.rundir)
    if not val_preds:
        print(
            f"[{state}] no fold{{N}}_reg_val_preds.parquet found under {args.rundir!r} -- "
            "was this run launched with MTL_DUMP_VAL_PREDS=1? Writing nothing."
        )
        return

    idx_to_centroid = load_region_centroids(state)

    per_fold = {}
    for fold_n, df in sorted(val_preds.items()):
        per_fold[fold_n] = near_miss_for_fold(df, idx_to_centroid)
        idm, odm = per_fold[fold_n]["in_distribution_misses"], per_fold[fold_n]["ood_misses"]
        print(
            f"[{state}] fold {fold_n}: n={per_fold[fold_n]['n_total']} "
            f"misses(in-dist)={idm['n']} P50={_fmt_km(idm['p50_km'])} P90={_fmt_km(idm['p90_km'])} | "
            f"misses(OOD)={odm['n']} P50={_fmt_km(odm['p50_km'])} P90={_fmt_km(odm['p90_km'])}"
        )

    pooled = _pool_folds(per_fold)

    out_json = f"{out_prefix}.json"
    out_md = f"{out_prefix}.md"
    with open(out_json, "w") as fh:
        json.dump(
            {"state": state, "rundir": args.rundir, "per_fold": per_fold, "pooled": pooled},
            fh, indent=2,
        )
    write_markdown(state, per_fold, pooled, out_md)
    print(f"Wrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
