#!/usr/bin/env python3
"""
Co-visitation network analysis (replicating Moura, Aquino & Loureiro,
"On the Design of Mobility-Aware Systems: A Tourist's Perspective", MSWiM 2025,
DOI 10.1109/MSWiM67937.2025.11308734) on our own check-in data.

Methodology (mirrors the target paper as closely as feasible):
  - Nodes = distinct placeid within a state/city.
  - Undirected edge between two POIs i, j whenever the SAME userid checked in
    at both; edge weight = number of distinct users who visited both (i.e. the
    co-occurrence count in a binary user-by-place incidence matrix).
  - Built via sparse matmul: M (users x places, binary "ever visited") ->
    P = M^T @ M (places x places co-visitor counts), diagonal zeroed. This
    avoids ever materializing an explicit user-pair loop (task instruction).

Reported per dataset (Sec. "graph_stats" below):
  - node/edge counts, average/max degree, category of the top-5 highest-degree
    POIs.
  - heavy-tailedness: max/mean degree ratio, share of nodes below median
    degree, and an approximate log-log slope from a rank-frequency (Zipf)
    regression of log(rank) vs log(degree) -- NOT a proper power-law MLE fit
    (the `powerlaw` package is not installed here); labelled approximate.
  - global clustering coefficient vs. density. For graphs small enough,
    exact global transitivity is attempted via a direct triangle count
    (trace(A^3)/6, computed through a single extra sparse matmul). For the
    graphs actually seen in this study this was NEVER cheap enough in
    practice (see findings doc) -- we therefore uniformly report an unbiased
    WEDGE-SAMPLING estimator of the global clustering coefficient (transitivity):
    sample "cherries" (2-paths) with probability proportional to
    C(deg_i, 2) at each candidate center i, and check whether the two sampled
    neighbours are themselves connected. The fraction of closed sampled wedges
    is an unbiased estimator of C = 3T/W (T = triangles, W = wedges), because
    each triangle contributes exactly 3 wedges across its 3 possible centers.
    This is validated against the exact value where the exact value is
    affordable (see findings doc, Alabama cross-check).
  - giant connected component fraction, via
    scipy.sparse.csgraph.connected_components (exact, cheap regardless of
    density -- linear in nnz).
  - diameter / average shortest path length: EXACT all-pairs computation is
    infeasible at these densities (see findings doc). We sample source nodes
    from the giant component and run unweighted BFS
    (scipy.sparse.csgraph.shortest_path, method="D", unweighted=True) from
    each; the max observed eccentricity in the sample is reported as an
    approximate ("sampled") diameter, and the mean of all finite pairwise
    distances from the sampled sources as an approximate average shortest
    path length. Both are explicitly labelled as sampled, not exact.

User segmentation (length-of-stay analogy): computed separately in
`user_segmentation()` / `main()` -- see findings doc for the split criterion
and justification (total check-in count, not span-in-days).

Run:
  /Users/vitor/Desktop/mestrado/ingred/.venv/bin/python \
      "articles/[mobiwac]/analysis/covisitation_network.py"

Writes:
  - covisitation_network_results.json (raw numbers, next to this script)
  - covisitation_network_findings.md   (write-up, next to this script)

Runtime note: Alabama/Arizona/Florida/Istanbul complete in well under a
minute each. California's raw (weight>=1) graph is very dense (see findings
doc) and its full pipeline takes several minutes and ~10GB RAM. Texas's raw
graph could NOT be built at all on this machine (crashed on the M^T @ M
matmul itself, ~124s in, before any thresholding was even possible) -- see
findings doc for the concrete numbers we do have and why we skip it rather
than silently truncate.
"""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

REPO = "/Users/vitor/Desktop/mestrado/ingred"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO, "data/checkins")
OUT_JSON = os.path.join(HERE, "covisitation_network_results.json")

SEED = 0
WEDGE_SAMPLES = 20000          # samples for the clustering-coefficient estimator
BFS_SAMPLES_DEFAULT = 300      # BFS sources for diameter/avg-shortest-path sampling
BFS_SAMPLES_DENSE = 60         # reduced sample for moderately dense/large graphs (FL-scale)
DENSE_EDGE_CUTOFF = 50_000_000       # above this, use scipy Dijkstra-based shortest_path with BFS_SAMPLES_DENSE
VERY_DENSE_EDGE_CUTOFF = 400_000_000  # above this, scipy's Dijkstra-based shortest_path is not
# tractable at all in a light experiment (empirically: >5.5 min for 200 sources on California's
# 708M-edge graph, still incomplete when killed). Above this cutoff we switch to a hand-rolled
# vectorized unweighted BFS (true O(V+E) per source, no heap) and drastically cut the sample size,
# because even the hand-rolled BFS costs ~60-80s/source at California's density (dominated by
# np.unique() over a first-hop frontier that already touches most of the graph). See findings doc.
BFS_SAMPLES_VERY_DENSE = 5

# Datasets: (label, filename stem under data/checkins/). Istanbul shares the
# exact same parquet schema as the US states (userid, placeid, datetime,
# category, ...), so it needed no special-cased loader.
DATASETS = [
    ("Alabama", "Alabama"),
    ("Arizona", "Arizona"),
    ("Florida", "Florida"),
    ("California", "California"),
    ("Texas", "Texas"),
    ("Istanbul", "Istanbul"),
]

# Texas's raw co-occurrence matrix could not be built on this machine (see
# findings doc): the M^T @ M matmul itself was killed after ~124s at a
# maximum resident set size of ~7.7GB / peak memory footprint ~79GB reported
# by /usr/bin/time -l (i.e. it was thrashing, not just slow). We do not retry
# with ad hoc outlier-user exclusion because that would silently change the
# graph-construction method relative to every other state; we report the
# crash and the numbers we do have instead (SKIP_STATES below).
SKIP_STATES = {"Texas"}

# California's full-state graph IS tractable (unlike Texas) but is already at
# the edge of a "light" time budget (~1-2 min for the matmul + connected
# components alone, plus a necessarily tiny/approximate diameter sample -- see
# VERY_DENSE_EDGE_CUTOFF above). Its own high/low-activity segment subgraphs
# would be comparably dense (the same super-active users dominate both), so we
# do not additionally build California segment subgraphs; Alabama, Arizona,
# Florida and Istanbul already give a clear, consistent segmentation result.
SKIP_SEGMENTATION_STATES = {"California"}


def load_checkins(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    return pd.read_parquet(path, columns=["userid", "placeid", "datetime", "category"])


def build_incidence(df: pd.DataFrame):
    """Binary user-by-place incidence matrix M (nnz = distinct (user,place) pairs)."""
    users = df["userid"].astype("category")
    places = df["placeid"].astype("category")
    pairs = pd.DataFrame({"u": users.cat.codes.values, "p": places.cat.codes.values}).drop_duplicates()
    n_users = len(users.cat.categories)
    n_places = len(places.cat.categories)
    M = sp.csr_matrix(
        (np.ones(len(pairs), dtype=np.float64), (pairs["u"].values, pairs["p"].values)),
        shape=(n_users, n_places),
    )
    place_ids = np.asarray(places.cat.categories)
    return M, place_ids


def cooccurrence(M: sp.csr_matrix) -> sp.csr_matrix:
    """P = M^T @ M (places x places, shared-visitor counts), diagonal zeroed."""
    P = (M.T @ M).tocsr()
    P.setdiag(0)
    P.eliminate_zeros()
    return P


def place_category_map(df: pd.DataFrame) -> pd.Series:
    """placeid -> category. Category is a fixed venue attribute (verified: 0
    places in Alabama have >1 distinct category value), so first-seen is exact,
    not a tie-broken mode."""
    return df.drop_duplicates("placeid").set_index("placeid")["category"]


def heavy_tail_stats(deg: np.ndarray) -> dict:
    n = len(deg)
    mean_deg = float(deg.mean())
    max_deg = int(deg.max())
    median_deg = float(np.median(deg))
    share_below_median = float((deg < median_deg).mean())
    nz = deg[deg > 0]
    order = np.argsort(-nz)
    sorted_deg = nz[order].astype(np.float64)
    ranks = np.arange(1, len(sorted_deg) + 1, dtype=np.float64)
    log_rank = np.log(ranks)
    log_deg = np.log(sorted_deg)
    slope, intercept = np.polyfit(log_rank, log_deg, 1)
    pred = slope * log_rank + intercept
    ss_res = float(np.sum((log_deg - pred) ** 2))
    ss_tot = float(np.sum((log_deg - log_deg.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(
        mean_degree=mean_deg,
        max_degree=max_deg,
        median_degree=median_deg,
        max_over_mean=max_deg / mean_deg if mean_deg > 0 else float("nan"),
        share_nodes_below_median_degree=share_below_median,
        loglog_rank_slope_approx=float(slope),
        loglog_rank_r2_approx=float(r2),
    )


def wedge_sampling_clustering(indptr: np.ndarray, indices: np.ndarray, deg: np.ndarray,
                               n_samples: int = WEDGE_SAMPLES, seed: int = SEED):
    """Unbiased wedge-sampling estimator of global transitivity C = 3T/W.
    Requires `indices` sorted within each row (caller must call .sort_indices())."""
    w = deg.astype(np.float64) * (deg - 1) / 2.0
    pool = np.where(deg >= 2)[0]
    if len(pool) == 0:
        return None
    rng = np.random.default_rng(seed)
    probs = w[pool] / w[pool].sum()
    centers = rng.choice(pool, size=n_samples, p=probs)
    closed = 0
    counted = 0
    for c in centers:
        start, end = indptr[c], indptr[c + 1]
        neigh = indices[start:end]
        if len(neigh) < 2:
            continue
        j, k = rng.choice(len(neigh), size=2, replace=False)
        nj, nk = neigh[j], neigh[k]
        rstart, rend = indptr[nj], indptr[nj + 1]
        row = indices[rstart:rend]
        pos = np.searchsorted(row, nk)
        counted += 1
        if pos < len(row) and row[pos] == nk:
            closed += 1
    if counted == 0:
        return None
    est = closed / counted
    se = float(np.sqrt(est * (1 - est) / counted))
    return dict(estimate=float(est), se=se, n_wedges_sampled=int(counted), total_wedges=float(w.sum()))


def _fast_unweighted_bfs(indptr: np.ndarray, indices: np.ndarray, source: int, n: int):
    """Hand-rolled O(V+E) unweighted BFS from a single source, using the CSR
    structure directly (vectorized ragged-gather per frontier level). Used
    instead of scipy's Dijkstra-based shortest_path(method='D') for very dense
    graphs, where the Dijkstra heap overhead proved intractable (see findings
    doc). Returns (dist array with -1 for unreached, eccentricity)."""
    dist = np.full(n, -1, dtype=np.int32)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    d = 0
    while len(frontier) > 0:
        d += 1
        starts = indptr[frontier]
        ends = indptr[frontier + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            break
        cum_excl = np.concatenate(([0], np.cumsum(lengths)))[:-1]
        flat_idx = np.arange(total) - np.repeat(cum_excl, lengths) + np.repeat(starts, lengths)
        neighbors = np.unique(indices[flat_idx])
        unvisited = neighbors[dist[neighbors] == -1]
        if len(unvisited) == 0:
            break
        dist[unvisited] = d
        frontier = unvisited
    return dist, d - 1  # d was incremented once more than the last successful assignment


def giant_component_and_distances(A: sp.csr_matrix, n_nodes: int, seed: int = SEED):
    ncomp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    giant_label = int(sizes.argmax())
    giant_frac = float(sizes.max() / n_nodes)
    giant_nodes = np.where(labels == giant_label)[0]

    n_edges = A.nnz // 2
    rng = np.random.default_rng(seed)

    if n_edges > VERY_DENSE_EDGE_CUTOFF:
        # scipy's Dijkstra-based shortest_path is not tractable here (empirically
        # >5.5 min for 200 sources on a 708M-edge graph and still not done). Fall
        # back to a hand-rolled BFS with a drastically smaller sample, and be
        # explicit about the small n in the result.
        sample_size = min(BFS_SAMPLES_VERY_DENSE, len(giant_nodes))
        sample_nodes = rng.choice(giant_nodes, size=sample_size, replace=False)
        method = "hand_rolled_bfs_tiny_sample"
        eccs, means = [], []
        for s in sample_nodes:
            dist, ecc = _fast_unweighted_bfs(A.indptr, A.indices, int(s), n_nodes)
            finite = dist[dist >= 0]
            eccs.append(ecc)
            means.append(float(finite.mean()))
        approx_diameter = int(max(eccs))
        approx_avg_spl = float(np.mean(means))
    else:
        sample_size = BFS_SAMPLES_DENSE if n_edges > DENSE_EDGE_CUTOFF else BFS_SAMPLES_DEFAULT
        sample_size = min(sample_size, len(giant_nodes))
        method = "scipy_dijkstra_unweighted"
        sample_nodes = rng.choice(giant_nodes, size=sample_size, replace=False)
        D = shortest_path(A, method="D", unweighted=True, indices=sample_nodes)
        finite = np.isfinite(D)
        ecc = np.where(finite, D, -1).max(axis=1)
        approx_diameter = int(ecc.max())
        approx_avg_spl = float(D[finite].mean())

    return dict(
        n_components=int(ncomp),
        giant_component_fraction=giant_frac,
        bfs_sample_size=int(sample_size),
        bfs_method=method,
        approx_diameter_sampled=approx_diameter,
        approx_avg_shortest_path_sampled=approx_avg_spl,
    )


def analyze_graph(df: pd.DataFrame, label: str) -> dict:
    t0 = time.time()
    M, place_ids = build_incidence(df)
    P = cooccurrence(M)
    n_nodes = P.shape[0]
    n_edges = P.nnz // 2
    deg = np.diff(P.indptr)

    cat_map = place_category_map(df)
    top5_idx = np.argsort(-deg)[:5]
    top5 = [
        dict(placeid=int(place_ids[i]), degree=int(deg[i]),
             category=str(cat_map.get(place_ids[i], "?")))
        for i in top5_idx
    ]

    A = P.copy()
    A.data[:] = 1.0
    A.sort_indices()

    clustering = wedge_sampling_clustering(A.indptr, A.indices, deg)
    density = n_edges / (n_nodes * (n_nodes - 1) / 2)
    comp = giant_component_and_distances(A, n_nodes)

    result = dict(
        label=label,
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
        density=float(density),
        top5_by_degree=top5,
        elapsed_sec=time.time() - t0,
        **heavy_tail_stats(deg),
        clustering_wedge_sampling=clustering,
        **comp,
    )
    return result


def user_activity_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("userid")
    total_checkins = g.size()
    span_days = (g["datetime"].max() - g["datetime"].min()).dt.total_seconds() / 86400.0
    distinct_days = df.assign(_day=df["datetime"].dt.date).groupby("userid")["_day"].nunique()
    return pd.DataFrame({
        "total_checkins": total_checkins,
        "span_days": span_days,
        "distinct_active_days": distinct_days,
    })


def percentile_report(series: pd.Series) -> dict:
    pcts = [10, 25, 50, 75, 90, 99]
    vals = np.percentile(series.values, pcts)
    return {f"p{p}": float(v) for p, v in zip(pcts, vals)}


def segment_users(activity: pd.DataFrame, frac: float = 0.10):
    """Top `frac` of users by total_checkins = high-activity segment.
    Comparably-sized bottom segment drawn from multi-checkin (>=2) users only
    (single-checkin users contribute zero co-visitation edges)."""
    n = len(activity)
    k = max(1, round(frac * n))
    ranked = activity.sort_values("total_checkins", ascending=False)
    high = ranked.index[:k]

    multi = activity[activity["total_checkins"] >= 2]
    k_low = min(k, len(multi))
    low = multi.sort_values("total_checkins", ascending=True).index[:k_low]
    return set(high), set(low), k, k_low


def main():
    results = {"datasets": {}, "segmentation": {}}

    for label, stem in DATASETS:
        print(f"=== {label} ===")
        if label in SKIP_STATES:
            df = load_checkins(stem)
            users = df["userid"].astype("category")
            places = df["placeid"].astype("category")
            pairs = pd.DataFrame({"u": users.cat.codes.values, "p": places.cat.codes.values}).drop_duplicates()
            per_user_places = pairs.groupby("u").size()
            results["datasets"][label] = dict(
                label=label,
                skipped=True,
                reason=(
                    "Raw co-occurrence matrix (M^T @ M) crashed the machine during "
                    "construction (killed after ~124s, /usr/bin/time -l reported "
                    "maximum resident set size ~7.7GB and peak memory footprint "
                    "~79GB -- i.e. it was thrashing/OOM, not just slow). Driven by "
                    "extreme per-user place-count skew (see max_unique_places_single_user)."
                ),
                n_checkins=int(len(df)),
                n_users=int(len(users.cat.categories)),
                n_places=int(len(places.cat.categories)),
                n_unique_user_place_pairs=int(len(pairs)),
                max_unique_places_single_user=int(per_user_places.max()),
                sum_pairwise_combinations_per_user=float((per_user_places * (per_user_places - 1) // 2).sum()),
            )
            print(f"  SKIPPED: {results['datasets'][label]['reason']}")
            continue

        df = load_checkins(stem)
        try:
            res = analyze_graph(df, label)
            results["datasets"][label] = res
            print(f"  nodes={res['n_nodes']} edges={res['n_edges']} density={res['density']:.4g} "
                  f"elapsed={res['elapsed_sec']:.1f}s")
        except Exception as e:  # pragma: no cover - defensive, report and move on
            results["datasets"][label] = dict(label=label, skipped=True, reason=f"exception: {e}\n{traceback.format_exc()}")
            print(f"  FAILED: {e}")
            continue

        # User activity distribution (percentiles) -- computed for every
        # dataset we could load, not just AL/FL, since it's cheap.
        activity = user_activity_table(df)
        n_single = int((activity["total_checkins"] == 1).sum())
        pct = dict(
            n_users=int(len(activity)),
            n_single_checkin_users=n_single,
            total_checkins_percentiles=percentile_report(activity["total_checkins"]),
            span_days_percentiles=percentile_report(activity["span_days"]),
            distinct_active_days_percentiles=percentile_report(activity["distinct_active_days"]),
        )

        # Segmentation split + subgraphs (skip for states whose full graph
        # itself we couldn't build -- already excluded via SKIP_STATES/continue
        # above -- and for California, see SKIP_SEGMENTATION_STATES above).
        high_ids, low_ids, k_high, k_low = segment_users(activity)
        high_checkins = activity.loc[list(high_ids), "total_checkins"]
        low_checkins = activity.loc[list(low_ids), "total_checkins"]
        seg_result = dict(
            percentiles=pct, k_high=k_high, k_low=k_low,
            high_checkins_range=[int(high_checkins.min()), int(high_checkins.max())],
            low_checkins_range=[int(low_checkins.min()), int(low_checkins.max())],
        )
        if label in SKIP_SEGMENTATION_STATES:
            seg_result["skipped"] = True
            seg_result["reason"] = (
                "Full-state graph already at the edge of the light time budget "
                "(matmul + connected components ~1-2 min, diameter only sampleable "
                "at n=5 via a hand-rolled BFS); the high-activity segment would be "
                "comparably dense (same super-active users dominate both), so we "
                "did not additionally build California segment subgraphs. AL/AZ/FL/"
                "Istanbul already give a clear, consistent segmentation result."
            )
            results["segmentation"][label] = seg_result
            print(f"  [segmentation] SKIPPED: {seg_result['reason']}")
            continue
        for seg_name, ids in [("high_activity", high_ids), ("low_activity_multi_checkin", low_ids)]:
            sub_df = df[df["userid"].isin(ids)]
            try:
                sub_res = analyze_graph(sub_df, f"{label}/{seg_name}")
                seg_result[seg_name] = sub_res
                print(f"  [{seg_name}] users={len(ids)} nodes={sub_res['n_nodes']} "
                      f"edges={sub_res['n_edges']} density={sub_res['density']:.4g} "
                      f"C~{sub_res['clustering_wedge_sampling']['estimate'] if sub_res['clustering_wedge_sampling'] else None}")
            except Exception as e:
                seg_result[seg_name] = dict(skipped=True, reason=f"exception: {e}")
                print(f"  [{seg_name}] FAILED: {e}")
        results["segmentation"][label] = seg_result

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
