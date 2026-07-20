# Shortlist spatial compactness -- cross-state results (A40-6, P7 support)

> Consolidated summary of `shortlist_compactness.py` across states. For each
> validation visit, how tightly the model's **top-10 predicted regions** cluster
> in space: **centroid spread** (mean haversine km from each of the 10 shortlist
> centroids to their geographic mean) and **bounding-box diagonal** (km).
> In-distribution and out-of-distribution (OOD -- true region absent from the
> fold's training vocabulary) visits are reported SEPARATELY, never pooled. Bare
> percentiles only; no service-radius / coverage-threshold overlay (venue-bridge
> guardrail, MOBILITY_PLAN.md §0/§3.1). This describes the geometry of the
> model's own shortlists on the validation folds, not a measured service result.

## Why this exists (P7 / C4)

The paper's P7 usage sketch argues a top-10 region shortlist is actionable via
its enrichment over chance (~78-547x). Enrichment is a *relative* claim; the
sharpest residual reviewer counter is "is the shortlist also spatially compact,
or scattered across a whole state?" These numbers answer it from data that
already existed on this machine (the near-miss `MTL_DUMP_VAL_PREDS=1` dumps),
no retraining. It changes no verdict; it strengthens P7's C4 argument
(camera-ready / rebuttal). Spec: `MOBILITY_SCIENCE_BRIDGE_PLAN.md §12.7`,
`docs/studies/closing_data/v17_completion/A40.md · A40-6`.

## Provenance

- **Recipe:** the SAME champion v17 runs the near-miss metric used (bs8192 +
  per-head cat-lr, gated stride-1 overlap engine `check2hgi_dk_ovl`, fp32,
  seed 0 x 5 folds, `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower`
  (reg) + `next_gru` (cat)). Same four rundirs as PR #59.
- **Per-sample predictions:** the `fold{N}_reg_val_preds.parquet` dumps -- the
  top-10 predicted region indices per validation visit.
- **Region geometry:** `output/check2hgi/<state>/temp/{checkin_graph.pt,
  boroughs_area.csv}` via `load_region_centroids` reused verbatim from
  `near_miss_distance.py` (region_idx -> GEOID -> WKT-polygon centroid; 11-char
  GEOID zero-pad handled; **0 shortlist entries dropped for a missing centroid**
  at every state).

## Caveat — Istanbul is a single-seed point estimate

Istanbul's numbers here come from ONE **seed-0 x 5-fold** v17 `check2hgi_dk_ovl`
run (the same PR #59 near-miss dumps). The paper reports Istanbul at **n=20
(4 seeds {0,1,7,100})** for its headline region cell, and that cell is still on
the older **stride-1 GCN** substrate (the `dk_ovl` Istanbul rebuild is the
pending A40-2/H3 task) -- so the Istanbul figure differs from the paper standard
on BOTH seed count and substrate. This is acceptable for a motivation-only
metric, but it is a single-seed point estimate, not the multi-seed standard.
AL/AZ/FL do NOT carry this caveat: their paper cells are themselves n=5 seed-0
provisional (P1 pending), so seed-0 near-miss/compactness matches their current
headline standard; only Istanbul's paper standard is already n=20.

**Top-up:** when the n=20 `dk_ovl` Istanbul dumps exist (A40-2/H3 rebuild, or
the P1/H100 lane, re-running v17 Istanbul at seeds 1/7/100 with
`MTL_DUMP_VAL_PREDS=1`), regenerate this metric from them and pool. Istanbul is
small (520 regions), so the top-up is A40-feasible, not strictly H100-only.

## Summary (pooled over the 5 folds, per state)

Reported over EVERY validation visit (the shortlist is what a downstream system
would act on for every visit, whether or not the top-1 is right -- so, unlike
the near-miss metric, this is not conditioned on a top-1 miss).

| State | regions | spread P50 (km) | spread P90 (km) | bbox-diag P50 (km) | bbox-diag P90 (km) | OOD spread P50 / P90 (km) |
|---|---:|---:|---:|---:|---:|---:|
| Istanbul |   520 | **2.86** |  7.55 | 10.91 |  24.22 |  3.66 /   9.11 |
| Alabama  | 1,109 | **6.24** | 26.67 | 23.67 | 102.96 | 12.55 /  96.81 |
| Arizona  | 1,547 | **6.09** | 15.64 | 23.62 |  57.47 | 14.00 / 145.91 |
| Florida  | 4,703 | **7.53** | 15.09 | 32.14 |  57.92 | 11.99 /  59.43 |

_spread = median / 90th-percentile of the per-visit mean distance from each of
the 10 shortlist centroids to their geographic mean; bbox-diag = the shortlist's
bounding-box diagonal. Full P1..P99 grid + per-visit max spread in each
`shortlist_compactness_<state>.json`._

## Reading (with its reference point)

The model's top-10 shortlist is **spatially tight**: a median in-distribution
spread of about **3 km (Istanbul) to 8 km (Florida)**, with 90% of shortlists
inside ~8-27 km. The reference point (GLOSSARY par.4: never a number without
one) is the map's own scale -- the random-pair inter-centroid P50 from
`near_miss_floor.py` (Istanbul 20.4, Alabama 170.7, Arizona 120.3, Florida
241.2 km): the model's shortlist spread is roughly **7x (Istanbul), 27x
(Alabama), 20x (Arizona), 32x (Florida) tighter** than the characteristic
distance between two random candidate regions of the same map. So the shortlist
is not scattered across the state; it concentrates on a small neighborhood of
regions. Istanbul is the tightest, consistent with its geography (520 dense
mahalle in one metropolitan area vs. 1,109-4,703 census tracts across a whole
US state). OOD visits (true region never in the fold's training vocabulary, so
no in-vocabulary correct answer exists) have wider, longer-tailed shortlists and
are reported separately, never mixed into the in-distribution figure.

## Health (no anomaly cited)

0 shortlist entries dropped for a missing centroid at any state (every predicted
region_idx resolved to a polygon centroid). Per-fold spread P50 is stable across
the 5 folds at every state (e.g. Alabama 5.80-6.94, Florida 7.38-7.79). OOD
counts are small (Istanbul 180, Alabama 527, Arizona 523, Florida 1,172 pooled)
and kept separate. The heavy right tail in the pooled mean vs. median (e.g.
Alabama mean 12.6 vs. P50 6.2) is a handful of scattered shortlists, which is
why the median is the headline, not the mean.

## Source rundirs (on the A40; `results/` is gitignored)

Identical to the near-miss rundirs (same dumps).

| State | training rundir |
|---|---|
| Alabama  | `results/check2hgi_dk_ovl/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260707_163142_347078` |
| Arizona  | `results/check2hgi_dk_ovl/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260707_164331_349463` |
| Istanbul | `results/check2hgi_dk_ovl/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260707_164911_350709` |
| Florida  | `results/check2hgi_dk_ovl/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260707_182009_363321` |

## Reproduce

No GPU, no training. Against the existing dumps (CPU-minutes total):

```bash
export PYTHONPATH=src
for spec in \
  "alabama  results/check2hgi_dk_ovl/alabama/mtlnet_lr1.0e-04_bs8192_ep50_20260707_163142_347078" \
  "arizona  results/check2hgi_dk_ovl/arizona/mtlnet_lr1.0e-04_bs8192_ep50_20260707_164331_349463" \
  "istanbul results/check2hgi_dk_ovl/istanbul/mtlnet_lr1.0e-04_bs8192_ep50_20260707_164911_350709" \
  "florida  results/check2hgi_dk_ovl/florida/mtlnet_lr1.0e-04_bs8192_ep50_20260707_182009_363321" ; do
  set -- $spec
  python "articles/[mobiwac]/analysis/shortlist_compactness.py" --state "$1" --rundir "$2"
done
```

To regenerate the dumps from scratch (only if the rundirs are gone), see the
near-miss `Reproduce` recipe in `near_miss_RESULTS.md` (same runs;
`MTL_DUMP_VAL_PREDS=1`), then point `--rundir` at the fresh rundir.

## Note on JSON format (deliberate)

Each `shortlist_compactness_<state>.json` stores a **dense bare percentile grid**
(P1..P99 + mean/max/std/n) per statistic, per subset, per fold and pooled --
NOT the raw per-visit arrays. A raw all-visits x 3-statistic dump is ~1.3M rows
at Florida (~100 MB of JSON), well past the near-miss dumps' already-heavy 17 MB;
the grid fully characterizes the distribution shape, is literally "bare
percentiles" (guardrail-safe), and keeps the artifact at ~20 KB.

## Status

All four A40-feasible states complete (2026-07-08, A40, CPU-only). CA/TX are
out of scope by the A40-6 spec (their `catx_v17_seed0_5f/` runs did not carry
the dump flag; C4 scopes honestly to "the four datasets where we measured it").
The paper placement (whether this enters §6.2 / a supplementary) is an author
decision the plan defers.

## Matched random comparator — TEN regions drawn at random (added 2026-07-20)

A readability reviewer flagged that the section-7 sentence compared asymmetric
quantities: the TEN-region shortlist's spread vs the TWO-region random-pair
inter-centroid floor (`near_miss_floor.py`, 20.45-241.22 km P50). Author
ruling: recompute the comparator as TEN regions drawn at random, so both sides
are the same quantity — the spread of a ten-region set around its own centroid.
`shortlist_compactness_matched.py` does that (results in
`shortlist_compactness_matched.json`); the shortlist side is untouched.

**Matched quantity** (identical to the shortlist side, reusing
`_shortlist_stats` / `_grid_summary` verbatim): per draw of 10 regions, the
MEAN haversine km from each of the 10 centroids to their spherical unit-vector
geographic mean; median (P50) + IQR over 10,000 draws, fixed seed 0.
**Pool** (identical to the pair floor): the model's candidate-region vocabulary
(`checkin_graph.pt` `region_to_idx` joined to `boroughs_area.csv` polygon
centroids), uniform draw, 10 distinct regions per draw (without replacement,
matching the shortlist's 10 distinct predicted indices; the pair floor's
ordered-pairs-with-replacement scheme has no 10-region verbatim analog).

**Reproduce-first gate (passed 2026-07-20, built into the script):** the
two-region pair floor recomputes bit-exactly from the local
`output/check2hgi/<state>/temp/` inputs (P50/P90/mean all match the published
table above to the printed rounding), and the shortlist-side pooled in-dist
spread P50s verify against the recorded `shortlist_compactness_<state>.json`
grids (2.8583 / 6.2368 / 6.0927 / 7.5289; the raw `fold{N}_reg_val_preds.parquet`
dumps are A40-only, so the shortlist side is verified against its artifact of
record, not recomputed from raw dumps).

| State | pool regions | matched 10-random P50 (km) | IQR (P25-P75) | shortlist P50 (km) | shortlist is | old 2-random P50 |
|---|---:|---:|---:|---:|---:|---:|
| Istanbul |   520 |  16.64 | 13.14-20.45  | 2.86 | ~5.8x tighter  |  20.45 |
| Arizona  | 1,547 |  87.79 | 69.15-107.28 | 6.09 | ~14.4x tighter | 120.32 |
| Alabama  | 1,109 | 135.97 | 115.12-156.48| 6.24 | ~21.8x tighter | 170.67 |
| Florida  | 4,703 | 176.16 | 152.36-204.59| 7.53 | ~23.4x tighter | 241.22 |

Reading: the matched comparator is, as expected, SMALLER than the pair floor
(a ten-point set's mean spread around its own centroid is tighter than a
random pair's separation), so the contrast is somewhat weaker than the old
"20 to 241 km" (now "17 to 176 km"), but the story is unchanged: the model's
shortlist is roughly 6x (Istanbul) to 23x (Florida) tighter than ten random
candidate regions of the same map. Sensitivity: using the per-draw MEDIAN
distance to the centroid instead of the pipeline's per-draw MEAN gives P50s of
12.64 / 55.04 / 124.34 / 156.48 km (Istanbul/AZ/AL/FL) — same order, same
conclusion; the headline stays on the pipeline statistic (mean), matching the
shortlist side exactly.

Section-7 replacement sentence this feeds (author to apply in
`src/sections/07_discussion.tex`): "On four datasets (Alabama, Arizona,
Florida, and Istanbul; a single seed over five folds), the ten shortlisted
regions lie a median of 3 to 8 kilometers from the shortlist's centroid,
against 17 to 176 kilometers for ten regions drawn at random from the same
candidate pool (median over 10,000 draws)."
