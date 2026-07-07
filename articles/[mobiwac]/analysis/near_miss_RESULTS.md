# Geographic near-miss distance -- cross-state results

> Consolidated summary of `near_miss_distance.py` across states. For each
> validation visit whose top-1 predicted region is wrong, the haversine distance
> (km) between the predicted and true region centroid. In-distribution and
> out-of-distribution (OOD -- true region absent from the fold's training
> vocabulary) misses are reported SEPARATELY, never pooled. Bare percentiles
> only; no service-radius / coverage-threshold overlay (venue-bridge guardrail,
> MOBILITY_PLAN.md 0/3.1). This is a property of the model's own predictions on
> the validation folds, not a measured service result.

## Provenance

- **Recipe:** champion v17 (bs8192 + per-head cat-lr, `MTL_ONECYCLE_PER_HEAD_LR=1`),
  gated stride-1 overlap engine `check2hgi_dk_ovl`, fp32, seed 0 x 5 folds,
  `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (reg) + `next_gru` (cat),
  static_weight cw=0.75, geom_simple selector. Run on the A40, 2026-07-07.
- **Per-sample predictions:** the `MTL_DUMP_VAL_PREDS=1` dump
  (`fold{N}_reg_val_preds.parquet`), written at each reg diagnostic-best
  (`top10_acc_indist`) epoch -- the same event the reported region cell is
  selected on, so the near-miss is measured at the same checkpoint as the paper's
  Acc@10 number.
- **Region geometry:** `output/check2hgi/<state>/temp/{checkin_graph.pt, boroughs_area.csv}`
  (region_idx -> GEOID -> WKT-polygon centroid; GEOID zero-pad handled; 0 dropped
  centroids at every state).

## Summary (pooled over the 5 folds, per state)

| State | regions | reg Acc@10 (this run) | in-dist miss P50 (km) | P90 (km) | mean (km) | OOD miss P50 / P90 (km) |
|---|---:|---:|---:|---:|---:|---:|
| Alabama  | 1,109 | 69.8 | **8.13** | 38.47 | 20.38 | 22.48 / 156.14 |
| Arizona  | 1,547 | 59.7 | **8.05** | 30.71 | 17.35 | 17.74 / 128.87 |
| Istanbul |   520 | 75.4 | **3.16** | 14.91 |  5.75 | 11.58 /  28.59 |
| Florida  | 4,703 | (running) | -- | -- | -- | -- |

_reg Acc@10 = mean over 5 folds of `top10_acc_indist` at the reg diagnostic-best
epoch (matches the board cell within run-to-run variation)._

## Reading

When the model's single most-likely next region is wrong, the region it did
predict sits a median of about **3 km (Istanbul) to 8 km (Alabama / Arizona)**
from the true one, with 90% of in-distribution misses within ~15 km (Istanbul) to
~30-38 km (Arizona / Alabama). The tighter Istanbul figure is consistent with its
geography: 520 dense mahalle units in one metropolitan area versus 1,109-1,547
census tracts spread across a whole US state. OOD misses (visits whose true region
was never in the training vocabulary, so no in-vocabulary correct answer exists)
are larger and are reported separately, never mixed into the in-distribution
figure.

## Health (no collapsed fold cited)

Per-fold reg diagnostic-best epochs land late at every state (no early-epoch
precision collapse): Alabama 27/31/37/43/35, Arizona 36/38/33/25/30, Istanbul
42/33/36/35/39. The per-fold `top10_acc_indist` tracks the board region cell at
each state.

## Status

Alabama, Arizona, Istanbul complete (2026-07-07). Florida is running (the
stride-1 overlap large state, ~1.3M rows; runs alone for host-RAM + throughput)
and will be appended here when its 5 folds land.
