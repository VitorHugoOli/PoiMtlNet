# Bridging metrics — interpretability anchors for the headline cells (P4)

> **Why.** The paper's headline region metric is **Acc@10** (over thousands of tract-regions) and category
> is **macro-F1** (7 classes); neither has a published reference scale. This supplementary gives the
> **metrics ladder** (Acc@1/@5/@10/MRR for region; Acc@1 for category) and the **floors** so a reader can
> calibrate "is 65.66 Acc@10 good?". The key floors are already stated inline in §6.2 (the metric-calibration
> clause); this is the fuller record + the cells that still need a cheap re-score. Numbers in **percent**,
> mean±std over 5 folds (seed 0) unless noted. Every cell cites its source JSON.

## Floors (the calibration anchors)

| State | regions | region **random** Acc@10 | region **Markov-1** Acc@10 | category **majority** Acc@1 | category **majority** macro-F1 |
|---|---:|---:|---:|---:|---:|
| AL | 1109 | 0.90 | 47.01 | 34.19 | 7.28 |
| AZ | 1547 | 0.65 | 42.96 | 34.01 | 7.25 |
| FL | 4703 | 0.21 | 65.05 | 24.69 | 5.66 |
| CA | 8501 | 0.12 | 52.09 | 32.72 | 7.04 |
| TX | 6553 | 0.15 | 54.94 | 30.98 | 6.76 |
| Istanbul | 520 | 1.92 | 52.5 | 33.31 | 7.14 |

Random top-10 = 10/n_regions (`docs/results/P0/simple_baselines/<state>/next_region.json`, `random/acc10_mean`).
The headline reg Acc@10 (~60–77) should be read against the **Markov-1 floor**, not the ~1 % random floor.
Majority floors from `docs/baselines/next_category/results/<state>.json` (`floors/majority_class`) +
`docs/results/P0/simple_baselines/istanbul/next_category.json`. Majority **macro-F1** (~5.7–7.3) is the right
anchor for the headline macro-F1; "about 7 %" is the value cited in §6.2.

## Region metrics ladder — directly available in committed JSONs

### STL region ceiling — `check2hgi_dk_ovl` + `next_stan_flow`, seed 0 × 5f (leak-free per-fold prior)
Source: `docs/results/P1/region_head_<state>_region_5f_50ep_<state>_ovl_stl_reg_s0.json` (Istanbul: `..._istanbul_stride1_stl_reg_s0.json`)

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 31.03 ± 2.56 | 58.70 ± 3.58 | 69.99 ± 3.56 | 44.00 ± 2.86 |
| AZ | 27.30 ± 0.78 | 49.37 ± 1.71 | 59.40 ± 2.15 | 37.97 ± 1.04 |
| FL | 49.61 ± 0.97 | 69.64 ± 1.20 | 76.71 ± 1.09 | 58.96 ± 1.01 |
| CA | 33.92 ± 0.49 | 55.13 ± 0.36 | 63.48 ± 0.31 | 43.98 ± 0.40 |
| TX | 31.03 ± 0.55 | 55.39 ± 0.63 | 64.96 ± 0.52 | 42.54 ± 0.57 |
| Istanbul | 34.42 | 64.49 | 74.80 | 48.23 |

### Faithful STAN (region external) — converged, audited v5/v6, seed 0 × 5f
Source: `docs/results/baselines/faithful_stan_{alabama,arizona}_5f_200ep_v5_compiled.json`,
`faithful_stan_istanbul_5f_200ep_v5_bf16c.json`, `faithful_stan_florida_5f_200ep_v6_opt.json`

| State | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| AL | 21.44 ± 2.72 | 47.92 ± 5.44 | 60.72 ± 5.20 | 34.05 ± 3.61 |
| AZ | 18.71 ± 5.32 | 39.45 ± 10.18 | 49.86 ± 11.53 | 28.87 ± 7.23 |
| FL | 42.17 ± 0.24 | 65.00 ± 0.21 | 72.99 ± 0.34 | 53.22 ± 0.23 |
| Istanbul | 25.30 ± 0.67 | 50.83 ± 0.60 | 61.86 ± 0.61 | 38.05 ± 0.69 |

(CA/TX faithful-STAN footnoted infeasible at scale. Old v4 numbers superseded — do not cite.)

## Cells that need a cheap re-score (saved logits, NOT a re-train)

Three re-scores would complete the ladder; the saved logits/rundirs exist but the k>10 metrics were not
serialized (and the HMT-GRN raw per-fold JSONs are gitignored / not in this checkout):

1. **Our MTL champion-G reg** — Acc@1/@5/MRR (currently only `reg_full_top10`). Rundirs in each
   `docs/results/closing_data/{h100,a40}/<state>_s0_mtl_*_score.json` `rundir` field.
2. **HMT-GRN** — Acc@1/@5/MRR (only Acc@10 in `docs/baselines/next_region/hmt_grn.md`).
3. **Category Acc@1** for MTL cat + STL cat ceiling (only macro-F1 serialized).

> Status: the **floors** and the **STL-reg + faithful-STAN ladders** are paper-ready (above). The three
> re-score items are deferred (need the gitignored logits); they are nice-to-have anchors, not blockers —
> the §6.2 calibration clause already gives the reader the random/Markov/majority reference scales.
> A **fourth deferred metric (geographic near-miss)** is registered below; unlike items 1–3 it is NOT
> computable from any saved artifact and must ride the P1 re-runs.

## Geographic near-miss — deferred item 4 (registered 2026-07-06, NOT yet computed)

**Definition.** For each test visit whose **top predicted region is wrong**: the distance in kilometers
between the centroid of the predicted census tract (mahalle for Istanbul) and the centroid of the true
one; report per-state **P50/P90** and the full distribution, per fold (seed 0 at minimum). Visits whose
true region is absent from training (the OOD share the Acc@10 discount removes) are reported
**separately** — they have no in-vocabulary correct answer.

**Why.** The prepared answer to "Acc@10 over thousands of regions says nothing about how bad the misses
are": it interprets region error at tract granularity for the mobility motivation **without measuring any
service**. Venue-bridge guardrails (settled ruling, motivation-only): bare percentiles or a bare CDF
only — **no service-radius/threshold overlay** (that would be the banned coverage curve in disguise);
keep the measurement verb on the prediction, service value as explicitly untested interpretation.

**Source plan (why it can't be computed today).** Per-sample predictions are never serialized (the eval
path computes top-k per sample and discards it) and no checkpoints are written — there is nothing to
re-score from. Plan: an env-gated dump flag (e.g. `MTL_DUMP_VAL_PREDS=1`) in `mtl_eval.py`/`mtl_cv.py`
writes `<rundir>/metrics/fold{N}_reg_val_preds.parquet` (true + top-10 predicted region idx, int32+zstd,
~10–20 MB/fold at TX/CA) at each **reg diagnostic-best** improvement (⚠ align the trigger to
`top10_acc_indist`, the matched scorer's selection metric), riding the **P1 n=20 top-up** on the H100 at
<1 % overhead. Geometry is already local: `output/check2hgi/<state>/temp/boroughs_area.csv` (GEOID + WKT
polygons; ⚠ GEOID int64 drops the leading zero — zero-pad to 11 chars) + `region_to_idx` in
`temp/checkin_graph.pt`. Offline join + haversine is CPU-minutes. Full pipeline, placement verdict, and
ready-to-paste (placeholder-gated) prose: [`MOBILITY_PLAN.md §3`](MOBILITY_PLAN.md).

**Status (updated 2026-07-08). COMPUTED — decoupled from P1.** Ran standalone on the A40 (PR #59,
2026-07-07: AL/AZ/FL/Istanbul, v17 `dk_ovl` seed-0 5f with the dump flag; the old "A40 infeasible at
FL/CA/TX" was wrong for FL, which completed run-alone in fp32; CA/TX remain unmeasured, author-triggered
only). Median in-distribution miss 3.16–8.13 km vs. a 20–241 km random-pair floor
(`analysis/near_miss_floor.py`, 2026-07-08) — misses ~6–34× closer than chance. Record:
[`analysis/near_miss_RESULTS.md`](analysis/near_miss_RESULTS.md). Paper placement graduated from "at most
one sentence" to the certified §7 usage-sketch rewrite (C4 in
[`MOBILITY_SCIENCE_BRIDGE_PLAN.md §12.3/§13`](MOBILITY_SCIENCE_BRIDGE_PLAN.md); tracking
`CLOSER_HANDOFF.md §P7`) — still bare percentiles, still no fig4 panel, still no service-radius overlay.
