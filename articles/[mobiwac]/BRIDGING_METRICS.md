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
