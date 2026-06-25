# Massive-STEPS corpora — statistics (T1-style)

**Study:** second_dataset · **Phase:** E (ETL) · **Built:** 2026-06-15
**Cities:** `istanbul` (PRIMARY, non-US) · `nyc` (secondary, US) · State token == city key.

T1-style stats paralleling the Gowalla per-state profile, so the Massive-STEPS
cities drop into the same comparison frame. Generated from the parsed corpora
(`data/checkins/<Token>.parquet`) + the graph artifacts.

## T1 — Massive-STEPS cities vs Gowalla canon states (raw corpus)

| Dataset | Source | Users | Check-ins | POIs | Regions | Categories |
|---|---|---:|---:|---:|---:|---:|
| Alabama | Gowalla | 3,858 | 113,846 | 11,848 | 1,109 (tracts) | 7 |
| Arizona | Gowalla | 7,869 | 236,450 | 20,666 | 1,547 (tracts) | 7 |
| **NYC** | **Massive-STEPS** | **6,929** | **272,368** | **45,804** | **1,912 (TIGER tracts)** | **7** |
| **Istanbul** | **Massive-STEPS** | **23,700** | **544,471** | **42,795** | **520 (mahalle, PRIMARY) / 2,585 (H3)** | **7** |
| Florida | Gowalla | 21,052 | 1,407,034 | 76,544 | 4,703 (tracts) | 7 |
| California | Gowalla | 37,090 | 3,171,380 | 169,145 | 8,501 (tracts) | 7 |
| Texas | Gowalla | 38,644 | 4,089,892 | 160,938 | 6,553 (tracts) | 7 |

**External-validity framing:** Gowalla = US; **NYC = US** (tests *source* diversity only);
**Istanbul = non-US** (tests *geographic/cultural* generalization — the stronger claim,
hence PRIMARY). Both Massive-STEPS cities are small-to-mid by region count → expect the
**H3-alt** recipe in Phase V.

## Input-window data comparison (within-user, window=9 — identical construction every row)

The headline "how much training data after building input windows" — within-user
non-overlapping window=9 sequences (set (a), the primary protocol). Gowalla rows from
the existing canonical `sequences_next.parquet`; NYC/Istanbul from this study.

| Dataset | Source | Raw check-ins | Substrate check-ins | POIs | Regions | **Input windows** | win/check-in |
|---|---|---:|---:|---:|---:|---:|---:|
| Alabama | Gowalla | 113,846 | 113,846 | 11,848 | 1,109 | **12,709** | 0.112 |
| Arizona | Gowalla | 236,450 | 236,450 | 20,666 | 1,547 | **26,396** | 0.112 |
| **NYC** | Massive-STEPS | 272,368 | 252,057 | 40,371 | 1,912 | **30,155** | 0.120 |
| **Istanbul** | Massive-STEPS | 544,471 | 462,615 (mahalle) / 479,229 (H3) | 29,816 | **520 (mahalle, primary)** / 2,585 (H3) | **58,075** (mahalle) / 60,091 (H3) | 0.126 |
| Florida | Gowalla | 1,407,034 | 1,407,034 | 76,544 | 4,703 | **159,175** | 0.113 |
| California | Gowalla | 3,171,380 | 3,171,380 | 169,145 | 8,501 | **358,302** | 0.113 |
| Texas | Gowalla | 4,089,892 | 4,089,892 | 160,938 | 6,553 | 460,976 | 0.113 |

- **Istanbul (60,091 windows) ≈ between Arizona and Florida; NYC (30,155) ≈ Arizona.**
  Both Massive-STEPS cities are small-to-mid → fit the H3-alt small-state recipe.
- win/check-in ≈ 0.11–0.12 everywhere (≈ 1 non-overlapping window per 9 check-ins) — sanity passes.
- Gowalla `substrate==raw` (those parquets are pre-filtered to mapped POIs); NYC/Istanbul keep
  null-coord rows in the corpus and drop them at the substrate. Istanbul window count differs
  slightly by region def (H3 60,091 vs mahalle 58,075) because each region polygon set drops a
  slightly different POI/check-in tail; both are internally consistent.

## OVERLAP board Tbl 1 (paper windowing — gated stride-1, MIN_SEQ=10, emit_tail=False)

The paper is on the **overlap** board (`check2hgi_dk_ovl`). Window counts below are the overlap windowing,
recomputed 2026-06-25 from the per-user check-in parquets and **validated** against the one on-disk dk_ovl
parquet: AL recompute = 96,326 = `output/check2hgi_dk_ovl/alabama/input/next.parquet` exactly; FL = 1,274,418 =
the value cited in `CLOSE_BLOCKERS_HANDOFF.md`. Gowalla raw == substrate; **Istanbul = mahalle substrate**
(post null-coord drop). Recompute script: `scripts/closing_data/tbl1_overlap_stats.py` (uses `generate_sequences`).

| Dataset | Source | Check-ins | Users | POIs | Regions | **Windows (overlap)** | Max seq | Avg seq | Sparsity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alabama | Gowalla | 113,846 | 3,858 | 11,848 | 1,109 | **96,326** | 3,835 | 29.5 | 0.9975 |
| Arizona | Gowalla | 236,450 | 7,869 | 20,666 | 1,547 | **200,895** | 5,589 | 30.1 | 0.9985 |
| Florida | Gowalla | 1,407,034 | 21,052 | 76,544 | 4,703 | **1,274,418** | 16,679 | 66.8 | 0.9991 |
| California | Gowalla | 3,171,380 | 37,090 | 169,145 | 8,501 | **2,925,466** | 14,855 | 85.5 | 0.9995 |
| Texas | Gowalla | 4,089,892 | 38,644 | 160,938 | 6,553 | **3,830,414** | 42,300 | 105.8 | 0.9993 |
| Istanbul | Massive-STEPS | 462,615 (mahalle) | 23,694 | 29,816 | 520 | **270,217** | 817 | 19.5 | 0.9993 |

- Sparsity = 1 − check-ins / (users × POIs). Avg/Max seq = per-user check-in count (verified on the parquet).
- Overlap windows ≈ check-ins minus the per-user tail (stride-1, emit_tail=False drops the OOB last-POI windows
  and users with < 10 check-ins) — e.g. FL 1,274,418 / 1,407,034 raw ≈ 0.906 win/check-in (vs 0.113 non-overlap).
- Istanbul row is the **mahalle substrate** (post null-coord drop; raw corpus = 544,471 ck / 23,700 users).

## NYC (Massive-STEPS-New-York; regions = NY TIGER 2022 tracts)

| Quantity | Raw corpus | After region join (substrate-eligible) |
|---|---:|---:|
| Check-ins | 272,368 | 252,057 (−20,311: 19,834 null-coord + 477 outside NY tracts) |
| POIs | 45,804 | 40,371 |
| Regions | — | **1,912** tracts (POIs/tract: median 6, mean 21, max 356) |
| Users / trails | 6,929 / 92,041 | — |
| Null-coord check-ins | 19,834 (7.3%) | dropped |

7-root dist (check-in-weighted): Food 27.8 · Community 15.7 · Shopping 13.7 · Outdoors 12.9 · Travel 12.2 · Nightlife 10.5 · Entertainment 7.2.
Shipped split (per-trail): train 190,740 / val 27,201 / test 54,427 check-ins.

## Istanbul (Massive-STEPS-Istanbul; TWO region defs — no clean gov TIGER-equivalent)

Region options (both built; **mahalle = PRIMARY** per the methodology advisor 2026-06-15 —
real-admin task identity matches every Gowalla state + NYC, and the gap-to-ceiling framing
already absorbs the granularity gap; H3 = granularity-matched robustness/sensitivity variant.
See `PHASE_E_REPORT.md` §Istanbul regions):
- **mahalle (520 populated of 972) — PRIMARY** — REAL administrative units (OSM `admin_level=8`,
  ODbL), the practical TIGER-equivalent; coarser (median 16 POIs/region). No clean *government*
  mahalle polygon exists (TÜİK/TKGM/HGM gate/fragment it). Path: `output/check2hgi/istanbul/` (top level).
- **H3 res-9 (2,585 cells) — SECONDARY** — synthetic grid; granularity-matched to NYC (1,912)
  and the Gowalla band; retires the "is the 520-way result an artifact of coarseness?" objection.
  Path: `output/check2hgi/istanbul/h3/`.

| Quantity | Raw corpus | mahalle (PRIMARY) | H3 (secondary) |
|---|---:|---:|---:|
| Check-ins | 544,471 | 462,615 | 479,229 |
| POIs | 42,795 | 29,816 | 29,807 |
| Regions (populated) | — | **520** (median 16, mean 57, max 1,367) | **2,585** (median 4, max 478) |
| set (a) input windows | — | 58,075 | 60,091 |
| Users / trails | 23,700 / 216,411 | — | — |
| Null-coord check-ins | 62,621 (11.5%) | dropped | dropped |

7-root dist (check-in-weighted): **Food 27.0 · Outdoors 26.8 · Community 21.2** · Travel 8.2 · Shopping 7.7 · Nightlife 4.8 · Entertainment 4.3.
> The category profile is markedly **different from NYC** (Outdoors 26.8% vs 12.9%; Community 21.2% vs 15.7%; Nightlife 4.8% vs 10.5%) — driven by Istanbul's many universities, neighborhoods/plazas, bridges, and cafés. This cultural divergence is exactly what makes Istanbul a stronger external-validity probe.

H3 region build: bbox (lat 40.70–41.50, lon 28.40–29.60) captures 98.4% of POIs-with-coords; out-of-bbox points (raw lon spans −73→40, i.e. mislocated to as far as NYC) drop naturally via the sjoin. Shipped split: train 380,986 / val 54,600 / test 108,885 check-ins.

## Built sequence sets (window=9, E4) — per city

| Protocol | Unit | Split | NYC seqs | Istanbul seqs |
|---|---|---|---:|---:|
| (a) Gowalla-parity [PRIMARY] | within-user | user-grouped 5-fold CV, seeds {0,1,7,100,42} | 30,155 | 60,091 |
| (b) native-trail [secondary] | within-trail | shipped split | 9,509 | 7,245 |

> Trails are tiny in both (NYC median 2, Istanbul median 2; <12% have ≥5 check-ins),
> so within-trail window=9 keeps few sequences and is power-session-biased — set (a)
> within-user is the primary protocol. A 3rd "native-shape" set (within-trail,
> window≈5, overlapping/per-step) is recommended for leaderboard-comparable
> robustness (see `PHASE_E_REPORT.md` §sequence-construction).
