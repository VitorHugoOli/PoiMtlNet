# second_dataset — Phase E (ETL) report + hand-off

**Status:** Phase E COMPLETE (2026-06-15) · **Machine:** Mac M2 Pro (CPU ETL) ·
**Branch:** `study/second-dataset` (worktree; NOT merged to main) ·
**Corpora:** `istanbul` (PRIMARY, non-US; regions = mahalle primary, H3 secondary) · `nyc` (secondary, US, TIGER tracts).
Dataset family: Massive-STEPS (Apache-2.0). Reviewed against source + paper (arXiv:2505.11239).

> Phase V (champion G + STL ceilings + Markov-1 floor, 4 seeds) remains BLOCKED on a
> CUDA box + the FROZEN champion/substrate. Below is the leak-free, embedding-independent
> ETL that lets Phase V plug in the frozen substrate and run.

---

## ⚠ Findings (verified against the Massive-STEPS source + paper)

### F1 — The shipped split is a USER-STRATIFIED RANDOM split over trails, **not temporal**
`future_work.md §8` motivated this corpus partly as a free **temporal-split protocol
bridge**. **That does not hold.** Source: `create_next_poi_dataset.py` does
`train_test_split(..., stratify=user_id, random_state=42)` at a 7:1:2 *trail* ratio.
So: trails are the split unit; users **deliberately** appear in all three splits (the
paper: "we ensure all test users appear in train/val"); timestamps overlap (only 14.3%
of train∩test users have test-after-train). Conclusion (non-temporal) is correct — a
temporal bridge needs a **Gowalla-side chronological re-split** (roadmap A5), not this.

### F2 — The shipped split is incompatible with the repo's window=9 task
Trails are 8-hour sessions: median 2 check-ins (NYC + Istanbul), <12% have ≥5. The
repo's window=9 next-POI task is inherently **within-user**; within-*trail* windowing
keeps <15% of trails (power-session-biased). Within-user windows cross trails → cross
the shipped split → leakage. The two cannot both be honored.

### F3 — Massive-STEPS' native sequence construction is WITHIN-TRAIL; we deliberately diverge
Every bundled baseline (GETNext, Flashback, …) windows **strictly within a trail**
(GETNext: `for traj_id ...`; even the POI graph blocks cross-trail edges). Massive-STEPS
also has **only a next-POI task — no category and no region task** (both are OUR additions,
ported from the Gowalla pipeline). So our within-user set (a) is a *deliberate, documented*
Gowalla-parity divergence, NOT a Massive-STEPS-leaderboard-comparable result.

---

## Sequence construction — decision + recommendation (methodology review, 2026-06-15)

**User decision: build BOTH** protocols; an independent advisor reviewed which to lead with.

- **LEAD with set (a): within-user window=9 + user-grouped 5-fold CV.** It is the only
  *controlled* comparison — same task/window/fold protocol as Gowalla, only the corpus
  changes — which is exactly what an external-validity claim requires. The cross-trail
  "noisy target" lowers the ceiling *equally on both corpora* (fair, not leakage); the
  3600s substrate temporal-decay means embeddings carry little spurious long-range signal;
  and user-grouped CV (users disjoint across folds) is a *stronger* generalization test
  than the shipped split (which keeps users in all splits by design).
- **DEMOTE set (b): within-trail window=9** to a footnote — ~85–89% trail-discard →
  power-session selection bias + padding-dominated. Not headline evidence.
- **RECOMMENDED 3rd set (not yet built): within-trail, window≈5, overlapping/per-step,
  shipped split** — matches Massive-STEPS' native windowing, leaderboard-comparable,
  triangulates the claim. ~1 day CPU; build via a `build_inputs_native.py` variant.

**Phase-V non-negotiables (from the review):**
1. **Freeze the Istanbul/NYC substrate to the bit-identical Gowalla recipe.** Transductivity
   is the one live leak risk; it must be *matched* across corpora so it cancels in the comparison.
2. **Report Markov-1 floor + per-task STL ceilings in the SAME table** and compare
   **gap-to-ceiling / lift-over-floor**, NOT absolute Acc@k (region counts + base rates
   differ across corpora — NYC 1,912, Istanbul 2,585 regions).
3. **Drop temporal-bridge language** from any Phase-V / paper claim (F1).
4. Re-verify per-fold `log_T` freshness after embeddings regenerate (CLAUDE.md stale-log_T rule).

---

## Artifacts (per city; under the `<city>` state token — NO Gowalla path touched)

`data/`, `output/` are gitignored (worktree symlinks them to the main repo, so artifacts
materialize canonically). Verified: no existing Gowalla check-in / `output/check2hgi/<state>/`
artifact was modified; only `Nyc.parquet`, `Istanbul.parquet`, and `output/check2hgi/{nyc,istanbul}/` are new.

| Artifact | Path (`<city>` ∈ {nyc, istanbul}) |
|---|---|
| Parsed corpus | `data/checkins/<Token>.parquet` (Nyc / Istanbul) |
| Category map | `docs/studies/second_dataset/category_map.md` + `data/massive_steps_<city>/category_map.{csv,json}` |
| Graph artifact | `output/check2hgi/<city>/temp/checkin_graph.pt` (+ `boroughs_area.csv`) |
| (a) sequences / labels / folds | `output/check2hgi/<city>/temp/sequences_next.parquet` · `input/next_region_labels.parquet` · `folds/fold_spec_userGrouped.json` |
| (a) priors | `output/check2hgi/<city>/region_transition_log_seed{S}_fold{N}.pt` (25 files: seeds {0,1,7,100,42} × 5 folds) |
| (b) sequences / prior | `output/check2hgi/<city>/shipped_split/sequences_next_trail.parquet` · `region_transition_log_shipped_train.pt` |
| Reports | `data/massive_steps_<city>/{parse,graph,inputs}_report.json` |
| ETL scripts (city-generic) | `scripts/second_dataset/{cities,acquire,build_category_map,parse_city,build_h3_boroughs,build_graph,build_inputs,build_region_variant}.py` |
| Istanbul mahalle (PRIMARY) | top-level `output/check2hgi/istanbul/` + `data/miscellaneous/istanbul_mahalle/istanbul_mahalle.geojson` |
| Istanbul H3 (secondary) | `output/check2hgi/istanbul/h3/{temp,input,shipped_split,region_transition_log_*}` |
| Stats | `docs/studies/second_dataset/STATS_T1.md` |

### Regions (per city) — Istanbul: mahalle PRIMARY, H3 secondary
- **NYC** = US Census TIGER 2022 tracts (point-in-polygon), 1,912 populated.
- **Istanbul — TWO region defs built** (user: "build both"; methodology advisor 2026-06-15 set the primary):
  - **mahalle (520 populated of 972) — PRIMARY**, at `output/check2hgi/istanbul/` (top level) —
    REAL admin units (OSM `admin_level=8`, ODbL, EPSG:4326; `data/miscellaneous/istanbul_mahalle/`),
    the practical TIGER-equivalent. `region_mode: admin` in `cities.py`.
  - **H3 res-9 (2,585 cells) — SECONDARY/robustness**, at `output/check2hgi/istanbul/h3/` —
    synthetic grid; granularity-matched to NYC/Gowalla. Reproduce: `build_region_variant.py
    --city istanbul --variant h3 --source h3`.

**Why mahalle is primary (advisor verdict):** external validity is a claim about the *same task*
transferring to a new corpus. Every Gowalla state + NYC defines next-region over REAL geography
(census tracts), so mahalle preserves the **task identity** (predict a real admin region) while
varying only source+geography — the actual experiment. H3 would silently change the *task kind*
(synthetic grid) on the one city where real geography is hard. The agreed **gap-to-ceiling /
lift-over-floor** framing already absorbs the only thing mahalle changes (granularity: Gowalla
already spans 1,109→8,501 regions); it cannot absorb a real-vs-synthetic task change. Substrate:
the region encoder (POI→region aggregation, real adjacency, area-weighted city pooling) is
*semantically* matched to real polygons; H3 gives it synthetic adjacency + uniform area. Risk
(520-way coarseness compressing floor/champion/ceiling) is retired by reporting H3 as a same-table
sensitivity check in Phase V.

**TIGER-for-Istanbul search (2026-06-15):** no clean *government* mahalle polygon exists for all
Istanbul — TÜİK/TKGM/HGM gate or fragment it; İBB's mahalle-polygon request was closed unresolved;
the official ULASAV/CBS portal only has per-district fragments. geoBoundaries reaches only ilçe
(39 districts, too coarse). Best real-admin layer = OSM mahalle (~972; ~520 populated).

## Reproduce (Mac, ~1 min/city, CPU)

```bash
CITY=istanbul   # or nyc
python scripts/second_dataset/acquire.py            --city $CITY  # HF parquets + FSQ v1 tree (+ NY TIGER for nyc)
python scripts/second_dataset/build_category_map.py --city $CITY  # FSQ v1 → 7-root (self-builds fine inventory)
python scripts/second_dataset/parse_city.py         --city $CITY  # → data/checkins/<Token>.parquet
python scripts/second_dataset/build_graph.py        --city $CITY  # → checkin_graph.pt (tiger or H3 regions)
python scripts/second_dataset/build_inputs.py       --city $CITY  # → sequences/labels/folds/priors (both protocols)
# Istanbul: build_graph/build_inputs above produce the PRIMARY mahalle build (region_mode=admin).
# Add the SECONDARY granularity-matched H3 variant -> output/check2hgi/istanbul/h3/ :
python scripts/second_dataset/build_region_variant.py --city istanbul --variant h3 --source h3
```
City registry (HF repo, region mode, admin geojson, H3 res/bbox) lives in `scripts/second_dataset/cities.py`.

## Correctness guarantees

- **Bit-parity with Phase-V folds:** set (a) sequences use the repo's own
  `core.convert_user_checkins_to_sequences`; per-fold priors use
  `StratifiedGroupKFold(5, shuffle=True, random_state=seed)` on `(rows, next_category, userid)`
  — identical to the trainer's split (depends only on row count, labels, userids, seed).
  Geography (`poi_to_region`) is deterministic from the shapefile/H3 + corpus, so a graph
  rebuild preserves it.
- **Leak-free, verified (both cities):** set (a) folds are user-disjoint (train∩val users = ∅);
  priors are built from train userids only; row-normalized log-probs; seed/fold-tagged.
  set (b) sequences never cross a trail (max 1 split/trail); its prior uses train-split trails only.
- **Labels validated:** region_idx ∈ [0, n_regions), no NaN categories, all 7 roots; no
  duplicate (userid, placeid, datetime) keys.

## Corrections folded in vs the first NYC pass

- **Timestamps are LOCAL** (upstream `preprocess_std.py` strips the tz offset; UTC
  unrecoverable). `datetime` = `local_datetime` = local wall-clock. (My earlier "UTC"
  label + double-shifted `local_datetime` are fixed. The `datetime` *value* never changed,
  so graphs/priors built earlier remain valid.) Minor divergence from Gowalla (datetime=UTC);
  features stay semantically sound (local hour-of-day).
- Split reframed as **"user-stratified random over trails (seed 42, 7:1:2)"** — users in
  all splits is by design, not a quirk.
- `docs/context/DATASETS.md` assumed FSQ v2 + a bundled `categories.csv`; actual data is
  FSQ **v1** ids with **no** taxonomy file → map built from a verified FSQ v1 tree (100%
  leaf-id coverage for both cities). DATASETS.md also listed FSQ-NYC TIST2015 as priority 1;
  superseded by the Massive-STEPS choice per `AGENT_PROMPT.md`.

## What Phase V still needs (NOT done here — CUDA + frozen recipe)

1. Generate each city's **check2hgi substrate embeddings** with the FROZEN encoder/recipe
   (bit-identical to Gowalla — see non-negotiable #1), then the embedding-bearing
   `next.parquet` / `category.parquet` / `next_region.parquet` via the standard pipeline;
   the folds/priors here line up bit-for-bit.
2. Run **champion G + per-task STL ceilings + Markov-1 floor**, 4 seeds, under set (a)
   (primary). Optionally the 3rd native-shape set for a leaderboard cell.
3. Expect the **H3-alt** recipe (both cities small-to-mid by region count).
4. The Acc@k **bridge metric** on native FSQ labels is available (`fsq_category` preserved).
