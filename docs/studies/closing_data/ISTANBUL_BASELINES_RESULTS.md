# ISTANBUL BASELINES — results + run log (2026-06-25, M4 Pro / MPS, fp32)

> ⚠ **STAN-VARIANT NOTE SUPERSEDED (2026-06-26).** The "Variant alignment" directive below (STAN = `stl_hgi`,
> faithful STAN removed) was **reversed** by the region-refooting decision + PR #53. The paper's headline STAN is now
> **faithful-from-raw** (audited v5, converged): Istanbul Acc@10 **61.86** (`STAN_REFOOTING_HANDOFF.md`,
> `FAITHFUL_STAN_FINDINGS.md`). **STAN-`stl_hgi`** (Istanbul 71.13, below) is now a **future-headroom signal, NOT a
> paper baseline** (user steer). The Istanbul **Markov / POI-RGNN category** numbers below are unaffected (they are
> the canonical Table-3 cat baselines: Markov-K best 24.55 / POI-RGNN 30.12). **ReHDM** stays faithful, deferred to CUDA.

> Execution record for [`ISTANBUL_BASELINES_HANDOFF.md`](ISTANBUL_BASELINES_HANDOFF.md). Completes the
> §6.3 external-validity box so Istanbul carries the same baseline columns as the Gowalla states.
> Protocol: **seed 0 × 5 folds (n=5)**, leak-free per-fold train-only priors, user-disjoint folds, MPS = fp32.
> **n=5 provisional** — report **gap-to-ceiling / lift-over-floor**, not absolute Acc@k.
>
> **Variant alignment (user directive 2026-06-25):** **STAN = `stl_hgi`** everywhere (incl. Istanbul — run from
> the HGI substrate); **ReHDM = faithful** everywhere. The earlier `stl_check2hgi` STAN and the from-scratch
> "faithful STAN" were the wrong variant and have been removed (see §3).

## 1 · Results (this run)

| Cell | Variant | Metric | Value | Source JSON |
|---|---|---|---|---|
| Markov majority (cat) | — | macro-F1 | **7.14** | `P0/simple_baselines/istanbul/next_category.json` |
| **Markov-1-POI** (cat floor, P(cat\|last POI)) | — | macro-F1 | **17.55 ± 0.44** | `…/next_category.json` (`markov_1step`) |
| **Markov-K-cat** (apples-to-apples POI-RGNN floor) | — | macro-F1 | **24.55 ± 0.30** (best, k5) | `…/next_category_markov_kstep.json` |
| **POI-RGNN** (cat native) | faithful | macro-F1 | **30.12 ± 0.84** | `baselines/faithful_poi_rgnn_istanbul_5f_35ep_*.json` |
| **STAN** (region) | **stl_hgi** | Acc@10 | **71.13 ± 0.68** | `P1/region_head_istanbul_..._STAN_HGI_istanbul_5f50ep.json` |
| **ReHDM** (region) | **faithful** | Acc@10 | ⏸ **deferred to CUDA** (user) | (ETL+adapter verified; training pending) |

Markov-K-cat full series (macro-F1): k1=11.45 · k3=24.14 · **k5=24.55** · k7=23.14 · k9=22.55.
POI-RGNN aux: Acc=41.64 · Acc@5=93.67 · MRR=62.62. STAN stl_hgi aux: Acc@1=32.29 · Acc@5=60.38 · MRR=45.37 · cat-F1(diag)=12.23.

### Reference cells (already done before this run — for gap/lift framing)
Champion-G MTL (set-a, 4 seeds) cat 60.16 / reg 69.79 · STL cat ceiling (`next_gru`) 52.10 · STL reg ceiling
(`next_stan_flow`) 70.37 · CTLE-SC 25.92 ± 0.31 · Check2HGI-SC ceiling 54.53 ± 0.69 · Markov-1 region floor 52.5 ·
HMT-GRN region SOTA **60.4** (stride-1, see §2).

**Sanity:** STAN stl_hgi Acc@10 = 71.13 sits on the STL reg ceiling (70.37) — i.e. NOT inflated by a stale log_T
(a stale prior would push it ~+8–12 pp; see §4.3). POI-RGNN (30.12) > Markov-K-cat (24.55) and < STL cat ceiling
(52.10) — the expected ordering.

## 2 · HMT-GRN 60.4-vs-56.56 conflict — RESOLVED
The handoff flagged these as "same run, two recorded numbers." They are **two distinct builds**, both correctly recorded:

| Build dir | Windowing | n_train (fold 0) | reg Acc@10 |
|---|---|---|---|
| `results/baseline_b3_hmt_grn_style/istanbul/` | stride-9 (non-overlap) | 46,638 | **56.56** |
| `results/baseline_b3_hmt_grn_style/istanbul_stride1/` | stride-1 (overlap) | 217,333 (≈4.7×) | **60.42 ≈ 60.4** |

(Both `b3_seed0_folds5.json`; the JSON `windowing` field is a stale generic label in both — trust dir name + n_train.)
`RESULTS_BOARD.md §4`'s 60.4 is correct; cite stride-1 when tabulating.

## 3 · STAN substrate = HGI (new substrate built for Istanbul)
Istanbul previously had **no HGI substrate** (handoff: "check2hgi is the only one"). To run STAN=stl_hgi we **built
one** (2026-06-25):
- Pre-placed the **mahalle** boroughs CSV at `output/hgi/istanbul/temp/boroughs_area.csv` (copied from
  `output/check2hgi/istanbul/temp/` — same 520/964-region taxonomy), so HGI's `shapefile=None` consumes mahalle,
  not a synthetic grid.
- Ran the 5-stage HGI pipeline (`hgi.pipe.py::process_state('istanbul', {shapefile:None, cross_region_weight:0.7})`),
  CPU-pinned, ~13 min. Output: POIs 29,945 / Regions 520, `region_embeddings.parquet (520, 64)`.
- **Alignment verified:** HGI `region_id` == check2hgi `region_idx` for **29,945/29,945** POIs (consistent 520-region
  bijection) — so STAN's HGI region-embedding lookup row-aligns with check2hgi's region targets + per-fold log_T.
- Reproducibility entry added (commented) to `hgi.pipe.py::STATES`. The substrate lives under gitignored `output/`.

**Superseded / removed (wrong variant):** STAN `stl_check2hgi` (Acc@10 70.39) and from-scratch "faithful STAN"
(Acc@10 57.60 ± 0.71). The faithful-STAN result JSON was removed, but its enabling **mahalle adapter** in
`stan/etl.py` (+ the MPS-generator fix in `stan/train.py`) is kept — both are correct, additive, and let faithful
STAN run on Istanbul if ever needed.

## 4 · Protocol corrections + lessons (handoff defects caught)
1. **`--task next_poi` is invalid** (§2.A). `compute_simple_baselines.py` only accepts `next_category`/`next_region`.
   The category floors come from **`--task next_category`** (→ `majority`+`markov_1step` macro-F1). Output is
   `next_category.json`, not `next_poi.json`.
2. **STAN defaults to seed 42** (the p1 ablation + both faithful `train.py` omit-default), but the protocol is
   **seed 0**. Always pass `--seed 0`.
3. **Stale per-fold log_T** (CLAUDE.md trap). On-disk seed-0 log_T (Jun 24 05:47) predated `next_region.parquet`
   (Jun 24 07:20) → would silently inflate reg Acc@10 by +8–12 pp. Rebuilt before STAN:
   `compute_region_transition.py --state istanbul --engine check2hgi --per-fold --seed 0 --n-splits 5`.
   The `assert_log_t_fresh` guard correctly hard-blocked the stale run.
4. **RAM guard over-trips on the tiny Istanbul dataset.** `load_next_data._guard_cpu_resident_ram` demands 16 GB
   head-room; Istanbul's next dataset needs ~0.3 GB. Set **`MTL_RAM_HEADROOM_GB=4`** for every run.
5. **macOS `stat`**: §5's freshness check uses Linux `stat -c`; on darwin use `stat -f '%m %N'`.
6. **HGI build on FSQ/Massive-STEPS** needed a one-line fix: `preprocess.py` now drops POIs with **null geometry**
   (FSQ check-ins carry missing coords; absent from US Gowalla) — mirrors POI-RGNN's NaN-coord drop, no-op for US.
7. **HGI build under a custom runner** must use the `if __name__ == '__main__'` + `freeze_support()` idiom — HGI
   stages spawn workers (macOS spawn start-method) that re-import the launcher.

## 5 · Faithful baselines feasibility — handoff "blocked" claim was OVERSTATED
- **ReHDM faithful runs on Istanbul as-is** — its ETL is data-driven (`_assign_regions` sjoins POIs against the
  mahalle `boroughs_area.csv`; quadkey region is pure lat/lon). ETL completed in **~2 s** (290 regions, 113,903
  trajectories). On MPS, training is ~96 s/epoch → 5 seeds × 50 ep ≈ multi-hour, so **deferred to CUDA** (user).
- **STAN faithful** needed only the `_shapefile_path` KeyError fixed (mahalle geojson `@id`→GEOID) — a small adapter,
  not the "~30 h region-assignment rebuild" the handoff feared. (But the chosen STAN variant is `stl_hgi`, not faithful.)

## 6 · Open items
- **ReHDM faithful** (5 seeds, seed 0) — run on CUDA: `python -m research.baselines.rehdm.train --state istanbul
  --folds 5 --epochs 50 --seed 0 --tag REHDM_istanbul_5seeds_50ep` (ETL already built; add Acc@10 to §1 + the board).
- **Tabulation** into `docs/baselines/next_{category,region}/` + the Istanbul column of `RESULTS_BOARD.md §1`
  still pending — apply the n=5-provisional / gap-to-ceiling rule, the resolved HMT-GRN value (60.4), and the
  STAN=stl_hgi / ReHDM=faithful variant alignment.
