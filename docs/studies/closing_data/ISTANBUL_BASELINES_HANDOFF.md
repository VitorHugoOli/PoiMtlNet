# ISTANBUL BASELINES — handoff (M4 Pro, MPS) · self-contained · 2026-06-25

> ✅ **EXECUTED 2026-06-25 — see [`ISTANBUL_BASELINES_RESULTS.md`](ISTANBUL_BASELINES_RESULTS.md)** for numbers,
> the resolved HMT-GRN conflict (60.4 = stride-1 / 56.56 = non-overlap), and the handoff defects caught.
> **Variant alignment (user):** STAN = **stl_hgi** (built a new HGI substrate for Istanbul — see RESULTS §3);
> ReHDM = **faithful** (ETL+adapter verified, training **deferred to CUDA**). Done: Markov cat floors · POI-RGNN
> (30.12) · **STAN stl_hgi (Acc@10 71.13 ± 0.68)**. The `stl_check2hgi` (70.39) + from-scratch "faithful STAN"
> (57.60) were the wrong variant and were removed. §2.E not run.
> **Defects to note before re-running:** §2.A `--task next_poi` is invalid → use `--task next_category`;
> §2.C/STAN omits `--seed` (defaults to 42) → pass `--seed 0`; rebuild per-fold log_T first (it was stale);
> set `MTL_RAM_HEADROOM_GB=4` (the RAM guard over-trips on Istanbul's tiny dataset). Istanbul has **no HGI
> substrate by default** — build it (RESULTS §3) for STAN=stl_hgi.

> **Goal.** Complete the **Istanbul §6.3 external-validity box** so it carries the *same baseline set* as the
> Gowalla states. Istanbul is non-US (Massive-STEPS Foursquare, **mahalle** regions = 520 primary admin), so a
> few US-tied baselines need adaptation — see the feasibility table. Protocol mirrors the board:
> **seed 0 × 5 folds (n=5)**, **gated stride-1 overlap**, leak-free per-fold train-only priors, user-disjoint
> folds, **MPS = fp32** (no fp16 confound). Decisions/roles inherited from `RESULTS_BOARD.md §4` +
> `BASELINE_M4.md`. Report **gap-to-ceiling / lift-over-floor** (the F2 external-validity rule), NOT absolute Acc@k.
>
> ⚠ **RAM (M4 24 GB):** keep ≤2–3 concurrent under `scripts/closing_data/ram_watchdog.sh` (4+ → OOM-reboot zone).
> Wrap long runs in `caffeinate -i env ...` (SIGBUS-on-sleep, G4). Do NOT run a CA/TX kNN here (OOM-rebooted the box once).

## 0 · Setup (once)
```bash
cd "/Volumes/Vitor's SSD/ingred"; git checkout main && git pull   # (currently on study/pre-freeze-gates — branch as needed)
export PYTHONPATH=src
PY=.venv/bin/python
```

## 1 · Istanbul box — what is DONE vs NEEDED
The reduced board's Istanbul cell is **already complete** (champion-G + the W3 cat gate). The cells below marked
**NEEDED** are *full-table completeness* (region SOTA + cat-native baseline + Markov cat floor) — they make the
Istanbul row carry the same baseline columns as AL/AZ/FL/CA/TX, not reduced-board-critical.

| Cell | Metric | Value | Status | Source |
|---|---|---|---|---|
| **Champion-G MTL** (set-a, GCN, 4 seeds) | cat macro-F1 / reg Acc@10 | **60.16 / 69.79** | ✅ done | `PHASE_V_ISTANBUL_S0.md` |
| **STL cat ceiling** `next_gru` (set-a) | macro-F1 | **52.10** | ✅ done | `PHASE_V_ISTANBUL_S0.md` |
| **STL reg ceiling** `next_stan_flow` (set-a) | Acc@10 | **70.37** | ✅ done | `PHASE_V_ISTANBUL_S0.md` |
| **CTLE-SC** (stride-1, W3 cat gate) | macro-F1 | **25.92 ± 0.31** | ✅ done | `baseline_compare/istanbul_ctle.json` |
| **Check2HGI-SC** ceiling (stride-1) | macro-F1 | **54.53 ± 0.69** | ✅ done | `baseline_compare/istanbul_check2hgi_sc.json` |
| **Markov-1 region** floor | Acc@10 | **52.5** | ✅ done | `P0/simple_baselines/istanbul/next_region.json` |
| **HMT-GRN** (region native-E2E) | Acc@10 | **60.4** ⚠ / 56.56 | ✅ done (value ⚠ see note) | `RESULTS_BOARD §4` vs `next_region/comparison.md` |
| **Markov-1-POI / Markov-9-cat** floors | macro-F1 | **17.55 / 22.55** (best k5 **24.55**) | ✅ done | `P0/simple_baselines/istanbul/next_category*.json` |
| **POI-RGNN** faithful (cat native) | macro-F1 | **30.12 ± 0.84** | ✅ done | `baselines/faithful_poi_rgnn_istanbul_5f_35ep_*.json` |
| **STAN** (region, **stl_hgi** seed 0) | Acc@10 | **71.13 ± 0.68** | ✅ done | `P1/region_head_istanbul_..._STAN_HGI_*.json` |
| **ReHDM** (region, **faithful**) | Acc@10 | deferred → CUDA | ⏸ user | (ETL+adapter verified; training pending CUDA) |
| Champion-G @ **stride-1** (windowing unify) | cat / reg | — | ⏳ LOW (BASELINE_M4 §2b) | run §2.E |

> ✅ **HMT-GRN conflict RESOLVED (2026-06-25):** NOT the same run — two distinct builds.
> `istanbul/` = stride-9 non-overlap (n_train 46,638) → **56.56**; `istanbul_stride1/` = stride-1 overlap
> (n_train 217,333) → **60.42 ≈ 60.4**. Board protocol = gated stride-1, so the **board-correct value is 60.4**;
> 56.56 is the older non-overlap build. (`windowing` field is a stale generic label in both — trust dir name + n_train.)

## 2 · Per-baseline feasibility + exact CLI

### Feasibility summary (the FSQ/non-US adaptations)
| Baseline | Needs | On Istanbul today? |
|---|---|---|
| Markov floors | check2hgi substrate (`next.parquet`, `sequences_next.parquet`) — **exist** | ✅ runs now |
| POI-RGNN faithful | `data/checkins/Istanbul.parquet` (Gowalla schema) + 7-class cat | ✅ runs after `parse_city` (taxonomy aligns — Istanbul `category_map.json` maps FSQ→**7 Gowalla roots** = POI-RGNN `CATEGORY_LABELS`) |
| **STAN = stl_hgi** (chosen) | an **HGI substrate** (Istanbul had none) + per-fold log_T | ✅ DONE 2026-06-25 — built HGI on mahalle (RESULTS §3); Acc@10 71.13 |
| STAN **faithful** (alt) | raw checkins + region polygons | ✅ unblocked — `stan/etl.py` now maps mahalle geojson (`@id`→GEOID); not the chosen variant |
| **ReHDM = faithful** (chosen) | raw checkins + 24h sessions + boroughs CSV | ✅ ETL runs as-is (mahalle `boroughs_area.csv`; ~2 s) — training **deferred to CUDA** |

> **The original "faithful blocked / US-tied" framing was OVERSTATED.** ReHDM's ETL is **data-driven** (sjoins
> POIs vs `output/check2hgi/<city>/temp/boroughs_area.csv` + a pure-lat/lon quadkey) → mahalle works with no code
> change. STAN faithful needed only a one-line `_shapefile_path`/`@id`→GEOID adapter. The chosen STAN variant is
> **stl_hgi** (run from a newly-built HGI substrate, RESULTS §3), not faithful. Istanbul's mahalle (520) are defined
> by `data/miscellaneous/istanbul_mahalle/istanbul_mahalle.geojson`.

---

### 2.A · Markov floors (cat) — runs NOW, cheap
```bash
# region floor already exists (52.5); build the CATEGORY floors:
$PY scripts/compute_simple_baselines.py --state istanbul --task next_poi      # majority-class + Markov-1-POI cat floor
$PY scripts/compute_markov_kstep_cat.py --state istanbul                      # Markov-K-cat (apples-to-apples POI-RGNN floor)
#  -> docs/results/P0/simple_baselines/istanbul/next_poi.json + next_category_markov_kstep.json
```

### 2.B · POI-RGNN faithful (cat native) — runs after one ETL prep
```bash
# 1. build the Gowalla-schema Istanbul checkins (one pass; writes data/checkins/Istanbul.parquet):
$PY scripts/second_dataset/parse_city.py --city istanbul
#    verify: category col = {Community,Entertainment,Food,Nightlife,Outdoors,Shopping,Travel}; lat/lon present.
# 2. POI-RGNN ETL + train (category-only; no region geometry needed):
caffeinate -i env PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -m research.baselines.poi_rgnn.etl --state istanbul
caffeinate -i env PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -m research.baselines.poi_rgnn.train \
    --state istanbul --folds 5 --epochs 35 --tag FAITHFUL_POIRGNN_istanbul_5f35ep
#  ~minutes on MPS (AL was ~70 s). NaN-coord rows are dropped by build_windows (acceptable).
```

### 2.C · STAN region — **stl_hgi** (the chosen variant; HGI substrate built 2026-06-25)
```bash
# Prereq: build the Istanbul HGI substrate ONCE (it has none by default — regions are mahalle, not TIGER).
# See ISTANBUL_BASELINES_RESULTS.md §3: cp the mahalle boroughs CSV to output/hgi/istanbul/temp/, then run
# hgi.pipe.py process_state('istanbul', {shapefile:None, cross_region_weight:0.7}). Verify HGI region_id ==
# check2hgi region_idx before trusting STAN. Then:
caffeinate -i env PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 MTL_RAM_HEADROOM_GB=4 $PY -u scripts/p1_region_head_ablation.py \
    --state istanbul --heads next_stan --folds 5 --epochs 50 --input-type region --seed 0 \
    --region-emb-source hgi --per-fold-transition-dir output/check2hgi/istanbul \
    --tag STAN_HGI_istanbul_5f50ep
#  → Acc@10 71.13 ± 0.68. (stl_check2hgi was the earlier wrong-variant substitute — superseded.)
```

### 2.D · ReHDM region — **faithful** (the chosen variant; runs as-is — NOT blocked)
```bash
# ETL is data-driven: sjoins POIs vs the mahalle boroughs CSV (output/check2hgi/istanbul/temp/boroughs_area.csv);
# quadkey region is pure lat/lon. Runs as-is in ~2 s (no US dependency). Needs data/checkins/Istanbul.parquet
# (built by parse_city, §2.B step 1).
caffeinate -i env PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 MTL_RAM_HEADROOM_GB=4 $PY -u -m research.baselines.rehdm.etl --state istanbul
# Faithful training = 5 SEEDED runs (--folds = #seeds; seed 0 → seeds 0..4). ~96 s/epoch on MPS (multi-hour) →
# DEFERRED to CUDA:
$PY -m research.baselines.rehdm.train --state istanbul --folds 5 --epochs 50 --seed 0 --tag REHDM_istanbul_5seeds_50ep
#  (per-batch sub-hypergraphs → RAM bounded; still run ALONE on MPS. stl_check2hgi was the wrong variant — superseded.)
```

### 2.E · Champion-G @ stride-1 (LOW priority — windowing unify, BASELINE_M4 §2b)
```bash
# per-fold seeded log_T for the stride-1 base (build if absent):
$PY scripts/compute_region_transition.py --state istanbul --engine check2hgi --per-fold --seed 0 --n-splits 5
# champion-G H3-alt recipe, --canon none + explicit heads, fp32, seed 0, 5f, on the stride-1 Phase-V base.
# Mirror the PHASE_V_ISTANBUL_S0 §provenance MTL invocation with --device mps; keep the 4-seed set-a result too.
```

## 3 · The scope fork — RESOLVED 2026-06-25 (user): STAN=stl_hgi, ReHDM=faithful
Per the cross-state convention: **STAN = `stl_hgi`** (substrate-fed, run from HGI), **ReHDM = faithful** (paper
protocol). Both done/runnable for Istanbul — the original "faithful blocked / ~30 h ETL" framing was **overstated**:

- **STAN stl_hgi**: needed a one-time **HGI substrate build** for Istanbul (it had none — mahalle ≠ TIGER). Built
  via the mahalle boroughs CSV (RESULTS §3); region indices verified == check2hgi. Result Acc@10 71.13 ± 0.68.
- **ReHDM faithful**: ETL runs **as-is** (data-driven mahalle sjoin; ~2 s). Only the training is heavy on MPS
  (~96 s/ep) → **deferred to CUDA**. No region-adapter rebuild was needed.
- (STAN's `stan/etl.py` also got a mahalle adapter so *faithful* STAN can run on non-US regions if ever wanted —
  but the chosen STAN variant is stl_hgi, not faithful.)

Earlier substitutes (`stl_check2hgi` STAN 70.39, from-scratch faithful STAN 57.60) are **superseded/removed**.

## 4 · Sequencing (respects ram_watchdog ≤2–3 concurrent) — AS EXECUTED 2026-06-25
1. **Group A (parallel):** §2.A Markov · §2.B POI-RGNN (after `parse_city`) · §2.C STAN — all done.
2. **HGI substrate build** (CPU, ~13 min) — prereq for STAN stl_hgi.
3. **ReHDM faithful** ETL done; **training deferred to CUDA** (heavy on MPS, run ALONE there).
4. **LOW / not run:** §2.E champion-G stride-1.

## 5 · Outputs + traps
- All paths train-only per fold (vocab / prior / OOD / fold-split); val users disjoint from train (asserted) — don't "optimise" away.
- Markov → `docs/results/P0/simple_baselines/istanbul/`; POI-RGNN → `docs/results/baselines/`; STAN stl_hgi → `docs/results/P1/`; ReHDM faithful → its tagged JSON (on CUDA). Then tabulate into `docs/baselines/next_{category,region}/` + the Istanbul column of `RESULTS_BOARD.md §1`.
- **n=5 provisional** everywhere. Report Istanbul as **gap-to-ceiling / lift-over-floor**, not absolute. Resolve the HMT-GRN 60.4-vs-56.56 conflict (§1 note) before tabulating.
- Verify per-fold log_T freshness (`region_transition_log_seed0_fold*.pt` mtime > `next_region.parquet`) before any STL/MTL run that uses `--per-fold-transition-dir`.
