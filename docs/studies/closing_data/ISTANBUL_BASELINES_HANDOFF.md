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
| STAN **faithful** | raw checkins **+ TIGER shapefile** for region assignment | ⛔ blocked — `stan/etl.py:_shapefile_path` raises `KeyError` for non-US; Istanbul = mahalle. Needs a region-assignment swap (mahalle geojson) |
| STAN **stl_check2hgi** | check2hgi region substrate + per-fold log_T — **exist** | ✅ runs now |
| ReHDM **faithful** | raw checkins + 24h sessions + borough assignment | ⛔ blocked — borough assignment is US-tied; needs adaptation (mahalle/`boroughs_area.csv`) |
| ReHDM **stl_check2hgi** | check2hgi substrate (`next.parquet` 9-step seq + `sequences_next.parquet`) — **exist** | ✅ runs now |

> **Why faithful STAN/ReHDM were never done for Istanbul:** both ETLs assign regions via US-only geometry
> (`data/checkins/<State>.parquet` + TIGER tracts / boroughs). Istanbul's regions are **mahalle** (520), defined
> by `data/miscellaneous/istanbul_mahalle/istanbul_mahalle.geojson`. The substrate-fed **`stl_check2hgi`**
> variants sidestep this entirely (they consume the already-built, mahalle-correct check2hgi region targets).

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

### 2.C · STAN region — **stl_check2hgi** (runs now) ⚠ faithful is blocked (§3)
```bash
caffeinate -i env PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u scripts/p1_region_head_ablation.py \
    --state istanbul --heads next_stan --folds 5 --epochs 50 --input-type region \
    --region-emb-source check2hgi --per-fold-transition-dir output/check2hgi/istanbul \
    --tag STAN_CHECK2HGI_istanbul_5f50ep
#  (no stl_hgi variant — Istanbul has no HGI substrate; check2hgi is the only one.)
```

### 2.D · ReHDM region — **stl_check2hgi** (runs now, ALONE) ⚠ faithful is blocked (§3)
```bash
caffeinate -i env PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=1 $PY -u -m research.baselines.rehdm.train_stl_study \
    --state istanbul --engine check2hgi --epochs 50 \
    --batch-size 256 --lr 1e-4 --max-lr 3e-3 --max-intra 3 --max-inter 3 \
    --tag REHDM_STL_STUDY_istanbul_check2hgi_5f50ep
#  Hypergraph build is RAM-heavy → run this one ALONE (no co-scheduled jobs), watch ram_watchdog.
```

### 2.E · Champion-G @ stride-1 (LOW priority — windowing unify, BASELINE_M4 §2b)
```bash
# per-fold seeded log_T for the stride-1 base (build if absent):
$PY scripts/compute_region_transition.py --state istanbul --engine check2hgi --per-fold --seed 0 --n-splits 5
# champion-G H3-alt recipe, --canon none + explicit heads, fp32, seed 0, 5f, on the stride-1 Phase-V base.
# Mirror the PHASE_V_ISTANBUL_S0 §provenance MTL invocation with --device mps; keep the 4-seed set-a result too.
```

## 3 · The scope fork — faithful vs stl for STAN/ReHDM (DECISION NEEDED)
The **reduced board** uses HMT-GRN (done) + Markov-1 (done) as Istanbul's region externals — so STAN/ReHDM are
*not* reduced-board-critical. Two ways to give Istanbul the STAN/ReHDM columns the Gowalla table has:

- **(A) `stl_check2hgi` only** (substrate-fed, §2.C/§2.D): runs **today**, zero new ETL, mahalle-correct. But it
  is the *substrate-as-input* variant (cold-user holdout), not the paper-faithful protocol.
- **(B) faithful too** (paper protocol): requires a **region-assignment adaptation** (swap TIGER/borough geometry
  for the mahalle geojson in `stan/etl.py` + `rehdm/etl.py`) — real code + the ReHDM-faithful 24h-session ETL is
  also expensive (~30 h on M4 at FL scale; Istanbul is smaller but still heavy).

**Recommendation:** do (A) now (cheap, unblocks the table); treat (B) as opt-in only if a reviewer demands
faithful-protocol parity for the external-validity row. Confirm before building the FSQ→mahalle region adapter.

## 4 · Sequencing (respects ram_watchdog ≤2–3 concurrent)
1. **Group A (parallel, all cheap/light):** §2.A Markov · §2.B POI-RGNN (after `parse_city`) · §2.C STAN stl.
2. **Then ALONE:** §2.D ReHDM stl (hypergraph RAM).
3. **Last, LOW:** §2.E champion-G stride-1.
(Faithful STAN/ReHDM only if §3 decision = B.)

## 5 · Outputs + traps
- All paths train-only per fold (vocab / prior / OOD / fold-split); val users disjoint from train (asserted) — don't "optimise" away.
- Markov → `docs/results/P0/simple_baselines/istanbul/`; POI-RGNN → `docs/results/baselines/`; STAN-stl → `docs/results/P1/`; ReHDM-stl → its tagged JSON. Then tabulate into `docs/baselines/next_{category,region}/` + the Istanbul column of `RESULTS_BOARD.md §1`.
- **n=5 provisional** everywhere. Report Istanbul as **gap-to-ceiling / lift-over-floor**, not absolute. Resolve the HMT-GRN 60.4-vs-56.56 conflict (§1 note) before tabulating.
- Verify per-fold log_T freshness (`region_transition_log_seed0_fold*.pt` mtime > `next_region.parquet`) before any STL/MTL run that uses `--per-fold-transition-dir`.
