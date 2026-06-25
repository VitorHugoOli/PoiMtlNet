# ISTANBUL BASELINES — results + run log (2026-06-25, M4 Pro / MPS, fp32)

> Execution record for [`ISTANBUL_BASELINES_HANDOFF.md`](ISTANBUL_BASELINES_HANDOFF.md). Completes the
> §6.3 external-validity box so Istanbul carries the same baseline columns as the Gowalla states.
> Protocol: **seed 0 × 5 folds (n=5)**, gated stride-1 overlap, leak-free per-fold train-only priors,
> user-disjoint folds, MPS = fp32. **n=5 provisional** — report **gap-to-ceiling / lift-over-floor**, not absolute Acc@k.

## 1 · Results (this run)

| Cell | Metric | Value | Source JSON |
|---|---|---|---|
| Markov majority (cat) | macro-F1 | **7.14** | `P0/simple_baselines/istanbul/next_category.json` |
| **Markov-1-POI** (cat floor, P(cat\|last POI)) | macro-F1 | **17.55 ± 0.44** | `…/next_category.json` (`markov_1step`) |
| **Markov-K-cat** (apples-to-apples POI-RGNN floor) | macro-F1 | **24.55 ± 0.30** (best, k5) | `…/next_category_markov_kstep.json` |
| **POI-RGNN** faithful (cat native) | macro-F1 | **30.12 ± 0.84** | `baselines/faithful_poi_rgnn_istanbul_5f_35ep_*.json` |
| **STAN** region (stl_check2hgi, seed 0) | Acc@10 | **70.39** | `P1/region_head_istanbul_region_5f_50ep_STAN_CHECK2HGI_*.json` |
| **ReHDM** region (stl_check2hgi) | Acc@10 | ⏳ running (fold-by-fold) | `baselines/REHDM_STL_STUDY_istanbul_check2hgi_5f50ep_fold*.json` |

Markov-K-cat full series (macro-F1): k1=11.45 · k3=24.14 · **k5=24.55** · k7=23.14 · k9=22.55.
POI-RGNN auxiliary: Acc=41.64 · Acc@5=93.67 · MRR=62.62. STAN auxiliary: Acc@1=31.98 · Acc@5=59.73 · MRR=44.96 · cat-F1(diag)=10.86.

### Reference cells (already done before this run — for gap/lift framing)
Champion-G MTL (set-a, GCN, 4 seeds) cat 60.16 / reg 69.79 · STL cat ceiling (`next_gru`) 52.10 · STL reg ceiling
(`next_stan_flow`) 70.37 · CTLE-SC 25.92 ± 0.31 · Check2HGI-SC ceiling 54.53 ± 0.69 · Markov-1 region floor 52.5.

**Sanity:** STAN-stl Acc@10 = 70.39 lands right on the STL reg ceiling (70.37) — i.e. NOT inflated by a stale
log_T (a stale prior would have pushed it ~+8–12 pp to ~80; see §3.3). POI-RGNN (30.12) sits above Markov-K-cat
(24.55) and below the STL cat ceiling (52.10) — the expected ordering.

## 2 · HMT-GRN 60.4-vs-56.56 conflict — RESOLVED
The handoff flagged these as "same run, two recorded numbers." They are **two distinct builds**, both correctly recorded:

| Build dir | Windowing | n_train (fold 0) | reg Acc@10 |
|---|---|---|---|
| `results/baseline_b3_hmt_grn_style/istanbul/` | stride-9 (non-overlap) | 46,638 | **56.56** |
| `results/baseline_b3_hmt_grn_style/istanbul_stride1/` | stride-1 (overlap) | 217,333 (≈4.7×) | **60.42 ≈ 60.4** |

(Both `b3_seed0_folds5.json`; the JSON `windowing` field is a stale generic label in both — the directory name +
n_train are the real signal.) The board protocol mandates **gated stride-1 overlap** (matching STAN/POI-RGNN/champion-G),
so the **board-correct HMT-GRN value is 60.4 (stride-1)**. The 56.56 in `next_region/comparison.md` is the older
non-overlap build. `RESULTS_BOARD.md §4`'s 60.4 is correct; cite stride-1 when tabulating.

## 3 · Protocol corrections + lessons (handoff defects caught)
1. **`--task next_poi` is invalid** (§2.A). `compute_simple_baselines.py` only accepts `next_category` / `next_region`.
   The category floors come from **`--task next_category`** (computes `macro_f1` for `majority` + `markov_1step` = the
   "Markov-1-POI cat floor"). Output is `next_category.json`, not `next_poi.json`.
2. **STAN silently defaults to seed 42** (§2.C omits `--seed`), but the protocol is **seed 0**. Always pass `--seed 0`.
   The header protocol is authoritative; §2.E already uses seed 0 + builds `region_transition_log_seed0_*`.
3. **Stale per-fold log_T** (CLAUDE.md trap). The on-disk seed-0 log_T (Jun 24 05:47) predated `next_region.parquet`
   (Jun 24 07:20) → would silently inflate reg Acc@10 by +8–12 pp. Rebuilt before STAN:
   `compute_region_transition.py --state istanbul --engine check2hgi --per-fold --seed 0 --n-splits 5`.
   The `assert_log_t_fresh` guard in `p1_region_head_ablation.py` correctly hard-blocked the stale seed-42 run.
4. **RAM guard over-trips on the tiny Istanbul dataset.** `load_next_data._guard_cpu_resident_ram` demands 16 GB
   head-room; the Istanbul next dataset needs only ~0.3 GB. Set **`MTL_RAM_HEADROOM_GB=4`** for every run.
5. **macOS `stat`**: §5's freshness check uses Linux `stat -c`, which fails on darwin. Use `stat -f '%m %N'`.
6. **ReHDM RAM oscillation**: the hypergraph build spikes free-RAM to ~22–23% per fold then recovers to ~35%.
   Benign on Istanbul (small); never approached the 18% OOM-reboot floor. Run it ALONE (no co-scheduled jobs), as specified.

## 4 · Scope fork (handoff §3) — decision taken
Took **recommendation (A): `stl_check2hgi` only** for STAN + ReHDM (substrate-fed, mahalle-correct, runs today, zero
new ETL). **Faithful (B) NOT pursued** — both ETLs assign regions via US-only geometry (TIGER tracts / boroughs); a
faithful Istanbul run needs an FSQ→mahalle region-assignment adapter. Reduced board uses HMT-GRN + Markov-1 as
Istanbul's region externals, so STAN/ReHDM are full-table-completeness, not reduced-board-critical. Build (B) only if a
reviewer demands faithful-protocol parity for the external-validity row.

## 5 · Open items
- **ReHDM stl** (§2.D) still running at write time — add Acc@10 to §1 + the board when complete.
- **§2.E champion-G @ stride-1** (LOW, windowing-unify) — NOT run (out of this session's scope).
- **Tabulation** into `docs/baselines/next_{category,region}/` + the Istanbul column of `RESULTS_BOARD.md §1`
  still pending — apply the n=5-provisional / gap-to-ceiling rule and the resolved HMT-GRN value (60.4, stride-1).
