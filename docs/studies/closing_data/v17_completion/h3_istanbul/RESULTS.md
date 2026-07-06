# H3 — Istanbul rebuilt on `check2hgi_dk_ovl` + v17 (n=20)

> **Done 2026-07-06.** Istanbul is now on the **same substrate identity** as the 5 Gowalla states (v14/design_k
> re-windowed at stride-1 overlap = `check2hgi_dk_ovl`), removing the cross-substrate ("stride-1 GCN base check2hgi")
> caveat. Recipe = **v17** (bs8192 + per-head cat-lr 1e-3), fp32, n=20 {0,1,7,100} × 5f. Cat ceiling = small-state
> recipe **bs2048@0.005**; reg ceiling = `next_stan_flow` prior-off (same as the Gowalla top-up).

## Result (n=20)

| metric | MTL (v17) | STL ceiling | **Δ** | cross-seed σ |
|---|---:|---:|---:|---:|
| **cat** macro-F1 | **63.329** ± 0.020 | 54.738 ± 0.092 | **+8.59** ✅ beats | tiny |
| **reg** FULL top10 | **75.440** ± 0.041 | 75.158 ± 0.012 | **+0.28** ✅ matches/beats | tiny |

Per-seed — MTL cat [63.318, 63.332, 63.359, 63.306]; MTL reg [75.407, 75.484, 75.392, 75.477];
STL cat [54.706, 54.863, 54.771, 54.610]; STL reg [75.156, 75.175, 75.141, 75.161].

**The champion-G signature replicates and STRENGTHENS on the matched substrate.** vs the old board Istanbul
(stride-1 GCN base check2hgi: cat 59.89 / reg 74.28, **Δcat +6.69 / Δreg −0.52**): the v14+overlap substrate lifts
both heads (cat +3.4, reg +1.2) and, crucially, moves reg from "matches (slightly negative)" to **beats (+0.28)** —
Istanbul now matches the FL/CA/TX pattern (beats both) instead of being the lone "reg slightly-negative" small state.
s0 sanity gate passed (Δcat +8.61 / Δreg +0.25) before committing {1,7,100}.

## Build provenance (the full H3 chain)
1. **v14/design_k substrate** — `build_design_k_delaunay.py --state istanbul --out-suffix resln_mae_l0_1 --epochs 500
   --device cuda` (~5 min; encoder=resln, mae-poi-λ=0.3, Delaunay reg GCN, anchor λ=0.1). Fixed 2 blockers:
   (a) Istanbul had **no v14 substrate** (the handoff assumed it existed — it was on base check2hgi only);
   (b) a **filename-casing bug** — the POI2Vec loader wants `poi2vec_poi_embeddings_Istanbul.csv` (capitalized, the
   Gowalla convention) but Istanbul's file was lowercase; fixed with a non-destructive symlink. Postbuild's *non-overlap*
   `next_region` guard trips (base check2hgi Istanbul is 271666 overlap seqs vs v14 default 58297 non-overlap) — **irrelevant**,
   the dk_ovl pipeline regenerates overlap windowing from checkins and never reads v14's non-overlap inputs.
2. **dk_ovl engine** — `build_overlap_probe_engine.py istanbul 1 10` (symlinks v14 embeddings; builds overlap
   next/next_region = 271666 rows / 520 regions / 0% pad).
3. **n=20 cells** — `run_step3_n20.sh` (MTL + cat + reg, interleaved by seed, s0 gate). fp32, `MTL_STRICT=1`,
   `--canon none` + explicit recipe, skip-inert log_T (champion prior off → no per-fold log_T needed).

## Remaining for H3 acceptance
- ☐ **Re-foot substrate-bound Istanbul baselines** on dk_ovl: HGI (Table 2), CTLE-SC. Region externals (HMT-GRN,
  faithful STAN 61.86) are **substrate-independent → re-confirm, do NOT re-run**.
- ☐ **Paper**: update Table 3 Istanbul cells + drop the cross-substrate prose in §5/§6 (the .tex — flag for the author).
- ☑ RESULTS_BOARD §1 Istanbul row → dk_ovl+v17, "stride-1 GCN" note dropped (this commit).
