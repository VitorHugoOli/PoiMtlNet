# Comprehensive Results Table — Check2HGI MTL Study

**Date:** 2026-04-20. Consolidates every paper-relevant number measured in this study, plus external-baseline anchors. Complements `BASELINES_AND_BEST_MTL.md` with a stricter "one row per method × state × task" layout ready for the paper's §Results.

All runs: check2HGI engine, fair user-disjoint `StratifiedGroupKFold`, `OneCycleLR(max_lr)` per the noted value, AdamW, batch 2048, 50 epochs, seed 42 unless otherwise noted.

Legend: **Acc@10** in §Task B Region rows is `top10_acc_indist` (restricted to regions seen in training set of the fold) for MTL rows, and `top10_acc` for STL rows (which already filter OOD by construction).

---

## Task A · next_category (7 classes, macro-F1 primary)

Comparable external baselines on the same state split are documented in `docs/baselines/BASELINE.md` (POI-RGNN, HAVANA, PGC-NN on FL/CA/TX). Here we report only our own pipeline numbers for this metric; the external comparison is a separate block in the paper.

| Row | Method | Substrate | State | Protocol | macro-F1 | Acc@1 | MRR | Notes |
|:---:|:---|:---:|:---:|:---:|---:|---:|---:|:---|
| **Baselines** | | | | | | | | |
| A-B1 | Random (uniform over 7 classes) | — | AL | theoretical | ~14.3 | ~14.3 | — | floor |
| A-B2 | Majority-class | — | AL | closed form | — | 34.20 | — | |
| A-B3 | Markov-1-POI | — | AL | closed form | ~31.7 | ~32 | — | P0 |
| **STL (single-task)** | | | | | | | | |
| A-S1 | HGI + NextHeadMTL | HGI POI emb | AL | 5f × 50ep | 20.29 ± 1.34 | 24.10 ± 1.54 | 50.29 ± 1.32 | P1.5b |
| A-S2 | **Check2HGI + NextHeadMTL** ⭐ | Check2HGI checkin | AL | 5f × 50ep | **38.58 ± 1.23** | 40.18 ± 0.95 | **62.35 ± 0.48** | **P1.5b champion** |
| A-S3 | Check2HGI + NextHeadMTL | Check2HGI checkin | AZ | 5f × 50ep | 42.08 ± 0.89 | 42.97 ± 0.75 | — | P1.5b (AZ) |
| A-S4 | Check2HGI + NextHeadMTL | Check2HGI checkin | FL | 1f × 50ep | 63.17 | 65.25 | 78.48 | P1.5b (FL, n=1) |
| **MTL (2-task: category + region)** | | | | | | | | |
| A-M1 | mtlnet_dselectk + pcgrad + GRU | Check2HGI | AL | 5f × 50ep | 36.08 ± 1.96 | 38.22 ± 1.43 | 61.52 ± 1.03 | P2 champion |
| A-M2 | mtlnet_dselectk + MTLoRA r=8 | Check2HGI | AL | 5f × 50ep | 35.61 ± 1.54 | 38.32 ± ? | 61.54 ± ? | Step 4a |
| A-M3 | **mtlnet_crossattn + pcgrad + GRU** ⭐ | Check2HGI | AL | 5f × 50ep | **38.58 ± 0.98** | 40.40 ± 0.39 | 63.00 ± ? | **P2 ablation 06 — ties STL** |
| A-M4 | mtlnet_crossattn + pcgrad + STAN d=128 | Check2HGI | AL | 5f × 50ep | 39.07 ± 1.18 | 40.48 ± 1.20 | 63.36 ± 0.96 | P8 MTL-STAN |
| A-M5 | mtlnet_crossattn + pcgrad + STAN d=256 | Check2HGI | AL | 5f × 50ep | 38.11 ± 1.11 | 40.37 ± 0.58 | 63.21 ± 0.67 | P8 MTL-STAN hp-tuned |
| A-M6 | mtlnet_crossattn + pcgrad + GRU | Check2HGI | AZ | 5f × 50ep | **43.13 ± 0.55** | 44.00 ± 0.51 | 65.48 ± 0.40 | P2 az1 |
| A-M7 | mtlnet_crossattn + pcgrad + STAN d=128 | Check2HGI | AZ | 5f × 50ep | 42.64 ± 0.26 | 44.07 ± 0.43 | 65.65 ± 0.18 | P8 MTL-STAN |
| A-M8 | mtlnet_crossattn + pcgrad + STAN d=256 | Check2HGI | AZ | 5f × 50ep | 42.74 ± 0.45 | 44.05 ± 0.66 | 65.58 ± 0.29 | P8 MTL-STAN hp-tuned |
| A-M9 | mtlnet_dselectk + pcgrad + GRU | Check2HGI | FL | 1f × 50ep | **64.78** | 67.44 | 79.70 | FL P2-validate (n=1) |
| A-M10 | mtlnet_crossattn + pcgrad + GRU | Check2HGI | FL | 1f × 50ep | 66.46 | 68.92 | 80.47 | FL P2 (n=1) |
| A-M11 | mtlnet_crossattn + pcgrad + STAN d=256 | Check2HGI | FL | 1f × 50ep | 66.16 | 68.76 | 80.35 | P8 sanity (n=1, ties GRU) |

**Task-A observations:**

- **AL:** STL (A-S2, 38.58) and MTL cross-attn (A-M3, 38.58) are byte-tied within σ. MTL-STAN variants (A-M4, A-M5) are within σ of both — **the region head choice does not shift the category head** on AL.
- **AZ:** STL (A-S3, 42.08) and all MTL variants (A-M6/M7/M8, 42.64–43.13) are within σ of each other. MTL **slightly lifts** cat on AZ (~1 pp). The region head is irrelevant.
- **FL:** MTL cross-attn (A-M10, 66.46, n=1) is +3.29 pp above STL (A-S4, 63.17, n=1). Both numbers are n=1 and need 5-fold confirmation per Phase 7.
- **Category F1 is region-head-invariant.** Across AL + AZ (4 MTL-vs-MTL head swaps), all Δcat are within ±1 pp. This is a clean finding for the paper: the cross-attention architecture's cat-closer property lives in the shared backbone + cross-attn exchange, not in head-pairing.

---

## Task B · next_region (Acc@10 primary, MRR secondary)

| Row | Method | Substrate | State | Protocol | Acc@1 | **Acc@10** | MRR | Notes |
|:---:|:---|:---:|:---:|:---:|---:|---:|---:|:---|
| **Baselines — Alabama** | | | | | | | | |
| B-B1 | Random | — | AL | theoretical | 0.09 | 0.90 | 0.68 | 1109 classes |
| B-B2 | Majority | — | AL | closed form | 1.97 | 1.97 | 1.97 | |
| B-B3 | Top-K popular | — | AL | closed form | 1.97 | 14.67 | 4.87 | |
| B-B4 | **Markov-1-region** (paper floor) ⭐ | — | AL | closed form | 25.40 ± 2.73 | **47.01 ± 3.55** | 32.17 ± 2.90 | **P0 floor** |
| B-B5 | Markov-5-region (w/ backoff) | — | AL | closed form | 20.80 ± 2.65 | 33.42 ± 2.16 | 24.99 ± 2.53 | P0 |
| B-B6 | Markov-9-region (ctx-matched) | — | AL | closed form | 20.49 ± 2.57 | 32.79 ± 1.92 | 24.54 ± 2.36 | P0 |
| **STL — Alabama** | | | | | | | | |
| B-S1 | STL GRU (hd=256) | Check2HGI region | AL | 5f × 50ep | 23.60 ± 1.86 | 56.94 ± 4.01 | 34.57 ± 2.34 | P1 |
| B-S2 | STL TCN-residual | Check2HGI region | AL | 5f × 50ep | 21.76 ± 2.35 | 56.11 ± 4.02 | 32.93 ± ? | P1 |
| B-S3 | STL HGI region emb (TCN head) | HGI region | AL | 5f × 50ep | ? | 57.02 ± 2.92 | 33.14 ± 1.87 | P1.5 (CH15, ties) |
| B-S4 | **STL STAN (Luo WWW'21 adapt)** ⭐ | Check2HGI region | AL | 5f × 50ep | **24.64 ± 1.38** | **59.20 ± 3.62** | **36.10 ± 1.96** | **P8 STL SOTA** |
| **MTL — Alabama** | | | | | | | | |
| B-M1 | mtlnet_dselectk + pcgrad + GRU | Check2HGI | AL | 5f × 50ep | 13.31 | 48.88 ± 6.26 | 24.43 | P2 champion (prior) |
| B-M2 | mtlnet_dselectk + MTLoRA r=8 + GRU | Check2HGI | AL | 5f × 50ep | 13.95 | 50.72 ± 4.36 | 25.36 | Step 4a (prior best MTL) |
| B-M3 | mtlnet_crossattn + static λ=0.5 + GRU | Check2HGI | AL | 5f × 50ep | 11.34 | 50.26 ± 4.34 | 25.35 | Step 2 |
| B-M4 | mtlnet_crossattn + pcgrad + GRU | Check2HGI | AL | 5f × 50ep | 10.06 | 45.09 ± 5.37 | 20.94 | B13 (ablation 06) |
| B-M5 | **mtlnet_crossattn + pcgrad + STAN d=128** ⭐ | Check2HGI | AL | 5f × 50ep | 12.48 ± 1.44 | **50.27 ± 4.47** | 24.16 ± 2.25 | **P8 MTL-STAN best-mean, low-σ** |
| B-M6 | mtlnet_crossattn + pcgrad + STAN d=256, 8h | Check2HGI | AL | 5f × 50ep | 13.86 ± 3.43 | 51.60 ± 10.09 | 25.69 ± 5.34 | P8 MTL-STAN hp-tuned (high σ) |
| **Baselines — Arizona** | | | | | | | | |
| B-B7 | Random (AZ) | — | AZ | theoretical | — | 0.65 ± 0.00 | — | 1540 classes |
| B-B8-AZ | Majority (AZ) | — | AZ | closed form | 7.43 | 7.43 ± 0.70 | — | P0 |
| B-B9-AZ | Top-K popular (AZ) | — | AZ | closed form | 7.43 | 20.82 ± 1.28 | — | P0 |
| B-B10-AZ | **Markov-1-region (AZ)** ⭐ | — | AZ | closed form | 23.98 ± 1.13 | **42.96 ± 2.05** | — | **P0 floor** |
| B-B11-AZ | Markov-9-region (AZ, ctx-matched) | — | AZ | closed form | — | 33.38 ± 1.33 | — | P0 |
| **STL — Arizona** | | | | | | | | |
| B-S5 | STL GRU (hd=256) | Check2HGI region | AZ | 5f × 50ep | 23.63 ± 2.04 | 48.88 ± 2.48 | 32.13 ± 2.21 | P1 |
| B-S6 | **STL STAN (Luo WWW'21 adapt)** ⭐ | Check2HGI region | AZ | 5f × 50ep | **24.48 ± 2.29** | **52.24 ± 2.38** | **33.70 ± 2.36** | **P8 STL SOTA** |
| **MTL — Arizona** | | | | | | | | |
| B-M7 | mtlnet_crossattn + pcgrad + GRU | Check2HGI | AZ | 5f × 50ep | 13.20 ± 1.99 | **41.07 ± 3.46** | 22.49 ± 2.49 | P2 az1 |
| B-M8 | mtlnet_crossattn + pcgrad + STAN d=128 | Check2HGI | AZ | 5f × 50ep | 9.79 ± 1.98 | 37.47 ± 4.01 | 18.53 ± 2.54 | P8 MTL-STAN (bottleneck) |
| B-M9 | mtlnet_crossattn + pcgrad + STAN d=256, 8h | Check2HGI | AZ | 5f × 50ep | 11.53 ± 2.11 | 41.04 ± 4.55 | 20.93 ± 2.86 | P8 MTL-STAN hp-tuned (ties GRU) |
| **Baselines — Florida** | | | | | | | | |
| B-B8 | Random | — | FL | theoretical | 0.02 | 0.21 | 0.19 | 4702 classes |
| B-B9 | Majority | — | FL | closed form | 22.25 | 22.25 | 22.25 | |
| B-B10 | Top-K popular | — | FL | closed form | 22.25 | 33.82 | 25.65 | |
| B-B11 | **Markov-1-region** (paper floor) ⭐ | — | FL | closed form | 46.36 ± 0.89 | **65.05 ± 0.93** | 52.37 ± 0.90 | **P0 floor** |
| B-B12 | Markov-5-region (w/ backoff) | — | FL | closed form | 42.95 ± 0.69 | 54.91 ± 0.74 | 47.07 ± 0.71 | P0 |
| B-B13 | Markov-9-region (ctx-matched) | — | FL | closed form | 42.56 ± 0.68 | 54.10 ± 0.72 | 46.53 ± 0.71 | P0 |
| **STL — Florida** | | | | | | | | |
| B-S7 | **STL GRU (hd=256)** ⭐ | Check2HGI region | FL | 5f × 50ep | **44.49 ± 0.51** | **68.33 ± 0.58** | **52.74 ± 0.45** | **P1 champion** |
| B-S8 | STL STAN | Check2HGI region | FL | 5f × 50ep | TBD | TBD | TBD | Paper-blocking — not yet run |
| **MTL — Florida** | | | | | | | | |
| B-M10 | mtlnet_dselectk + pcgrad + GRU | Check2HGI | FL | 1f × 50ep | 15.43 | **57.05** | 27.49 | FL P2-validate (n=1) |
| B-M11 | mtlnet_crossattn + pcgrad + GRU | Check2HGI | FL | 1f × 50ep | — | 57.60 | — | FL P2 (n=1) |
| B-M12 | mtlnet_crossattn + pcgrad + STAN d=256, 8h | Check2HGI | FL | 1f × 50ep | 12.09 | **57.71** (indist) | 24.51 | P8 FL sanity (n=1, **ties GRU on Acc@10**; MRR −3.6 pp, Acc@5 −14 pp — worse fine-grained ranking) |

**Task-B observations:**

- **STL STAN consistently > STL GRU** across states (AL +2.26 pp, AZ +3.36 pp Acc@10). Margin grows with scale. STAN is the universal STL ceiling.
- **Markov-1-region is a strong classical floor** that grows with state density (AL 47.01 → FL 65.05). Models must beat this; ours do by +12 pp (AL) to +3 pp (FL).
- **Markov monotonically degrades with k** (sparsity outpaces context gain). At context-matched k=9, STL neural beats Markov by +24 pp (AL) / +14 pp (FL).
- **MTL region is capped below STL everywhere**: AL −9 pp (B-M5 vs B-S4), AZ −11 pp (B-M9 vs B-S6), FL estimate −11 pp (B-M11 vs B-S7). CH-M1 holds.
- **MTL head-swap effect is scale-dependent**: AL STAN-d=128 > GRU (+5 pp); AZ STAN-d=128 < GRU (−3 pp); **AZ STAN-d=256 = GRU** (hp-tuned config matches backbone output dimension). See `research/MTL_WITH_STAN_HEAD.md`.

---

## Cross-task summary — "which ceiling binds?"

For each state × task, the two binding reference points are the **simple floor** (Markov for region, Majority for category) and the **STL ceiling** (best single-task). MTL performance sits between these two. Below are the headline gaps.

| State | Task | Floor | STL ceiling | Best MTL | MTL − Floor | MTL − STL | Verdict |
|---|---|---:|---:|---:|---:|---:|:---|
| AL | cat F1 | 31.7 (Markov-POI) | **38.58** (A-S2) | 38.58 (A-M3) | +6.88 pp | 0.00 pp | MTL ties STL |
| AL | reg Acc@10 | 47.01 (B-B4) | **59.20** (B-S4) | 50.27 (B-M5) / 51.60 (B-M6) | +3.26 / +4.59 pp | −8.93 / −7.60 pp | MTL capped below STL |
| AZ | cat F1 | — | 42.08 (A-S3) | 43.13 (A-M6) | — | +1.05 pp | MTL slightly lifts |
| AZ | reg Acc@10 | 42.96 (B-B10-AZ) | **52.24** (B-S6) | 41.07 (B-M7) / 41.04 (B-M9) | −1.89 / −1.92 pp | −11.17 / −11.20 pp | MTL capped, head-invariant at d=256, BELOW Markov floor |
| FL | cat F1 | 37.2 (Markov-POI) | 63.17 (A-S4, n=1) | 66.46 (A-M10, n=1) | +29.26 pp | +3.29 pp | MTL lifts (n=1, 5-fold pending) |
| FL | reg Acc@10 | **65.05** (B-B11) | 68.33 (B-S7) | 57.60 (B-M11) / 57.71 (B-M12, STAN-d256) | −7.34 pp | −10.62 pp | MTL regresses below Markov at both heads (n=1) |

**Key cross-state patterns:**

1. **The region task is harder for MTL than for STL at every scale.** Gap ranges from 7 to 11 pp. This is a **shared-backbone dilution** effect confirmed by the λ=0 decomposition (`results/P2/ablation_architectural_overhead.md`): the region head loses ~25 pp of capacity at FL when forced through a shared backbone, of which ~14 pp is recovered via category→region transfer at FL scale.
2. **The category task benefits from MTL only at scale.** AL null, AZ +1 pp, FL +3.3 pp (pending 5-fold).
3. **STAN as head adds value only in specific configs:** STL (+2.2 to +3.4 pp universal), MTL-AL (+5 pp at d=128), MTL-AZ-FL (ties GRU at d=256). It is **not a universal MTL lift**.

---

## External baseline anchors (paper's Comparable column)

For the category track, `docs/baselines/BASELINE.md` documents POI-RGNN, HAVANA, PGC-NN reproduced on the same FL/CA/TX state splits. Summary numbers to reference in the paper:

| External baseline | FL cat F1 | CA cat F1 | TX cat F1 | Comparability |
|---|---:|---:|---:|:---|
| POI-RGNN (reproduced) | 34.49 | 31.78 | 33.03 | same split, same task |
| HAVANA (paper-reported) | ~62.9 | ~46.9 | ~59.8 | semantic annotation task (not sequential) |
| PGC-NN (reproduced) | 40.79 | 33.64 | 41.39 | labeling task (not sequential) |
| **Ours — Check2HGI STL next_cat (A-S4)** | **63.17** | TBD | TBD | our pipeline |
| **Ours — Check2HGI MTL crossattn (A-M10)** | **66.46** | TBD | TBD | our pipeline |

Our STL next_category on FL (n=1) is already **+28.68 pp above POI-RGNN** and roughly on par with HAVANA (despite HAVANA being a different task). The MTL lift sits +3.29 pp on top. These numbers are n=1 and need 5-fold confirmation in Phase 7 for the paper.

**Next-region external baselines:** HMT-GRN and STAN are concept-aligned but use different datasets (Foursquare-NYC/TKY, full Gowalla). `research/POSITIONING_VS_HMT_GRN.md` and `research/SOTA_STAN_BASELINE.md` explain why direct numerical comparison is not apples-to-apples. Our adapted-STAN (B-S4/B-S6) lands in the **competitive attention-baseline ballpark** (Acc@1 ~24%, Acc@10 ~52-59% on 1.1–1.5K-class region) — which is in the same regime as STAN's reported NYC/TKY Acc@1 ~25% on ~2K POIs.

---

## Gaps to close before paper freeze

Ranked by paper-blocking priority:

1. **FL STL STAN 5-fold** — to complete the B-S8 row. Scheduled alongside Phase 7 or at FL STL-STAN 1-fold then 5-fold.
2. **FL MTL cross-attn + pcgrad + GRU 5-fold** (replaces A-M10/B-M11 n=1 → n=5). Already planned in Phase 7.
3. **FL STL cat 5-fold** (replaces A-S4 n=1 → n=5). Phase 7.
4. **CA + TX headline configs** (5 methods × 2 states × 5-fold). Phase 7 parallel-machine plan exists.
5. **AZ Markov baseline archive** (B-B7). ~2 min re-run of `compute_simple_baselines.py --state arizona`.

Deferred (not paper-blocking):

6. GETNext-style graph-prior head (from `STAN_CRITICAL_REVIEW.md §4.5`).
7. ALiBi-decay init for STAN d=256 on AL (variance reduction, optional).
8. PCGrad vs static_weight × STAN ablation (isolates optimizer × head interaction).

---

## Index of source artefacts

| Group | Location |
|---|---|
| STL next_category (P1.5b) | `results/P1_5b/next_category_{alabama,arizona,florida}_check2hgi_*.json` |
| STL next_category HGI | `results/P1_5b/next_category_alabama_hgi_5f_50ep_fair.json` |
| STL region (P1) | `results/P1/region_head_{alabama,arizona,florida}_region_5f_50ep_*.json` |
| STL STAN (P1) | `results/P1/region_head_{alabama,arizona}_region_5f_50ep_STAN_*.json` |
| MTL ablations (P2) | `results/P2/ablation_*.json`, `results/P2/az*.json`, `results/P2/fl_*.json`, `results/P2/rerun_R*.json` |
| MTL STAN (P8) | `results/P8_sota/mtl_crossattn_pcgrad_*_stan_*.json` |
| Markov simple baselines | `results/P0/simple_baselines/{alabama,florida}/next_region.json` |
| External comparisons | `docs/baselines/BASELINE.md` |
| Positioning docs | `research/POSITIONING_VS_HMT_GRN.md`, `research/SOTA_STAN_BASELINE.md`, `research/STAN_CRITICAL_REVIEW.md`, `research/MTL_WITH_STAN_HEAD.md` |
