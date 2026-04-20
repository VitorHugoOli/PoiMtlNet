# Baselines + Best MTL Results — Paper Comparison Table

**Date:** 2026-04-18. Fair folds (user-disjoint `StratifiedGroupKFold`) throughout.

All numbers are from commits in this worktree; see §Sources at the bottom for the JSON path for each row.

## Task A — next-category (7 classes, macro-F1 primary)

### Alabama (AL, 10K training rows, 5 folds × 50 epochs)

| Row | Method | macro-F1 | Acc@1 | MRR | Source |
|:---:|--------|---------:|------:|----:|:------:|
| A1 | Random (uniform over 7 classes) | ~14.3 | ~14.3 | — | theoretical |
| A2 | Majority (most-common class) | — | 34.20 | — | P0 simple baselines |
| A3 | Markov-1-POI | ~31.7 | ~32 | — | P0 simple baselines |
| A4 | HGI STL (POI-level embeddings, 5f×50ep) | **20.29 ± 1.34** | 24.10 ± 1.54 | 50.29 ± 1.32 | P1.5b refair |
| A5 | **Check2HGI STL (check-in embeddings, 5f×50ep)** ⭐ | **38.58 ± 1.23** | **40.18 ± 0.95** | **62.35 ± 0.48** | **P1.5b refair** |
| | *MTL with shared-backbone architectures:* | | | | |
| A6 | MTL dselectk + pcgrad (P2 champion) | 36.08 ± 1.96 | 38.22 ± 1.43 | 61.52 ± 1.03 | P2-validate |
| A7 | MTL dselectk + MTLoRA r=8 | 35.61 ± 1.54 | 38.32 ± ? | 61.54 ± ? | Ablation step 4a |
| | *MTL with cross-attention architecture:* | | | | |
| A8 | **MTL mtlnet_crossattn + pcgrad** ⭐ **best MTL** | **38.58 ± 0.98** | **40.40 ± 0.39** | **63.00 ± ?** | **Ablation step 6** |

**Key observations:**
- **A8 closes the MTL → STL gap entirely on category** (38.58 = 38.58 within σ). Cross-attention is the only MTL architecture we tested that reaches STL-parity.
- A5 beats A4 by **+18.30 pp F1** with non-overlapping std envelopes (CH16 confirmed). This is the paper's primary substrate claim.
- Check2HGI STL (A5) beats published POI-RGNN (~31.8–34.5% F1 on Gowalla states) by 4–7 pp.

### Florida (FL, 127K training rows, 1 fold × 50 epochs — STL cat pending 5-fold)

| Row | Method | macro-F1 | Acc@1 | MRR | Notes |
|:---:|--------|---------:|------:|----:|:-----:|
| A9 | Majority | — | 22.25 | 22.25 | P0 simple baselines |
| A10 | Markov-1-POI | ~37.2 | — | — | P0 simple baselines |
| A11 | Check2HGI STL (1f×50ep fair) | 63.17 | 65.25 | 78.48 | P1.5b-FL (1 fold; 5-fold pending) |
| A12 | **MTL dselectk + pcgrad (1f×50ep fair)** | **64.78** | **67.44** | **79.70** | FL P2-validate |

**Key observation (CH01 asymmetric):** On FL, MTL **improves** category F1 by **+1.61 pp** over STL (64.78 vs 63.17). This is the **first positive MTL lift** we've measured — and contrasts with AL where MTL ≤ STL on category. **Data scale matters for the weaker-head task**: FL's 127K rows support MTL's shared-backbone signal transfer; AL's 10K do not.

---

## Task B — next-region (~1109 cls AL / ~4702 cls FL, Acc@10 + MRR primary)

### Alabama

| Row | Method | Acc@1 | Acc@10 | MRR | Source |
|:---:|--------|------:|-------:|----:|:------:|
| B1 | Random | 0.09 | 0.90 | 0.68 | theoretical |
| B2 | Majority | 1.97 | 1.97 | 1.97 | P0 |
| B3 | Top-K popular | 1.97 | 14.67 | 4.87 | P0 |
| B4 | Markov-1-POI | ~12 | **21.33 ± 1.55** | ~15.00 | P0 (legacy floor) |
| B5 | **Markov-1-region** (paper floor, best k) | 25.40 ± 2.73 | **47.01 ± 3.55** | 32.17 ± 2.90 | P0 |
| B5a | Markov-2-region (w/ backoff) | 22.66 ± 2.91 | 37.87 ± 2.77 | 27.76 ± 2.92 | P0 |
| B5b | Markov-3-region (w/ backoff) | 21.77 ± 2.96 | 35.22 ± 2.55 | 26.28 ± 2.86 | P0 |
| B5c | Markov-5-region (w/ backoff) | 20.80 ± 2.65 | 33.42 ± 2.16 | 24.99 ± 2.53 | P0 |
| B5d | Markov-7-region (w/ backoff) | 20.59 ± 2.57 | 32.87 ± 1.85 | 24.63 ± 2.34 | P0 |
| B5e | **Markov-9-region** (context-matched to model) | 20.49 ± 2.57 | **32.79 ± 1.92** | 24.54 ± 2.36 | P0 |
| | *STL neural heads, single-task:* | | | | |
| B7 | STL TCN-residual standalone | 21.76 ± 2.35 | 56.11 ± 4.02 | 32.93 ± ? | P1 |
| B8 | STL GRU standalone | 23.60 ± 1.86 | 56.94 ± 4.01 | 34.57 ± 2.34 | P1 |
| B8a | **STL STAN (Luo WWW'21, adapt)** ⭐ | **24.64 ± 1.38** | **59.20 ± 3.62** | **36.10 ± 1.96** | **SOTA note** |
| B9 | STL HGI region embeddings (TCN head, same pipeline) | ? | 57.02 ± 2.92 | 33.14 ± 1.87 | P1.5 (CH15) |
| | *MTL with shared-backbone architectures:* | | | | |
| B10 | MTL dselectk + pcgrad (P2 champion) | 13.31 | 48.88 ± 6.26 | 24.43 | P2-validate |
| B11 | **MTL dselectk + MTLoRA r=8** ⭐ **best MTL reg** | 13.95 | **50.72 ± 4.36** | **25.36** | **Ablation step 4a** |
| B12 | MTL λ=0.5 equal-weight (static) | 11.34 | 50.26 ± 4.34 | 25.35 | Ablation step 2 |
| | *MTL with cross-attention:* | | | | |
| B13 | MTL mtlnet_crossattn | 10.06 | 45.09 ± 5.37 | 20.94 | Ablation step 6 |

**Key observations:**
- **B8a (STL STAN 59.20%) is the region-task ceiling** — the literature-aligned Luo et al. WWW'21 architecture, adapted to our check2HGI substrate (see `research/SOTA_STAN_BASELINE.md`). STAN beats `next_gru` (B8) by +2.26 pp Acc@10 within σ on AL. `next_gru` is kept in the paper as the MTL-architecture's region head per §CH-M4 / C06 framing. No MTL architecture we tested exceeds STL STAN on AL — B11 at 50.72% is the closest MTL result (−8.48 pp gap vs STAN).
- **B5 Markov-1-region is the binding simple floor** (47.01%). Our best STL beats it by +9.93 pp; our best MTL exceeds it by +3.71 pp.
- **B5→B5e context-matched Markov monotonically DEGRADES with k** (47.01 → 32.79). N-gram sparsity outpaces training data beyond k=1 even with backoff. The context-matched (k=9) Markov is 32.79% — **STL STAN beats Markov-9 by +26.41 pp** at the same 9-step context. This isolates the neural contribution as representation-learning, not context-length.
- **B9 (HGI region embeddings through same pipeline): tied with Check2HGI** at the region task (57.02 vs 56.11). Expected — pooling to region erases check-in-level variance. This is CH15.
- **Legacy "Markov-1-POI" (B4) is a degenerate baseline** — top-K fallback inflates variance. B5 (region-level) is the paper-reported floor.

### Florida

| Row | Method | Acc@1 | Acc@10 | MRR | Source |
|:---:|--------|------:|-------:|----:|:------:|
| B14 | Random | 0.02 | 0.21 | 0.19 | theoretical |
| B15 | Majority | 22.25 | 22.25 | 22.25 | P0 |
| B16 | Top-K popular | 22.25 | 33.82 | 25.65 | P0 |
| B17 | **Markov-1-region** (paper floor, best k) | 46.36 ± 0.89 | **65.05 ± 0.93** | 52.37 ± 0.90 | P0 |
| B17a | Markov-2-region (w/ backoff) | 44.47 ± 0.77 | 59.17 ± 0.86 | 49.50 ± 0.80 | P0 |
| B17b | Markov-3-region (w/ backoff) | 43.63 ± 0.75 | 56.65 ± 0.87 | 48.13 ± 0.79 | P0 |
| B17c | Markov-5-region (w/ backoff) | 42.95 ± 0.69 | 54.91 ± 0.74 | 47.07 ± 0.71 | P0 |
| B17d | Markov-7-region (w/ backoff) | 42.74 ± 0.68 | 54.43 ± 0.74 | 46.76 ± 0.71 | P0 |
| B17e | **Markov-9-region** (context-matched to model) | 42.56 ± 0.68 | **54.10 ± 0.72** | 46.53 ± 0.71 | P0 |
| B19 | **STL GRU standalone** ⭐ **ceiling** | **44.49 ± 0.51** | **68.33 ± 0.58** | **52.74 ± 0.45** | **P1 champion** |
| B20 | MTL dselectk + pcgrad (1f×50ep) | 15.43 | **57.05** | 27.49 | FL P2-validate |

---

### Arizona (mid-scale validation, 26K rows, 1540 regions)

| Row | Method | Acc@1 | Acc@10 | MRR | Source |
|:---:|--------|------:|-------:|----:|:------:|
| B-AZ-gru | STL GRU standalone | 23.63 ± 2.04 | 48.88 ± 2.48 | 32.13 ± 2.21 | P1 (AZ confirm) |
| B-AZ-stan | **STL STAN (Luo WWW'21, adapt)** ⭐ | **24.48 ± 2.29** | **52.24 ± 2.38** | **33.70 ± 2.36** | **SOTA note** |

Scale-curve confirmation: **STAN > GRU delta grows with scale** (AL +2.26 pp → AZ +3.36 pp on Acc@10). Consistent with transformer-family being more data-hungry than recurrent networks.

---

**Key observations (CH01 asymmetric):**
- **MTL regresses region by −11.28 pp** on FL (57.05 vs 68.33 STL). Consistent with AL pattern — region is capacity-ceiling-bound regardless of data scale.
- The region GRU standalone (68.33%) is only 3.28 pp above Markov-1-region (65.05%) — FL is a dense-data regime where near-term transitions saturate signal extraction. MTL cannot find additional signal to add.
- **B17→B17e context-matched Markov monotonically DEGRADES with k** (65.05 → 54.10). N-gram sparsity penalty holds even at FL's 127K rows. At k=9 (model's context), STL GRU beats Markov by **+14.23 pp** — again attributable to learned representations, not context-length.

---

## The asymmetric MTL story (CH-M1, paper's primary new claim)

| State | Task | STL | Best MTL | Δ (MTL−STL) | Verdict |
|:-----:|------|----:|---------:|------------:|:--------|
| AL | cat F1 | 38.58 | **38.58** (cross-attn) | **0.00 pp** | **MTL closes gap** |
| AL | reg Acc@10 | 56.94 | 50.72 (MTLoRA r=8) | **−6.22 pp** | MTL capped below STL |
| FL | cat F1 | 63.17 | **64.78** (dselectk+pcgrad) | **+1.61 pp** | **MTL lifts** |
| FL | reg Acc@10 | 68.33 | 57.05 (dselectk+pcgrad) | **−11.28 pp** | MTL capped below STL |

**The pattern is consistent across states:**

- **Category (weaker head task):** MTL matches or exceeds STL **once the architecture allows content-based exchange** (cross-attention) OR **once data is abundant** (FL).
- **Region (strong standalone head):** MTL is capacity-ceiling-bound across all 7 intervention families + both data regimes. The STL ceiling reflects a *head-input saturation* — GRU already extracts all signal from the 9-step region sequence; there is no untapped signal for MTL to transfer to the region head.

---

## Sources

- **B8a STAN (AL + AZ):** `docs/studies/check2hgi/results/P1/region_head_{alabama,arizona}_region_5f_50ep_STAN_*_5f50ep.json`. See `docs/studies/check2hgi/research/SOTA_STAN_BASELINE.md`.
- **A5, A4:** `docs/studies/check2hgi/results/P1_5b/next_category_alabama_{check2hgi,hgi}_5f_50ep_fair.json`
- **A11:** `docs/studies/check2hgi/results/P1_5b/next_category_florida_check2hgi_1f_50ep_fair.json`
- **A8:** `docs/studies/check2hgi/results/P2/ablation_06_crossattn_al_5f50ep.json`
- **A6, B10:** `docs/studies/check2hgi/results/P2/validate_dselectk_pcgrad_gru_al_5f_50ep.json`
- **A7, B11:** `docs/studies/check2hgi/results/P2/ablation_04_mtlora_r8_al_5f50ep.json`
- **A12, B20:** `docs/studies/check2hgi/results/P2/validate_fl_dselectk_pcgrad_gru_1f_50ep.json`
- **B8:** `docs/studies/check2hgi/results/P1/region_head_alabama_region_5f_50ep_E_confirm_gru_region.json`
- **B7:** `docs/studies/check2hgi/results/P1/region_head_alabama_region_5f_50ep_E_confirm_tcn_region.json`
- **B19:** `docs/studies/check2hgi/results/P1/region_head_florida_region_5f_50ep_E_confirm_fl_gru_region.json`
- **B9:** `docs/studies/check2hgi/results/P1/region_head_alabama_region_5f_50ep_P15_hgi_al_tcn.json`
- **B5, B17:** `docs/studies/check2hgi/results/P0/simple_baselines/{alabama,florida}/next_region.json`
- **B12:** `docs/studies/check2hgi/results/P2/ablation_02_1227.json`

---

## TL;DR for the paper's results section

| Claim | AL (dev) | FL (headline) | Evidence strength |
|-------|:--------:|:-------------:|:-----------------:|
| Check2HGI > HGI on category | +18.30 pp (5f, fair) | TBD (need FL HGI STL) | ✅ strong (CH16) |
| Per-task input modality > shared/concat | directional | TBD | ✅ directional (CH03) |
| MTL cross-attn matches STL on cat | 0.00 pp gap | TBD (NS-1 pending) | 🟡 AL-only so far |
| MTL dilutes region regardless of architecture | −6.22 pp best | −11.28 pp | ✅ strong (7-family ablation) |
| MTL lifts category at scale | No lift on 10K | **+1.61 pp on 127K** | ✅ one data point FL |
| Architectural overhead = 5.4 pp of 8-pp gap | (λ=0.0 isolation) | — | ✅ unique to this study |
