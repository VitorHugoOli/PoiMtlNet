# Results Table — Check2HGI MTL Study

> ⚠ **2026-05-02 v11 PAPER CLOSURE — FL §0.1 upgraded to n=20 (both axes paper-grade); all five states now multi-seed on the headline architectural-Δ axis.**
> Cross-state P3 (CA + TX) + multi-seed at AL/AZ/FL/CA landed leak-free per-fold log_T.
> Numbers in pre-2026-05-01 rows used the legacy unseeded log_T and are
> **leak-inflated by 13-27 pp** (state-dependent). The leak-free architectural-Δ
> picture is in [`../archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md`](../archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md) (background provenance only — superseded by the current §0 below);
> paired Wilcoxon JSONs: [`../research/PAPER_CLOSURE_WILCOXON.json`](../research/PAPER_CLOSURE_WILCOXON.json) + [`../research/GAP_FILL_WILCOXON.json`](../research/GAP_FILL_WILCOXON.json) (cat-Δ multi-seed v8) + [`../research/ARCH_DELTA_WILCOXON.json`](../research/ARCH_DELTA_WILCOXON.json) (CA/TX arch-Δ n=20 v10).
>
> **Headline (v11, leak-free, F51 canonical extraction, both tasks):**
>
> | State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat |
> |---|---:|---:|---:|---:|---:|
> | AL | 20 | **−11.04** | **1.9e-06** | **−0.78** (small-significantly negative; magnitude < 2 % relative on a 41 % F1 scale) | **0.036** |
> | AZ | 20 | **−12.27** | **1.9e-06** | **+1.20** | **<1e-04** |
> | FL | 20 | **−7.34** | **1.9e-06** | **+1.40** | **2e-06** |
> | **CA** | **20** | **−9.50** | **2e-06** | **+1.68** | **2e-06** |
> | **TX** | **20** | **−16.59** | **2e-06** | **+1.89** | **2e-06** |
>
> Δ = MTL B9 − STL ceiling (paired Δ from per-fold values). Reg metric: per-fold max `top10_acc_indist` for ep ≥ 5.
> Cat metric: per-fold max unweighted `f1` for ep ≥ 5. **AL: significant in the negative direction at p = 0.036 (n=20 multi-seed); magnitude small, ~1.9 % relative.**
>
> **Recipe selection (B9 vs H3-alt) — current canonical (CA n=20 in v8; TX n=20 in v9):**
>
> | State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat |
> |---|---:|---:|---:|---:|---:|
> | AL | 20 | −0.35 | **1.9e-03** | **−2.22** | **1.9e-06** |
> | AZ | 20 | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** |
> | FL | 25 | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 |
> | **CA** | **20** | **+4.18** | **<1e-04** | **+0.51** | **<1e-04** |
> | **TX** | **20** | **+1.87** | **7.0e-04** | **+0.52** | **2.0e-04** |
>
> **B9 is FL/CA/TX-scale paper-grade (all large-scale states, n=20).** At AL/AZ H3-alt is paper-grade better on cat;
> reg tied. Recipe-selection narrative: **"scale-conditional optimal recipe"**.
> Wilcoxon JSONs: [`../research/PAPER_CLOSURE_RECIPE_WILCOXON.json`](../research/PAPER_CLOSURE_RECIPE_WILCOXON.json) + [`../research/GAP_FILL_WILCOXON.json`](../research/GAP_FILL_WILCOXON.json) (CA/TX n=20 v9).
>
> v6 body (§0 below) landed 2026-05-01; v7 added STL cat multi-seed; v8 landed cat-Δ Wilcoxon (AL/AZ/FL) + CA recipe multi-seed; v9 landed TX recipe multi-seed (n=20, both axes paper-grade); v10 landed CA+TX §0.1 arch-Δ upgraded to n=20 (all four tests p=2e-06); v11 closes FL §0.1 at n=20 (last remaining headline asymmetry — all five states now paper-grade on §0.1). The legacy §1–§5 + Phase-1 / F49 cells are
> preserved unchanged underneath as audit; **the §0 tables are paper-canonical**.

**Last updated:** 2026-05-02 (v11 — §0.1 FL row upgraded to n=20; all five states now paper-grade on §0.1, last headline asymmetry closed). Prior: v10 (§0.1 CA+TX rows n=20, p=2e-06). v9 (§0.4 TX n=20, B9 paper-grade at FL/CA/TX). v8 (CA n=20 + §0.1 Δ_cat p-values). v7 (STL cat multi-seed means). v6 (§0 initial).

---

## 0 · Paper-headline tables (v11, leak-free canon with 2026-05-02 upgrades)

These tables supersede §1–§5 below for paper drafting. All numbers are seed=42 leak-free under per-fold log_T (`region_transition_log_seed42_fold{N}.pt`) on `StratifiedGroupKFold(groups=userid, seed=42)` under sklearn 1.8.0; 5 folds × 50 epochs. Cat metric: per-fold max F1 for ep ≥ 5. Reg metric: per-fold max `top10_acc_indist` (MTL) or `top10_acc` (STL) for ep ≥ 5 — the F51 canonical extraction. Multi-seed aggregations are pooled across paired (seed, fold) tuples.

**MTL recipe:** B9 = `mtlnet_crossattn + static_weight(cat=0.75) + next_gru (cat) + next_stan_flow (reg) + per-head LR (cat=1e-3 / reg=3e-3 / shared=1e-3) + cosine + alt-SGD + α-no-WD + per-fold log_T`. H3-alt = B9 minus alt-SGD, cosine, α-no-WD; replace with `--scheduler constant`. **Recipe is scale-conditional** — see §0.4.

### 0.1 · Five-state architectural-Δ (MTL B9 vs matched-head STL ceiling)

| State | n_regions | MTL B9 reg Acc@10 | STL `next_stan_flow` Acc@10 | **Δ_reg pp** | p_reg | MTL B9 cat F1 | STL `next_gru` F1 | **Δ_cat pp** | p_cat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **AL** (n=20) | 1,109 | 50.17 ± 0.24 | 61.21 ± 0.18 | **−11.04** | **1.9e-06** | 40.57 ± 0.24 | 41.35 ± 0.17 (n=4 seeds) | **−0.78** (small-significantly negative; magnitude ~1.9 % relative) | **0.036** |
| **AZ** (n=20) | 1,547 | 40.78 ± 0.07 | 53.06 ± 0.15 | **−12.27** | **1.9e-06** | 45.10 ± 0.19 | 43.90 ± 0.17 (n=4 seeds) | **+1.20** | **<1e-04** |
| **FL** (n=20) | 4,703 | 63.27 ± 0.10 | 70.62 ± 0.09 | **−7.34** | **1.9e-06** | 68.56 ± 0.79 | 67.16 ± 0.13 | **+1.40** | **2e-06** |
| **CA** (n=20) | 8,501 | 47.35 ± 0.11 | 56.85 ± 0.09 | **−9.50** | **2e-06** | 64.07 ± 0.14 | 62.39 ± 0.13 | **+1.68** | **2e-06** |
| **TX** (n=20) | 6,553 | 42.84 ± 0.14 | 59.44 ± 0.09 | **−16.59** | **2e-06** | 65.00 ± 0.11 | 63.11 ± 0.13 | **+1.89** | **2e-06** |

n = paired Wilcoxon sample size: all five states = 4 seeds × 5 folds (multi-seed pooled, n=20). All states pooled across seeds {0, 1, 7, 100}.

**v11 update (2026-05-02):** FL §0.1 row upgraded from n=5 (seed=42 only) to n=20 (seeds {0,1,7,100}, canonical B9 recipe with `--cat-head next_gru --reg-head next_getnext_hard`). FL now paper-grade significant on both axes (p=1.9e-06 reg, p=2e-06 cat). The last remaining headline asymmetry is closed; all five states are now multi-seed on §0.1. Wilcoxon JSON: [`../research/FL_CAT_DELTA_WILCOXON.json`](../research/FL_CAT_DELTA_WILCOXON.json).

**v10 update (2026-05-02):** CA and TX §0.1 rows upgraded from n=5 (seed=0 only) to n=20 (seeds {0,1,7,100}). Both CA and TX now paper-grade significant on both axes (p=2e-06 for all four tests). CA: Δ_reg = −9.50 pp, Δ_cat = +1.68 pp. TX: Δ_reg = −16.59 pp, Δ_cat = +1.89 pp. Wilcoxon JSON: [`../research/ARCH_DELTA_WILCOXON.json`](../research/ARCH_DELTA_WILCOXON.json).

**v8 update (2026-05-01 PM):** STL `next_gru` cat F1 for AL/AZ/FL refreshed from multi-seed runs {0,1,7,100}. Δ_cat p-values now computed from paired Wilcoxon MTL B9 cat vs multi-seed STL cat (n=20 pairs at AL/AZ; n=5 single-seed at FL): **AL p = 0.036 (small-significantly negative at Δ = −0.78 pp; magnitude ~1.9 % relative on a 41 % F1 scale)**; AZ p < 1e-04 (significantly positive at +1.20 pp); FL p = 0.0625 (sign-consistent positive at +1.52 pp, n = 5 ceiling). CA recipe-selection multi-seed (n=20) also landed in v8 — see §0.4. Wilcoxon JSON: [`../research/GAP_FILL_WILCOXON.json`](../research/GAP_FILL_WILCOXON.json).

**Headline (the classic MTL tradeoff, sign-consistent on reg across all 5 states, now paper-grade at all five):**
- **Reg side:** MTL B9 < STL by 7–17 pp at every state (sign-consistent). Significant (p≤1.9e-06) at all five states (n=20).
- **Cat side:** MTL B9 > STL at four of five states, significant (p≤2e-06) at AZ/CA/TX/FL (n=20). **AL is small-significantly negative** (Δ = −0.78 pp, p = 0.036 across n = 20 multi-seed fold-pairs; magnitude small at ~1.9 % relative on a 41 % F1 scale).
- The cross-attention architecture's expressiveness gets spent on cat-helps-cat (joint training transfers signal to the easier 7-class task at most states) at the cost of the harder ~1k–9k-class region task that already has its own `α·log_T` graph prior to learn from in `next_stan_flow`.

Source: this file (canonical). Wilcoxon JSONs: [`../research/PAPER_CLOSURE_WILCOXON.json`](../research/PAPER_CLOSURE_WILCOXON.json) + [`../research/GAP_FILL_WILCOXON.json`](../research/GAP_FILL_WILCOXON.json) (cat-Δ multi-seed v8) + [`../research/ARCH_DELTA_WILCOXON.json`](../research/ARCH_DELTA_WILCOXON.json) (CA/TX arch-Δ n=20 v10). Background provenance (superseded): [`../archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md §4a`](../archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md).

### 0.2 · Joint Δm scoreboard (CH22, leak-free 2026-05-01)

`PRIMARY = cat F1 + reg MRR`; `SECONDARY = cat F1 + reg Acc@10`. Per-fold paired Wilcoxon at seed=42.

| State | n_pairs | Δm-MRR (%) | n+/n− | p_greater | p_two | Δm-Acc@10 (%) | n+/n− | p_two |
|---|--:|---:|:-:|---:|---:|---:|:-:|---:|
| AL | 5  | **−24.84** | 0/5 | 1.0000 | **0.0625** | **−22.41** | 0/5 | **0.0625** |
| AZ | 5  | **−12.79** | 1/4 | 0.9688 | 0.1250 | **−14.53** | 0/5 | **0.0625** |
| **FL** | **25** | **+2.33** | **25/0** | **2.98e-08** | **5.96e-08** | **−1.12** | 4/21 | **3.20e-05** |
| CA | 5  | −1.61 | 1/4 | 0.9375 | 0.1875 | −6.85 | 0/5 | **0.0625** |
| TX | 5  | −4.63 | 0/5 | 1.0000 | **0.0625** | **−11.60** | 0/5 | **0.0625** |

FL multi-seed = 5 seeds {42, 0, 1, 7, 100} × 5 folds; AL/AZ/CA/TX = single seed (=42), n=5 ceiling. FL multi-seed numerical artefact: [`paired_tests/F50_T0_delta_m_FL_multiseed.json`](paired_tests/F50_T0_delta_m_FL_multiseed.json). FL per-seed Δm-MRR = {+2.17, +2.46, +2.27, +2.33, +2.43}% (range 0.29 pp, 25/25 fold-pairs positive). FL per-seed Δm-Acc@10 = {−1.31, −1.00, −1.09, −1.17, −1.04}% (4/25 fold-pairs positive).

**Verdict:** MTL is Pareto-negative on Δm at 4/5 states; **FL on the MRR axis is the lone Pareto-positive cell, now at p ≈ 3×10⁻⁸** (was n=5 ceiling 0.0312 before multi-seed extension). The MRR-vs-Acc@10 split at FL is paper-grade significant in both directions — Δm-MRR positive at p < 1e-7, Δm-Acc@10 negative at p ≈ 3e-5. Reframes the prior 2026-04-28 leaky scoreboard which had inverted signs at AL/AZ/FL. Source: [`../research/F50_DELTA_M_FINDINGS_LEAKFREE.md`](../research/F50_DELTA_M_FINDINGS_LEAKFREE.md). Numerical artefact: [`paired_tests/F50_T0_delta_m_leakfree.json`](paired_tests/F50_T0_delta_m_leakfree.json).

### 0.3 · Substrate axis (CH16 + CH18-cat) — leak-free 5-state survey

Cat STL `next_gru` matched-head, 5f × 50ep, seed=42. Source: [`../FINAL_SURVEY.md §2`](../FINAL_SURVEY.md).
C2HGI F1 for AL/AZ/FL updated to multi-seed mean ± seed σ (seeds {0,1,7,100}, 2026-05-01 PM) — substrate Δ is unaffected; Wilcoxon p-values retain seed=42 basis.

| State | C2HGI cat F1 | HGI cat F1 | **Δ pp** | Wilcoxon p_greater | Pos/Neg |
|---|---:|---:|---:|---:|:-:|
| AL | **41.35 ± 0.17** (multi-seed) | 25.26 ± 1.18 | **+15.50** | **0.0312** | 5/0 |
| AZ | **43.90 ± 0.17** (multi-seed) | 28.69 ± 0.79 | **+14.52** | **0.0312** | 5/0 |
| FL | **63.43 ± 0.98** | 34.41 ± 1.05 | **+29.02** | **0.0312** | 5/0 |
| CA | **59.94 ± 0.59** | 31.13 ± 1.04 | **+28.81** | **0.0312** | 5/0 |
| TX | **60.24 ± 1.84** | 31.89 ± 0.55 | **+28.34** | **0.0312** | 5/0 |

**CH16 confirmed at 5/5 states with paper-grade significance; Δ scales broadly with data — two-band: ~15 pp at small states (AL/AZ) and ~28–29 pp at large states (FL/CA/TX). Within the large-state band the ordering is FL +29.02 ≥ CA +28.81 ≥ TX +28.34 — directional but not strictly monotone in state size.**

Reg STL `next_stan_flow` matched-head, leak-free per-fold log_T, 5f × 50ep, seed=42:

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p_greater | TOST δ=2pp | TOST δ=3pp |
|---|---:|---:|---:|---:|:-:|:-:|
| AL | 59.15 ± 3.48 | **61.86 ± 3.29** | −2.71 | 1.0000 | ✗ | ✗ |
| AZ | 50.24 ± 2.51 | **53.37 ± 2.55** | −3.13 | 1.0000 | ✗ | ✗ |
| FL | 69.22 ± 0.52 | **71.34 ± 0.64** | −2.12 | 1.0000 | ✗ | ✓ |
| CA | 55.92 ± 1.20 | **57.77 ± 1.12** | −1.85 | 1.0000 | ✓ | ✓ |
| TX | 58.89 ± 1.28 | **60.47 ± 1.26** | −1.59 | 1.0000 | ✓ | ✓ |

**CH15 reframing rejected at AL/AZ/FL** (HGI nominally above C2HGI by 1.6–3.1 pp under TOST δ=2pp); tied at CA/TX. **Sign-flipped vs Phase-2 leaky reference at every state** — the leaky CH15 sign came from substrate-asymmetric F44 leak (FINAL_SURVEY §6). Per-visit context (Check2HGI) is the load-bearing substrate for **next_category**; for **next_region**, POI-stable HGI is at parity (CA/TX) or marginally ahead (AL/AZ/FL).

### 0.4 · Recipe selection (B9 vs H3-alt) — scale-conditional

| State | n_pairs | Δ_reg pp | p_reg | Δ_cat pp | p_cat | Verdict |
|---|--:|---:|---:|---:|---:|---|
| AL | 20 | −0.35 | **1.9e-03** | **−2.22** | **1.9e-06** | **H3-alt > B9 on cat; reg tied** |
| AZ | 20 | −0.09 | 0.23 (n.s.) | **−0.96** | **7.1e-04** | **H3-alt > B9 on cat; reg tied** |
| FL | 25 | **+3.48** | **3.0e-08** | +0.42 | 1.3e-05 | **B9 > H3-alt on both** (F51) |
| CA | **20** | **+4.18** | **<1e-04** | **+0.51** | **<1e-04** | **B9 > H3-alt, both tasks significant** |
| **TX** | **20** | **+1.87** | **7.0e-04** | **+0.52** | **2.0e-04** | **B9 > H3-alt, both tasks significant** |

**v9 update (2026-05-02):** TX row updated from n=5 (seed=42) to n=20 (seeds {0,1,7,100}) — B9 vs H3-alt now paper-grade significant at TX on both tasks (reg +1.87 pp p=7e-04, cat +0.52 pp p=2e-04). **B9 is now paper-grade at FL/CA/TX (large-scale states); H3-alt remains better at AL/AZ (small-scale).** Wilcoxon JSON: [`../research/GAP_FILL_WILCOXON.json`](../research/GAP_FILL_WILCOXON.json).

**B9 is FL/CA/TX-scale paper-grade, NOT universal.** B9's three additions over H3-alt (alt-SGD + cosine + α-no-WD) hurt cat at AL/AZ. Mechanism: B9 targets the reg-saturation problem at large transition graphs (D5); at smaller transition graphs the saturation is less severe AND alt-SGD's per-step temporal gradient separation costs cat-side signal small states can't afford to lose. **Paper recipe-selection narrative:** *"B9 is paper-grade at FL/CA/TX (all large-scale states, n=20 multi-seed); H3-alt remains the better recipe at small scale (AL/AZ); the optimal MTL recipe is scale-conditional."* Source: this file (canonical). Wilcoxon JSONs: [`../research/PAPER_CLOSURE_RECIPE_WILCOXON.json`](../research/PAPER_CLOSURE_RECIPE_WILCOXON.json) + [`../research/GAP_FILL_WILCOXON.json`](../research/GAP_FILL_WILCOXON.json) (CA/TX n=20 v9). Background provenance (superseded): [`../archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md §4a-bis`](../archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md).

### 0.5 · External literature baselines — `next_region` (Acc@10)

Source: [`../GAP_A_CLOSURE_20260430.md`](../GAP_A_CLOSURE_20260430.md). Faithful axis closed at AL/AZ/FL; CA/TX faithful runs scoped out per H100 budget (~5–7 h/fold STAN at CA, ~75–120 h/state REHDM).

| Baseline | Variant | AL | AZ | FL | CA | TX |
|---|---|---:|---:|---:|---:|---:|
| Markov-1-region (floor) | — | 47.01 ± 3.55 | 42.96 ± 2.05 | 65.05 ± 0.93 | 52.09 ± 0.80 | 54.94 ± 0.46 |
| **STAN** | `faithful` | 34.46 ± 3.88 | 38.96 ± 3.41 | 65.36 ± 0.69 | ⚪ | ⚪ |
| **STAN** | `stl_check2hgi` | 59.20 ± 3.62 | 52.24 ± 2.38 | 72.62 ± 0.52 | 58.82 ± 1.04 | 61.35 ± 0.36 |
| **STAN** | `stl_hgi` | **62.88 ± 3.90** | **54.86 ± 2.84** | **73.58 ± 0.43** | **60.45 ± 0.97** | **62.70 ± 0.37** |
| **ReHDM** ‡ | `faithful` | **66.06 ± 0.98** | 54.65 ± 0.77 | 65.68 ± 0.26 | ⚪ | ⚪ |
| ReHDM | `stl_check2hgi` | 26.22 ± 1.58 | 23.24 ± 1.27 | 38.74 ± 0.49 | ⚪ | ⚪ |
| ReHDM | `stl_hgi` | 42.78 ± 2.82 | 34.00 ± 3.02 | 54.49 ± 0.32 | ⚪ | ⚪ |

⚪ = scoped out (see GAP_A_CLOSURE). ‡ ReHDM faithful uses paper protocol (chronological 80/10/10 + 24h sessions + 5 seeds, b=128 + 4× LR scaling validated against b=32 paper baseline within 1σ). Substrate-axis covers 5 states (controlled architecture); faithful axis covers 3 states (sufficient for trend confirmation).

### 0.6 · External literature baselines — `next_category` (macro-F1)

Source: [`../baselines/next_category/comparison.md`](../baselines/next_category/comparison.md) + GAP_A_CLOSURE.

| Baseline | AL | AZ | FL | CA | TX |
|---|---:|---:|---:|---:|---:|
| Majority class (floor) | 34.20 | — | — | — | — |
| Markov-1-POI (floor) | ≈31.7 | — | ≈37.2 | — | — |
| **POI-RGNN** faithful | 31.78–34.5 (state-level range) | — | 34.49 | 31.78 | 33.03 |
| **MHA+PE** faithful | (closed all 5 states) | (closed) | (closed) | (closed) | (closed) |
| Substrate linear probe — C2HGI / HGI / Δ | 30.84 / 18.70 / **+12.14** | 34.12 / 22.54 / **+11.58** | 40.77 / 25.74 / **+15.03** | 37.45 / 21.32 / **+16.13** | 38.38 / 22.33 / **+16.06** |
| C2HGI cat — `next_gru` STL (matched-head) | 40.76 ± 1.68 | 43.21 ± 0.87 | 63.43 ± 0.98 | 59.94 ± 0.59 | 60.24 ± 1.84 |

C2HGI **matched-head STL** lifts cat F1 by **+27–29 pp over our faithful POI-RGNN reproduction** at FL/CA/TX; the **MTL** row widens this external gap to roughly **+32–34 pp**. MHA+PE faithful per-state JSONs at `baselines/next_category/results/<state>.json`.

---

> ⚠ **HISTORICAL AUDIT — DO NOT CITE FOR PAPER DRAFTING.**
> Sections §1–§5 below are **superseded by §0 above** and preserved verbatim from 2026-04-26..27 as audit-only.
> Several `next_region` numbers in §1–§3 used the legacy unseeded `region_transition_log.pt`
> and are leak-inflated by 13–27 pp (state-dependent; substrate-asymmetric, hurting
> reproducibility of CH15/CH18-reg). **Use §0 above for paper drafting.**
> The cat-side numbers in §1–§5 are mostly leak-free by construction (cat heads
> don't read log_T); the F49 3-way decomposition cells in the late-file section
> were refuted by F37 + paper-closure findings (the leak-free architectural-Δ reading is in `archive/post_paper_closure_2026-05-01/OBJECTIVES_STATUS_TABLE.md §2.5`, archived in the 2026-05-01 cleanup).

---

> ⚠ **The "F48-H3-alt champion" framing below was SUPERSEDED by the leak-free 2026-05-01 paper closure.** Under leak-free measurement, the AL "+6.25 pp MTL > STL" claim is a leak artefact and MTL trails STL on `next_region` at every state by 7–17 pp (see §0.1 for the current canonical view). The H3-alt vs B9 comparison is a *recipe-selection finding*, not a champion-vs-predecessor relation; B9 is paper-grade at FL/CA/TX (§0.4); H3-alt is the small-state recipe. **Read §0 above for the current paper-canonical view.**

**Champion candidate (2026-04-26, SUPERSEDED — see banner above):** **F48-H3-alt** = B3 architecture + per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3`, `--scheduler constant`). The original 2026-04-26 framing claimed F48-H3-alt closed the F21c STL gap (CH18 Tier B → A): AL exceeded STL by +6.25 pp; AZ closed 75%; FL validated at 5-fold. **All these "MTL > STL on reg" claims are leak artefacts** under leak-free measurement (see current §0 above). Audit / derivation pointers (also superseded): [`../NORTH_STAR.md`](../NORTH_STAR.md), [`../MTL_ARCHITECTURE_JOURNEY.md`](../MTL_ARCHITECTURE_JOURNEY.md), [`../research/F48_H3_PER_HEAD_LR_FINDINGS.md`](../research/F48_H3_PER_HEAD_LR_FINDINGS.md).

**Predecessor (2026-04-24, SUPERSEDED — kept as historical comparand):** **B3** = `mtlnet_crossattn + static_weight(cat=0.75) + next_gru (cat) + next_stan_flow (reg, alias of legacy next_getnext_hard)`, OneCycleLR max=0.003, 50ep.

**Common protocol** (unless noted per row): user-disjoint `StratifiedGroupKFold(groups=userid)`, 5 folds × 50 epochs, seed 42, AdamW (lr=1e-4 → OneCycleLR max_lr=0.003 for predecessor B3 / per-head constant for H3-alt), batch 2048 (1024 for FL H3-alt to avoid MPS OOM), `gradient_accumulation_steps=1`. Reg metrics are `*_indist` (restricted to regions seen in training set of the fold) where applicable.

**Legend:**
- ✅ best-in-table for that metric · ⭐ champion config · 🔴 pending · 🟡 known limitation
- `(n=1)` = single-fold; all others are 5-fold mean ± std unless noted
- `(SUPERSEDED)` = value replaced by a later run
- **Cat F1** = macro-F1 on 7 categories. **Reg Acc@10** = `top10_acc_indist`. **Reg Acc@5** = `top5_acc_indist`. **Reg MRR** = `mrr_indist`. **Reg F1** = macro-F1 on ~1K–4.7K regions.

---

## 1 · Ablation states (AL + AZ + FL-1f) — mechanism + validation

These three blocks are the study's ablation evidence (`PAPER_STRUCTURE.md §2.1`). They are not the paper's headline numbers but justify the method's behavior.

### 1.1 Alabama (AL, 10 K rows, 1,109 regions, 5f × 50ep unless noted)

**Baselines — next-category**

| Method | cat F1 | cat Acc@1 | Source |
|---|---:|---:|---|
| Random (1/7) | ≈14.3 | ≈14.3 | theoretical |
| Majority | — | 34.20 | P0 |
| Markov-1-POI | ≈31.7 | ≈32 | P0 |
| POI-RGNN (state-level range reference) | 31.8–34.5 | — | `docs/baselines/BASELINE.md` (Capanema et al.) |

**Baselines — next-region**

| Method | Acc@1 | **Acc@10** | Acc@5 | MRR | F1 | Source |
|---|---:|---:|---:|---:|---:|---|
| Random | 0.09 | 0.90 | — | — | — | theoretical |
| Majority | 1.97 | 1.97 | — | — | — | P0 |
| Top-K popular | 1.97 | 14.67 | — | — | — | P0 |
| **Markov-1-region** (simple floor) | 25.40 ± 2.73 | **47.01 ± 3.55** | — | 32.17 ± 2.90 | — | P0 |
| Markov-5-region w/ backoff | 20.80 ± 2.65 | 33.42 ± 2.16 | — | 24.99 ± 2.53 | — | P0 |
| Markov-9-region (ctx-matched) | 20.49 ± 2.57 | 32.79 ± 1.92 | — | 24.54 ± 2.36 | — | P0 |

**STL — matched-class & literature-aligned**

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| STL Check2HGI cat (`next_mtl`) | **38.58 ± 1.23** | — | — | — | — | — | P1_5b refair |
| STL HGI cat (substrate ablation, CH16) | 20.29 ± 1.34 | — | — | — | — | — | P1_5b refair |
| STL GRU reg | — | 23.60 ± 1.86 | 56.94 ± 4.01 | — | 34.57 ± 2.34 | — | P1 |
| STL TCN-residual reg | — | 21.76 ± 2.35 | 56.11 ± 4.02 | — | 32.93 | — | P1 |
| **STL STAN reg** (Luo WWW'21 adapt) | — | 24.64 ± 1.38 | **59.20 ± 3.62** | — | 36.10 ± 1.96 | 24.64 ± 1.38 | P1 SOTA |
| STL HGI reg (substrate ablation) | — | ? | 57.02 ± 2.92 | — | 33.14 ± 1.87 | — | P1.5 (tied with Check2HGI reg) |
| **STL STAN-Flow reg (matched-head)** — **F21c** ⭐ | — | 24.07 ± 1.94 | **68.37 ± 2.66** | 53.62 ± 3.02 | 41.17 ± 2.28 | 11.91 ± 0.86 | **F21c 2026-04-24 — STL ceiling for H3-alt comparison** |

**MTL — all variants**

| # | Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B-M1 | dselectk + pcgrad + GRU | 36.08 ± 1.96 | 13.31 | 48.88 ± 6.26 | — | 24.43 | — | P2 prior champion |
| B-M2 | ~~dselectk + MTLoRA r=8 + GRU~~ | — | — | ~~50.72 ± 4.36~~ | — | — | — | SUPERSEDED (MTL_PARAM_PARTITION_BUG) |
| B-M2b | dselectk + MTLoRA r=8 + GRU (post-fix) | 36.53 ± 1.24 | 17.48 ± 1.35 | 53.71 ± 3.80 | 40.54 ± 3.17 | 29.60 ± 2.01 | 8.31 ± 1.02 | `results/P5_bugfix/a7_mtlora_r8_al_5f50ep_postfix_pcgrad.json` |
| B-M3 | cross-attn + static λ=0.5 + GRU | ? | 11.34 | 50.26 ± 4.34 | — | 25.35 | — | P2 ablation step 2 |
| B-M4 | cross-attn + pcgrad + GRU | 38.58 ± 0.98 | 10.06 | 45.09 ± 5.37 | — | 20.94 | — | P2 ablation step 6 |
| B-M5 | cross-attn + pcgrad + STAN d=128 | 39.07 ± 1.18 | 12.48 ± 1.44 | 50.27 ± 4.47 | — | 24.16 ± 2.25 | — | P8 MTL-STAN |
| B-M6 | cross-attn + pcgrad + STAN d=256 | 38.11 ± 1.11 | 13.86 ± 3.43 | 51.60 ± 10.09 | — | 25.69 ± 5.34 | — | P8 high-σ |
| B-M6a | + ALiBi on STAN d=256 | ? | 14.09 ± 3.71 | 51.64 ± 8.92 | — | 25.69 ± 5.40 | — | P8 ALiBi null |
| B-M6b | **cross-attn + pcgrad + GETNext-SOFT d=256** | 38.56 ± 1.45 | 15.72 ± 2.74 | 56.49 ± 4.25 | 43.40 ± 4.60 | 28.93 ± 3.20 | 8.66 ± 1.20 | P8 soft champion (prior north-star) |
| B-M6c | + static (cat=0.50) | 8.65 ± 0.56 (F1_macro) | 15.60 ± 2.06 | 56.45 ± 4.34 | 43.08 ± 4.42 | 28.88 ± 2.55 | 8.52 ± 0.64 | attribution test: pcgrad ≈ static on soft |
| B-M6d | + ALiBi on soft | 9.05 ± 1.02 | 16.37 ± 2.13 | 57.27 ± 4.17 | 43.87 ± 3.84 | 29.52 ± 2.63 | 9.05 ± 1.02 | B7 AL — optional stabiliser |
| B-M6e | cross-attn + pcgrad + GETNext-HARD d=256 | 38.50 ± 1.56 | 15.03 ± 3.04 | 57.96 ± 5.09 | 44.22 ± 5.58 | 28.93 ± 3.88 | 9.47 ± 0.71 | B5 AL hard — +1.47 over soft (tied within σ) |
| B-M_B3 (pre-F27) | cross-attn + static (cat=0.75) + GETNext-HARD d=256 + NextHeadMTL cat | 39.28 ± 0.80 | ? | 56.33 ± 8.16 | 42.81 ± 7.89 | 28.55 ± 5.33 | 9.43 ± 0.71 | B3 validation AL (F18, 2026-04-23). Superseded by post-F27 row below. |
| **B3 (post-F27)** | cross-attn + static(0.75) + **next_gru cat** + GETNext-HARD reg (OneCycleLR 50ep) | **42.71 ± 1.37** | ? | 59.60 ± 4.09 | ? | 28.55 ± 5.33 | ? | **B3 post-cat-head-swap (F31, 2026-04-24)** — predecessor champion. +3.43 cat F1 over pre-F27 row. |
| **MTL-H3-alt** | ⭐ B3 + **per-head LR (cat=1e-3, reg=3e-3, shared=1e-3, constant)** | **42.22 ± 1.00** | 34.93 ± 3.26 | ✅ **74.62 ± 3.11** | 61.65 ± 4.06 | 47.49 ± 3.29 | ? | **F48-H3-alt 2026-04-26** — **MTL EXCEEDS STL F21c ceiling by +6.25 pp on reg** while preserving cat F1 within σ of B3. CH18 Tier B → A. |
| — | F40 scheduled-static (cat_weight 0.75 → 0.25 ramp, OneCycleLR) | 42.63 ± 1.26 | 16.61 ± 2.20 | 60.81 ± 3.10 | 47.33 ± 3.58 | 30.85 ± 2.64 | ? | **negative control (2026-04-26)** — cat preserved, reg only +1.21 pp over B3 (Pareto fails). Refutes loss-side mechanism. |
| — | F48-H1 single LR const 1e-3, 150ep | 40.99 ± 1.80 | ? | 61.43 ± 9.60 | ? | ? | ? | gentle const refuted — reg-best ep collapses [4..10], σ inflated. |
| — | F48-H2 warmup_constant (50ep ramp + 100ep plateau @ 3e-3, single LR) | 41.35 ± 0.78 | 17.35 ± 1.58 | 57.84 ± 4.48 | 44.32 ± 3.26 | 30.17 ± 1.69 | ? | **negative control (2026-04-26)** — cat survives warmup but reg DROPS below B3 (-1.76 pp). Cat-vs-reg compete for shared cross-attn at plateau LR. |
| — | F45 single LR const 3e-3, 150ep | **10.44 ± 0.04 💀** | ? | **74.20 ± 2.95** | ? | ? | ? | breakthrough — proves reg arch CAN exceed STL but cat collapses. Mechanism source for H3-alt. |
| — | F48-H3 per-head (sh=3e-3, instead of 1e-3) | 11.53 ± 1.63 💀 | 34.93 ± 3.26 | 74.24 ± 2.58 | 61.64 ± 4.01 | 47.41 ± 3.33 | ? | reproduces F45 — confirms shared cross-attn at 3e-3 destabilises cat path. Refutes "throttle cat encoder alone" hypothesis. |
| — | MTLoRA rank sweep r=16 / r=32 (post-fix) | — | 15.83 / 17.01 | 51.62 / 53.28 | — | 27.78 / 29.24 | — | rank-insensitive — `P5_bugfix` |
| — | AdaShare (post-fix) | — | 10.66 ± 3.76 | 44.51 ± 6.87 | — | 21.62 ± 4.68 | — | below Markov — not competitive |

### 1.2 Arizona (AZ, 26 K rows, 1,547 regions, 5f × 50ep unless noted)

**Baselines**

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR |
|---|---:|---:|---:|---:|---:|
| Majority | — | 7.43 | 7.43 ± 0.70 | — | — |
| Top-K popular | — | 7.43 | 20.82 ± 1.28 | — | — |
| **Markov-1-region** | — | 23.98 ± 1.13 | **42.96 ± 2.05** | — | — |
| Markov-9-region (ctx-matched) | — | — | 33.38 ± 1.33 | — | — |

**STL**

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| STL Check2HGI cat (`next_mtl`) | **42.08 ± 0.89** | — | — | — | — | — | P1_5b |
| STL HGI cat (CH16 extension) | 🔴 F3 pending | — | — | — | — | — | — |
| STL GRU reg | — | 23.63 ± 2.04 | 48.88 ± 2.48 | — | 32.13 ± 2.21 | — | P1 |
| **STL STAN reg** | — | 24.48 ± 2.29 | 52.24 ± 2.38 | 40.41 ± ? | 33.70 ± 2.36 | 24.48 ± 2.29 | P1 SOTA |
| **STL STAN-Flow reg (matched-head)** — **F21c** ⭐ | — | 25.13 ± 2.07 | **66.74 ± 2.11** | 52.18 ± 2.20 | 41.15 ± 2.13 | 12.28 ± 0.91 | **F21c 2026-04-24 — STL ceiling for H3-alt comparison** |

**MTL**

| # | Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B-M7 | cross-attn + pcgrad + GRU | 43.13 ± 0.55 | 13.20 ± 1.99 | 41.07 ± 3.46 | — | 22.49 ± 2.49 | 13.20 ± 1.99 | P2 az1 |
| B-M8 | + STAN d=128 | 42.64 ± 0.26 | 9.79 ± 1.98 | 37.47 ± 4.01 | — | 18.53 ± 2.54 | — | MTL-STAN d=128 bottleneck |
| B-M9 | + STAN d=256 | 42.74 ± 0.45 | 11.53 ± 2.11 | 41.04 ± 4.55 | — | 20.93 ± 2.86 | — | P8 hp-tuned |
| B-M9a | + STAN d=256 + ALiBi | 42.74 ± 0.45 | 11.24 ± 1.41 | 41.04 ± 3.26 | — | 20.79 ± 2.03 | — | null on mean, −28% σ |
| B-M9b | **soft GETNext d=256** | 42.82 ± 0.96 | 12.63 ± 1.79 | 46.66 ± 3.62 | 35.70 ± 3.38 | 23.81 ± 2.30 | 6.93 ± 0.68 | P8 AZ soft |
| B-M9c | + static (cat=0.50) | 7.30 ± 0.46 (F1_macro) | 12.79 ± 1.98 | 47.32 ± 3.02 | 36.16 ± 3.17 | 24.16 ± 2.27 | 7.13 ± 0.54 | attribution: pcgrad ≈ static on soft |
| B-M9d | cross-attn + pcgrad + GETNext-HARD | 42.22 ± 0.53 | 14.55 ± 2.53 | 53.25 ± 3.44 | 40.06 ± 3.36 | 26.89 ± 2.62 | 8.95 ± 0.52 | B5 AZ hard — **+6.59 pp Acc@10 over soft**, all 4 region metrics p=0.0312 (F1 Wilcoxon) |
| B-M_B3 (pre-F27) | cross-attn + static(0.75) + GETNext-HARD + NextHeadMTL cat | 43.62 ± 0.74 | 14.27 ± 2.53 | 52.76 ± 3.92 | 39.68 ± 3.61 | 26.40 ± 2.45 | 9.17 | B3 validation AZ (F19, 2026-04-23). Wilcoxon p=0.0312 on cat F1 vs STL. Superseded by post-F27 row. |
| **B3 (post-F27)** | cross-attn + static(0.75) + **next_gru cat** + GETNext-HARD reg (OneCycleLR 50ep) | **45.81 ± 1.30** | 49.30 ± 0.67 | 53.82 ± 3.11 | 40.54 ± 3.40 | 27.66 ± 2.41 | ? | **F31 post-cat-head-swap (2026-04-24)** — predecessor champion. +2.19 cat F1 over pre-F27 row. |
| **MTL-H3-alt** | ⭐ B3 + **per-head LR (cat=1e-3, reg=3e-3, shared=1e-3, constant)** | **45.11 ± 0.32** | 32.42 ± 2.88 | **63.45 ± 2.49** | 51.96 ± 3.41 | 42.15 ± 3.02 | ? | **F48-H3-alt 2026-04-26** — closes 75% of B3-vs-STL gap (53.82 → 66.74) while preserving cat F1. CH18 Tier B → A. |
| — | F40 scheduled-static (cat_weight 0.75 → 0.25 ramp, OneCycleLR) | 44.98 ± 1.05 | 15.49 ± 1.82 | 54.39 ± 3.15 | 41.70 ± 3.24 | 28.04 ± 2.17 | ? | **negative control (2026-04-26)** — cat preserved, reg only +0.57 pp over B3 (Pareto fails). |
| — | F48-H1 single LR const 1e-3, 150ep | 45.34 ± 0.84 | ? | 50.68 ± 6.89 | ? | ? | ? | gentle const refuted — reg-best ep collapses [7..10], σ inflated. |
| — | F48-H2 warmup_constant (50ep ramp + 100ep plateau @ 3e-3, single LR) | 44.45 ± 0.54 | 14.63 ± 1.89 | 48.91 ± 5.12 | 37.83 ± 4.29 | 25.81 ± 2.65 | ? | **negative control (2026-04-26)** — cat preserved but reg DROPS by 4.91 pp vs B3. Cat-vs-reg compete for shared cross-attn. |
| — | F45 single LR const 3e-3, 150ep | **12.23 ± 0.16 💀** | ? | **63.34 ± 2.46** | ? | ? | ? | breakthrough — proves reg arch gains; cat collapses. |
| — | F48-H3 per-head (sh=3e-3, instead of 1e-3) | 19.61 ± 13.34 💀 | ? | 62.04 ± 1.90 | ? | ? | ? | reproduces F45 — confirms shared cross-attn at 3e-3 destabilises cat path. |
| — | MTLoRA r=8 pcgrad (post-fix) | — | 11.31 ± 2.90 | 39.51 ± 3.83 | — | 20.95 ± 2.96 | — | `results/P5_bugfix/az2_mtlora_r8_fairlr_5f50ep_postfix.json` |

### 1.3 Florida (FL, 127 K rows, 4,702 regions)

#### FL 5-fold

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Markov-1-region** (baseline) | — | 46.36 ± 0.89 | 65.05 ± 0.93 | — | 52.37 ± 0.90 | P0 |
| Markov-5-region | — | 42.95 ± 0.69 | 54.91 ± 0.74 | — | 47.07 ± 0.71 | P0 |
| Markov-9-region | — | 42.56 ± 0.68 | 54.10 ± 0.72 | — | 46.53 ± 0.71 | P0 |
| Majority | — | 22.25 | 22.25 | — | 22.25 | P0 |
| Top-K popular | — | 22.25 | 33.82 | — | 25.65 | P0 |
| STL GRU reg | — | 44.49 ± 0.51 | 68.33 ± 0.58 | — | 52.74 ± 0.45 | P1 |
| STL STAN reg | 🔴 F6 pending | | | | | |
| STL STAN-Flow reg (matched-head F37) | 🔴 F37 4050-assigned | | | | | pending — STL ceiling for FL H3-alt comparison |
| **MTL-H3-alt** ⭐ | **67.92 ± 0.72** | 50.27 ± 0.55 | ✅ **71.96 ± 0.68** | 63.62 ± 0.80 | 56.96 ± 0.55 | **F48-H3-alt FL 2026-04-26** — first 5f H3-alt FL run. σ excepcionalmente baixa (N=127k). cat +2.20 pp over F32 B3 1f (65.72); reg +6.70 pp over F32 B3 1f (65.26). Used `--batch-size 1024` to avoid MPS OOM at fold 2 (bs=2048 silent kill). |

#### FL 1-fold (ablation evidence — k=2 CV first fold)

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | Notes |
|---|---:|---:|---:|---:|---:|---|
| STL Check2HGI cat | **63.17** | — | — | — | — | P1_5b FL 1f |
| dselectk + pcgrad + GRU MTL | 64.78 | 15.43 | 57.05 | — | 27.49 | FL P2-validate |
| cross-attn + pcgrad + GRU MTL | 66.46 | — | 57.60 | — | — | FL P2 |
| cross-attn + pcgrad + STAN d=256 MTL | 66.16 | 12.09 | 57.71 | — | 24.51 | P8 sanity |
| **cross-attn + pcgrad + GETNext-SOFT d=256 MTL** (B-M13) | 66.01 | 12.74 | 60.62 | 36.01 | 25.55 | prior north-star FL 1f |
| cross-attn + pcgrad + GETNext-HARD d=256 MTL | 55.43 | 13.70 | 58.88 | 49.54 | 28.01 | **hard-under-pcgrad cat starvation** — diagnosed in `research/B5_FL_SCALING.md` |
| **B3 (F2 Phase B3)** — static cat=0.75 + hard ⭐ | 66.23 | 13.46 | 65.82 | 39.88 | 27.94 | **Pareto-dominates soft**: +0.22 cat, +5.20 Acc@10, +3.87 Acc@5, +2.39 MRR |
| **B3 (F17 fold 1)** — independent replicate ⭐ | **67.06** | 16.36 | **66.55** | 53.60 | 31.29 | diagnostic-task-best epoch · +1.05 cat, +5.93 Acc@10 vs B-M13 |

---

## 2 · Headline states (FL + CA + TX 5-fold) — paper primary table

**Status as of 2026-04-26:** FL 5-fold landed for MTL-H3-alt (this session). CA + TX data pipelines not yet built (F22-F25).

| State | cat POI-RGNN | cat STL | cat **MTL-H3-alt** | reg Markov-1 | reg STL STAN | reg STL STAN-Flow | reg STL GRU | reg **MTL-H3-alt** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FL** | 34.49 | 63.17 (n=1) | **67.92 ± 0.72** ✅ | 65.05 ± 0.93 | 🔴 F6 | 🔴 F37 4050 | 68.33 ± 0.58 | **71.96 ± 0.68** ✅ |
| **CA** | 31.78 | 🔴 F24 | 🔴 F24 | 🔴 F24 | 🔴 F24 | 🔴 F21+F24 | 🔴 F24 | 🔴 F24 |
| **TX** | 33.03 | 🔴 F25 | 🔴 F25 | 🔴 F25 | 🔴 F25 | 🔴 F21+F25 | 🔴 F25 | 🔴 F25 |

FL row now populated. Compared to predecessor B3 (1f): cat 65.72 → H3-alt 67.92 (+2.20 pp), reg 65.26 → 71.96 (+6.70 pp). MTL-H3-alt strictly beats Markov-1-region (+6.91 pp) and STL GRU (+3.63 pp) on reg without losing cat F1.

---

## 3 · Cross-state "best-of-each" summary (5-fold data available)

| State | cat F1 best | Method | reg Acc@10 best | Method | Joint (H3-alt) |
|---|---:|---|---:|---|---|
| AL | **42.71 ± 1.37** (B3 post-F27) | next_gru cat | **74.62 ± 3.11** (MTL-H3-alt) ✅ | per-head LR (1e-3/3e-3/1e-3) | **cat 42.22 / reg 74.62** ⭐ |
| AZ | **45.81 ± 1.30** (B3 post-F27) | next_gru cat | 66.74 ± 2.11 (STL F21c) | STL STAN-Flow matched | **cat 45.11 / reg 63.45** (75% gap closed) |
| FL | **67.92 ± 0.72** (MTL-H3-alt) ✅ | per-head LR | **71.96 ± 0.68** (MTL-H3-alt) ✅ | per-head LR | **cat 67.92 / reg 71.96** ⭐ |

**Reading under the H3-alt champion constraint ("single MTL model for both tasks, joint cat+reg paper claim"):**

- **H3-alt jointly wins on FL** (best cat AND best reg, both ⭐) — first time MTL is the per-state champion on both heads simultaneously at the headline state.
- **H3-alt exceeds STL F21c ceiling on AL** (+6.25 pp reg Acc@10), preserving cat F1 within σ of B3. **Strict MTL-over-matched-STL win on reg at AL.**
- **H3-alt closes 75% of the F21c gap on AZ** (53.82 → 63.45 vs 66.74 ceiling). MTL no longer trails by 12-14 pp; residual gap is 3.3 pp within ~1.5σ.
- **CH18 reframed Tier B → A.** The "MTL trails STL by 12-14 pp on reg" finding was a single LR-schedule confound; per-head LR resolves it. See `MTL_ARCHITECTURE_JOURNEY.md`.
- **Three negative controls bracket H3-alt as unique** in this design space — F40 (loss-side), F48-H1 (gentle const), F48-H2 (warmup+plateau). See `research/F40_*` and `research/F48_H2_*`.

---

## 4 · Archived / superseded rows

Kept in source of truth for audit but not referenced in the paper:

| Row | What | Reason |
|---|---|---|
| Pre-fix B-M2 MTLoRA r=8 AL (50.72 ± 4.36) | pre-partition-bug value | SUPERSEDED by B-M2b (53.71 ± 3.80) |
| All AdaShare MTL rows (pre-fix + post-fix) | AdaShare trails MTLoRA by ~9 pp post-fix | Drop from paper; not a competitive MTL sharing mechanism on this task |
| Pre-B5 "MTL-dselectk is the champion" framing | Pre-B5 incorrect narrative | Replaced by B3 (post-F2) |
| FL hard + pcgrad (55.43 / 58.88) | FL-scale gradient starvation | Known failure mode; kept in table as the mechanism anchor for `research/B5_FL_SCALING.md` |

---

## 5 · Index of source JSONs

| Group | Location |
|---|---|
| **F48-H3-alt (AL + AZ 5f, batch=2048)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260425_18*/summary/full_summary.json` |
| **F48-H3-alt FL 5f, batch=1024** | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045/summary/full_summary.json` |
| **F40 scheduled-static (AL + AZ 5f)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260426_08*/summary/full_summary.json` |
| **F48-H2 warmup_constant (AL + AZ 5f, 150ep)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep150_20260426_09*/summary/full_summary.json` |
| **F45 / F48-H1 / F48-H3 control runs** | per-state under `results/check2hgi/<state>/mtlnet_*_20260425_*/summary/full_summary.json` (see `research/F44_F48_LR_REGIME_FINDINGS.md` for matrix) |
| F31 B3 post-F27 (AL + AZ 5f) | superseded older B3_validation by post-F27 — see `results/F27_validation/al_5f50ep_b3_cathead_gru.json` and AZ counterpart |
| F21c STL STAN-Flow (AL + AZ 5f) | `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json` |
| B3 validation (pre-F27, AL + AZ 5f) | `results/B3_validation/{al,az}_5f50ep_b3.json` |
| B5 hard-index (AL + AZ 5f + FL 1f) | `results/B5/{al_5f50ep,az_5f50ep,fl_1f50ep}_next_getnext_hard.json (legacy filename; head is STAN-Flow)` |
| F2 diagnostic (4 × FL 1f) | `results/F2_fl_diagnostic/fl_1f50ep_hard_{pcgrad_ckpt,static_cat0.25,static_cat0.50,static_cat0.75}.json` |
| F17 partial (FL fold 1 only) | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260423_0630/folds/fold1_info.json` |
| MTLoRA post-fix suite | `results/P5_bugfix/*.json` |
| P8 SOTA MTL-STAN / TGSTAN / STA-Hyper / GETNext | `results/P8_sota/*.json` |
| STL cat (P1.5b) | `results/P1_5b/next_category_{alabama,arizona,florida}_*_5f_50ep_fair.json` |
| STL reg heads (P1) | `results/P1/region_head_*_region_5f_50ep_*.json` |
| Simple baselines (Markov k=1..9, Majority, Top-K, Random) | `results/P0/simple_baselines/{alabama,florida}/*.json`; AZ pending |
| Historical comparisons (pre-B3 framing) | `BASELINES_AND_BEST_MTL.md` (kept for audit; annotated 2026-04-23) |
| **Phase-1 substrate validation** (per-fold) | `results/phase1_perfold/` (probe + cat STL × 4 heads × 2 substrates × 2 states + reg STL × 2 substrates × 2 states + MTL counterfactual × 2 states + C4 POI-pooled). Index in `research/SUBSTRATE_COMPARISON_FINDINGS.md` Appendix. |
| **Phase-1 paired tests** | `results/paired_tests/` (Wilcoxon + paired-t + TOST outputs from `scripts/analysis/paired_test_analyser.py`). |
| **Phase-1 linear probe** | `results/probe/{state}_{check2hgi,hgi,check2hgi_pooled}_last.json` (head-free substrate probe). Generator: `scripts/probe/substrate_linear_probe.py`. |
| **Phase-1 baseline ports** | `baselines/next_category/results/{state}.json` + `baselines/next_region/results/{state}.json` (faithful STAN, POI-RGNN, MHA+PE, REHDM ports). |
| **F49 3-way decomposition (AL+AZ 5f, FL 5f via F49c)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_2026042{7_1008,7_1019,7_1029,7_1049}/summary/full_summary.json` (AL+AZ); `results/check2hgi/florida/f49c_{lossside,frozen}_5f_2026042715/2026042718/summary/full_summary.json` (FL n=5); `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260427_1452/summary/full_summary.json` (F49b reproduction gate). Index in `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §6. |

---

## Phase-1 substrate-comparison cells (NEW 2026-04-27)

Source: `research/SUBSTRATE_COMPARISON_FINDINGS.md` (full verdict). Per-fold JSONs at `results/phase1_perfold/` and `results/probe/`. Paired tests at `results/paired_tests/`.

### Cat STL substrate Δ (8 head-state probes; 8/8 positive at p=0.0312, 5/5 folds positive each)

| Probe | AL C2HGI F1 | AL HGI F1 | AL Δ | AZ C2HGI F1 | AZ HGI F1 | AZ Δ |
|---|---:|---:|---:|---:|---:|---:|
| Linear (head-free) | 30.84 ± 2.02 | 18.70 ± 1.38 | **+12.14** | 34.12 ± 1.22 | 22.54 ± 0.45 | **+11.58** |
| **next_gru (matched-head MTL)** | **40.76 ± 1.50** | 25.26 ± 1.06 | **+15.50** ✓ | **43.21 ± 0.78** | 28.69 ± 0.71 | **+14.52** ✓ |
| next_single (head-sensitivity) | 38.71 ± 1.32 | 26.76 ± 0.36 | **+11.96** | 42.20 ± 0.72 | 29.69 ± 0.97 | **+12.50** |
| next_lstm (head-sensitivity) | 38.38 ± 1.08 | 23.94 ± 0.84 | **+14.44** | 41.86 ± 0.84 | 26.50 ± 0.29 | **+15.36** |

### Reg STL matched-head (STAN-Flow (`next_stan_flow`)) — CH15 reframed

| State | C2HGI Acc@10 | HGI Acc@10 | Δ Acc@10 | Wilcoxon p | TOST δ=2pp |
|---|---:|---:|---:|---:|---|
| AL | **68.37 ± 2.66** | 67.52 ± 2.80 | +0.85 | 0.0625 marginal | non-inferior ✅ |
| AZ | **66.74 ± 2.11** | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ | non-inferior ✅ |

### MTL B3 substrate-counterfactual (CH18) — HGI substitution breaks B3

| State | Substrate | cat F1 | reg Acc@10_indist | Δ_cat (C2HGI − HGI) | Δ_reg (C2HGI − HGI) |
|---|---|---:|---:|---:|---:|
| AL | C2HGI (B3) | **42.71 ± 1.37** | **59.60 ± 4.09** | — | — |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 | **+16.75** | **+29.65** |
| AZ | C2HGI (B3) | **45.81 ± 1.30** | **53.82 ± 3.11** | — | — |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 | **+17.11** | **+31.72** |

MTL+HGI is *worse than STL+HGI* on reg by ~37 pp at AL.

### C4 mechanism — per-visit context contribution (AL)

| Substrate | Linear probe F1 | Matched-head STL F1 (`next_gru`) |
|---|---:|---:|
| Check2HGI (canonical) | 30.84 ± 2.02 | 40.76 ± 1.50 |
| Check2HGI POI-pooled | 23.20 ± 1.08 | 29.57 |
| HGI | 18.70 ± 1.38 | 25.26 ± 1.06 |

Decomposition: per-visit context = +7.64 pp (~63%) linear / +11.19 pp (~72%) matched-head; training signal residual = +4.50 (~37%) / +4.31 (~28%).

---

## F49 3-way decomposition cells (NEW 2026-04-27)

Source: `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §10 + §13. Per-fold JSONs at the paths in `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` §6 + §13.

H3-alt regime: `static_weight + cat_lr=1e-3 + reg_lr=3e-3 + shared_lr=1e-3 + scheduler constant`.

| State | STL F21c | encoder-frozen λ=0 | loss-side λ=0 | Full MTL H3-alt | (frozen − STL) **arch** | (loss − frozen) **co-adapt** | (Full − loss) **transfer** |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL (5f) | 68.37 ± 2.66 | 74.85 ± 2.38 | 74.94 ± 2.01 | 74.62 ± 3.11 | **+6.48** ~2.7σ | +0.09 (null) | −0.32 (null) |
| AZ (5f) | 66.74 ± 2.11 | 60.72 ± 1.64 | 62.70 ± 3.01 | 63.45 ± 2.49 | **−6.02** ~3.7σ | +1.98 (small+) | +0.75 (null) |
| FL (5f, F49c) | TBD F37 | 64.22 ± 12.03 | 72.48 ± 1.40 | 71.96 ± 0.68 | TBD | +8.27 (~0.68σ) | −0.52 (~0.34σ null) |

All cells are reg `top10_acc_indist`. **Cat-supervision transfer ≤ |0.75| pp on all 3 states (within σ of zero)** — refutes legacy "+14.2 pp transfer at FL" claim by ≥9σ on FL alone, ≥18σ aggregate.

F49b reproduction gate (AL static_weight λ=0 + max_lr=3e-3 + OneCycleLR + next_gru, 5f×50ep): `top10_acc_indist = 53.18 ± 4.56` vs published 52.27 ± 5.03 → Δ +0.91 pp at ~0.13σ (σ-tight match). Infra validated.
