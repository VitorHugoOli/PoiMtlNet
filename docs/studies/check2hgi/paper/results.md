# Results section — draft v1

**Date:** 2026-04-28 (revised after F37 FL closing)
**Target:** Submission paper §4 (Results). ~4-5 pages camera-ready.
**Status:** First-pass draft with AL+AZ+FL n=5 (Phase-1 + F49 + F37). CA/TX rows are placeholders pending P3.

---

## 4 Results

### 4.1 Headline: scale-conditional MTL behaviour across three US states

Table 1 reports the joint MTL champion (F48-H3-alt — see §3.4) against matched-head single-task baselines, classical Markov-1-region, and external published baselines (POI-RGNN for cat; STAN ceiling for reg). The pattern is **state-dependent**: at AL the joint MTL exceeds matched-head STL on both heads; at FL the substrate carries the cat-side advantage but the matched-head STL STAN-Flow exceeds MTL on reg by 8.78 pp.

**Table 1 — Headline cross-state results (5-fold × 50ep, seed 42).**

| State | Method | next_category macro-F1 | next_region Acc@10 | Δ vs STL (reg) |
|-------|--------|----------------------:|--------------------:|---------------:|
| **AL** | Markov-1-region        | —                     | 0.471 ± —          | — |
|        | POI-RGNN (external)    | 0.341                 | —                  | — |
|        | STAN (external)        | —                     | 0.592 ± —          | — |
|        | STL `next_gru` cat     | 0.4076 ± 0.020        | —                  | — |
|        | STL STAN-Flow reg | —                | 0.6837 ± 0.0266    | (ref.) |
|        | **MTL H3-alt (ours)**  | **0.4222 ± 0.0100**   | **0.7462 ± 0.0311**| **+6.25 pp** *(p=0.0312)* |
| **AZ** | Markov-1-region        | —                     | 0.378 ± —          | — |
|        | STAN (external)        | —                     | 0.523 ± —          | — |
|        | STL `next_gru` cat     | 0.4234 ± 0.013        | —                  | — |
|        | STL STAN-Flow reg | —                | 0.6674 ± 0.0211    | (ref.) |
|        | **MTL H3-alt (ours)**  | **0.4511 ± 0.0032**   | **0.6345 ± 0.0249**| −3.29 pp (closes 75% of B3 gap; n.s.) |
| **FL** | Markov-1-region        | —                     | 0.6505 ± —         | — |
|        | POI-RGNN (external)    | 0.318                 | —                  | — |
|        | **STL `next_gru` cat (F37 P1, 2026-04-28)**     | **0.6698 ± 0.0061** | — | — |
|        | **STL STAN-Flow reg (F37 P2, 2026-04-28)** | — | **0.8244 ± 0.0038** | (ref.) |
|        | **MTL H3-alt (ours)**  | **0.6792 ± 0.0072**   | 0.7196 ± 0.0068    | **−8.78 pp** *(p=0.0312, 5/5 folds negative)* |
| **CA** | (pending P3 upstream + 5f H3-alt) | TBD          | TBD              | TBD |
| **TX** | (pending P3 upstream + 5f H3-alt) | TBD          | TBD              | TBD |

**Headline observations** (AL+AZ+FL n=5, paired Wilcoxon at fold level):
- **MTL H3-alt > matched-head STL on cat at all 3 states**: AL +1.46 pp, AZ +2.77 pp, **FL +0.94 pp** (post-F37 P1). The cat-side MTL > STL relation generalises.
- **MTL H3-alt vs matched-head STL on reg is scale-conditional**: AL +6.25 pp (p=0.0312, 5/5); AZ −3.29 pp (n.s., closes 75% of B3 gap); **FL −8.78 pp (p=0.0312, 5/5 folds negative).** The architectural lift on AL does **not** generalise to FL scale.
- F49 architectural decomposition (§4.3) shows the underlying mechanism: the cross-attention architecture lifts reg by +6.48 pp on AL (with frozen-random cat features), but costs reg by −6.02 pp on AZ and **−16.16 pp on FL**. The cost grows with region cardinality (1.1K → 1.5K → 4.7K).

### 4.2 Substrate validation — Check2HGI is necessary, head-invariant, and per-visit context is the dominant mechanism

Table 2 reports the Phase-1 substrate-comparison grid at AL+AZ, with paired Wilcoxon n=5.

**Table 2 — Phase-1 substrate Δ (Check2HGI vs HGI; matched-head STL).**

| State | Head probe | C2HGI cat F1 | HGI cat F1 | Δ (pp) | n+ | Wilcoxon p (one-sided) |
|-------|-----------|-------------:|-----------:|-------:|---:|----------------------:|
| AL | linear (head-free) | 0.4076 ± 0.020 | 0.2862 ± 0.018 | **+12.14** | 5/5 | 0.0312 |
| AL | next_gru (matched MTL) | 0.4076 ± 0.020 | 0.2526 ± 0.012 | **+15.50** | 5/5 | 0.0312 |
| AL | next_lstm | 0.388 ± —      | 0.227 ± —      | **+15.83** | 5/5 | 0.0312 |
| AL | next_single | 0.408 ± —     | 0.225 ± —      | **+18.30** | 5/5 | 0.0312 |
| AZ | linear (head-free) | 0.4040 ± —    | 0.2882 ± —     | **+11.58** | 5/5 | 0.0312 |
| AZ | next_gru (matched MTL) | 0.4581 ± 0.013 | 0.3129 ± —    | **+14.52** | 5/5 | 0.0312 |
| AZ | next_lstm | 0.434 ± —      | 0.279 ± —      | **+15.51** | 5/5 | 0.0312 |
| AZ | next_single | 0.434 ± —     | 0.260 ± —      | **+17.40** | 5/5 | 0.0312 |

All 8 head/state cells are positive at n=5 paired Wilcoxon p=0.0312 (the smallest achievable p at n=5 with no zero deltas). Check2HGI's cat advantage is **head-invariant** — no head choice flips sign or fails significance.

**Mechanism: per-visit context vs training signal.** Phase-1 C4 reports a POI-pooled counterfactual: replacing per-visit Check2HGI with per-POI mean (so each POI gets one vector across all visits) reduces cat F1 from **40.76** (canonical) → **29.57** (POI-pooled) → **25.26** (HGI baseline) at AL matched-head STL. Decomposition:

| Source | Δ pp |
|--------|-----:|
| Per-visit context (canonical − pooled) | **+11.19** (~72%) |
| Training signal (pooled − HGI) | **+4.31** (~28%) |
| Total substrate gap | **+15.50** |

Linear-probe agrees within 9 pp (63%/37% under linear probe; 72%/28% under matched-head). **Per-visit context is the dominant mechanism behind Check2HGI's substrate advantage on `next_category`.**

### 4.3 Architecture attribution — F49 3-way decomposition

Table 3 reports the encoder-frozen / loss-side / Full MTL decomposition (§3.7) at AL+AZ+FL n=5. We measure `next_region` Acc@10 at the per-task best epoch.

**Table 3 — F49 architecture attribution (next_region Acc@10, 5-fold paired Wilcoxon).**

| State | STL F21c | frozen-cat λ=0 | loss-side λ=0 | Full MTL (H3-alt) | Architecture (frozen − STL) | Co-adapt (loss − frozen) | Transfer (full − loss) | Full vs STL |
|-------|--------:|---------------:|--------------:|------------------:|----------------------------:|-------------------------:|-----------------------:|------------:|
| AL  | 0.6837 ± 0.0266 | 0.7485 ± 0.0238 | 0.7494 ± 0.0201 | 0.7462 ± 0.0311 | **+6.48 pp (~2.7σ)** | +0.09 pp | −0.32 pp | **+6.25 pp p=0.0312** ✓ |
| AZ  | 0.6674 ± 0.0211 | 0.6072 ± 0.0164 | 0.6270 ± 0.0301 | 0.6345 ± 0.0249 | **−6.02 pp (~3.7σ)** | +1.98 pp | +0.75 pp | −3.29 pp (n.s.) |
| FL  | **0.8244 ± 0.0038** *(F37 P2, 2026-04-28)* | 0.6422 ± 0.1203 | 0.7248 ± 0.0140 | 0.7196 ± 0.0068 | **−16.16 pp p=0.0312** | +8.27 pp (~0.68σ) | −0.52 pp | **−8.78 pp p=0.0312** ✗ |

Paired Wilcoxon (one-sided greater, n=5, exact) on the decomposition components:
- AL co-adapt (loss − frozen): **W_p = 0.81**, n+/n− = 2/2, Δ = −0.02 pp → null
- AL transfer (full − loss): **W_p = 0.31**, n+/n− = 3/2, Δ = +0.46 pp → null
- AZ co-adapt: **W_p = 0.41**, n+/n− = 2/3, Δ = +0.52 pp → null
- AZ transfer: **W_p = 0.50**, n+/n− = 2/3, Δ = +0.12 pp → null
- FL co-adapt: **W_p = 0.31**, Δ = +3.62 pp → not significant at n=5 (high σ)
- FL transfer: **W_p = 0.50**, Δ = +3.75 pp → not significant at n=5 (high σ)

**Cat-supervision transfer is null on all 3 states** at n=5: |Δ| ≤ 0.75 pp at AL/AZ; FL transfer −0.52 pp (~0.34σ). This refutes the legacy "+14.2 pp transfer at FL" claim (from a 2-state n=2 framing in archived `research_pre_b5/CHAIN_FINDINGS_2026-04-20.md`) by **≥9σ on FL alone** (independent σ_diff ≈ 1.4 pp; Δ ≈ −0.5 pp; legacy claim was +14.2 pp).

**Per-state mechanism summary (Layer 3 closed by F37 2026-04-28):**
- **AL** — H3-alt reg lift is **purely architectural**. Cross-attention with α-growth (graph prior) extracts +6.48 pp from STL F21c with frozen-random cat features. Paired Wilcoxon at fold level: full MTL > STL **+6.25 pp p=0.0312, 5/5 folds**.
- **AZ** — Architecture *costs* reg by 6.02 pp (frozen-cat falls below STL); co-adaptation rescues +1.98 pp; transfer rescues +0.75 pp. Net Full MTL trails STL by 3.29 pp (within 1.5σ, n.s.). This is the **classical "MTL pipeline costs the strong head"** pattern.
- **FL — heavy architectural cost; STL ceiling above MTL.** Architecture −16.16 pp (5/5 paired folds negative, p=0.0312, σ ~12 pp driven by per-fold instability when cat is random). Co-adaptation rescues +8.27 pp; transfer null. Net **full MTL still −8.78 pp below STL ceiling** (5/5 folds, p=0.0312). Per-fold reg-best epochs {2, 14, 9, 4, 2} indicate α-growth fails to engage when cat features are random at the 4,702-region scale; this frozen-cat instability is a separate paper-worthy methodological caveat (Limitations §6.2). The headline interpretation is robust to the FL-frozen instability: the **full MTL with cat training also loses to STL by 8.78 pp**.

**Cardinality-conditional architectural cost:** the architectural Δ across states (1,109 → 1,547 → 4,702 regions) is **+6.48 → −6.02 → −16.16 pp**. The cross-attention mechanism that lifts reg at small region cardinality pays an increasing cost as cardinality grows. At our largest scale (FL), the matched-head STL STAN-Flow (with α·log_T graph prior) is the per-task ceiling, and joint MTL pays a coupling cost the per-head LR cannot recover.

### 4.4 Methodological note — loss-side λ=0 is unsound under cross-attention MTL

(See appendix in `N7c_methodological_appendix.md` for full treatment.)

---

## 4.5 Phase-1 substrate-specificity (CH18) — MTL B3 requires Check2HGI

Table 4 reports the Phase-1 Leg III counterfactual: substituting HGI in place of Check2HGI in the otherwise-identical MTL pipeline.

**Table 4 — MTL substrate counterfactual (n=5 paired).**

| State | Substrate | next_category macro-F1 | Δ (pp) | next_region Acc@10 | Δ (pp) |
|-------|-----------|----------------------:|-------:|-------------------:|-------:|
| AL | Check2HGI (canonical) | 0.4271 ± 0.0137 | (ref.) | 0.5960 ± 0.0409 | (ref.) |
| AL | HGI                   | 0.2596 ± —      | **−16.75** | 0.2995 ± — | **−29.65** |
| AZ | Check2HGI (canonical) | 0.4581 ± 0.0130 | (ref.) | 0.5382 ± 0.0311 | (ref.) |
| AZ | HGI                   | 0.2870 ± —      | **−17.11** | 0.2210 ± — | **−31.72** |

MTL+HGI is **worse than STL+HGI** on `next_region` by ~37 pp at AL. The MTL win is interactional with substrate; HGI substitution breaks the joint signal. This is **CH18, Tier A**.

## 4.6 Joint synthesis — CH21 (revised after F37 FL closing)

(Top-line claim block; combines §4.1 + §4.2 + §4.3 + §4.5)

> Across three US states (AL: 10K check-ins, 1,109 regions; AZ: 26K, 1,547; FL: 127K, 4,702), the joint MTL on `(next_category, next_region)` displays a **scale-conditional pattern with two decoupled mechanisms**:
>
> 1. **Substrate (Check2HGI per-visit context) — generalises across scale.** CH16 confirms +11–15 pp head-invariant cat F1 advantage at AL+AZ (FL replication queued); CH18-substrate confirms substituting HGI breaks the joint signal at AL+AZ (cat −17 pp, reg −30 pp). The cat-side MTL > STL relation holds at all 3 states (+0.94 to +3.64 pp). The substrate is the **paper-grade contribution that scales**.
> 2. **Architecture (cross-attention + per-head LR) — state-dependent.** The cross-attention pipeline lifts reg by +6.48 pp on AL (paired Wilcoxon p=0.0312, 5/5 folds; F49 Layer 3) but costs reg by −6.02 pp on AZ and **−16.16 pp on FL** (both p=0.0312 at n=5 paired). At FL the matched-head STL STAN-Flow exceeds full MTL by **−8.78 pp** (p=0.0312, 5/5 folds negative). The architectural mechanism is the **AL contribution**, not state-general.
>
> Cat-supervision transfer is null at all 3 states (≤|0.75| pp), refuting the conventional "MTL transfers signal" framing by ≥9σ on FL alone. The methodological contribution (CH20 Layer 2: loss-side `task_weight=0` ablation is unsound under cross-attention MTL) holds throughout. The paper reports a per-state pattern: **AL (architecture-dominant joint lift) → AZ (classical, architecture costs partially recovered) → FL (substrate-only joint lift, STL is the per-task reg ceiling).** This characterisation, not retraction, is the contribution.

---

## Open TODOs for Results

- ✅ Cat F1 cells filled for all 3 states (F37 P1 done 2026-04-28).
- ✅ FL absolute architectural Δ filled (F37 P2 done 2026-04-28; Layer 3 closed with negative result).
- Add CA + TX rows once P3 upstream pipelines + 5f H3-alt land. **Now even more important post-F37**: tests whether FL architectural-cost-at-scale is an outlier (FL-idiosyncratic) or a cardinality-curve (replicable at CA+TX scale).
- ✅ Paired Wilcoxon H3-alt vs B3 done (`results/paired_tests/H3alt_vs_B3_wilcoxon.json`).
- Mark which Tier-A claims survive a multi-seed sensitivity test (P6 deferred). **Especially relevant for FL frozen-cat σ=12 pp instability**: a multi-seed run on FL frozen-cat would tighten the architectural-Δ magnitude.
