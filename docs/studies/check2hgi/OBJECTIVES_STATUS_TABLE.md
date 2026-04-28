# Objectives Status Table — Check2HGI Study

**Date:** 2026-04-27 (v5, post-F49 + post-Phase-1 substrate validation). One-page snapshot of where we stand against the study's scientific objectives. Phase-1 substrate validation + F49 attribution are both paper-grade; Phase 2 (FL/CA/TX substrate replication) queued in [`PHASE2_TRACKER.md`](PHASE2_TRACKER.md), F37 (FL F21c) queued in `FOLLOWUPS_TRACKER.md`.

> **NEW Objective 3 (post-Phase-1):** *MTL B3 is substrate-specific* — substituting HGI into the same MTL configuration breaks the joint signal (cat −17 pp, reg −30 pp at AL+AZ). See §2.4 below + CH18 in `CLAIMS_AND_HYPOTHESES.md` + `research/SUBSTRATE_COMPARISON_FINDINGS.md`. AL+AZ confirmed; FL/CA/TX queued.

> **NEW Objective 4 (post-F49):** *MTL B3's reg lift is architecture-dominant, not transfer* — F49 3-way decomposition (encoder-frozen λ=0 / loss-side λ=0 / Full MTL) shows AL architecture alone gives +6.48 pp; cat-supervision transfer is null on all 3 states (≤|0.75| pp). Refutes legacy "+14.2 pp transfer" claim by ≥9σ on FL n=5 alone. See CH19 in `CLAIMS_AND_HYPOTHESES.md` + `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`. Layer 2 methodological contribution (loss-side ablation unsound under cross-attn) committable now.

> **Prior dates:** 2026-04-26 (v4, post-F48-H3-alt + F40 + F48-H2); 2026-04-23 (v2, joint-execution MTL rows). Built from `results/RESULTS_TABLE.md` + `results/B5/*.json` + `results/P5_bugfix/SUMMARY.md`.

> **North-star MTL config (unchanged by F49):** **F48-H3-alt** = B3 architecture + per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3`, `--scheduler constant`). See `NORTH_STAR.md` for the recipe; `MTL_ARCHITECTURE_JOURNEY.md` for the derivation. All paper-relevant MTL comparisons use H3-alt as the primary row from 2026-04-26 onward. Predecessor B3 (50ep + OneCycleLR) remains as a reported comparand. Pre-F27 v2 rows using `GETNext-soft` are retained below for audit but are no longer the headline.

> **Mechanism attribution sharpened (F49, 2026-04-27).** F49's 3-way decomposition (encoder-frozen λ=0 / loss-side λ=0 / Full MTL) reveals the H3-alt reg lift on AL is **architecture-dominant** (+6.48 ± 2.4 pp from architecture alone, ~2.7σ); cat-supervision transfer is null on all 3 states (≤|0.75| pp). The legacy "+14.2 pp transfer at FL" claim from `archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md` is empirically refuted at ≥9σ on FL n=5 alone. CH19 added (Tier A). Layer-2 paper-grade methodological contribution: loss-side `task_weight=0` ablation is unsound under cross-attention MTL (gradient-flow + 4 passing tests). See `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` for the full story.

> **Headline reversal (F48-H3-alt, 2026-04-26).** The F21c finding — STL `next_getnext_hard` dominates MTL by 12–14 pp on reg — is **resolved by the per-head LR recipe**. AL: MTL-H3-alt EXCEEDS STL F21c by +6.25 pp. AZ: closes 75% of the gap. FL: validates at scale, beats Markov-1 by +6.91 pp and STL GRU by +3.63 pp. CH18 promoted Tier B → Tier A. The MTL value proposition now includes paper-strength reg lift, joint single-model deployment, AND cat F1 lift over STL. **F49 then identifies the *cause* of that lift as architectural, not cat-supervision transfer** — paper claim sharpens from "MTL coupling helps reg" to "cross-attention architecture + per-head LR extracts more reg signal."

> **Methodological note.** Each MTL row below is a **single execution** that jointly trains both heads. The cat-F1 and reg-Acc@10 columns on a given row come from the **same model, same fold set, same seed** — compared side-by-side against the cat-task baselines (POI-RGNN, Majority, Markov-1-POI) and the region-task baselines (Markov-1-region, STL STAN, STL GETNext-hard matched-head). Earlier drafts of this file cherry-picked the best MTL per (state × task) cell across different runs; that mixed runs and wasn't an honest joint claim. This version fixes that.

---

## Baseline definitions

| Track | Baseline | Type | Notes |
|---|---|:-:|---|
| **next_category** | **POI-RGNN** (Capanema et al. 2019, reproduced) | external published | Gowalla state-level, same 7-class taxonomy. Reproduced on FL/CA/TX. |
| next_category | Majority-class | internal floor | Per-state most-common category. |
| next_category | Markov-1-POI | internal floor | Previous-POI → next-category transition counts. |
| **next_region** | **Markov-1-region** | internal floor / strong classical | Previous-region → next-region transition counts with Laplace smoothing + top-K fallback. Context-matched Markov-k (k=1..9) reported for completeness; k=1 binds. |
| **next_region** | **STL STAN** (Luo WWW'21 adapt) | internal neural ceiling | Single-task STAN on Check2HGI region embeddings, 5f × 50ep. Strongest single-task neural number we have. |
| next_region | STL GRU (hd=256) | internal neural ceiling (2nd) | Literature-aligned (HMT-GRN uses GRU). Used as the MTL architecture's region head in pre-B5 variants. |
| next_region | HMT-GRN, MGCL (reference only) | external concept-aligned | Different datasets (FSQ-NYC/TKY), not directly comparable (CH10 declared limitation). Framing anchor only. |

---

## Objective 1 — Check2HGI > HGI on next_category (CH16)

Single-task substrate comparison; same pipeline, different embeddings.

### Phase 1 (closed 2026-04-27) — matched-head + head-agnostic + linear probe at AL+AZ

Matched-head probe is `next_gru` (the post-F27 MTL B3 cat head). 5f × 50ep, seed 42:

| State | Probe | C2HGI F1 | HGI F1 | Δ | Wilcoxon p_greater | Evidence |
|:-:|:-|:-:|:-:|:-:|:-:|:-:|
| AL | Linear (head-free) | 30.84 ± 2.02 | 18.70 ± 1.38 | **+12.14** | n/a | `results/probe/alabama_*_last.json` |
| AL | next_gru (matched) | **40.76 ± 1.50** | 25.26 ± 1.06 | **+15.50** | **0.0312** ✅ | `results/phase1_perfold/AL_*_cat_gru_5f50ep.json` |
| AL | next_single | 38.71 ± 1.32 | 26.76 ± 0.36 | **+11.96** | **0.0312** ✅ | (also includes existing `P1_5b/*` evidence Δ=+18.30 leaky-vs-fair) |
| AL | next_lstm | 38.38 ± 1.08 | 23.94 ± 0.84 | **+14.44** | **0.0312** ✅ | C2 head sweep |
| AZ | Linear (head-free) | 34.12 ± 1.22 | 22.54 ± 0.45 | **+11.58** | n/a | substrate-only Leg I |
| AZ | next_gru (matched) | **43.21 ± 0.78** | 28.69 ± 0.71 | **+14.52** | **0.0312** ✅ | matched MTL cat head |
| AZ | next_single | 42.20 ± 0.72 | 29.69 ± 0.97 | **+12.50** | **0.0312** ✅ | head-sensitivity probe |
| AZ | next_lstm | 41.86 ± 0.84 | 26.50 ± 0.29 | **+15.36** | **0.0312** ✅ | head-sensitivity probe |

8/8 head-state probes positive at maximum significance. Phase-2 row (FL/CA/TX) in `PHASE2_TRACKER.md`.

**Status:** `confirmed at AL+AZ matched-head, head-invariant, paired-Wilcoxon p=0.0312 each`. Previous AL-only `next_single` evidence (Δ=+18.30, σ-clean) is preserved as a head-sensitivity probe row. Cross-state replication queued for Phase 2.

### CH15 reframing (reg substrate, head-coupled)

| State | Probe | C2HGI Acc@10 | HGI Acc@10 | Δ | Wilcoxon p (Acc@10) |
|:-:|:-|:-:|:-:|:-:|:-:|
| AL | STAN (existing CH15) | 59.20 ± 3.62 | **62.88 ± 3.90** | −3.68 (HGI > C2HGI under STAN) | — |
| AL | next_getnext_hard (matched MTL) | **68.37 ± 2.66** | 67.52 ± 2.80 | +0.85 (TOST non-inf at δ=2 pp) | 0.0625 marginal |
| AZ | STAN | 52.24 ± 2.38 | **54.86 ± 2.84** | −2.62 | — |
| AZ | next_getnext_hard (matched MTL) | **66.74 ± 2.11** | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ |

Previous "HGI > C2HGI on reg" was head-coupled to STAN's preference for POI-stable smoothness. Under the matched MTL reg head (graph prior), C2HGI ≥ HGI at both states (AL tied, AZ significantly C2HGI).

---

## Objective 2 — Joint-execution MTL comparison

Each row is one MTL run. Compare its cat output vs cat baselines and its reg output vs reg baselines, side-by-side.

### 2.1 Alabama (AL, 5 folds × 50 epochs, seed 42)

**Baselines for context:**
- cat: Majority 34.20 Acc@1 · Markov-1-POI ~31.7 F1 · POI-RGNN (state-level FL/CA/TX range) 31.8–34.5 F1 · STL Check2HGI cat (A-S2) **38.58 ± 1.23** F1
- reg: Markov-1-region 47.01 ± 3.55 Acc@10 · STL GRU 56.94 ± 4.01 · **STL STAN (B-S4) 59.20 ± 3.62**

| MTL run | cat F1 | reg Acc@10 | Cat vs POI-RGNN | Cat vs STL | Reg vs Markov-1 | Reg vs STL STAN |
|:-|-:|-:|:-:|:-:|:-:|:-:|
| A-M1/B-M1 dselectk+pcgrad+GRU | 36.08 ± 1.96 | 48.88 ± 6.26 | ≥ POI-RGNN (+2 pp) | −2.50 (σ-overlap) | +1.87 pp | −10.32 |
| A-M2/B-M2 dselectk+MTLoRA r=8 (post-fix) | 36.53 ± 1.24 | **53.71 ± 3.80** | ≥ POI-RGNN (+2 pp) | −2.05 (σ-overlap) | **+6.70 pp** | −5.49 |
| A-M3/B-M4 cross-attn+pcgrad+GRU | **38.58 ± 0.98** | 45.09 ± 5.37 | **+4–7 pp over POI-RGNN** | **0.00 (tied)** | −1.92 | −14.11 |
| A-M4/B-M5 cross-attn+pcgrad+STAN d=128 | 39.07 ± 1.18 | 50.27 ± 4.47 | **+5–8 pp over POI-RGNN** | +0.49 | +3.26 pp | −8.93 |
| A-M5/B-M6 cross-attn+pcgrad+STAN d=256 | 38.11 ± 1.11 | 51.60 ± 10.09 | +4–7 | −0.47 | +4.59 pp | −7.60 |
| B-M6b cross-attn+pcgrad+GETNext-SOFT d=256 | 38.56 ± 1.45 | 56.49 ± 4.25 | **+4–7** | −0.02 | **+9.48 pp** | −2.71 (σ-overlap) |
| **B-M6e cross-attn+pcgrad+GETNext-HARD d=256** ⭐ | 38.50 ± 1.56 | **57.96 ± 5.09** | **+4–7** | −0.08 | **+10.95 pp** | **−1.24 (σ-overlap)** |

**AL joint-execution reading (B-M6e is the joint champion):**
- Cat head **ties STL** (38.50 vs 38.58) and **beats POI-RGNN** by +4–7 pp.
- Reg head **beats Markov-1-region** by +10.95 pp and **ties STL STAN** within σ (σ_MTL=5.09, σ_STL=3.62; envelopes overlap).
- **Objective 2 on AL:** ✅ beats both baselines jointly on both tasks; MTL matches STL on cat; MTL within σ of STL on reg.

### 2.2 Arizona (AZ, 5 folds × 50 epochs, seed 42)

**Baselines for context:**
- cat: STL Check2HGI cat (A-S3) 42.08 ± 0.89 F1 · POI-RGNN not reported on AZ
- reg: Markov-1-region 42.96 ± 2.05 Acc@10 · STL GRU 48.88 ± 2.48 · **STL STAN (B-S6) 52.24 ± 2.38**

| MTL run | cat F1 | reg Acc@10 | Cat vs STL | Reg vs Markov-1 | Reg vs STL STAN |
|:-|-:|-:|:-:|:-:|:-:|
| A-M6/B-M7 cross-attn+pcgrad+GRU | **43.13 ± 0.55** | 41.07 ± 3.46 | **+1.05** | −1.89 | −11.17 |
| A-M7/B-M8 cross-attn+pcgrad+STAN d=128 | 42.64 ± 0.26 | 37.47 ± 4.01 | +0.56 | −5.49 | −14.77 |
| A-M8/B-M9 cross-attn+pcgrad+STAN d=256 | 42.74 ± 0.45 | 41.04 ± 4.55 | +0.66 | −1.92 | −11.20 |
| B-M9b cross-attn+pcgrad+GETNext-SOFT d=256 | 42.82 ± 0.96 | 46.66 ± 3.62 | +0.74 | **+3.70 pp** | −5.58 |
| **B-M9d cross-attn+pcgrad+GETNext-HARD d=256** ⭐⭐ | 42.22 ± 0.53 | **53.25 ± 3.44** | +0.14 (σ-overlap) | **+10.29 pp** | **+1.01** ✅ |

**AZ joint-execution reading (B-M9d is the joint champion):**
- Cat head **matches STL** (42.22 vs 42.08, within σ).
- Reg head **beats Markov-1-region** by +10.29 pp and **strictly beats STL STAN** by +1.01 pp (mean-over-mean; paired Wilcoxon pending — F1 in tracker).
- **Objective 2 on AZ:** ✅ **first (state × task) cell where a joint MTL execution strictly beats STL on region**. This is the study's strongest single finding.

### 2.3 Florida (FL, 1 fold × 50 epochs, seed 42)

**Baselines for context:**
- cat: STL Check2HGI cat (A-S4) 63.17 F1 (n=1) · POI-RGNN reproduced **34.49 F1**
- reg: Markov-1-region **65.05 ± 0.93 Acc@10** · STL GRU 68.33 ± 0.58 · STL STAN **not yet run at FL** (F6)

| MTL run (n=1) | cat F1 | reg Acc@10 | Cat vs POI-RGNN | Cat vs STL | Reg vs Markov-1 | Reg vs STL GRU |
|:-|-:|-:|:-:|:-:|:-:|:-:|
| A-M9/B-M10 dselectk+pcgrad+GRU | 64.78 | 57.05 | **+30.29 pp** | +1.61 | −8.00 | −11.28 |
| A-M10/B-M11 cross-attn+pcgrad+GRU | **66.46** | 57.60 | **+31.97 pp** | **+3.29** | −7.45 | −10.73 |
| A-M11/B-M12 cross-attn+pcgrad+STAN d=256 | 66.16 | 57.71 | +31.67 | +2.99 | −7.34 | −10.62 |
| **B-M13 cross-attn+pcgrad+GETNext-SOFT d=256** ⭐ | 66.01 | **60.62** | +31.52 | +2.84 | −4.43 | −7.71 |
| B-M14 cross-attn+pcgrad+GETNext-HARD d=256 | 55.43 | 58.88 | +20.94 | −7.74 ❌ | −6.17 | −9.45 |

**FL joint-execution reading (B-M13 is the joint champion):**
- Cat head **beats POI-RGNN by +31.5 pp** and lifts STL by +2.84 pp (n=1 — needs F4 5-fold confirmation).
- Reg head **below Markov-1-region** by −4.43 pp. No FL MTL run we've measured beats the classical 1-gram floor on Acc@10.
- B-M14 (hard variant) trades cat for reg and still loses to Markov on reg — diagnosed in `research/B5_FL_SCALING.md`.
- **Objective 2 on FL:** 🟡 clean win on cat (vs POI-RGNN and vs STL); clean fail on reg vs Markov. Paper has to frame FL as a scale-dependent / Markov-saturated regime OR report Acc@5 / MRR on the reg side (MTL beats Markov on both).

---

### 2.4 Objective 3 — MTL B3 is substrate-specific (NEW, post-Phase-1)

MTL B3 with HGI substrate (5f × 50ep, seed 42), compared to existing MTL B3 with C2HGI:

| State | Substrate | cat F1 | reg Acc@10_indist | Δ_cat (C2HGI − HGI) | Δ_reg (C2HGI − HGI) |
|:-:|:-|:-:|:-:|:-:|:-:|
| AL | C2HGI (B3) | **42.71 ± 1.37** | **59.60 ± 4.09** | — | — |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 | **+16.75** | **+29.65** |
| AZ | C2HGI (B3) | **45.81 ± 1.30** | **53.82 ± 3.11** | — | — |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 | **+17.11** | **+31.72** |

**MTL+HGI is *worse than STL+HGI* on reg** (-37 pp Acc@10 at AL: STL HGI gethard 67.52 → MTL HGI 29.95). The B3 configuration was tuned around Check2HGI's per-visit context and does not generalise to HGI substrate. Status: `confirmed at AL+AZ`. FL/CA/TX queued.

### 2.5 Objective 4 — MTL B3's reg lift is architecture-dominant on AL only; architecture costs reg on AZ+FL (NEW, post-F49 + F37 closing 2026-04-28)

F49 3-way decomposition under the H3-alt champion regime, 5-fold paired Wilcoxon (n=5, exact):

| State | STL F21c | encoder-frozen λ=0 | loss-side λ=0 | Full MTL H3-alt | (frozen − STL) **arch** | (loss − frozen) **co-adapt** | (Full − loss) **transfer** | (Full − STL) MTL vs ceiling |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| AL (5f) | 68.37 ± 2.66 | 74.85 ± 2.38 | 74.94 ± 2.01 | 74.62 ± 3.11 | **+6.48** ~2.7σ | +0.09 | −0.32 | **+6.25 pp p=0.0312** ✓ |
| AZ (5f) | 66.74 ± 2.11 | 60.72 ± 1.64 | 62.70 ± 3.01 | 63.45 ± 2.49 | **−6.02** ~3.7σ | +1.98 | +0.75 | −3.29 pp (n.s.) |
| **FL (5f, F37+F49c, 2026-04-28)** | **82.44 ± 0.38** | 64.22 ± 12.03 | 72.48 ± 1.40 | 71.96 ± 0.68 | **−16.16 pp p=0.0312** | +8.27 (~0.68σ) | −0.52 (~0.34σ) | **−8.78 pp p=0.0312** ✗ |

**Per-state mechanism (closed across all 3 headline states):**

- **AL — architecture-dominant lift.** Cross-attention with frozen-random cat features extracts +6.48 pp from STL ceiling (paired Wilcoxon p=0.0312, 5/5 folds). Cat-supervision transfer null. Full MTL exceeds STL by +6.25 pp.
- **AZ — classical MTL pattern.** Architecture costs −6.02 pp (3.7σ); co-adaptation rescues +1.98 pp; transfer negligible. Full MTL closes 75% of B3 gap but still trails STL by 3.29 pp (within 1.5σ, n.s.).
- **FL — heavy architectural cost; STL ceiling above MTL.** Architecture −16.16 pp (5/5 paired folds negative, p=0.0312); co-adapt +8.27 partially recovers; full MTL still −8.78 pp below STL ceiling (5/5 folds, p=0.0312). The architectural cost grows steeply with region cardinality (1.1K → 1.5K → 4.7K → +6.5 / −6.0 / −16.2 pp).

**Cat-supervision transfer ≤ |0.75| pp on all 3 states n=5** — refutes legacy "+14.2 pp transfer at FL" by ≥9σ on FL alone. Layer 2 paper-grade methodological contribution: loss-side `task_weight=0` ablation is unsound under cross-attention MTL (silenced cat encoder co-adapts via attention K/V). Status: `Layers 1 + 2 + 3 closed across AL+AZ+FL n=5`. Sources: `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`, `research/F37_FL_RESULTS.md`, `results/paired_tests/FL_layer3_after_f37.json`.

---

## 3 · Condensed objectives scorecard (north-star = F48-H3-alt, v5)

Using the **2026-04-26 champion candidate** (B3 architecture + per-head LR `cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3, --scheduler constant`) at every state:

| State | Row | Cat verdict vs (POI-RGNN / STL matched `next_mtl`) | Reg verdict vs (Markov-1 / STL STAN / STL GETNext-hard — CH18) | Joint success |
|:-:|:-|:-|:-|:-:|
| AL 5f | F48-H3-alt | cat F1 **42.22 ± 1.00** : ≈ B3 post-F27 (-0.49) · **+8 pp > POI-RGNN** · **+3.64 pp > STL `next_mtl` (38.58)** | reg Acc@10 **74.62 ± 3.11** : **+27.61 pp > Markov-1 (47.01)** · **+15.42 pp > STL STAN (59.20)** · ✅ **+6.25 pp > STL GETNext-hard (68.37)** CH18 RESOLVED | ✅ joint cat + reg above all STL ceilings |
| AZ 5f | F48-H3-alt | cat F1 **45.11 ± 0.32** : ≈ B3 post-F27 (-0.70) · **+3.03 pp > STL `next_mtl` (42.08)** | reg Acc@10 **63.45 ± 2.49** : **+20.49 pp > Markov-1 (42.96)** · **+11.21 pp > STL STAN (52.24)** · 🟡 **−3.29 pp vs STL GETNext-hard (66.74)** — 75% of B3-vs-STL gap closed | ✅ cat strict-gain; ✅ reg crosses STAN; 🟡 75% of CH18 closed |
| FL 5f | F48-H3-alt | cat F1 **67.92 ± 0.72** : **+33.4 pp > POI-RGNN (34.49)** · **+0.94 pp > STL `next_gru` (66.98 ± 0.61, F37 P1 2026-04-28)** | reg Acc@10 **71.96 ± 0.68** : **+6.91 pp > Markov-1 (65.05)** · **+3.63 pp > STL GRU (68.33)** · 🔴 STL STAN pending (F6) · ⚠️ **−8.78 pp vs STL GETNext-hard (82.44 ± 0.38, F37 P2 2026-04-28) p=0.0312** | 🟡 cat-side ✓ (substrate-driven); reg-side STL ceiling above MTL — CH18 scale-conditional |

**Objective 2 overall under H3-alt (post-F37 2026-04-28):**
- **MTL > per-task baselines:** ✅ clean on cat at every state (+8 to +33 pp over POI-RGNN). ✅ All three states beat Markov-1-region by +6.91 to +27.61 pp.
- **MTL > STL cat (matched-head `next_gru` post-F27):** ✅ AL **+3.64 pp** · ✅ AZ **+3.03 pp** · ✅ FL **+0.94 pp** (post-F37 P1: STL `next_gru` 66.98 ± 0.61, MTL H3-alt 67.92 ± 0.72). Cat-side MTL > STL holds at all 3 states.
- **MTL > STL STAN reg:** ✅ AL **+15.42 pp** · ✅ AZ **+11.21 pp** · 🔴 FL pending (F6).
- **MTL > matched-head STL GETNext-hard reg (CH18, scale-conditional):** ✅ AL EXCEEDS by **+6.25 pp p=0.0312** · 🟡 AZ closes 75% of B3 gap (−3.29 pp residual, n.s.) · ⚠️ **FL: STL ceiling above MTL by −8.78 pp p=0.0312, 5/5 folds negative** (F37 P2 2026-04-28). The architectural lift on AL **does not generalise to FL scale**. CH18 reframed scale-conditional in `CLAIMS_AND_HYPOTHESES.md §CH18`.

**What changed vs v3 (B3 post-F27 scorecard):** H3-alt's per-head LR recipe lifted reg Acc@10 by +14.02 pp at AL, +9.63 pp at AZ, and +6.70 pp at FL (vs predecessor B3 numbers), while keeping cat F1 within ~2 pp of B3 baseline.

**What changed vs v5 (post-F37 2026-04-28):** Layer 3 of the F49 attribution closed with the FL STL F21c run. The 3-state architectural-Δ pattern is **{AL +6.48, AZ −6.02, FL −16.16} pp** (paired Wilcoxon p=0.0312 at AL and FL). MTL H3-alt FL reg Acc@10 (71.96) is **−8.78 pp below** matched-head STL ceiling (82.44, p=0.0312). The CH18 "MTL exceeds STL on reg" claim is **scale-conditional: AL only**. The cat-side MTL > STL holds at all 3 states (+0.94 to +3.64 pp). Paper framing reads: **MTL H3-alt is the recommended joint-deployment recipe; AL is the architecture-dominant state where it exceeds matched-head STL on reg; FL's headline reg ceiling is STL `next_getnext_hard`. The architectural cost grows steeply with region cardinality** — this is a paper-grade per-state characterisation, not a retraction.

### Archived v3 scorecard (north-star = B3 post-F27 + OneCycleLR) — retained for audit

The v3 scorecard is preserved below with the original "trails matched-head STL by 12-14 pp" framing — kept to demonstrate the contribution of the per-head LR recipe between v3 and v4.

| State | Row | Cat (B3) | Reg (B3) | Joint |
|:-:|:-|:-|:-|:-:|
| AL 5f | F31 | 42.71 (+4.13 vs STL `next_mtl`) | 59.60 (+0.40 STL STAN, **−8.77 STL GETNext-hard**) | ✅ cat / 🔴 trails matched-head |
| AZ 5f | F19-F27 | 45.81 (Wilcoxon p=0.0312) | 53.82 (tied STAN, **−12.92 STL GETNext-hard**) | ✅ cat / 🔴 trails matched-head |
| FL 1f | F32 | 65.72 (n=1, +2-3 vs STL n=1) | 65.26 (n=1, tied Markov) | 🟡 n=1 |

### Archived v2 scorecard (north-star = GETNext-soft, pre-F27) — retained for audit

Before 2026-04-23 the committed north-star was `cross-attn + pcgrad + GETNext-soft d=256`. That scorecard is retained below as audit; do not cite against current claims.

| State | Row | Cat verdict | Reg verdict | Joint |
|:-:|:-|:-|:-|:-:|
| AL | B-M6b | cat: +4–7 pp > POI-RGNN · ties STL (−0.02) | reg: +9.48 pp > Markov · σ-tied STL STAN (−2.71) | ✅ |
| AZ | B-M9b | cat: ties STL (+0.74) | reg: +3.70 pp > Markov · −5.58 vs STL STAN | 🟡 |
| FL | B-M13 (n=1) | cat: +31.5 pp > POI-RGNN · +2.84 pp > STL | reg: −4.43 below Markov-1 · −7.71 vs STL GRU | 🟡 |

### Alternate-config scorecard (GETNext-hard) — kept for paper ablation row

| State | Row | Cat | Reg | Joint success |
|:-:|:-|:-|:-|:-:|
| AL | B-M6e | ties STL (0.00) | +10.95 pp > Markov, σ-tied with STL STAN (−1.24) | ✅ |
| AZ | B-M9d | ties STL (+0.14) | **+10.29 pp > Markov, +1.01 pp > STL STAN** — Wilcoxon one-sided **p=0.0312** on Acc@10/Acc@5/MRR/F1 (2026-04-23 F1, every fold positive, minimum-achievable p at n=5) | ✅ strict MTL-over-STL, statistically supported |
| FL (n=1) | B-M14 | **−7.74 pp cat F1** — training pathology diagnosed (cat best-val F1 ceiling 0.554 across 50 epochs) | −4.43 below Markov, −9.45 vs STL GRU | ❌ cat head fails to train (F2 task-weight sweep in progress 2026-04-23) |

Hard is strictly superior on AZ (statistically significant across all 4 region metrics), tied on AL, and has a diagnosed training failure on FL. Reported in the paper as an ablation showing the mechanism works at ≤ 1.5K regions. Significance evidence: `research/B5_AZ_WILCOXON.md`.

---

## 4 · What closes each remaining gap

| Gap | Follow-up | Cost | Expected outcome |
|---|---|:-:|---|
| AZ reg Δ (+1.01) needs significance | F1 — paired Wilcoxon on 5-fold deltas | 30 min CPU | p-value; almost certainly < 0.05 with all 5 deltas positive |
| FL cat lift is n=1 | F4 — FL MTL-GETNext-soft 5f × 50ep | ~5–6 h MPS | σ on the +2.84 pp cat lift; either confirmed or retracted |
| FL reg below Markov | F2 — task-weight sweep on FL-hard | ~2.25 h MPS | tests whether cat-task rebalancing rescues FL reg without breaking cat |
| FL reg below STL | F6 — STL STAN 5f × 50ep at FL | ~5–6 h MPS | completes the STL ceiling story; may re-frame the gap |
| CH16 only at AL | F3 — AZ HGI STL cat 5f × 50ep | ~3 h MPS | extends Objective 1 to n=2 states |
| Seed robustness | F8 — multi-seed n=3 on champions | ~20 h total, parallelisable | BRACIS-style seed σ |

See `FOLLOWUPS_TRACKER.md` for the complete, acceptance-criterion-tagged queue.

---

## 5 · Sources

- `results/RESULTS_TABLE.md` — canonical row-per-method table.
- `results/B5/{al,az,fl}_*next_getnext_hard.json` — B5 joint-execution JSONs (both task heads' metrics in the same file).
- `results/P5_bugfix/SUMMARY.md` — post-partition-bug MTLoRA reruns.
- `results/P1/region_head_*_STAN_*.json` — STL STAN ceiling runs.
- `results/P1_5b/next_category_*_{check2hgi,hgi}_*_fair.json` — CH16 substrate evidence.
- `results/P0/simple_baselines/{alabama,florida}/next_region.json` — Markov-k floors.
- `docs/baselines/BASELINE.md` — POI-RGNN / HAVANA / PGC-NN reproduced numbers.
- `review/2026-04-23_critical_review.md` — analytical state write-up.
