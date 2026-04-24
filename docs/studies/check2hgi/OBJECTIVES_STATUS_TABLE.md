# Objectives Status Table — Check2HGI Study

**Date:** 2026-04-24 (v3, post-F2 / F21c / F27). One-page snapshot of where we stand against the study's two scientific objectives, built from `results/RESULTS_TABLE.md` + `results/B5/*.json` + `results/B3_validation/*.json` + `results/F27_validation/*.json` + `results/P5_bugfix/SUMMARY.md`.

> **North-star MTL config (committed 2026-04-24):** **B3** = `mtlnet_crossattn + static_weight(category_weight=0.75) + next_gru (task_a) + next_getnext_hard (task_b) d=256, 8h` — see `NORTH_STAR.md` for the decision memo. All paper-relevant MTL comparisons now use B3 as the primary row. The soft-probe variant (prior north-star) remains as a reported ablation. Pre-F27 v2 rows using `GETNext-soft` are retained below for audit but are no longer the headline.

> **Headline caveat (F21c, 2026-04-24).** Matched-head STL `next_getnext_hard` (graph-prior alone, single-task) dominates MTL-B3 on reg Acc@10 by 12–14 pp on AL and AZ (CH18). The MTL value proposition now lives primarily on cat F1 + joint single-model deployment, not on reg. See `research/F21C_FINDINGS.md`.

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

| State | Task | Check2HGI STL | HGI STL | Δ (C2HGI − HGI) | σ-overlap? | Evidence |
|:-:|:-|:-:|:-:|:-:|:-:|:-:|
| AL | cat F1 | **38.58 ± 1.23** | 20.29 ± 1.34 | **+18.30 pp** | **No** | ✅ `P1_5b/next_category_alabama_{check2hgi,hgi}_5f_50ep_fair.json` |
| AL | reg Acc@10 (pooled) | 56.11 ± 4.02 | 57.02 ± 2.92 | −0.91 pp | Yes (tied) | CH15 — expected tie after region-pooling |
| AZ | cat F1 | 42.08 ± 0.89 | **not run** | — | — | 🔴 F3 in `FOLLOWUPS_TRACKER.md` |
| FL | cat F1 | 63.17 (n=1) | **not run** | — | — | 🔴 F9 |

**Status:** `confirmed on AL only`. AL delta is 14× the larger σ. AZ HGI STL cat (F3, ~3 h MPS) is the cheapest next step to get n=2 states.

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

## 3 · Condensed objectives scorecard (north-star = B3 post-F27)

Using the **committed north-star MTL config** (`mtlnet_crossattn + static_weight(cat=0.75) + next_gru (task_a) + next_getnext_hard (task_b) d=256`) at every state:

| State | Row | Cat verdict vs (POI-RGNN / STL matched `next_mtl`) | Reg verdict vs (Markov-1 / STL STAN / STL GETNext-hard — CH18) | Joint success |
|:-:|:-|:-|:-|:-:|
| AL 5f | F31 (post-F27) | cat F1 **42.71** : **+8 pp > POI-RGNN** · **+4.13 pp > STL `next_mtl` (38.58)** | reg Acc@10 **59.60** : **+12.59 pp > Markov-1 (47.01)** · **+0.40 pp > STL STAN (59.20)** first cross · **−8.77 pp vs STL GETNext-hard (68.37)** CH18 | ✅ cat+reg beat STAN; 🔴 trails matched-head STL |
| AZ 5f | F19-F27 | cat F1 **45.81** : ties STL-next_mtl (42.08) **+3.73 pp, Wilcoxon p=0.0312, 5/5 folds** | reg Acc@10 **53.82** : **+10.86 pp > Markov-1 (42.96)** · +1.58 σ-tied with STL STAN (52.24) · **−12.92 pp vs STL GETNext-hard (66.74)** CH18 | ✅ cat strict-gain; 🟡 reg tied STAN; 🔴 trails matched-head |
| FL 1f | F32 (post-F27) | cat F1 **65.72** (n=1) : **+31.2 pp > POI-RGNN (34.49)** · +2–3 pp over STL (63.17 n=1) | reg Acc@10 **65.26** (n=1) : tied with Markov-1 (65.05) · 🔴 STL STAN pending (F6) · 🔴 STL GETNext-hard pending (F21c FL) | 🟡 n=1; F33 decides FL-scale cat head |

**Objective 2 overall under B3 (post-F27):**
- **MTL > per-task baselines:** ✅ clean on cat at every state (+8 to +31 pp over POI-RGNN, all > Majority / Markov-POI). ✅ AL + AZ region over Markov-1 by +10.86 to +12.59 pp. 🟡 FL region at Markov-saturation (approach (a) in `PAPER_STRUCTURE.md §6`).
- **MTL > STL cat:** ✅ AL **+4.13 pp** (F31) · ✅ AZ **+3.73 pp Wilcoxon p=0.0312** (F27 5f) · 🟡 FL +2–3 pp (n=1, F33 5f pending).
- **MTL > STL STAN reg:** ✅ AL first-cross **+0.40 pp** (F31) · 🟡 AZ tied σ (+1.58) · 🔴 FL pending.
- **MTL > matched-head STL GETNext-hard reg (CH18):** 🔴 **AL and AZ lose by 12–14 pp** — F21c 2026-04-24 finding reframes the paper. FL pending.

**What changed vs v2 (pre-F27 scorecard):** the F27 cat-head swap unlocked cat lifts that were previously tied-σ (AZ +0.74 → +3.73 p=0.0312; AL 0.00 → +4.13). AL reg Acc@10 now *crosses* STL STAN for the first time under MTL. But F21c exposed a stricter-than-STL-STAN ceiling: STL with the graph-prior head alone dominates the MTL coupling on reg by 12–14 pp. The paper's MTL contribution is therefore restated as: **joint-single-model deployment with a strict cat-F1 win over STL, accepting a 12–14 pp reg Acc@10 cost vs matched-head STL**.

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
