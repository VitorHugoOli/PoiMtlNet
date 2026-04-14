# Critical Review — Report v2 & BRACIS Submission

**Date:** 2026-04-13
**Reviewer:** Internal critical review (LLM agent cross-checking data, docs, and claims)
**Subject:** `articles/report_orientador/report_v2.tex` and the BRACIS abstracts at its end

This document captures the issues that should be addressed before sending the report to the advisor and — more importantly — before submitting the paper to BRACIS 2026.

---

## 1. Summary of Claims the Report Makes

1. **CBIC 2025** published that MTL did not help (DGI + FiLM + NashMTL): Florida Cat F1 = 47.7%, Next F1 = 33.9%.
2. **This work** replaces embedding (Fusion 128D), architecture (DSelectK), optimizer (Aligned-MTL) and jumps to Alabama Cat F1 = 80.8% / Next F1 = 27.3% / Joint = 0.540, and Florida Cat F1 = 78.3% / Next F1 = 36.8% / Joint = 0.575 (final 5-fold, confirmed).
3. **Headline finding:** multi-source fusion *requires* gradient surgery (CAGrad / Aligned-MTL). Equal-weight fusion is ~25% worse.
4. Proposes two BRACIS abstracts (Option A: conference short; Option B: journal full system).

**Numerical claims traceable to the CSVs match.** The narrative arc is compelling and substantially supported. The issues below are about *claims that go slightly beyond what the evidence currently shows* and about *controls that were not run*.

---

## 2. Critical Issues — Must Fix Before Submission

### 2.1 Missing control: single-task-fusion baseline

The CBIC paper's headline finding was "MTL ≈ single-task". The new report shows MTL-fusion beats CBIC-MTL by +34.5 p.p., but **never shows single-task-fusion numbers**. If single-task-fusion also gets Cat F1 ≈ 80%, then "MTL helps" is *again* wrong — the improvement would just be from richer embeddings, not from multi-task learning itself.

**This is the #1 concern a reviewer will raise.** The entire "MTL helps when configured right" narrative depends on showing that MTL-fusion > single-task-fusion. Without this control, the paper claims something it hasn't actually tested.

**Action:** Run single-task-fusion (category alone and next alone) on Alabama and Florida before submission.

### 2.2 Batch-size confound — documented in STAGE_1_ANALYSIS.md but hidden from the report

CAGrad and Aligned-MTL force `gradient_accumulation_steps=1` (effective batch 4096); equal_weight, db_mtl, and uncertainty_weighting use `gradient_accumulation_steps=2` (effective batch 8192). The 25% joint score gap between optimizer classes is *partly* a batch-size artifact.

STAGE_1_ANALYSIS.md explicitly flags this and recommends a matched-batch confirmation. **The report does not mention the confound**, and Stage 3 did not include an equal_weight fusion config at matched batch size.

**Action:** Run at least one equal_weight fusion configuration with `gradient_accumulation_steps=1, bs=4096` on Alabama. If it still loses by a large margin, the gradient-surgery claim is confirmed. If it closes the gap, the headline finding needs to be reframed.

### 2.3 Unfair comparison with HAVANA and POI-RGNN

HAVANA operates on **raw graph adjacency and spatial features**. Our model consumes **pretrained HGI + Sphere2Vec embeddings** that were themselves trained on mobility graphs. These are different problem setups with different information available — not the same task with a better solution.

- "supera HAVANA por 15,5 p.p." is technically true (78.4% vs 62.9%) but ignores the input asymmetry.
- The team's own HAVANA *reproduction* on Florida gave **54.23% macro F1**, not 62.9%. The report uses the paper-reported number.

**Action:** Soften the claim. Use "outperforms published HAVANA numbers, noting different input representations" or similar. Do not remove the comparison, but add a qualifying footnote/sentence.

---

## 3. Numerical Issues

### 3.1 Evolution table mixes training regimes

The "Evolução do MTLnet (Alabama)" table presents three rows as if they were the same experiment evolving:

| Config | Folds × Epochs |
|--------|---------------|
| CBIC 2025 (DGI) | 5f × 50ep |
| Fase 1 (HGI only) | **2f × 15ep** |
| Este trabalho (Fusão) | 5f × 50ep |

By Finding 9 in PAPER_FINDINGS.md, going from 15 to 50 epochs adds ~+0.02 joint. So HGI-only at 50 epochs would likely be ~0.50–0.51, not 0.486 — shrinking the fusion-vs-HGI gap.

**Action:** Run HGI-only at 5f × 50ep with the best optimizer, or footnote the regime difference.

### 3.2 The "25% gap" is never defined precisely

Depending on how you compute it, the gap between top and bottom optimizer classes is:
- Top-3 avg vs bottom-3 avg: **+67% relative**, or +40% normalized
- Mid-top vs mid-bottom: **+28% normalized**
- Best top vs best "bad" group: **+23% relative**

The "~25%" number used in the report is defensible as a mid-range normalized gap, but never stated precisely.

**Action:** Define it explicitly in text (e.g., "top-10 mean joint is ~25% higher than bottom-15 mean joint").

### 3.3 "Fusão com peso igual perde 4,8% para HGI"

Stage 0 CSV:
- **Best** fusion (CGC+equal_weight): joint = 0.368 → gap vs HGI (0.386) = **4.7%**. Match.
- **Worst** fusion (base+equal_weight): joint = 0.297 → gap vs HGI = **23%**.

The report says broadly "fusão com peso igual" (meaning all variants) but the 4.8% figure refers specifically to the best fusion variant. This is ambiguous.

**Action:** Specify which fusion variant is being compared, or widen the statement ("entre 5% e 23% pior, dependendo da arquitetura").

### 3.4 The 73%/7%/20% contribution decomposition is undocumented

The claim "Embedding = 73% do ganho; Arquitetura = 7%; Otimizador = 20%" does not appear in any of the stage analyses, PAPER_FINDINGS.md, or KNOWLEDGE_BASE. Its derivation is unclear.

Because the three components interact (the optimizer only matters with fusion), the effects are **non-additive** — a clean percentage decomposition is methodologically suspect.

**Action:** Either show the sequential-ablation math (DGI → HGI → Fusion, each with its best optimizer, measured at matched training length) or drop the percentages and keep the qualitative claim.

### 3.5 No standard deviations reported

Every headline number is a mean only (Cat 80.8%, etc.). Advisors and reviewers expect mean ± std for any claim.

**Action:** Always report std. Recompute from per-fold metrics in the run directories.

### 3.6 "Time2Vec beneficia da base maior" is unsupported

The argument: Florida Next F1 = 37.1% vs Alabama Next F1 = 27.3% → "Time2Vec scaling with dataset size."

But CBIC DGI (no Time2Vec) also jumps from Alabama 26.6% to Florida 33.9% (+7.3 p.p.). The Florida-vs-Alabama delta cannot be attributed to Time2Vec without a Florida HGI-only control.

**Action:** Drop this claim, or run Florida HGI-only to isolate the Time2Vec contribution.

---

## 4. Abstract Overclaims

### 4.1 "+34.5 p.p." is cherry-picked
That's Cat F1 on Alabama. Next F1 gain is only +0.7 p.p. on Alabama, +3.2 p.p. on Florida. "Up to +34.5 p.p." is honest framing; stating it unqualified is not.

### 4.2 Abstract B: "over 100 experiments across six US states"
The actual completed work is **46 experiments in 2 states** (Alabama + Florida). California and Texas are pending. The abstract as drafted is misleading.

**Action:** Rewrite Abstract B to match reality, or wait until CA/TX finish.

### 4.3 Abstract B: "8 embedding engines compared"
The engines exist in the codebase but no systematic 8-engine comparison is shown in this report. The abstract implicitly promises more than is delivered.

### 4.4 "15:1 scale imbalance"
True for the category task (Sphere2Vec vs HGI = 15.2×). For the next task, the ratio is **8.7:1** (Time2Vec vs HGI). Using "15:1" globally overstates.

### 4.5 "State-of-the-art" phrasing
"State-of-the-art" is a strong term. Given the input-asymmetry issue with HAVANA (Section 2.3), using it without qualification will draw reviewer fire.

**Action:** Downgrade to "outperforms published baselines on the same benchmark (with different input representations)" or similar.

---

## 5. Missing Context

### 5.1 No std dev on headline numbers — already covered in 3.5

### 5.2 No FLOPs / wall-clock comparison with CBIC
The CBIC paper's secondary finding was that MTL took ~4× wall-clock and ~2× FLOPs vs single-task. The new report says nothing about compute cost. If the new config is still 4× slower than single-task-fusion, one CBIC finding still holds.

### 5.3 No DSelectK hyperparameter sensitivity
The winner is `dselectk(e=4, k=2, temperature=0.5)`. Settings are arbitrary. A reviewer will ask.

### 5.4 Only seed=42
PAPER_FINDINGS.md Finding 10 shows seed variance ~0.005 on joint. BRACIS_GUIDE recommends multi-seed confirmation.

### 5.5 Mechanism claim lacks gradient-cosine evidence
The claim "CAGrad/Aligned-MTL resolvem o conflito de gradiente entre fontes" is plausible but unverified. Stage 3 design mentions gradient-cosine tracking as a diagnostic, but no data is shown.

**Action:** Present the gradient-cosine data now, or downgrade "resolvem" to "são consistentes com a resolução de."

### 5.6 "Estatisticamente indistinguíveis (p > 0,6)" overstates
Paired t-test on 5 folds has very low power. "p > 0.6" means we can't reject the null — it doesn't prove equivalence.

**Action:** Rephrase to "no significant difference detected at 5 folds" or "differences within estimated variance."

---

## 6. Priority of Action Items (ordered)

### Must-do before BRACIS submission

1. **Run single-task-fusion** on Alabama (and Florida if time): category alone + next alone with the same fusion embeddings. (~2h each on Alabama, ~4h on Florida)
2. **Run equal_weight fusion with matched batch** (`gradient_accumulation_steps=1`) on Alabama. (~25 min)
3. **Run HGI-only at 5f × 50ep** with the best optimizer on Alabama, for the evolution table. (~25 min)
4. **Report std dev** for all headline numbers.
5. **Flag Florida as final 5-fold** in the abstracts (now confirmed; update the text).
6. **Soften HAVANA/POI-RGNN comparison** with an input-asymmetry note.
7. **Drop or justify** the 73%/7%/20% decomposition.
8. **Define the "25%" and "4.8%"** figures precisely.

### Should-do

9. **Add gradient-cosine diagnostic** for the mechanism claim.
10. **DSelectK hyperparameter sensitivity** (e, k, temperature) — at least 2–3 points.
11. **Multi-seed confirmation** of Stage 3 winner (2 extra seeds).
12. **Wall-clock/FLOPs comparison** with CBIC to close the "MTL is 4× slower" narrative.

### Nice-to-have

13. California and Texas results (strengthens cross-state claim).
14. Reproduce HAVANA/PGC on our data pipeline to confirm paper-reported numbers are achievable in our setup.

---

## 7. Overall Assessment

**Strongest parts:**
- Stage 1 optimizer-class stratification (top-10 ca/al vs bottom-15 eq/db/uw) is a genuinely interesting finding.
- Fusion scale-imbalance / source-selection analysis is publishable.
- Florida Stage 4 replicates the Alabama pattern at 5-fold (joint = 0.575, confirmed).
- CBIC self-comparison is a legitimate and strong contribution because it controls for team, data, and protocol.

**Biggest weakness:**
The paper-as-framed rests on **three unrun controls**: (a) single-task-fusion, (b) matched-batch equal-weight fusion, (c) HGI-only at 50 epochs. Without these, each of the three central claims is only partially supported.

**Risk to submission:**
If the report is sent as-is to collaborators and then to BRACIS, the comparison-fairness issues and missing single-task baseline will likely be blocking reviewer concerns. These are fixable in the week before the 2026-04-20 deadline if the new runs are started immediately.

**Bottom line:**
The empirical work is real and the headline numbers are trustworthy. The *claims* are slightly ahead of the evidence and the abstracts overstate the comparison strength. With 2–3 additional controlled runs and honest qualifications on HAVANA/POI-RGNN, this becomes a strong BRACIS submission. Without them, it is a defensible internal report but a risky paper.

---

## 8. Sources Cross-Checked

- `results/ablations/full_fusion_study/s0_fusion_1f_10ep/summary.csv`
- `results/ablations/full_fusion_study/s0_hgi_ref_1f_10ep/summary.csv`
- `results/ablations/full_fusion_study/s1_screen_1f_10ep/summary.csv`
- `results/ablations/full_fusion_study/s1_promoted_2f_15ep/summary.csv`
- `results/ablations/full_fusion_study/s2_heads_2f_15ep/summary.csv`
- `results/ablations/full_fusion_study/s3_confirm_5f_50ep/summary.csv`
- `results/ablations/full_fusion_study/s4_florida_5f_50ep/summary.csv` (final)
- `results/baselines/dgi/alabama/.../full_summary.json`
- `results/baselines/dgi/florida/bests/.../full_summary.json`
- `docs/full_ablation_study/runs/STAGE_1_ANALYSIS.md` (batch-size confound disclosed)
- `docs/full_ablation_study/FUSION_RATIONALE.md` (15:1 scale ratio, 90% HGI dependence)
- `docs/BRACIS_GUIDE.md` ("Don't omit the batch-size confound")
- `docs/PAPER_FINDINGS.md` (Finding 9: 15→50 ep ~+0.02 joint)
- `docs/baselines/BASELINE.md` (team HAVANA reproduction = 54.23% vs paper 62.9%)
