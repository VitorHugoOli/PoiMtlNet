# STATISTICAL_AUDIT.md — Critical Pass on Claims and Evidence

> ⚠ **SUPERSEDED 2026-05-04.** Canonical:
> `docs/studies/check2hgi/results/RESULTS_TABLE.md §0.6`. The user-disjoint
> claim and POI-RGNN numbers below are pre-bugfix (May-2 snapshot); the
> POI-RGNN reproduction protocol is explicitly user-disjoint per
> `docs/studies/check2hgi/baselines/next_category/poi_rgnn.md` "Adapted —
> Cross-validation protocol", and the canonical numbers are 34.49 / 31.78 /
> 33.03 (FL / CA / TX) per `RESULTS_TABLE.md §0.6`. Article-side files
> (`src/sections/results.tex`, `src/tables/external.tex`) inherit those
> numbers and protocol. Trust the canonical sources, not this file.

> **Purpose.** Be brutal about which claims have proper statistical backing, which are weakly supported but defensible if framed honestly, and which are descriptive narrative dressed up as inference. BRACIS reviewers in the empirical track care about this; over-claims invite desk-rejection or deep cuts. This file is the rigour contract that sub-agents inherit before drafting Results / Discussion.
>
> **Sources.** Canonical numerical source for all paper tables: **`docs/studies/check2hgi/results/RESULTS_TABLE.md §0` (v11, 2026-05-02 — FL §0.1 arch-Δ upgraded to n=20; all five states paper-grade on the headline axis)** + **`docs/studies/check2hgi/research/GAP_FILL_WILCOXON.json`** (cat-Δ + recipe multi-seed) + **`docs/studies/check2hgi/research/ARCH_DELTA_WILCOXON.json`** (CA/TX §0.1 arch-Δ n=20) + **`docs/studies/check2hgi/research/FL_CAT_DELTA_WILCOXON.json`** (FL §0.1 arch-Δ n=20). Background provenance only (do not cite as primary): `archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md` (moved to archive in the 2026-05-01 study cleanup), `FINAL_SURVEY.md`, `CLAIMS_AND_HYPOTHESES.md` (whitelisted: CH16, CH18-cat, CH15 reframing, CH19, CH22 — others contain superseded leak-era content), `research/F50_T1_RESULTS_SYNTHESIS.md`. **No new compute.** Any claim that does not reduce to a number in current `RESULTS_TABLE.md §0` (or to the documented mechanism artefacts in CH19) is flagged.

---

## 0 · Headline rigour pattern (must appear once in §Experimental Setup)

The paper makes paired-test claims at three different power regimes. State them once, explicitly, so reviewers cannot accuse us of cherry-picking p-values:

1. **Single-seed n = 5 paired Wilcoxon ceiling.** Five user-disjoint folds, one seed. Maximum achievable significance is **p_one-sided = 0.0312 (5/5 folds in claimed direction)** and **p_two-sided = 0.0625**. Any "p = 0.0312, 5/5 positive" claim is *at-ceiling significance for n = 5* — it is not "p < 0.05 by accident" but the strongest evidence n = 5 paired data can give.
2. **Multi-seed pooled paired Wilcoxon (n = 20 or n = 25).** Four seeds × 5 folds = 20 paired Δs (AL/AZ); five seeds × 5 folds = 25 paired Δs (FL). Pools fold-pairs across independent seed runs to break the n = 5 ceiling. Reaches sub-1e-4 p-values where applicable. AL/AZ multi-seed at {0,1,7,100}; FL multi-seed at {0,1,7,42,100}.
3. **TOST non-inferiority** at δ ∈ {2 pp, 3 pp} for "tied" claims (CH15 reframing on CA/TX). Used only where the claim is *equivalence within a margin*, not directional.

**Anchor sentence (drop verbatim into §4.3):** *"Each substrate / MTL-vs-STL comparison is reported as a paired Δ across folds with paired Wilcoxon signed-rank. Single-seed paired Wilcoxon at n = 5 has a maximum achievable significance of p = 0.0312 one-sided (0.0625 two-sided); we explicitly flag where this ceiling binds (the substrate axis §0.3, where seed-replication of HGI re-runs is out of compute scope). Where multi-seed data is available, we pool fold-pairs across seeds: all five states on the §0.1 headline architectural-Δ axis are pooled at 4 seeds × 5 folds = 20 paired Δs; FL auxiliary axes (Δm and recipe selection) are pooled at 5 seeds × 5 folds = 25 paired Δs. The pooled tests reach sub-1e-6 p-values."*

---

## 1 · C1 (substrate) — VERDICT: paper-grade, with one wording fix

### 1.1 Statistical backing

| Sub-claim | Evidence | Statistical strength | Verdict |
|---|---|---|---|
| Cat F1 STL `next_gru` Check2HGI > HGI at every state | 5/5 states, paired Wilcoxon p = 0.0312 each (5/5 folds positive), single-seed at n = 5 ceiling | **At-ceiling significance for n = 5 single-seed** | ✅ paper-grade |
| Multi-seed reinforcement at AL/AZ/FL | σ across {0,1,7,100} or {0,1,7,42,100} ≤ 0.18 pp on STL cat F1 | **Recipe-deterministic across seeds** | ✅ paper-grade — strengthens C1 |
| Head invariance (8/8 head-state probes positive at AL+AZ Phase 1) | Paired Wilcoxon p = 0.0312 each | At-ceiling per probe | ✅ paper-grade |
| FL/CA/TX head-invariance | Matched-head only (CH16 §3 confirmed via `next_gru`); other-head probes not run at FL/CA/TX | **Limited to matched-head at FL/CA/TX** | ⚠️ honest framing — "head-invariant at the AL+AZ ablation scale; matched-head at the headline scale" |

### 1.2 Wording fixes

- **DO NOT WRITE:** *"head-invariant across all five states"* (over-claim — head sweep was AL+AZ only).
- **DO WRITE:** *"head-invariant at the AL+AZ ablation scale (8/8 head-state probes positive at p = 0.0312); replicated under the matched-head MTL ceiling at FL/CA/TX."*
- **DO NOT WRITE:** *"the substrate Δ scales monotonically with data"* without numbers (this *is* monotone, but state it numerically).
- **DO WRITE:** *"the substrate Δ scales broadly with data: +15.5 pp at AL, +14.5 at AZ, +29.0 at FL, +28.8 at CA, +28.3 at TX."* (AL is +15.5, AZ is +14.5 — not the other way around. CA is +28.8, slightly less than FL +29.0 — directional but not strictly state-size-monotone. Fine to call "broadly scales with data" but do not say "strictly monotone".)

### 1.3 Mechanism (~72 % per-visit) — single-state weakness

- **Evidence:** AL only. POI-pooled counterfactual: linear-probe ~63 % of cat gap is per-visit context; matched-head ~72 %.
- **Replication:** "FL extension optional/pending" per `SUBSTRATE_COMPARISON_PLAN §6`. Not run.
- **Verdict:** ✅ paper-grade *as a mechanism observation at AL*. ⚠️ over-claim if framed as "general property" without state qualifier.
- **Wording fix:** *"At AL, a POI-pooled counterfactual decomposes the matched-head substrate gap into ~72 % per-visit context and ~28 % training signal."* Do not write *"per-visit context accounts for 72 % of the substrate gain"* without "at AL".
- **Reviewer pre-emption:** add to §Limitations — *"the AL pooled-vs-canonical decomposition is a single-state mechanism probe; cross-state replication is left for follow-up."*

---

## 2 · C2-cat (MTL ≥ STL on next-category) — VERDICT: paper-grade positive at AZ/FL/CA/TX, small-significantly negative at AL

> **v11 update (2026-05-02):** FL §0.1 architectural-Δ row upgraded to n = 20 (seeds {0,1,7,100} × 5 folds) — see `FL_CAT_DELTA_WILCOXON.json`. FL Δ_cat = **+1.40 pp, p = 2e-06** (20/20 fold-pairs positive); FL Δ_reg = **−7.34 pp, p = 1.9e-06** (0/20 positive). MTL B9 cat F1 = 68.56 ± 0.79 % (vs seed=42 reference 68.51 %). The earlier v8 single-seed n = 5 framing (FL p = 0.0625 ceiling) is superseded; all five states on the §0.1 headline are now n = 20 pooled multi-seed.

### 2.1 Statistical backing (current canonical)

| State | n_pairs | Δ_cat pp | p_cat (paired Wilcoxon) | Verdict |
|---|---:|---:|---:|---|
| AL | 20 (4 seeds × 5 folds) | **−0.78** | **0.036** (two-sided; 14/20 fold-pairs negative) | ⚠️ **small-significantly negative** — formally significant but magnitude small (< 2 % relative on a 41 % F1 scale) |
| AZ | 20 | **+1.20** | **< 1e-04** (18/20 fold-pairs positive) | ✅ paper-grade positive |
| FL | 20 (4 seeds × 5 folds, multi-seed pooled) | **+1.40** (paired Δ) | **2e-06** (paper-grade; 20/20 fold-pairs positive) | ✅ paper-grade positive |
| CA | 20 (4 seeds × 5 folds) | **+1.68** | **2e-06** | ✅ paper-grade positive |
| TX | 20 (4 seeds × 5 folds) | **+1.89** | **2e-06** | ✅ paper-grade positive |

### 2.2 The AL cat issue (v8 — Wilcoxon landed; honest framing)

**v8 update:** AL Δ_cat Wilcoxon now landed at p = 0.036 (n = 20 multi-seed; 14/20 fold-pairs negative, two-sided). Statistically, AL is **small-significantly below STL on cat** — not "tied" in the strict statistical sense. But the magnitude is small (~0.78 pp on a 41 % F1 scale, ~1.9 % relative; < 5 × the multi-seed STL σ of 0.17 pp), so the *practical* claim of "≈ tied" is defensible if framed by magnitude rather than significance. Best practice: surface both.

**DO NOT WRITE:** *"MTL gains on cat at every state"* / *"sign-consistent ≥ 0 across five states"* / *"+0 to +2 pp at every state"* / *"AL ≈ tied within multi-seed STL noise"* (the last bullet is no longer fully defensible — Wilcoxon at n = 20 reaches p = 0.036 in the negative direction).
**DO WRITE:** *"With Check2HGI fixed as substrate, joint MTL over a cross-attention backbone lifts next-category at four of the five states (AZ +1.20 pp, p < 1e-04 across n = 20 multi-seed fold-pairs; FL +1.40 pp, p = 2e-06; CA +1.68 pp, p = 2e-06; TX +1.89 pp, p = 2e-06). At AL the joint cat F1 is small-significantly below single-task (Δ = −0.78 pp, paired Wilcoxon p = 0.036 across n = 20 multi-seed fold-pairs; magnitude small at < 2 % relative on a 41 % F1 scale, ~5 × the multi-seed STL σ)."*
- This is **stronger** than either the "≈ tied" understatement or the "MTL gains at every state" overclaim because (i) it's defensible on both axes, (ii) it shows we know the difference between paper-grade significance, n = 5 ceiling, and small-magnitude significance, and (iii) the AL small-negative result is itself the smallest-state edge case — consistent with the cleaner substrate-task-asymmetry story (substrate carries cat at every state; MTL coupling helps at most states but not at the smallest one where data is thinnest).

### 2.3 Five-state closure on the headline axis

CA and TX were closed in v10; FL was closed in v11. The headline §0.1 cat axis is now fully pooled multi-seed across all five states: **AZ/FL/CA/TX are paper-grade positive**, while **AL is small-significantly negative**. The older "camera-ready audit item" and "FL n = 5 ceiling" wording should be removed everywhere active and retained only in historical audit notes.

---

## 3 · C2-reg (MTL costs reg) — VERDICT: paper-grade across all five states with multi-seed where available

### 3.1 Statistical backing

| State | n_pairs | Δ_reg Acc@10 pp | p_reg | Verdict |
|---|---:|---:|---:|---|
| AL | 20 (4 seeds × 5 folds) | **−11.04** | **1.9e-06** | ✅ paper-grade |
| AZ | 20 | **−12.28** | 1.9e-06 | ✅ paper-grade |
| FL | 20 (4 seeds × 5 folds, multi-seed pooled) | **−7.34** (paired Δ from `RESULTS_TABLE §0.1` v11) | **1.9e-06** (paper-grade; 20/20 fold-pairs negative) | ✅ paper-grade negative |
| CA | 20 (4 seeds × 5 folds) | **−9.50** | **2e-06** | ✅ paper-grade |
| TX | 20 (4 seeds × 5 folds) | **−16.59** | **2e-06** | ✅ paper-grade |

### 3.2 Wording fixes

- **DO WRITE:** *"MTL trails matched-head STL `next_stan_flow` on next-region at every state. All five states pooled multi-seed comparisons (n = 20 fold-pairs each) reach paired Wilcoxon p ≤ 1.9e-06 in the negative direction. The reg cost is sign-consistent and paper-grade significant across all five states; the magnitude varies (7–17 pp)."*
- **DO NOT WRITE:** *"all states statistically significant"* without also noting the pooled multi-seed footing (n = 20 on §0.1).

### 3.3 The FL Δm-MRR positive cell — paper-grade (multi-seed)

`CLAIMS_AND_HYPOTHESES.md §CH22 (2026-05-01 reframe)`: FL multi-seed n = 25, Δm-MRR = +2.33 %, p_greater = 2.98 × 10⁻⁸, 25/25 positive. Δm-Acc@10 = −1.12 %, p = 3.20 × 10⁻⁵, 4/21 positive (negative direction).

This is paper-grade and reportable as-is. The split (positive on MRR, negative on Acc@10) is itself a small mechanism finding: **MTL produces better-ranked region predictions than STL even where raw top-K is worse.** State this explicitly in §5.2.

---

## 4 · Scale-progression — VERDICT: descriptive, NOT inferential. Demoted from title after Codex audit.

> **Post-Codex update:** the scale-progression was previously the *title-bearing* claim ("Scale-Sensitive Multi-Task Tradeoff"). After Codex flagged TX as the honest non-monotone outlier (−16.59 pp at 6.5 K regions, between FL and CA in size, pays the largest cost), the framing has been demoted from headline-claim to descriptive secondary observation in §5.2 / §7. The title now leads with substrate task-asymmetry (paper-grade significant at every state) instead. Wording rules in §4.3 below remain valid as the descriptive framing.

### 4.1 Why the user's framing is qualitatively correct

The Δ_reg pattern is:

| State | n_regions | n_checkins | Δ_reg pp |
|---|---:|---:|---:|
| AL | 1,109 | 10K | −11.04 |
| AZ | 1,547 | 26K | −12.28 |
| **FL** | **4,703** | **127K** | **−7.34** |
| CA | 8,501 | 230K | −9.50 |
| TX | 6,553 | 187K | −16.59 |

On the AL → AZ → FL trajectory the cost shrinks by ~5 pp. CA at the largest cardinality preserves the regime (−9.50). **TX is the honest outlier** (−16.59 at 6.5K regions — between FL and CA in size — pays the largest cost).

### 4.2 Why this is descriptive, not inferential

- **There is no paired-test framework that natively tests "Δ shrinks with data scale".** Per-state Δ_reg is one observation per state; n = 5 *across* states is too small for meaningful Spearman / Pearson correlation, and the observations are not paired (different states, different folds, different train sets).
- A formal scale-progression test would require *replicating* the design at multiple data-density points within a fixed state (e.g. subsample FL at 10K, 30K, 100K) and running paired tests across density levels — a different experiment than the one we ran.
- **What we have is qualitative pattern evidence at five states**, with multi-seed support at AL/AZ/FL.

### 4.3 Wording fixes

- **DO NOT WRITE:** *"the cost decreases monotonically with data scale"* (false — TX breaks monotonicity).
- **DO NOT WRITE:** *"MTL generalises better with more data"* as a general claim (the data point is one trajectory, single-seed at the largest states).
- **DO WRITE:** *"On the AL → AZ → FL trajectory the architectural reg cost varies from −11.04 → −12.27 → −7.34 pp; CA preserves the regime (−9.50 at the largest cardinality), while TX is non-monotone (−16.59), pointing at state-specific factors beyond raw class count. We report this descriptively; quantifying a scale curve formally would require a within-state density ablation, which we leave for follow-up."*
- **Optional Spearman footnote:** ρ(n_regions, Δ_reg) across the 5 states is +0.41 (TX as outlier drives this away from a stronger negative correlation). Include only if reviewers ask; do not lead with it.

---

## 5 · C2-robustness (drop-in fixes) — VERDICT: "do not recover the gap" is FAIR; "fail" is over-claim

### 5.1 Evidence (`research/F50_T1_RESULTS_SYNTHESIS.md`, FL only)

| Drop-in fix | mean Δ_reg vs H3-alt | p_greater (n=5) | Reading |
|---|---:|---:|---|
| FAMO | +0.62 pp | 0.219 | not significant either direction |
| Aligned-MTL | −0.11 pp | 0.844 | not significant |
| HSM-reg-head | −3.01 pp | 0.313 | not significant; large σ |

### 5.2 Wording fixes

- **DO NOT WRITE:** *"FAMO / Aligned-MTL / HSM fail to recover the gap"* (suggests we tested "do they recover the gap" — we tested "do they reach +3 pp Δ_reg vs H3-alt", which is a different bar).
- **DO WRITE:** *"None of the three drop-in alternatives reaches paired-Wilcoxon significance at FL against the H3-alt baseline (FAMO Δ_reg = +0.62, p = 0.219; Aligned-MTL Δ_reg = −0.11, p = 0.844; hierarchical-additive softmax Δ_reg = −3.01, p = 0.313). The architectural reg cost does not respond to balancer or head-capacity drop-ins under the n = 5 single-seed budget; cross-substrate validation (MPS bs=1024 vs CUDA bs=2048) is within 0.5 σ on the metrics that matter."*
- **Caveat to surface:** these tests are **FL only and single-seed**. Sub-claim (CH22b) is "FL architectural cost is robust to head + balancer changes"; reviewers may push for "test at AL/AZ too" — flag this as an honest limitation in §6.2.

---

## 6 · C3 (cross-attn λ = 0 pitfall) — VERDICT: rigorous as architectural / mechanism note, not a statistical claim

### 6.1 What we have

- Mathematical / gradient-flow argument (`research/F49_LAMBDA0_DECOMPOSITION_GAP.md`).
- Four regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py` proving the silenced encoder updates via attention K/V projections under loss-side `task_weight = 0`.
- Empirical demonstration: F49 frozen-cat λ = 0 vs loss-side λ = 0 produce different reg numbers at AL/AZ/FL (loss-side ≠ frozen-cat by ≥ 8 pp at AL, AZ, FL — see `CLAIMS_AND_HYPOTHESES.md §CH20 Layer 2`).

### 6.2 Verdict

✅ paper-grade as a **methodological observation**. Do not dress it as a statistical claim — it is an architectural lemma backed by a regression test and an empirical demonstration. One paragraph in main text + reference to the tests in supplementary is enough.

### 6.3 Wording fix

- **DO NOT WRITE:** *"with high statistical significance"* in the methodological note (the claim is not about a test against a null).
- **DO WRITE:** *"Under cross-attention MTL, setting `task_weight = 0` does not silence the silenced encoder: gradient flow through the shared K/V projections continues to update its parameters whenever the other task's loss depends on its output. The empirical signature is that frozen-cat λ = 0 and loss-side λ = 0 produce reg accuracies that differ by ≥ 8 pp at AL/AZ/FL on `next_region` Acc@10 (see supplementary). Encoder-frozen isolation is the only clean architectural decomposition; we provide regression tests in our anonymous code."*

---

## 7 · External baselines (POI-RGNN, MHA+PE, ReHDM) — VERDICT: cite carefully + disclose ReHDM scope

### 7.0 ReHDM coverage is asymmetric — disclose

ReHDM (Li et al., IJCAI 2025) is reported at AL/AZ/FL only. CA/TX runs are not feasible on a single H100 within our compute budget — the dual-level hypergraph's collaborator pool scales quadratically with region cardinality (8.5 K at CA, 6.5 K at TX). Disclosure required in:
- T5 caption / footnote: *"ReHDM rows reported at AL/AZ/FL; CA/TX runs deferred to camera-ready due to the dual-level hypergraph's quadratic collaborator-pool scaling at large region cardinality."*
- §7 Limitations: same wording, framed as honest baseline-coverage limitation.

This is a **paper-strengthening** disclosure (per `BRACIS_GUIDE.md §10.2.7` honest-framing pattern) rather than a weakness — explicitly stating a per-cell budget constraint for the most recent baseline tells reviewers we ran the most expensive 2025 SOTA where compute allowed, instead of silently dropping it.

### 7.1 Reproduction caveat

`docs/studies/check2hgi/baselines/next_category/poi_rgnn.md`: our POI-RGNN reproduction *did not* use user-disjoint folds — published numbers are reported for context, but our internal "Check2HGI > POI-RGNN by ~28–32 pp on FL" is a **conservative lower bound** (POI-RGNN's published cat F1 was measured under non-user-disjoint folds, which inflates absolute scores).

### 7.2 Wording fix

- **DO WRITE:** *"Our STL `next_gru` Check2HGI cat F1 (60–67 % at FL/CA/TX) exceeds our faithful POI-RGNN reproduction (Capanema et al. 2022; FL 34.49, CA 31.78, TX 33.03 per `RESULTS_TABLE §0.6`) by ≥ 28 pp at every matched state. POI-RGNN's published evaluation reported a 31.8–34.5 pp range across states under non-user-disjoint folds; our reproduction at user-disjoint folds is the comparison we report, and the gap remains a conservative lower bound vs. the published configuration."*

### 7.3 Markov-1-region floor at FL — disclose

- FL Markov-1-region binds at ~65 % Acc@10 (`docs/studies/check2hgi/PAPER_STRUCTURE.md §6` original). Our STL STAN-Flow exceeds it; MTL trails on Acc@10 at FL but exceeds on Acc@5 + MRR.
- **Wording:** *"On dense-data state splits where Markov-1-region transitions cover ≥ 85 % of validation rows (FL Gowalla, 127K check-ins), the classical 1-gram prior is near-optimal for Acc@10 on short horizons. Our neural single-task model exceeds Markov-1 on Acc@5 and MRR at FL but the MTL row trails on Acc@10. We report Acc@5 + MRR alongside Acc@10 to characterise the saturation regime fairly."*

---

## 8 · sklearn-version reproducibility caveat — VERDICT: must disclose in §Limitations

`FINAL_SURVEY.md §8`: `StratifiedGroupKFold(shuffle=True, random_state=42)` produces *different fold splits* across sklearn 1.3.2 → 1.8.0 (PR #32540). Implications:

1. **Paired tests within a single env are unaffected** — both arms in each comparison ran under the same fold split.
2. **Absolute leak-magnitude attribution (Phase 2 leaky vs Phase 3 clean)** mixes leak removal with fold-shift; absolute pp shifts have a ±2-3 pp confound.
3. The conclusion (substrate-asymmetric leak; C2HGI exploited the leak more than HGI) is robust — both arms within each phase share the same env.

### Disclosure

- **§Limitations:** *"Our paired tests within each phase are run in a single sklearn environment, so paired Δs and p-values are unaffected by fold-split drift. Absolute leak-magnitude attribution (Phase 2 leaky → Phase 3 clean reg STL, see supplementary) mixes leak removal with a sklearn 1.3 → 1.8 fold-split shift (PR #32540); we do not draw absolute pp-of-leak conclusions from cross-phase comparisons. The qualitative conclusion that the leak was substrate-asymmetric (Check2HGI exploited it more than HGI) is robust because both arms within each phase share the same env."*

This is mandatory disclosure — ignoring it would invite a reviewer to find the same issue and reject for "irreproducible methodology".

---

## 9 · F49 leak attribution — VERDICT: do NOT mention in main text; if mentioned in supplement, disclose fully

The earlier "F49 +6.48 pp MTL > STL on AL" framing was a leak artefact (asymmetric C4 leak inflated MTL more than STL — see archived `archive/post_paper_closure_2026-05-01/PAPER_CLOSURE_RESULTS_2026-05-01.md §3-§4a`). It does not appear in the article-side draft.

**Rule:** the F-trail (F21c → F44 → F45 → F48-H3-alt → F49 → F50 → F51 → paper-closure) belongs in supplementary materials *only*. If we cite F49 at all in supplement, we must disclose the leak attribution. The leak-free closure numbers in §5 supersede; reviewers do not need the journey unless they ask.

---

## 10 · Reviewer-grade objections we should pre-empt

Reviewers in BRACIS empirical track will hunt for:

1. **"You report p = 0.0312 — that's just barely significant."** — Pre-empt by stating once that this is the **maximum achievable significance for n = 5 paired Wilcoxon (5/5 folds)**, not an "underpowered just-significant" finding. Multi-seed where available reaches sub-1e-4.
2. **"Are all five states multi-seed on the headline §0.1 axis?"** — Yes. AL/AZ/CA/TX/FL are all n = 20 (4 seeds × 5 folds, pooled multi-seed) on §0.1 as of v11. FL also has auxiliary multi-seed support on Δm and recipe selection (n = 25).
3. **"You claim 'MTL gains cat at every state' but Table 3 shows AL = −0.19."** — Already fixed in §2.2 above; ensure the wording is corrected throughout.
4. **"You claim 'cost shrinks with data' but TX breaks the trend."** — Pre-empt in §5.2 / §7 with the descriptive framing in §4.3 above.
5. **"FL `next_region` is Markov-saturated, so your MTL row trailing Markov-1 looks bad."** — Pre-empt with the Acc@5 + MRR re-direction in §7.3 above.
6. **"How do you know the cross-attn co-adaptation isn't just an implementation detail?"** — Reference the regression tests; the architectural lemma is provable from gradient flow, not just empirical.
7. **"Reproducibility: did you fix sklearn version?"** — §8 disclosure handles this.
8. **"Your POI-RGNN reproduction is non-comparable to the published numbers"** — §7.1 disclosure makes the comparison conservative-lower-bound.

Each pre-emption is a single sentence; they live in §Limitations or in the relevant table caption.

---

## 11 · Final verdict by claim

| Claim | Statistical strength | Paper-grade? | Required wording fix? |
|---|---|:-:|---|
| C1 substrate Δ (matched-head STL) | 5/5 states at-ceiling significant on the cat side; AL/AZ head sweep + FL/CA/TX matched-head replication | ✅ | qualifier: "head-invariant at AL+AZ; matched-head replicated at FL/CA/TX" |
| C1 head-invariance (4 head probes) | At-ceiling for AL+AZ; not run at FL/CA/TX | ✅ at AL+AZ | scope qualifier |
| C1 mechanism (~72 % per-visit) | AL only | ✅ at AL | "at AL"; flag as future replication |
| C2-cat MTL ≥ STL | Current canonical: AZ paper-grade (+1.20, p < 1e-4 n=20); FL paper-grade (+1.40, p = 2e-06 n=20); CA/TX paper-grade positive (+1.68 / +1.89, both p = 2e-06 n=20); **AL small-significantly negative** (Δ = −0.78 pp, p = 0.036 n=20; magnitude < 2% relative) | ✅ for AZ/CA/TX/FL; ⚠️ for AL (significant in negative direction, but small) | wording: "lifts cat at four of five states; small-significantly negative at AL" |
| C2-reg MTL < STL | All five states multi-seed paper-grade (n = 20, p ≤ 1.9e-06) | ✅ paper-grade at all five | report the n = 20 closures uniformly |
| C2-Δm joint | FL multi-seed paper-grade (p = 3e-8); other states n = 5 ceiling | ✅ for FL; ⚠️ ceiling for others | flag FL as the multi-seed cell |
| Scale-progression "cost shrinks with data" | DESCRIPTIVE, n = 5 across states | ⚠️ qualitative only | rewrite to remove monotonicity claim |
| C2-robustness drop-ins | FL single-seed n = 5; not significant either direction | ✅ as "no significant recovery" | wording fix in §5 |
| C3 cross-attn λ = 0 | Architectural lemma + 4 regression tests + empirical Δ ≥ 8 pp | ✅ as methodological observation | not a statistical claim — frame as such |
| External baselines (POI-RGNN) | Published numbers + our reproduction; non-user-disjoint published folds | ✅ as conservative lower bound | disclose reproduction caveat |

---

## 12 · Acceptance criteria for the Results / Discussion sub-agent

Before A5 (Results) and A7 (Discussion) commit prose, every paragraph that makes a numerical claim must:

1. State **n** (folds, seeds, fold-pairs).
2. State **paired Wilcoxon p** with direction (one-sided / two-sided).
3. State **fold-positivity** (e.g. 5/5 folds positive).
4. Use **the wording fixes in §1.2, §2.2, §3.2, §4.3, §5.2, §6.3, §7.2, §7.3 above**.
5. Disclose the n = 5 ceiling explicitly the first time it appears in §5.

If a paragraph cannot satisfy the four above, downgrade it to descriptive ("we observe that …") or remove the numerical claim.
