# Claims and Hypotheses Catalog — Check2HGI Study

Task pair: `{next_category (7 classes), next_region (~1K classes)}` on Check2HGI check-in-level embeddings.

## Paper's contribution — three intertwined claims (revised 2026-04-16 evening)

1. **Check2HGI (check-in-level contextual) improves next-category prediction over POI-level alternatives (HGI) and over published next-category baselines (POI-RGNN).** The check-in-level granularity — same POI visited twice yields different vectors — captures the user's *immediate* intent shifts that POI-level embeddings cannot. This is Check2HGI's design target and the paper's primary substrate claim. **CH16.**
2. **Per-task input modality** (check-in to the category head, region to the region head) is the Pareto-bidirectional MTL design; shared-modality or concat variants each collapse one head. Check2HGI uniquely enables this because it supplies both granularities simultaneously; HGI cannot produce per-visit variation. **CH03.**
3. **Bidirectional MTL preserves / improves both heads** under this design (category via Check2HGI's native strength, region via the shared-backbone bridge without regression). **CH01 / CH02.**

At the region level, Check2HGI and HGI converge (P1.5 confirmed tied Acc@10) because pooling to region smooths out per-visit variation. Future work (P6 encoder enrichment) targets the region side specifically; out of scope for BRACIS.

**Rule:** no claim enters the paper without `status ∈ {confirmed, partial}` and an evidence pointer.

---

## Tier A — Headline

### CH16 — Check2HGI check-in embeddings improve next-category F1 over HGI POI embeddings (PRIMARY SUBSTRATE CLAIM)

**Statement:** Single-task next-category macro-F1 on Check2HGI (check-in-level, per-visit contextual) is strictly greater than on HGI (POI-level, one-vector-per-POI) at matched compute on the Gowalla state-level corpus. Specifically, Δ = F1(Check2HGI) − F1(HGI) ≥ 2 pp on AL with non-overlapping std envelopes (5f × 50ep, seed 42, same head class).

**Why this claim exists:** Check2HGI was designed specifically as an HGI modification that introduces check-in-level contextual variation (same POI visited twice → different vectors). The *paper's primary contribution* is that this design choice lifts next-category prediction; the region-side tie (CH15) is expected because region-level pooling erases the per-visit variance Check2HGI adds. Without CH16 landing, the study's substrate justification collapses.

**Why HGI cannot match this architecturally:** HGI produces one vector per POI. Two check-ins at the same POI receive identical embeddings, discarding all temporal, co-visitor, and context-window structure. For next-category, where the user's *immediate* intent varies across visits of the same POI (e.g., visiting a café for breakfast vs. an evening meetup), this information is lost.

**Result (P1.5b-refair, 2026-04-17, AL, 5f × 50ep, seed 42, default `next_mtl` head, identical pipeline except substrate; user-disjoint folds via `StratifiedGroupKFold(groups=userid)`):**

| Metric | Check2HGI | HGI | Δ (pp) | std-overlap? |
|--------|-----------|-----|--------|--------------|
| Acc@1 | 40.18 ± 0.95 | 24.10 ± 1.54 | **+16.08** | No |
| **macro-F1** | **38.58 ± 1.23** | 20.29 ± 1.34 | **+18.30** | **No** |
| MRR | 62.35 ± 0.48 | 50.29 ± 1.32 | +12.06 | No |

All metrics favor Check2HGI with non-overlapping std envelopes over 5 folds. The +18.30 pp macro-F1 delta is approximately 14× the larger std — highly significant.

**Prior (leaky) numbers for reference** (same run but with non-grouped `StratifiedKFold`, 2026-04-16):

| Arm | Leaky F1 | Fair F1 | Leaky→Fair drop |
|-----|----------|---------|-----------------|
| Check2HGI | 39.16 ± 0.83 | 38.58 ± 1.23 | −0.57 pp (robust) |
| HGI | 23.48 ± 1.19 | 20.29 ± 1.34 | **−3.20 pp** (leaky) |
| Δ | +15.67 pp | **+18.30 pp** | CH16 grew +2.63 pp |

**Interpretation (crucial for paper):** Check2HGI's check-in-level contextual variance forces user-agnostic generalisation. HGI's POI-level vectors encode stable user-POI co-visit structure that memorises across leaky folds but doesn't generalise to unseen users. When folds enforce user-disjoint splits, HGI's advantage from memorisation evaporates (−3.20 pp) while Check2HGI holds up (−0.57 pp). The substrate claim is not just that Check2HGI wins — it wins *more* when the evaluation is harder and more realistic. This is the paper-quality finding.

**Source:** `docs/studies/check2hgi/results/P1_5b/next_category_alabama_{check2hgi,hgi}_5f_50ep_fair.json`.
**Test:** P1.5b — COMPLETE (refair 2026-04-17).
**Phase:** P1.5b.
**Status (pre-Phase-1):** `confirmed at AL only` under `next_single` head.

### CH16 update — 2026-04-27 Phase 1 substrate validation (matched-head + head-agnostic)

**Strengthens CH16 to head-invariant 2-state result.** Phase-1 grid (AL+AZ × {C2HGI, HGI} × {linear-probe head-free, next_gru matched-head, next_single, next_lstm}, 5f × 50ep, seed 42):

| State | Probe | C2HGI F1 | HGI F1 | Δ | Wilcoxon p_greater |
|---|---|---:|---:|---:|---:|
| AL | Linear (head-free) | 30.84 ± 2.02 | 18.70 ± 1.38 | **+12.14** | n/a |
| AL | next_gru (matched-head MTL) | 40.76 ± 1.50 | 25.26 ± 1.06 | **+15.50** | **0.0312** |
| AL | next_single | 38.71 ± 1.32 | 26.76 ± 0.36 | **+11.96** | **0.0312** |
| AL | next_lstm | 38.38 ± 1.08 | 23.94 ± 0.84 | **+14.44** | **0.0312** |
| AZ | Linear (head-free) | 34.12 ± 1.22 | 22.54 ± 0.45 | **+11.58** | n/a |
| AZ | next_gru (matched-head MTL) | 43.21 ± 0.78 | 28.69 ± 0.71 | **+14.52** | **0.0312** |
| AZ | next_single | 42.20 ± 0.72 | 29.69 ± 0.97 | **+12.50** | **0.0312** |
| AZ | next_lstm | 41.86 ± 0.84 | 26.50 ± 0.29 | **+15.36** | **0.0312** |

8/8 head-state probes positive at maximum-significance n=5 paired Wilcoxon (5/5 folds positive each). Δ range +11.58 to +15.50 pp.

**Source:** `docs/studies/check2hgi/results/probe/{alabama,arizona}_{check2hgi,hgi}_last.json` (Leg I) + `docs/studies/check2hgi/results/phase1_perfold/{AL,AZ}_{check2hgi,hgi}_cat_{gru,single,lstm}_5f50ep.json` (Leg II + C2) + `docs/studies/check2hgi/results/paired_tests/*_cat_*.json` (statistical tests).
**Status:** `confirmed at AL+AZ matched-head, head-invariant`. FL/CA/TX queued in `PHASE2_TRACKER.md`.

### CH18 — MTL B3 is substrate-specific (NEW, 2026-04-27)

**Statement:** The MTL B3 configuration (`mtlnet_crossattn + static_weight cat=0.75 + next_gru cat + next_getnext_hard reg, d=256, 8h`) requires the Check2HGI substrate. Substituting HGI into the same MTL configuration **breaks the joint signal**: at both AL and AZ, cat F1 drops by ~17 pp and reg Acc@10_indist drops by ~30 pp; MTL+HGI is even *worse than STL+HGI* on reg by ~37 pp at AL.

**Why this claim exists:** The substrate claim CH16 says C2HGI > HGI under matched-head STL. But the paper's deployment unit is MTL B3. We need to distinguish two possibilities: (i) MTL B3's win is "MTL > STL" and any reasonable substrate works; (ii) MTL B3's win is interactional — it specifically exploits Check2HGI's per-visit context. CH18 tests (ii) by direct counterfactual.

**Result (5f × 50ep, seed 42; "B3 (existing)" rows from `NORTH_STAR.md` post-F27 validation, "HGI counterfactual" rows new this Phase):**

| State | Substrate | cat F1 | reg Acc@10_indist | reg MRR | Δ_cat | Δ_reg Acc@10 |
|---|---|---:|---:|---:|---:|---:|
| AL | C2HGI (B3) | **42.71 ± 1.37** | **59.60 ± 4.09** | 30.74 ± 2.87 | — | — |
| AL | HGI (counterfactual) | 25.96 ± 1.61 | 29.95 ± 1.89 | (lower) | **−16.75** | **−29.65** |
| AZ | C2HGI (B3) | **45.81 ± 1.30** | **53.82 ± 3.11** | 27.66 ± 2.41 | — | — |
| AZ | HGI (counterfactual) | 28.70 ± 0.51 | 22.10 ± 1.63 | (lower) | **−17.11** | **−31.72** |

**Mechanism:** The MTL B3 configuration was tuned around Check2HGI's per-visit context. Substituting POI-stable HGI embeddings produces:
- **Cat head underutilises the embedding** — no per-visit variation to exploit; falls back to ≈ STL HGI cat F1 baseline.
- **Reg head's graph prior fails to combine productively** with HGI's smoother POI-level features — even worse than STL HGI gethard alone.

**Implication for paper:** the MTL win is *not* "MTL configuration over STL with any substrate". It is "MTL configuration paired with the Check2HGI substrate beats every alternative", which is a more constrained but also more interesting story — it means our two contributions (substrate + MTL configuration) are interdependent, not orthogonal.

**Source:** `results/hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260427_*` + `results/phase1_perfold/{AL,AZ}_hgi_mtl_{cat,reg}.json`. Existing C2HGI B3 numbers from `results/F27_validation/{al,az}_5f50ep_b3_cathead_gru.json` (per `NORTH_STAR.md`).
**Test:** Phase 1 Leg III — COMPLETE at AL+AZ. FL/CA/TX queued in PHASE2_TRACKER.
**Status:** `confirmed at AL+AZ`.

### CH19 — Per-visit context is the dominant mechanism behind CH16 (NEW, 2026-04-27)

**Statement:** ~72% of CH16's matched-head cat substrate gap is explained by Check2HGI's per-visit contextual variation. The residual ~28% is the embedding training signal itself (Check2HGI's graph topology + contrastive loss producing per-POI vectors that beat HGI's even after POI-mean pooling).

**Mechanism counterfactual:** POI-pooled Check2HGI = mean-pool the canonical Check2HGI vectors per `placeid` across all check-ins, applied uniformly to all visits at that POI. This kills per-visit variation while preserving Check2HGI's training signal.

**Result (AL, matched-head `next_gru` STL, 5f × 50ep, seed 42):**

| Substrate | Linear probe F1 | Matched-head STL F1 |
|---|---:|---:|
| Check2HGI (canonical) | 30.84 ± 2.02 | 40.76 ± 1.50 |
| **Check2HGI POI-pooled** | **23.20 ± 1.08** | **29.57** |
| HGI | 18.70 ± 1.38 | 25.26 ± 1.06 |

| Decomposition | Linear probe Δ | Matched-head Δ |
|---|---:|---:|
| Per-visit context (canonical − pooled) | +7.64 pp (~63%) | +11.19 pp (~72%) |
| Training signal (pooled − HGI) | +4.50 pp (~37%) | +4.31 pp (~28%) |

**Implication for paper:** the "per-visit variation" story is the dominant mechanism (matched-head STL gives even stronger per-visit signal than the head-free linear probe), but **not the whole story** — Check2HGI's training procedure produces per-POI vectors that outperform HGI's even before per-visit context enters the picture. Paper should acknowledge both contributions, not collapse them into one narrative.

**Source:** `results/probe/alabama_check2hgi_pooled_last.json` (linear probe) + `results/check2hgi_pooled/alabama/next_lr1.0e-04_bs1024_ep50_20260427_*` (matched-head STL). Code: `scripts/probe/build_check2hgi_pooled.py`.
**Test:** Phase 1 C4 — COMPLETE at AL.
**Status:** `confirmed at AL` (mechanism partial — ~72% per-visit, ~28% training signal). Extension to FL is `optional/pending` per `SUBSTRATE_COMPARISON_PLAN §6` — AL alone settles the mechanism unless reviewer asks for state-replication.

### CH17 — Check2HGI strongly surpasses published POI-RGNN next-category on Gowalla state-level

**Statement:** Our single-task next-category macro-F1 with Check2HGI exceeds POI-RGNN's published numbers (31.8% FL, 34.5% CA) on the matched states by ≥ 4 pp. The MTL configuration (P3) exceeds by a larger margin.

**Why this claim exists:** POI-RGNN is the most-cited comparable baseline for next-POI-category on Gowalla state-level data. Beating it establishes that Check2HGI is not merely incremental over HGI but competitive against the strongest published method on this data.

**Known prior values (HANDOFF.md, current study):**
- AL single-task next-category (Check2HGI, 5f × 50ep, prior run): **38.67%** macro-F1.
- POI-RGNN reported FL 31.8% / CA 34.5%. Direct AL comparison would need POI-RGNN run on AL — otherwise state-level comparison only.
- Δ on FL: Check2HGI single-task FL TBD (pending P1.5b extension to FL).

**Plus a related published HGI-based next-category number** (TBD reference — user's or related group's prior work using HGI for next-category on matched data). Frame as "HGI with the same downstream pipeline achieved X%; Check2HGI achieves Y%, a Δ of Z pp on matched data." This strengthens CH16 externally — CH16 is an internal controlled comparison on our pipeline; this is an external published comparison.

**Source:** POI-RGNN paper (FL/CA/TX published numbers); to-be-located HGI-for-next-category article (user-supplied reference TBD); our Check2HGI numbers (P1 + P1.5b + P3).
**Test:** reporting table in paper, not a separate training run. Numbers come from P3 headline + single-task baselines.
**Phase:** paper write-up after P3.
**Status:** `pending` — numeric comparisons pending. Need to locate the specific HGI-next-category reference the user plans to cite.

### CH01-INTERIM (AL development): **SUPERSEDED by F31 (2026-04-24)**

> **Status (2026-04-24):** this section is kept for audit. The MTL-dilution diagnosis it reports was specific to the pre-champion config (`mtlnet_dselectk + pcgrad + GRU`). The current champion B3 (`cross-attn + static(cat=0.75) + next_gru + next_getnext_hard`) on AL 5f × 50ep (F31, `results/F27_validation/al_5f50ep_b3_cathead_gru.json`) reaches **cat F1 42.71 ± 0.0137 (+4.13 pp over STL `next_mtl`)** and **reg Acc@10 59.60 ± 4.09 (+0.40 pp over STL STAN)** — the first AL cross of STL STAN by an MTL run. MTL no longer "dilutes" on AL under the champion config; the mechanistic finding below applies only to the superseded config.

First data point from 5f × 50ep MTL (mtlnet_dselectk+pcgrad, GRU region head + pad-mask fix, per-task modality, fair folds):

| | MTL | STL fair | Δ | σ-overlap |
|---|---|---|---|---|
| cat F1 | 36.08 ± 1.96 | **38.58 ± 1.23** | −2.50 | YES |
| reg Acc@10 | 48.88 ± 6.26 | **56.94 ± 4.01** | −8.06 | YES |

Δm = −14.12%; Pareto gate fails on both r_A and r_B.

**Mechanistic finding (important for paper framing):** MTL provides huge lift (+40 pp) when the task-b head is weak standalone (Transformer region: 7→47 via MTL), but **dilutes** when the task-b head is strong standalone (GRU region: 57→49 via MTL). The shared-backbone bottleneck creates a ceiling the strong standalone head already exceeds. See `results/P2/ch01_al_verdict.md`.

**Why CH01 is not abandoned:** AL is dev state; small data (10K) may not support 2-task capacity split. FL/CA/TX headline states (100K+ samples) have room to test whether MTL lift returns at scale. Also, different (arch, optim) combos may shift the dilution/transfer balance.

**Status `interim` — AL FAILS, FL/CA/TX verdict pending.**

### CH01 FL 1f × 50ep update (2026-04-17) — **ASYMMETRIC**, not uniformly failing

**Full FL verdict with both baselines in hand:**

| Task | FL MTL (1f×50ep) | FL STL fair (1f/5f) | Δ (pp) | r_i (rel) | Verdict |
|---|---|---|---|---|---|
| cat F1 | **64.78** | 63.17 (1f×50ep) | **+1.61** | **+2.55%** | ✅ MTL HELPS |
| reg Acc@10 | **57.05** | 68.33 ± 0.58 (5f GRU) | **−11.28** | **−16.50%** | ❌ dilutes |
| reg MRR | 27.49 | 52.74 (5f GRU) | **−25.25** | **−47.87%** | ❌ dilutes |

**Δm = ½(r_A + r_B) = −14.82%**
**Pareto gate: FAIL** (r_A > 0 ✓, r_B < 0 ✗)

**Refined mechanistic picture (combining AL + FL):**

| | AL (10K) | FL (127K) |
|---|---|---|
| cat F1 Δ | **−2.50 pp** (tied within σ) | **+1.61 pp** (clear lift) |
| reg Acc@10 Δ | −8.06 pp | −11.28 pp |

**Observations:**
1. **Category benefits from MTL when data is abundant.** On FL's 127K samples, MTL's shared-backbone signal transfer genuinely lifts cat F1 by +1.61 pp. On AL's 10K it doesn't have enough data to learn useful transfer. **Data-quantity IS a factor for the category side.**
2. **Region dilutes on both states.** The strong standalone GRU region head caps what the shared-backbone can provide regardless of data scale. **Backbone dilution is structural for the strong task.**

**Updated paper narrative:**

From "MTL helps both heads" (false) to:

> **"MTL on POI is a task-asymmetric tradeoff. The weaker-head task (category, 7 classes) benefits from shared-backbone signal transfer *when data is sufficient*. The stronger-head task (region, 1109 classes with GRU) is capped by backbone capacity and regresses. Per-task routing (MTLoRA, AdaShare) is needed to preserve the category lift while recovering region."**

This is a **much more interesting paper** than uniform failure. It motivates the ablation (CH03 per-task modality already + MTL_ABLATION_PROTOCOL's 4 techniques) as the solution to the asymmetry.

**Status: CH01 INTERIM updated to `failing-asymmetrically on 2/2 states, with a clean mechanism`.** Ablation proceeds next (step 1: RLW).

### CH01 — MTL {next_category, next_region} improves BOTH heads over single-task (HEADLINE, bidirectional)

**Statement:** On Check2HGI, the 2-task MTL with champion architecture + optimiser + per-task input modality (CH03) improves **both** heads over their respective single-task baselines at matched compute, on AL and FL. Specifically, all four of the following hold:

1. MTL next-category macro-F1 > single-task next-category macro-F1 (AL),
2. MTL next-category macro-F1 > single-task next-category macro-F1 (FL),
3. MTL next-region Acc@10 > single-task next-region Acc@10 (AL),
4. MTL next-region Acc@10 > single-task next-region Acc@10 (FL).

Multi-seed (n=15 = 3 seeds × 5 folds) so paired-test comparisons are well-powered.

**Single-task reference values (to beat):**
- AL `next_category`: 38.67% macro-F1 (Check2HGI check-in-emb input, 5f×50ep, prior run).
- AL `next_region`: **56.94% ± 4.01 Acc@10** (Check2HGI region-emb input, `next_gru` default hparams, 5f×50ep — see `results/P1/region_head_alabama_region_5f_50ep_E_confirm_gru_region.json`). **1.21× Markov-1-region (47.01%)**.
- FL `next_category`: TBD.
- FL `next_region`: **65.91% Acc@10** (Check2HGI region-emb input, `next_gru` default, 1f×30ep — see `results/P1/region_head_florida_region_1f_30ep_E4_fl_gru_region.json`). **Only 1.013× Markov-1-region (65.05%)** — 0.86pp margin. Dense-data regime where Markov coverage is high (~85% val rows have their (r_t) key in train transition map); neural generalisation has very little to add.

**Source:** HMT-GRN / MGCL show hierarchical MTL benefits both heads. Our contribution is testing this on check-in-level contextual embeddings with category (7 cls) as the coarsest task rather than POI (~11K cls).
**Test:** P3 — MTL champion vs. single-task per-head baselines (paired across folds × seeds).
**Phase:** P3.
**Status:** `pending`.

### CH02 — No per-head negative transfer under MTL (bidirectional statistical test via Δm)

**Statement:** Under the CH01 MTL config, the paired test rejects `H0: per-head delta ≤ 0` at α=0.05 for **both** heads on **both** states. This is the statistical teeth of CH01 — no head may silently regress.

**Joint score metric — Δm** (Maninis et al., CVPR 2019; Vandenhende et al., TPAMI 2021):

For fold i of a single (state, seed) run, define per-task relative improvements over single-task baselines:

```
r_A(i) = ( F1_mtl_A(i)   − F1_stl_A(i)  ) / F1_stl_A(i)                # next_category

r_B(i) = ½ · [ (Acc10_mtl_B(i) − Acc10_stl_B(i)) / Acc10_stl_B(i)
             + (MRR_mtl_B(i)   − MRR_stl_B(i))   / MRR_stl_B(i) ]      # next_region

Δm(i)  = ½ · (r_A(i) + r_B(i))
```

All metrics are higher-is-better, so no sign flip is needed (l_i = 0 in the Maninis formulation).

**Decision rule (bidirectional = primary gate):**

```
per-head success (state s)  ⇔  median_over_i(r_A(i)) > 0  AND  median_over_i(r_B(i)) > 0
CH02 passes (state s)       ⇔  Wilcoxon signed-rank test on {Δm(i)} vs 0  rejects H0 at α=0.05
                                AND  per-head success holds
```

The Pareto gate on `r_A`, `r_B` encodes the bidirectional thesis (no silent regression on either head); Wilcoxon on Δm carries the statistical power. Both must pass for both states.

**Sensitivity check (reported, not gating):** a geometric-mean variant `G = √((F1_mtl/F1_stl) · (HM(Acc10, MRR)_mtl / HM(Acc10, MRR)_stl))` using harmonic mean within Task B. If G agrees with Δm on the ranking, the Δm result is not a linear-averaging artifact.

**Test:** P3 paired comparison (MTL per-head vs single-task per-head), 15 paired samples per (state) from 3 seeds × 5 folds.
**Phase:** P3.
**Status:** `pending`.
**Source:** [Maninis et al., Attentive Single-Tasking of Multiple Tasks, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Maninis_Attentive_Single-Tasking_of_Multiple_Tasks_CVPR_2019_paper.pdf); [Vandenhende et al., MTL Dense Prediction Survey, TPAMI 2021 (arXiv:2004.13379)](https://arxiv.org/pdf/2004.13379); [Navon et al., NashMTL ICML 2022](https://arxiv.org/abs/2202.01017).
**Notes:** Δm is the reporting standard in the MTL balancing literature (NashMTL, PCGrad, LDC-MTL 2025, DB-MTL 2023 all use it). Adopting it makes our numbers directly comparable to the MTL-survey family. Closes C03.

### CH03 — Per-task input modality improves MTL over shared-modality input (ARCHITECTURAL CHOICE)

**Statement:** Feeding **different** input modalities to the two task heads — `check-in embedding sequence → category_encoder`, `region embedding sequence → next_encoder` — yields higher MTL metrics on **both** heads than any single-modality variant (both-check-in OR both-region) and higher than concat (both see `[check-in ⊕ region]`).

**Motivation:** P1 shows the two modalities have asymmetric task-value:
- Check-in emb: strong category signal, weak region signal (region head at 20% Acc@10 ≈ Markov-1-region floor 47% → actually WORSE than Markov).
- Region emb: strong region signal (region head at 54.68% Acc@10), unknown but likely weaker category signal (no check-in-level timing/context).
- Concat: `[B,9,128]` — forces both heads to see and filter out noise from the other modality; P1 showed concat is strictly worse than region-only for the region head (49.57% vs 53.33%).

The MTL architecture already has **two independent task-specific encoders** (`category_encoder`, `next_encoder`); routing different modalities through them is the natural design, with the **shared backbone** (FiLM / CGC / MMoE) as the cross-task bridge.

**Variants to test under identical MTL arch + optim:**

| Variant | Task A (cat) input | Task B (region) input |
|---|---|---|
| **per_task** (proposed) | check-in emb | region emb |
| concat | `[checkin ⊕ region]` | `[checkin ⊕ region]` |
| shared_checkin | check-in emb | check-in emb |
| shared_region | region emb | region emb |

**Prediction:** `per_task > concat > shared_checkin` (region side); `per_task ≥ concat > shared_region` (category side). If true, `per_task` is the CH01 headline config.

**P4-dev directional result** (AL, 1 fold × 20 epochs, default FiLM + NashMTL + `next_mtl` head; exploratory — final numbers will use P2 champion arch + `next_gru` head at 5f × 50ep × 3 seeds on CA/TX):

| Variant | Cat F1 | Reg Acc@10 | Reg MRR | Pareto |
|---------|--------|------------|---------|--------|
| **per_task** (cat=checkin, reg=region) | 36.66 | **33.19** | **16.38** | ✅ only bidirectionally strong |
| concat (both=[checkin⊕region], dim=128) | 35.10 | 12.16 | 5.53 | ❌ dominated by per_task |
| shared_checkin (both=checkin, dim=64) | **36.78** | 2.30 | 1.57 | ✅ cat-max (kills region) |
| shared_region (both=region, dim=64) | 20.19 | 34.44 | 16.38 | ✅ reg-max (kills category) |

**Reading:**

- `per_task` is the only variant that delivers usable performance on both heads simultaneously. Every other variant trades one task for the other or is dominated.
- `concat` is **strictly dominated** by `per_task`. Stacking both modalities into one input doubles the dim (64 → 128) without adding capacity and forces each head to filter noise from the wrong-modality channels. Confirms the P1 single-task concat < region-only observation.
- `shared_checkin` preserves category (check-in input is what category wants) but **crashes region to random** (2.3% Acc@10 on a 1109-class problem). Region needs region input — no amount of MTL sharing substitutes for the right modality.
- `shared_region` mirrors the failure: preserves region but **crashes category** (F1 36.78 → 20.19) because the category head loses check-in context.

**Note on absolute numbers:** these P4-dev runs used the legacy `next_mtl` transformer head for the region slot, which P1 showed collapses on 1109-class region (7.4% standalone). That caps the region numbers here. The final P4 run (after P2 champion + GRU head) will lift the per_task region number materially; the *ordering* across variants is what CH03 claims, and the ordering is robust to head choice because the input-signal argument is model-agnostic.

**Source:** `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep20_20260416_22{55,57,58}/summary/full_summary.json` (per_task, shared_checkin, shared_region); `mtlnet_lr1.0e-04_bs2048_ep20_20260416_2302/` (concat retry with `--embedding-dim 128`).
**Test:** P4 — 4-way comparison. Development signal on AL confirms the predicted ordering. Headline on CA/TX pending.
**Phase:** P4.
**Status:** `partial` — directional signal from AL development confirms per_task > all shared/concat variants on joint score; paper-ready confirmation needs CA/TX replication with P2 champion arch + GRU region head.
**Notes:** Supersedes the earlier "dual-stream concat" framing of CH03. Per-task modality is the architectural hypothesis this study is designed to validate, and AL dev data supports it.

---

## Tier B — Methodology & supporting

### CH04 — Region head is learnable beyond Markov-1-region floor (REPORTED COMPARISON, not gate)

**Statement:** The best-performing `next_region` head (from P1 head ablation) achieves Acc@10 strictly greater than the Markov-1-region floor on both AL and FL, with absolute improvement reported in percentage points.

**Floor:** `markov_1step_region` — previous-region transition counts with uniform top-K backoff. Honest, high-coverage. Markov-2/Markov-3 backoff hurts (transition sparsity), so Markov-1-region is the binding floor.

| State | Markov-1-region Acc@10 | Single-task GRU Acc@10 (5f×50ep) | Δ (pp) |
|-------|------------------------|----------------------------------|--------|
| AL | 47.01 ± 3.55 | **56.94 ± 4.01** | **+9.9** |
| FL | 65.05 ± 0.93 | **68.33 ± 0.58** | **+3.3** |

**Interpretation:** the region task is **learnable beyond Markov-1-region in both regimes**. The +9.9 pp on AL is substantial; the +3.3 pp on FL is modest, consistent with FL being near-saturated by short-horizon transition statistics (~85% val-row coverage in train transition map). Neural generalisation over 9-step context adds signal in both regimes, but diminishing as data density rises.

**History note:** earlier drafts set a "≥ 2×" multiplicative gate against a POI-level Markov baseline (21.3% AL / 45.9% FL). That baseline was degenerate (~50% fallback to top-K-popular because POI keys are too sparse in train transition map). The POI-level numbers are retained in `results/P0/simple_baselines/*.json` for continuity but are not used as a floor. The "2×" gate is retired; replaced with absolute pp deltas because pp-over-floor is the scale-invariant way to compare across regimes where floors differ by 18 pp. See CONCERNS.md §C08.

**Source:** `results/P1/region_head_{alabama,florida}_region_5f_50ep_E_confirm_*.json`; `results/P0/simple_baselines/{alabama,florida}/next_region.json`.
**Test:** P1 — COMPLETE.
**Phase:** P1.
**Status:** `confirmed` (as a reported comparison, not a gate).

### CH05 — Head choice matters for next_region: literature-aligned heads may outperform the default

**Statement:** Among `{next_mtl, next_gru, next_lstm, next_tcn_residual, next_temporal_cnn}`, the winner on next_region differs from the winner on next_category. Specifically, GRU-family heads (per HMT-GRN's approach) may beat the transformer default on the coarser region task.

**Test:** P1 head ablation — vary region head while keeping category head fixed.
**Phase:** P1.
**Status:** `confirmed` — `next_gru` wins Acc@10 on both check-in input (20.11%) and region-emb input (53.33%). Transformer (`next_mtl`) collapses at best_ep=1 under both LR settings (1e-2 and 5e-4) — genuine failure, not LR mis-tune. `next_temporal_cnn` is a close 2nd under region-emb input (51.83%).

### CH06 — Champion MTL architecture for {next_category, next_region} on Check2HGI

**Statement:** The full 5-arch × all-optimizer ablation identifies a champion (arch, optim) pair. Document whether expert-gating (CGC/MMoE/DSelectK) beats FiLM-only base MTLnet, and whether gradient-surgery optimisers beat equal_weight.

**Test:** P2 — screen (1f×10ep) → promote (2f×15ep) → confirm (5f×50ep).
**Phase:** P2.
**Status:** `pending`.
**Notes:** Pre-requisite: parameterise CGC/MMoE/DSelectK/PLE with TaskSet (~150 LOC × 4 variants).

### CH07 — Seed variance bound

**Statement:** The 3-seed × 5-fold std of next-category macro-F1 on the P3 champion is < 2pp.

**Test:** By-product of P3's multi-seed runs.
**Phase:** P3.
**Status:** `pending`.

### CH18 — Matched-head STL GETNext-hard dominated MTL-B3 on next-region at ≤1.5K-region scale (RESOLVED — Tier A as of 2026-04-26)

**Status update (2026-04-26):** This claim is **RESOLVED** by F48-H3-alt. The 12–14 pp STL-MTL gap was not structural to MTL — it was a single LR-schedule confound. Per-head LR (cat=1e-3, reg=3e-3, shared=1e-3, all constant) recovers the gap completely on AL (MTL **+6.25 pp above STL ceiling**) and closes 75% on AZ (-3.29 pp residual within 1.5σ). FL also closes the smaller B3-vs-STL gap. The mechanism: α (the graph-prior weight in `next_getnext_hard.head`) needs sustained-high LR to grow; per-head LR decouples α's regime from the cat-stability regime. Three orthogonal negative controls (F40 loss-side ramp, F48-H1 monolithic gentle constant, F48-H2 warmup-then-plateau) confirm H3-alt is the unique design that satisfies the joint cat+reg objective. **Tier B → Tier A** (gap closed; recipe paper-ready). Full derivation: `research/F48_H3_PER_HEAD_LR_FINDINGS.md` + `MTL_ARCHITECTURE_JOURNEY.md`.

**Updated headline numbers (5-fold × 50 epochs, seed 42, F48-H3-alt recipe):**

| State | STL `next_getnext_hard` Acc@10 | MTL-H3-alt Acc@10 | Δ (MTL − STL) | Outcome |
|:-:|---:|---:|---:|---|
| AL | 68.37 ± 2.66 | **74.62 ± 3.11** | **+6.25** | **MTL exceeds STL** |
| AZ | 66.74 ± 2.11 | 63.45 ± 2.49 | −3.29 | closes 75% of B3 gap |
| FL† | TBD (F37 4050) | 71.96 ± 0.68 | — | FL B3-vs-STL gap was small at 1f; H3-alt at 5f is the new reference |

The earlier B3 (50ep + OneCycleLR) numbers below remain accurate for the predecessor recipe — preserved as a comparand to demonstrate the per-head LR contribution.

---

**Original statement (preserved for audit, predecessor recipe B3 50ep + OneCycleLR):** On AL (1,109 regions) and AZ (1,547 regions), the single-task model `STL next_getnext_hard` (STAN backbone + hard `α · log_T[last_region_idx]` graph prior, trained single-task with the same aux side-channel pipeline as MTL-B3) delivers strictly higher **reg Acc@10, reg Acc@5, and reg MRR** than MTL-B3 at 5f × 50ep, with non-overlapping σ envelopes. The MTL coupling does not add value on the region head beyond the choice of head at ablation-state scale.

**Why this claim exists:** CH01/CH02 were formulated on the premise "joint training lifts both heads over their single-task baselines". F21c (2026-04-24) ran the matched-head STL control that CH01/CH02 never executed — using the same `next_getnext_hard` head as B3's task_b, but trained standalone via `scripts/p1_region_head_ablation.py`. The single-task run dominates the joint run on region at both ablation states, by 12–14 pp Acc@10. This is a methodological finding: the study's earlier "MTL > STL on region" framing compared MTL against unmatched STL heads (GRU, STAN without prior), overstating the MTL contribution.

**Why it is NOT a refutation of MTL:** MTL-B3 remains the single-model joint predictor. Deploying two STL models (one `GETNext-hard` for region + one matched cat head) doubles inference cost. MTL-B3 still lifts cat F1 over STL (CH01 post-F27: +4.13 pp AL, +3.73 pp AZ p=0.0312) — that contribution survives F21c. The paper's reframed contribution is "joint single-model deployment accepting a reg cost vs matched-head STL" rather than "MTL universally lifts both heads".

**Numbers (5-fold × 50 epochs, seed 42, `StratifiedGroupKFold(groups=userid)`):**

| State | Metric | STL `next_getnext_hard` | MTL-B3 | Δ (MTL − STL) | σ-overlap? |
|:-:|:-|---:|---:|---:|:-:|
| AL | Acc@10 | **68.37 ± 2.66** | 56.33 ± 8.16 | **−12.04** | No |
| AL | MRR | **41.17 ± 2.28** | 28.55 ± 5.33 | **−12.62** | No |
| AZ | Acc@10 | **66.74 ± 2.11** | 52.76 ± 3.92 | **−13.98** | No |
| AZ | MRR | **41.15 ± 2.13** | 26.40 ± 2.45 | **−14.75** | No |

Macro-F1 on region is *higher* for STL STAN (24.6 AL / 24.5 AZ) than for either STL GETNext-hard (~12) or MTL-B3 (~9) — the hard prior trades per-class calibration for top-K recall. Reported alongside Acc@10/MRR as the honest secondary metric.

**Source:** `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json`; full analysis in `research/F21C_FINDINGS.md`.
**Test:** F21c — COMPLETE on AL + AZ. FL 5f × 50ep pending (`FOLLOWUPS_TRACKER.md §F21c FL` — not launched; awaits headline-path decision).
**Phase:** post-B3 (matched-head STL baseline program).
**Status:** `confirmed at AL + AZ`; FL pending.

**Tier placement (updated 2026-04-26):** CH18 was originally filed as Tier B (methodological limitation). The escalation criterion was "(a) a future MTL variant closes ≥75% of the 12–14 pp gap without regressing cat F1". F48-H3-alt **satisfies (a) and exceeds it** — the gap is closed on AL (with surplus) and 75% closed on AZ. **Promoted to Tier A 2026-04-26.**

**F49 attribution (added 2026-04-27):** F49's 3-way decomposition shows the H3-alt reg lift on AL is architecture-dominant (+6.48 pp from architecture alone, +0.09 from co-adapt, −0.32 from cat-supervision). On AZ the architecture costs reg, and the multi-task wrap rescues part. On FL transfer is null. CH18's "MTL with per-head LR exceeds STL" stands; CH19 sharpens the *why* — the per-head LR enables the architecture (not cat-supervision) to do the work. See CH19 below for the full attribution chain and the methodological contribution this enables.

**Recipe (paper-strength MTL lift, validated 2026-04-26):**

```bash
--scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3
```

Cross-state validation: AL+AZ at 5-fold × 50ep on m4_pro (~30 min total), FL at 5-fold × 50ep on m4_pro (~4.3 h with batch=1024 to avoid MPS OOM). Cat preservation evidence (within ~2 pp of B3) holds on all three states; reg lift +6.7-15 pp over B3.

The earlier candidates (per-task weight clipping, prior-magnitude normalisation, per-fold transition matrix, PIF-style frequency prior) are no longer in the critical path — H3-alt's per-head LR achieves the lift with zero head-side or loss-side modifications, just an optimizer-level change.

---

### CH20 — Cat-supervision transfer is small (≤|0.75| pp) on AL/AZ/FL n=5; the H3-alt reg lift is architecture-dominant, not transfer (Tier A as of 2026-04-27)

(NB: this claim was originally numbered CH19 in F49 work. Renumbered to CH20 because the Phase-1 work independently took CH19 for the per-visit-context mechanism claim. Both Tier A; CH19 is the substrate-side mechanism, CH20 is the architecture-side.)

**Statement:** Decomposing the H3-alt reg lift via the F49 3-way isolation (encoder-frozen λ=0 / loss-side λ=0 / Full MTL) reveals that **cat-supervision transfer through L_cat is null/near-null on all three states n=5** (AL: −0.32 pp; AZ: +0.75 pp; FL: −0.52 pp; all within σ of zero). The conventional MTL framing — "cat training transfers signal that helps reg" — is empirically refuted at our scale. Per-state pattern:

- **AL (5f, ~2.7σ):** architecture +6.48 ± 2.4 pp over STL F21c; co-adapt +0.09 (null); transfer −0.32 (null). The H3-alt reg lift is **purely architectural** on AL.
- **AZ (5f, ~3.7σ):** architecture −6.02 ± 1.6 pp (overhead); co-adapt +1.98 (modest positive); transfer +0.75 (null/small). Classical MTL pattern (architecture costs reg, multi-task wrap rescues part).
- **FL (5f, F49c):** loss-side 72.48 ± 1.40, frozen 64.22 ± 12.03 (frozen-cat reg path is unstable on FL — per-fold Acc@10 spread 49–74), Full MTL 71.96 ± 0.68. Co-adapt = +8.27 (~0.68σ — direction matches AL/AZ); transfer = −0.52 (~0.34σ null). FL absolute architectural Δ vs STL pending F37.

**Why this claim exists:** the 2026-04-20 chain-of-4 framing (`archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md`) reported "+14.2 pp transfer at FL" + "uniform architectural overhead." Both were artefacts of (a) loss-side ablation under cross-attention not being a clean architectural isolation (the silenced cat encoder co-adapts via attention K/V — see Layer 2 below), and (b) `CONCERNS.md §C12` LR confound (MTL@1e-3 vs STL@3e-3). F49 fixes both via the encoder-frozen variant + H3-alt regime. The +14.2 pp transfer claim is now refuted at ≥9σ on FL n=5 alone, ≥18σ aggregate.

**Why this is a paper-grade contribution:**

- **Layer 1 (transfer claim):** "Cat-supervision transfer in cross-attention MTL on next_region is small at our scale; the H3-alt reg lift is architecture-dominant on AL and architectural-overhead-with-modest-rescue on AZ." Reframes the paper's MTL contribution from "we found a transfer mechanism" to "we found an architectural mechanism (the per-head LR + cross-attention pipeline) that lifts reg with cat training adding ≈ 0."
- **Layer 2 (methodological):** "Loss-side `task_weight=0` ablation is unsound under cross-attention MTL because the silenced task's encoder still co-adapts via attention K/V projections. Encoder-frozen isolation is the only clean architectural decomposition." Applies to MulT, InvPT, and any future cross-task interaction MTL with `task_weight=0` ablations.
- **Layer 3 (per-state mechanism):** AL/AZ patterns committable; FL absolute architectural Δ vs STL gated on F37 (4050-assigned).

**Source:** `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` (numbers, σ, Tree decision); `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` (gradient-flow analysis, design rationale); 4 regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`.
**Test:** F49 (planning), F49b (reproduction gate, PASSED), F49c (FL n=5).
**Phase:** F-series, post-H3-alt.
**Status:** `confirmed (Tier A) — 2026-04-27`. Layer 1 + Layer 2 paper-grade. Layer 3 pending F37 for FL absolute Δ.

---

### CH21 — MTL B3's lift over STL is interactional architecture × substrate, not transfer (TOP-LINE PAPER CLAIM, Tier A as of 2026-04-27)

**Statement:** The MTL B3 lift over single-task baselines on `next_region` and `next_category` is the joint outcome of two necessary mechanisms, neither sufficient alone:

1. **Substrate (CH18 + CH19):** Check2HGI's per-visit contextual variation is required. Substituting HGI into the same B3 configuration breaks the joint signal (cat −17 pp, reg Acc@10_indist −30 pp at AL+AZ; MTL+HGI is *worse than STL+HGI* on reg by ~37 pp at AL). Per-visit context accounts for ~72% of the cat substrate gap; the residual ~28% is Check2HGI's training signal.
2. **Architecture (CH20):** The cross-attention architecture under per-head LR (H3-alt) is what extracts the reg lift. Cat-supervision transfer is null on all 3 states n=5 (≤|0.75| pp); the H3-alt reg lift on AL is architecture-dominant (+6.48 pp from architecture alone vs STL F21c).

**The conventional MTL framing — "joint training transfers signal from cat to reg" — is empirically refuted at our scale** (≥9σ on FL n=5 alone vs the legacy +14.2 pp claim). The H3-alt reg lift comes from the cross-attention architecture extracting more reg signal from Check2HGI's per-visit-contextual substrate than STL can. Neither the substrate nor the architecture alone is the "cause" — both are necessary, and the win is interactional.

**Implications for paper framing:**

- **Headline claim** (Methods + Results): "We propose MTL B3 (`mtlnet_crossattn + static_weight + per-head LR + Check2HGI substrate`) which jointly delivers a paper-strength reg lift (+6.25 pp over matched-head STL on AL) and preserves cat F1 within ~2 pp of STL. We attribute the lift to **two necessary, complementary mechanisms** — substrate and architecture — and refute the conventional "MTL transfer" framing through a clean 3-way decomposition."
- **Mechanism story** (Discussion): "The substrate (Check2HGI) carries per-visit contextual variation that POI-level alternatives (HGI) cannot supply (CH16 + CH19); the architecture (cross-attention + per-head LR) extracts more reg signal from any input than STL can (CH20). Neither alone explains the lift — substituting HGI into the architecture breaks reg by 30 pp; the architecture alone with frozen-random cat features already extracts +6.48 pp on AL. The interaction is the win."
- **Methodological note** (CH20 Layer 2): loss-side `task_weight=0` ablation is unsound under cross-attention MTL because the silenced encoder co-adapts via attention K/V — encoder-frozen isolation is the only clean decomposition. Applies beyond our study.

**Source:** Combined evidence from CH18 (Phase-1 Leg III MTL substrate-counterfactual) + CH19 (Phase-1 C4 per-visit mechanism) + CH20 (F49 3-way decomposition). Synthesised in `SESSION_HANDOFF_2026-04-27.md §0.3` and `README.md` headline. The two studies were independent (Phase-1 substrate-side vs F49 architecture-side); they converge on this joint claim.
**Test:** Already done — both component findings are paper-grade. The joint claim itself doesn't require new experiments; it's the synthesis.
**Phase:** post-Phase-1 + post-F49.
**Status:** `confirmed (Tier A) — 2026-04-27`. Headline-paper-grade. **Most important claim entry in the catalog.**

---

## Tier C — Input-modality mechanism

### CH08 — Per-task modality gain is state-dependent

**Statement:** The Δ (MTL − single-task) on both heads differs between AL and FL. Prediction: FL's denser per-region data (34/class vs AL's 11/class) + stronger Markov-1-region floor (65% vs 47% Acc@10) leaves less headroom for MTL transfer on the region side; AL may show the larger proportional MTL gain.

**Source:** P0 simple-baselines show the two states have different learnability floors. P1 will quantify the single-task ceilings on FL.
**Test:** P4 cross-state comparison at the CH03 champion modality.
**Phase:** P4.
**Status:** `pending`.

---

## Tier D — Architecture exploration

### CH09 — Cross-attention between task-specific encoders > per-task modality (gated on CH03)

**Statement:** `MTLnetCrossAttn` with bidirectional cross-attention between the check-in-side and region-side encoders achieves higher MTL metrics on both heads than the per-task-modality champion from CH03.

**Motivation:** Per-task modality gives each head its preferred input but the only cross-task interaction is via the shared backbone. Cross-attention would let the region stream "peek" at check-in context and vice versa before the backbone, potentially capturing interactions the backbone can't.

**Test:** P5 — only runs if CH03 champion is identified AND the paired margin is not already saturating single-task ceilings (so additional arch capacity has room to contribute).
**Phase:** P5 (gated).
**Status:** `pending`.

---

## Tier E — Declared limitations

### CH10 — Gowalla state-level ≠ FSQ-NYC/TKY

**Statement:** Results not directly comparable to HMT-GRN/MGCL/GETNext on FSQ-NYC/TKY. External numbers in appendix only.
**Status:** `declared`.

### CH11 — Encoder enrichment is a separate research track (P6)

**Statement:** The P6 enrichment phase tests whether improving Check2HGI's input features (temporal, spatial, graph) lifts the downstream MTL task beyond what vanilla Check2HGI achieves. Requires literature review before implementation.
**Status:** `pending` (research phase).

---

## Tier F — Encoder enrichment (P6, research-gated)

### CH12 — Temporal enrichment (Time2Vec-like) improves next-category F1 over vanilla Check2HGI

**Statement:** Replacing the fixed 4D sin/cos temporal features in Check2HGI preprocessing with learnable multi-frequency time embeddings (Time2Vec-inspired) + time-gap + recency decay features improves the P3 champion's next-category macro-F1 by ≥ 1pp on AL.

**Source:** Time2Vec (Kazemi et al. 2019), TiSASRec (Li et al. 2020), ImNext (He et al. 2024) show learnable temporal encodings outperform fixed sin/cos in sequential recommendation.
**Test:** P6 ablation — enriched vs vanilla at matched MTL config.
**Phase:** P6.
**Status:** `pending` (requires literature review to finalise implementation).

### CH15 — Check2HGI and HGI region embeddings are statistically tied on the region task

**Statement:** Single-task `next_region` Acc@10 (5f × 50ep, `next_tcn_residual`, region-emb input) on Alabama is statistically indistinguishable between Check2HGI and HGI-sourced region embeddings.

| Substrate | Acc@1 | Acc@10 | MRR |
|-----------|-------|--------|-----|
| Check2HGI (region emb pooled from check-in vectors) | 21.76 ± 1.8 | **56.11 ± 4.02** | 33.4 ± 2.4 |
| HGI (region emb pooled from POI-level vectors) | 21.82 ± 1.50 | **57.02 ± 2.92** | 33.14 ± 1.87 |
| Δ (HGI − Check2HGI) | +0.06 | **+0.91** (within noise) | −0.26 |

Both std envelopes overlap heavily; the 0.91 pp delta is well within either arm's single-seed fold-to-fold std.

**Interpretation:** region-level embeddings converge to similar quality regardless of whether the upstream POI representation is check-in-level (Check2HGI) or POI-level (HGI). The pooling to region level smooths out the contextual variation that Check2HGI adds at the check-in level.

**Implication for paper framing (pivot from original hypothesis):**

The original claim expected Check2HGI to clearly win on the region task. It doesn't. The *meaningful* Check2HGI contribution is therefore not at the region-level input; it is at the **check-in-level** input where HGI *architecturally cannot compete* because HGI produces only POI-level embeddings (same POI visited twice → same vector). Check2HGI uniquely enables:

1. **Per-task MTL with distinct input modalities** (check-in emb for next-category, region emb for next-region) — P4 / CH03 tests whether this design choice is advantageous.
2. **Check-in-level contextual variation** — two check-ins at the same POI can differ in time, user, co-visitor structure, etc. HGI cannot see any of these at the POI-embedding level.

The paper's framing therefore shifts from "Check2HGI is a better embedding" to "**Check2HGI uniquely supports per-task MTL with distinct modalities, which we show (P4) gives a measurable advantage.**"

**Source:** `results/P1/region_head_alabama_region_5f_50ep_E_confirm_tcn_region.json` (Check2HGI), `results/P1/region_head_alabama_region_5f_50ep_P15_hgi_al_tcn.json` (HGI).
**Test:** P1.5 — COMPLETE.
**Phase:** P1.5.
**Status (pre-Phase-1):** `confirmed (tied) under next_tcn_residual`.

### CH15 reframing — 2026-04-27 head-coupled finding

**The CH15 verdict was head-coupled** to the sequence model (TCN, then STAN at the published-baseline rows). Phase-1 added two additional reg-head probes — STAN at AL/AZ/FL (`next_region/comparison.md`, existing) and matched-head `next_getnext_hard` (5f × 50ep, seed 42):

| State | Probe | C2HGI Acc@10 | HGI Acc@10 | Δ (C2HGI − HGI) | Wilcoxon p_greater |
|---|---|---:|---:|---:|---:|
| AL | STAN (existing CH15-style) | 59.20 ± 3.62 | 62.88 ± 3.90 | −3.68 (HGI > C2HGI) | — |
| AL | next_getnext_hard (matched MTL B3) | **68.37 ± 2.66** | 67.52 ± 2.80 | +0.85 | 0.0625 marginal · TOST δ=2 pp ✅ non-inf |
| AZ | STAN | 52.24 ± 2.38 | 54.86 ± 2.84 | −2.62 | — |
| AZ | next_getnext_hard (matched MTL B3) | **66.74 ± 2.11** | 64.40 ± 2.42 | **+2.34** | **0.0312** ✅ |

**Interpretation:** CH15's "HGI > C2HGI on reg" was an artefact of STAN's preference for POI-stable smoothness. Under the matched MTL reg head (`next_getnext_hard` = STAN + α·log_T graph prior), C2HGI's per-visit context combines productively with the prior — flipping the sign at AZ (significantly C2HGI) and closing the gap at AL within σ.

**Source:** `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json` (C2HGI side, F21c) + `results/P1/region_head_*_STL_*_hgi_reg_gethard_5f50ep.json` (HGI side, this Phase) + `results/paired_tests/{alabama,arizona}_acc10_reg_acc10.json`.
**Test:** Phase 1 reg STL grid — COMPLETE at AL+AZ. FL/CA/TX queued in PHASE2_TRACKER.
**Status:** `reframed (head-coupled finding)` — pure-substrate verdict on reg is now C2HGI ≥ HGI under matched-head; the STAN-head data is preserved as a head-sensitivity probe row, not refuted.

**Scope caveat (2026-04-24, post-F21c):** the tie was measured with `next_tcn_residual` STL on region-pooled embeddings. After F21c introduced `STL next_getnext_hard` (STAN + graph prior) as a new matched-head baseline, CH15 has not been re-tested under that head. A priori the graph-prior term `α · log_T[last_region_idx]` is substrate-agnostic (it reads from `region_transition_log.pt`, not from the embedding), so the tie is *expected* to hold — but the AL-pooled region embeddings do differ across Check2HGI vs HGI source, and the STAN backbone co-adapts differently. This is a minor follow-up item tracked in `FOLLOWUPS_TRACKER.md §3` (deferred, follow-up paper; only revisit if a reviewer asks whether CH16 replicates on reg under the graph-prior head).

---

### CH13 — Spatial enrichment improves next-category F1 over vanilla Check2HGI

**Statement:** Adding continuous geospatial positional encoding from (lat, lon) + distance-to-previous-POI + distance-to-region-centroid as node features improves the P3 champion's next-category macro-F1 by ≥ 1pp on AL.

**Source:** Sphere2Vec (Mai et al. 2023), Space2Vec (Mai et al. 2020). Current Check2HGI has no explicit spatial features — geography enters only via region assignment and graph structure.
**Test:** P6 ablation — enriched vs vanilla at matched MTL config.
**Phase:** P6.
**Status:** `pending` (requires literature review to finalise implementation).

---

## Summary dashboard

**Reconciled 2026-04-24** against `OBJECTIVES_STATUS_TABLE.md` v3 + `NORTH_STAR.md` (post-F27) + `review/2026-04-23_critical_review.md` + `research/F21C_FINDINGS.md` + `research/F27_CATHEAD_FINDINGS.md` + `research/B3_AZ_WILCOXON_VS_STL.md` + `research/B5_AZ_WILCOXON.md`.

| ID | Tier | Phase | Status | Decides |
|----|------|-------|--------|---------|
| **CH16** | **A** | **P1.5b** | **confirmed (AL only, +18.30 pp fair)** | **Check2HGI > HGI on next-category (PRIMARY SUBSTRATE).** AL only; F3 AZ HGI STL cat is the cheapest replication. |
| **CH17** | **A** | **paper** | **pending (audit done 2026-04-23)** | **Check2HGI > POI-RGNN.** Reproduced POI-RGNN FL 34.49 / CA 31.78 / TX 33.03 vs ours FL 63.17 STL / 66.01 MTL = +28–32 pp. Protocol audit in `docs/baselines/POI_RGNN_AUDIT.md`. |
| **CH01** | A | P3 / B5 | reframed → `OBJECTIVES_STATUS_TABLE.md §3` | "Bidirectional MTL lift" literal form no longer the headline. North-star (soft) gives ties/lifts on cat and region below STL; hard ablation delivers strict MTL-over-STL on AZ region (+1.01 pp, p=0.0312 F1). FL region below Markov under both variants. |
| **CH02** | A | P3 / B5 | reframed → F1 Wilcoxon done | "No negative transfer" now tested via paired Wilcoxon on AZ hard-vs-soft: cat Δ p=0.44 two-sided (no regression), region all 4 metrics p=0.0312 one-sided (lift). See `research/B5_AZ_WILCOXON.md`. |
| **CH03** | A | P4 | partial (AL dev only, 1f×20ep) | Per-task input modality > shared/concat, directional. Settled by design choice (north-star uses `task-a=checkin`, `task-b=region`); full P4 replication not executed and not paper-blocking — incorporated as configuration, not as a tested claim. |
| CH04 | B | P1 | retired | Region head validates — demoted to reported pp-delta over Markov-1-region. |
| CH05 | B | P1 | confirmed | Head choice matters for region task. Post-B5: GETNext-soft (north-star) > GRU ≫ NextHeadMTL on region. |
| CH06 | B | P2 / B5 / B3 / F27 | settled as `mtlnet_crossattn + static_weight(cat=0.75) + next_gru (task_a) + next_getnext_hard (task_b) d=256, 8h` (post-F27 2026-04-24) | See `NORTH_STAR.md`. |
| CH07 | B | P3 / F8 | pending (F8 multi-seed) | Seed-variance bound. Currently n=1 seed × 5 folds across champion rows. |
| CH08 | C | P4 / B5 | confirmed via scale-curve findings | AL → AZ → FL scale-dependent MTL behaviour documented in `research/B5_MACRO_ANALYSIS.md` + `research/B5_FL_SCALING.md`. |
| CH09 | D | P5 | done (cross-attn is the committed architecture) | Cross-attention IS the MTL architecture used across all P8/B5 champion runs. The `MTLnetCrossAttn` class with pcgrad + GETNext-soft is the north-star. |
| CH10 | E | — | declared | External-validity limit (Gowalla ≠ FSQ) |
| CH11 | E | P6 | deferred | Enrichment is a post-paper research track. |
| CH12 | F | P6 | deferred | Temporal enrichment (Time2Vec-like) |
| CH13 | F | P6 | deferred | Spatial enrichment (Sphere2Vec-like) |
| **CH15** | **B** | **P1.5** | **confirmed (tied)** | **Check2HGI ≈ HGI on region task** (POI2HGI not separately tested; HGI-region used as proxy). The meaningful Check2HGI advantage is on the cat-input side (CH16) — pivot documented 2026-04-16. **Scope caveat (2026-04-24):** tie measured with `next_tcn_residual` STL head. Not re-tested under STL `next_getnext_hard` or under the MTL coupling; CH18-class matched-head comparisons could change the picture. Tracked as low-priority follow-up. |
| **CH18** | **A** | **F21c → F48-H3-alt → F49** | **resolved 2026-04-26 + sharpened 2026-04-27** | F21c gap closed by H3-alt per-head LR (AL exceeds STL by +6.25 pp; AZ closes 75%; FL beats Markov+STL GRU). F49 attribution reveals the lift is architecture-dominant (AL +6.48 pp from architecture alone), not transfer-driven. Tier A. See full section above + CH19. |
| **CH19** | **A** | **Phase-1 C4** | **confirmed 2026-04-27** | **Per-visit context = ~72% of CH16 cat substrate gap**; training signal residual = ~28%. POI-pooled C2HGI counterfactual (mechanism partial): linear probe AL Δ canonical−pooled=+7.64 (~63%); matched-head AL Δ=+11.19 (~72%). Source: `results/probe/alabama_check2hgi_pooled_last.json` + `results/check2hgi_pooled/alabama/...`. |
| **CH20** | **A** | **F49** | **confirmed 2026-04-27** | **Cat-supervision transfer ≤ |0.75| pp on AL/AZ/FL n=5; H3-alt reg lift is architecture-dominant.** Refutes legacy +14.2 pp transfer claim at ≥9σ on FL alone. Methodological contribution: loss-side `task_weight=0` ablation is unsound under cross-attention MTL (silenced encoder co-adapts via K/V); encoder-frozen isolation gives the only clean architectural decomposition. Layer 1 + Layer 2 paper-grade now; FL absolute architectural Δ vs STL pending F37. |
| **CH21** | **A** | **synthesis** | **confirmed 2026-04-27 — TOP-LINE PAPER CLAIM** | **MTL B3's lift is interactional architecture × substrate, not transfer.** Substrate (CH18+CH19) is necessary: HGI substitution breaks reg by 30 pp. Architecture (CH20) is necessary: cat-supervision transfer is null; H3-alt reg lift is architecture-dominant. Neither alone explains the lift. The conventional MTL "transfer" framing is empirically refuted. See full statement above + `SESSION_HANDOFF_2026-04-27.md §0.3`. |

### Post-B5 additions (not in the original CH01-CH17 numbering, captured in `OBJECTIVES_STATUS_TABLE.md` and `review/2026-04-23_critical_review.md`)

| Ref | What | Status |
|---|---|---|
| CH-M4 | Cross-attn closes the MTL→STL cat gap | locked on AL; AZ +1.05 n=5; FL +3.29 n=1 |
| CH-M5 | Fair LR (max_lr=0.003) dominates architecture | locked |
| CH-M6 | Scale-curve (AL → AZ → FL) | documented, but FL n=1 limit noted |
| CH-M7 | Markov-k monotone degrade | locked |
| CH-B5-AZ | MTL-GETNext-hard (B-M9d, pcgrad) strictly > STL STAN on AZ region | confirmed on B-M9d only, n=5 paired (p=0.0312 on Acc@10/Acc@5/MRR/F1). **Caveat:** the current north-star B3 (`static + next_gru + hard`) lands at Acc@10 53.82 (tied σ with STAN; F19-followup Wilcoxon Δ=−0.81, 2/5 folds positive). The strict lift survives in B-M9d → kept as ablation row, not as B3's claim. |
| **CH18** | **Matched-head STL `next_getnext_hard` (graph prior alone) > MTL-B3 on reg Acc@10 at ≤1.5K-region scale** | see CH18 below (Tier B) |
| CH-B5-FL-fail | FL-hard cat head fails to train (gradient starvation) | diagnosed with JSON-level evidence; F2 is the rescue test |
