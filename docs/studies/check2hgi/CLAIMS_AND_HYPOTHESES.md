# Claims and Hypotheses Catalog — Check2HGI Study

> ⚠ **PAPER-FACING WHITELIST (current canon: v10, 2026-05-02).** Only the entries below are safe to cite as paper canon. The rest of this document contains pre-leak-free framing, superseded H3-alt narratives, and historic claim attributions that are **not paper-current**. When in doubt, cross-check against `results/RESULTS_TABLE.md §0` (v10) and the `CHANGELOG.md` timeline.
>
> **Whitelisted entries (paper-facing safe):**
> - **CH16** — Cat substrate advantage (Check2HGI > HGI on next-category, head-invariant at AL+AZ; matched-head replicated at FL/CA/TX). Numbers are leak-free and 5-state.
> - **CH18-cat** — Cat substrate advantage under MTL B9 (also paper-grade significant). The reg-side of the original CH18 ("MTL substrate-specific on reg") was a leak artefact and is **not whitelisted**.
> - **CH15 reframing** — Reg substrate parity / marginal HGI advantage under matched-head STL `next_stan_flow` (TOST tied at CA/TX, δ=2pp; FL δ=3pp). Sign-flipped from earlier framing under the F44 leak.
> - **CH19** — Per-visit context mechanism (~72 % AL / ~64 % AZ of cat substrate gap; two-state replicated 2026-05-03). Survives all leak-free re-measurements.
> - **CH22** — Δm joint score (leak-free 2026-05-01 reframe). FL multi-seed Δm-MRR positive at p = 2.98e-8 (n = 25); other states at n = 5 ceiling.
>
> **Not whitelisted (pre-leak-free or superseded):**
> - **CH01** / **CH02** / **CH03** — pre-Phase-1 framings; not paper-current.
> - **CH18-reg** (pre-2026-04-30) — was a leak artefact; CH15 reframing supersedes it.
> - **CH20** / **CH21** — F49 attribution narrative; the AL "+6.48 pp architecture-dominant" finding was a leak artefact; the Layer 2 methodological side-finding (cross-attn `task_weight=0` co-adaptation) survives independently and is referenced via `research/F49_LAMBDA0_DECOMPOSITION_GAP.md`.
> - **CH22b** — pre-multi-seed FL framing; superseded by the leak-free CH22 v8 numbers in `RESULTS_TABLE §0.2`.
>
> Sub-agents writing paper prose: read `articles/[BRACIS]_Beyond_Cross_Task/AGENT.md §8` for the same whitelist with article-side cross-references.

Task pair: `{next_category (7 classes), next_region (~1K classes)}` on Check2HGI check-in-level embeddings.

## Historical early-study framing (pre-leak-free, not paper-current)

The three bullets below are preserved for claim archaeology from the early study phase. They are **not** the current paper canon; the live paper-facing interpretation is the whitelist banner above plus `results/RESULTS_TABLE.md §0` (v10).

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

**AZ replication (added 2026-05-03, matched-head `next_gru` STL, 5f × 50ep, seed 42):**

| Substrate | Matched-head STL F1 |
|---|---:|
| Check2HGI (canonical) | 43.17 ± 0.28 |
| **Check2HGI POI-pooled** | **34.09 ± 0.63** |
| HGI | 28.99 ± 0.51 |

| State | Per-visit Δ | Training-signal Δ | Per-visit share |
|---|---:|---:|---:|
| AL | +11.19 pp | +4.31 pp | ~72% |
| AZ | +9.08 pp  | +5.10 pp | ~64% |

**Implication for paper:** the "per-visit variation" story is the dominant mechanism (matched-head STL gives even stronger per-visit signal than the head-free linear probe), but **not the whole story** — Check2HGI's training procedure produces per-POI vectors that outperform HGI's even before per-visit context enters the picture. Both components are positive at AL **and** AZ; §6.1 now carries a two-state replicated mechanism finding instead of single-state AL evidence.

**Source:** `results/probe/alabama_check2hgi_pooled_last.json` (linear probe) + `results/check2hgi_pooled/alabama/next_lr1.0e-04_bs1024_ep50_20260427_*` (AL matched-head STL) + `results/{check2hgi,check2hgi_pooled,hgi}/arizona/next_lr1.0e-04_bs1024_ep50_20260503_*` (AZ matched-head STL). Per-fold JSONs in `docs/studies/check2hgi/results/phase1_perfold/AZ_*_cat_gru_5f50ep_20260503.json`. Code: `scripts/probe/build_check2hgi_pooled.py`, launcher `scripts/run_AZ_pervisit_counterfactual.sh`.
**Test:** Phase 1 C4 — COMPLETE at AL+AZ.
**Status:** `confirmed at AL+AZ` (mechanism partial — per-visit dominant ~64–72%, training-signal residual ~28–36%). Two-state replicated mechanism. Extension to FL is `optional/pending` per `SUBSTRATE_COMPARISON_PLAN §6`.

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

### CH18 — Matched-head STL GETNext-hard dominated MTL-B3 on next-region (Tier A — scale-conditional, reframed 2026-04-28)

**Status update (2026-04-28, after F37 FL):** The original "RESOLVED by H3-alt" framing held on AL+AZ but **flips at FL scale**. With F37 STL F21c FL now landed (5f × 50ep, paired Wilcoxon), the per-state pattern is:
- **AL:** MTL H3-alt **+6.25 pp** above matched-head STL (architecture-dominant lift, F49 confirms).
- **AZ:** MTL H3-alt closes 75% of B3 gap; STL ceiling still −3.29 pp above MTL (within 1.5σ).
- **FL:** **Matched-head STL ceiling exceeds MTL H3-alt by −8.78 pp** (5/5 folds in negative direction, paired Wilcoxon p=0.0312, max-significance at n=5). The architectural Δ vs STL F21c (F49 Layer 3) is **−16.16 pp** at FL — heavy architectural cost when cat features are random-frozen.

The CH18 claim therefore is **scale-conditional, not state-general**. Tier A status retained for the AL paper-strength result; FL is reported as the contrasting regime. See `research/F37_FL_RESULTS.md` for full details.

**Updated headline numbers (5-fold × 50 epochs, seed 42, F48-H3-alt recipe):**

| State | STL `next_getnext_hard` Acc@10 | MTL-H3-alt Acc@10 | Δ (MTL − STL) | Wilcoxon (5/5 paired) | Outcome |
|:-:|---:|---:|---:|:-:|---|
| AL | 68.37 ± 2.66 | **74.62 ± 3.11** | **+6.25** | p=0.0312 (5/5 +) | **MTL exceeds STL** |
| AZ | 66.74 ± 2.11 | 63.45 ± 2.49 | −3.29 | n.s. (within 1.5σ) | closes 75% of B3 gap |
| FL | **82.44 ± 0.38** | 73.65 ± 1.25 | **−8.78** | **p=0.0312 (5/5 −)** | **STL ceiling above MTL** |

(FL MTL H3-alt is reported here at per-task best epoch on `top10_acc_indist`, matched to F49 conventions; STL F21c reports `top10_acc` over the full distribution. FL OOD share is ~0.6–1.1%; converting either metric to the other shifts the Δ by ≤0.7 pp — see `results/paired_tests/FL_layer3_after_f37.json`.)

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
- **FL (5f, F49c + F37 closing Layer 3, 2026-04-28):** STL F21c 82.44 ± 0.38; loss-side λ=0 72.48 ± 1.40; frozen-cat λ=0 64.22 ± 12.03 (per-fold reg-best epochs {2,14,9,4,2} → α-growth fails when cat is random); Full MTL H3-alt 71.96 ± 0.68. **Architectural Δ (frozen − STL) = −16.16 pp** (5/5 folds negative, paired Wilcoxon p=0.0312); co-adapt = +8.27 (~0.68σ); transfer = −0.52 (~0.34σ null). **Architecture is a heavy cost at FL scale; co-adapt only partially recovers, full MTL still −8.78 pp below STL ceiling p=0.0312.**

**Why this claim exists:** the 2026-04-20 chain-of-4 framing (`archive/research_pre_b5/CHAIN_FINDINGS_2026-04-20.md`) reported "+14.2 pp transfer at FL" + "uniform architectural overhead." Both were artefacts of (a) loss-side ablation under cross-attention not being a clean architectural isolation (the silenced cat encoder co-adapts via attention K/V — see Layer 2 below), and (b) `CONCERNS.md §C12` LR confound (MTL@1e-3 vs STL@3e-3). F49 fixes both via the encoder-frozen variant + H3-alt regime. The +14.2 pp transfer claim is now refuted at ≥9σ on FL n=5 alone, ≥18σ aggregate.

**Why this is a paper-grade contribution:**

- **Layer 1 (transfer claim):** "Cat-supervision transfer in cross-attention MTL on next_region is small at our scale (≤|0.75| pp on AL/AZ/FL n=5); the H3-alt reg lift is architecture-dominant on AL, architectural-overhead-with-modest-rescue on AZ, and **architectural-cost-at-scale on FL**." Reframes the paper's MTL contribution from "we found a transfer mechanism" to "we found an architectural mechanism that lifts reg on AL but **costs reg at FL scale**, with cat training adding ≈ 0 throughout."
- **Layer 2 (methodological):** "Loss-side `task_weight=0` ablation is unsound under cross-attention MTL because the silenced task's encoder still co-adapts via attention K/V projections. Encoder-frozen isolation is the only clean architectural decomposition." Applies to MulT, InvPT, and any future cross-task interaction MTL with `task_weight=0` ablations.
- **Layer 3 (per-state mechanism, CLOSED 2026-04-28):** With F37 STL F21c FL landed, the 3-state architectural-Δ pattern is **{AL +6.48, AZ −6.02, FL −16.16} pp**. AL is the only state where the cross-attention architecture (with frozen-random cat features) lifts reg above the matched-head STL ceiling. AZ and FL pay an architectural cost; on FL the cost is heavy and dominates the full MTL outcome (MTL still −8.78 pp below STL even with full cat training). Per-state mechanism: AL = architecture-dominant; AZ = classical (architecture costs, transfer/co-adapt rescue partial); FL = architecture costs heavily, co-adapt rescues ~half, transfer null.

**Source:** `research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md` (numbers, σ, Tree decision); `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` (gradient-flow analysis, design rationale); `research/F37_FL_RESULTS.md` (Layer 3 closure, 2026-04-28); 4 regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`; `results/paired_tests/FL_layer3_after_f37.json`.
**Test:** F49 (planning), F49b (reproduction gate, PASSED), F49c (FL n=5), **F37 (FL STL F21c, 2026-04-28 — Layer 3 closing)**.
**Phase:** F-series, post-H3-alt.
**Status:** `confirmed (Tier A) — Layers 1+2+3 closed 2026-04-28`. Per-state architectural Δ pattern is {AL +6.48, AZ −6.02, FL −16.16} pp; AL is the architecture-dominant outlier among the 3 headline states.

---

### CH21 — On AL, the MTL lift is interactional architecture × substrate; at scale (FL), the substrate carries the cat win while the architecture costs reg (TOP-LINE PAPER CLAIM, Tier A — reframed 2026-04-28 after F37)

**Statement (revised after F37 FL closing):** The MTL B3 result over single-task baselines is **scale-conditional**, with two state-dependent mechanisms:

1. **Substrate (CH18-substrate + CH19) — generalises across states:** Check2HGI's per-visit contextual variation is required for the cat win and prevents collapse. Substituting HGI into the same B3 configuration breaks the joint signal (cat −17 pp, reg Acc@10_indist −30 pp at AL+AZ; MTL+HGI is *worse than STL+HGI* on reg by ~37 pp at AL). Per-visit context accounts for ~72% of the cat substrate gap; the residual ~28% is Check2HGI's training signal. **CH16 (cat substrate advantage) is head-invariant at AL+AZ; FL replication queued via F36.**
2. **Architecture (CH20) — state-dependent:** The cross-attention architecture under per-head LR (H3-alt) is the *load-bearing* lever for reg, and its sign **depends on state**. AL: architecture lifts reg by +6.48 pp from cross-attn alone (frozen-random cat features). AZ: architecture costs reg by −6.02 pp; classical multi-task wrap rescues partially. **FL: architecture costs reg by −16.16 pp** (paired Wilcoxon p=0.0312, 5/5 folds negative); co-adaptation rescues ~half but full MTL still ends −8.78 pp below the matched-head STL ceiling. Cat-supervision transfer is null at all three states (≤|0.75| pp).

**The conventional MTL framing — "joint training transfers signal from cat to reg" — is empirically refuted at our scale (cat-transfer null at all 3 states; ≥9σ on FL alone vs the legacy +14.2 pp claim).** The architectural lift on AL is real and paper-grade, but **does not generalise to FL scale**. The cat-side MTL > STL relation does generalise (+0.94 pp at FL, +3-4 pp at AL/AZ), driven by the substrate.

**Implications for paper framing (revised):**

- **Headline claim** (Methods + Results): "We propose MTL H3-alt (`mtlnet_crossattn + static_weight + per-head LR + Check2HGI substrate`) and characterise its scale-dependent behaviour across three US states. **At Alabama (10K check-ins, 1,109 regions) the joint MTL exceeds matched-head single-task on both heads** (+0.94 to +6.25 pp). At Florida (127K check-ins, 4,702 regions) the substrate alone (Check2HGI per-visit contextual encoding) preserves the cat-side advantage, but the cross-attention architecture costs reg, so STL `next_getnext_hard` is the per-task ceiling. We attribute the AL win to a clean architectural mechanism via a 3-way decomposition, and refute the conventional MTL-transfer framing — cat-supervision contributes ≈0 to reg at all three states."
- **Mechanism story** (Discussion): "The substrate (Check2HGI) carries per-visit contextual variation that POI-level alternatives (HGI) cannot supply (CH16 + CH19); the architecture (cross-attention + per-head LR) extracts more reg signal from any input than STL can *at small region cardinality*, but the architectural cost grows steeply with cardinality (1.1K → 1.5K → 4.7K → architectural Δ +6.5 / −6.0 / −16.2 pp). The substrate-side and architecture-side mechanisms are **decoupled**: AL benefits from both; AZ and FL benefit only from substrate. The architecture is **not** a universal lever."
- **Methodological note** (CH20 Layer 2): loss-side `task_weight=0` ablation is unsound under cross-attention MTL because the silenced encoder co-adapts via attention K/V — encoder-frozen isolation is the only clean decomposition. Applies beyond our study.
- **Limitations note:** FL frozen-cat reg is unstable (per-fold Acc@10 spread {49.8, 51.4, 78.9, 76.6, 74.7}); architectural-Δ at FL has σ ~12 pp. The headline architectural cost at FL is robust (5/5 folds negative for the full MTL vs STL comparison; p=0.0312 max-significance) but the magnitude is uncertain pending multi-seed.

**Source:** Combined evidence from CH18 (Phase-1 Leg III MTL substrate-counterfactual + F37 FL flip), CH19 (Phase-1 C4 per-visit mechanism), CH20 (F49 3-way decomposition + F37 Layer 3 closing). Synthesised in `research/F37_FL_RESULTS.md` (2026-04-28). The two studies were independent (Phase-1 substrate-side vs F49 architecture-side); they now converge on a *scale-conditional* joint claim.
**Test:** All experiments done — both substrate (CH16 + CH18-substrate at AL+AZ; FL replication queued in F36) and architecture (CH20 Layers 1+2+3 at AL+AZ+FL) are paper-grade. Joint claim is the synthesis.
**Phase:** post-Phase-1 + post-F49 + post-F37.
**Status:** `confirmed (Tier A) — reframed 2026-04-28`. Headline-paper-grade. **Most important claim entry in the catalog.**

---

### CH22 — Joint Δm is Pareto-negative at 4/5 states; FL is the lone Pareto-positive cell on the MRR axis (Tier A, REFRAMED 2026-05-01 leak-free)

> ⚠ **2026-05-01 LEAK-FREE REFRAME — supersedes the original 2026-04-28 statement below.**
> The original CH22 result was extracted from pre-F44/F50 leaky log_T runs. Leak-free
> re-extraction on the Phase-3 / paper-closure pool (5 states, seed=42, per-fold paired Wilcoxon)
> **inverts the per-state Δm pattern:** AL/AZ flip from Pareto-positive → Pareto-negative;
> FL flips from Pareto-negative → Pareto-positive on MRR. CA + TX (new) both Pareto-negative.
>
> **Leak-free Δm scoreboard (cat F1 + reg MRR PRIMARY, cat F1 + reg Acc@10 SECONDARY):**
>
> | State | n_pairs | Δm-MRR (%) | n+/n− | p_greater | p_two | Δm-Acc@10 (%) | n+/n− | p_two |
> |---|--:|---:|:-:|---:|---:|---:|:-:|---:|
> | AL | 5  | **−24.84** | 0/5 | 1.0000 | **0.0625** | **−22.41** | 0/5 | **0.0625** |
> | AZ | 5  | **−12.79** | 1/4 | 0.9688 | 0.1250 | **−14.53** | 0/5 | **0.0625** |
> | **FL (multi-seed)** | **25** | **+2.33** | **25/0** | **2.98e-08** | **5.96e-08** | **−1.12** | 4/21 | **3.20e-05** |
> | CA | 5  | −1.61 | 1/4 | 0.9375 | 0.1875 | −6.85 | 0/5 | **0.0625** |
> | TX | 5  | −4.63 | 0/5 | 1.0000 | **0.0625** | **−11.60** | 0/5 | **0.0625** |
>
> FL is multi-seed (5 seeds × 5 folds = 25 paired Δs) using F51 paper-grade B9 dirs
> + paper_close + c4_clean STL reg multi-seed; AL/AZ/CA/TX are single-seed. The FL
> multi-seed extension lifts the FL Δm-MRR p-value from the n=5 ceiling (p=0.0312)
> to **p_greater = 2.98×10⁻⁸ across 25 fold-pairs**, sign-consistent at 25/25.
>
> **Verdict:** the joint Δm metric ratifies the classic MTL tradeoff finding from
> `PAPER_CLOSURE_RESULTS_2026-05-01.md` — at every state the MTL B9 architecture
> costs reg substantially while gaining ≤2 pp on cat; the joint metric reflects this.
> **The single MTL-positive cell is FL on MRR**, now at paper-grade significance
> (p ≈ 3×10⁻⁸). The MRR-vs-Acc@10 split at FL is paper-grade in both directions
> (Δm-MRR positive p < 1e-7; Δm-Acc@10 negative p ≈ 3e-5) — paper-worthy mechanism
> note: MTL produces better-ranked region predictions than STL on FL even where
> raw top-10 is worse.
>
> **Source:** `research/F50_DELTA_M_FINDINGS_LEAKFREE.md §3.5` (multi-seed) + `results/paired_tests/F50_T0_delta_m_FL_multiseed.json` + `results/paired_tests/F50_T0_delta_m_leakfree.json` (single-seed) + drivers `scripts/analysis/f50_delta_m_fl_multiseed.py` + `scripts/analysis/f50_delta_m_leakfree.py`.
> **Status:** `confirmed (Tier A) leak-free 2026-05-01 — reframed`. The original 2026-04-28 statement below is preserved for audit.

---

**Original statement (2026-04-28, leaky — kept for historical traceability):**

**Statement:** Under the MTL-survey-standard joint Δm (Maninis CVPR 2019 / Vandenhende TPAMI 2021), MTL H3-alt is Pareto-positive vs matched-head STL at small region cardinality (AL/AZ) and Pareto-negative at large cardinality (FL). The directional pattern is monotone in n_regions.

**Why this claim exists:** the per-state architectural-Δ pattern from F49 (AL +6.48 / AZ −6.02 / FL −16.16 pp) and the per-task gap pattern (AL +6.25 / AZ −3.29 / FL −8.78 pp on reg Acc@10) needed a *joint* metric to commit the scale-conditional reading rather than relying on per-task readings. Δm with paired Wilcoxon at n=5 max-significance closes that gap.

**Result (F50 Tier 0, 2026-04-28; PRIMARY metric = cat F1 + reg MRR, both clean across MTL/STL JSONs):**

| State | n_regions | Δm primary | n+/n− | Wilcoxon p_greater | Verdict |
|:-:|:-:|:-:|:-:|:-:|:-:|
| AL | 1,109 | **+8.70% ± 2.04** | 5/0 | **0.0312** ✓ | MTL Pareto-wins (n=5 ceiling) |
| AZ | 1,547 | **+3.19% ± 1.50** | 5/0 | **0.0312** ✓ | MTL wins on MRR (n=5 ceiling); marginal on top5/top10 |
| FL | 4,702 | **−1.63% ± 0.64** | 0/5 | 1.0 (p_two_sided=**0.0625**) | MTL Pareto-loses (n=5 ceiling on two-sided) |

n=5 minimum achievable p is 0.0312 (one-sided) / 0.0625 (two-sided). All three states cleanly hit their ceiling; the verdicts are at maximum-possible significance for n=5.

**Bonus finding (AZ MRR vs top-K asymmetry):** at AZ the MRR-based Δm is significantly positive (+3.19%, p=0.0312) while top5-based is null (−0.38%, p=0.500). MTL's reg head produces *better-ranked* predictions than STL even when raw top-K is similar — paper-worthy mechanism for the AZ-specific advantage.

**Implication for paper framing:** CH21's "scale-conditional" reading is now backed by a joint metric, not just per-task readings. The cat-side advantage (Δ_cat F1 in [+0.7%, +7.0%] across all 15 folds) is uniformly positive; the reg-side flip from AL win → AZ tie/marginal → FL loss tracks region cardinality monotonically.

**Source:** `research/F50_DELTA_M_FINDINGS.md` + `results/paired_tests/F50_T0_delta_m.json` + driver `scripts/analysis/f50_delta_m.py`.
**Test:** F50 Tier 0 — DONE 2026-04-28 (analysis-only, no compute).
**Phase:** F50.
**Status:** `confirmed (Tier A) 2026-04-28`. Backs CH21 with formal joint metric.

### CH22b — FL architectural cost is robust to head + balancer changes (Tier A sub-claim, 2026-04-29)

**Statement:** The negative-Δm at FL (CH22) is *not* an artefact of the H3-alt recipe; it survives three independent classes of architectural / optimisation drop-in alternatives at paper-grade n=5 paired Wilcoxon.

**Why this sub-claim exists:** without ruling out cheap drop-in alternatives (head capacity, magnitude balancing, direction alignment), CH22's "scale-conditional" reading could be argued to reflect a tunable parameter we haven't tuned. F50 Tier 1 closes that loop.

**Result (F50 Tier 1, 2026-04-29; per-task-best reg `top10_acc_indist` paired vs CUDA H3-alt 73.61 ± 0.83):**

| Test | Class of fix | mean Δreg | W+ (n=5) | Wilcoxon p_greater | Verdict |
|---|---|:-:|:-:|:-:|:-:|
| **T1.2-MTL HSM** | Head capacity (hierarchical-additive softmax) | **−3.01 pp** ± 9.95 | 10 | 0.3125 | ❌ FAIL +3 pp |
| **T1.3 FAMO** | Magnitude balancing (NeurIPS 2023) | **+0.62 pp** ± 1.50 | 11 | 0.2188 | ❌ FAIL +3 pp |
| **T1.4 Aligned-MTL** | Direction alignment (CVPR 2023) | **−0.11 pp** ± 0.62 | 4 | 0.8438 | ❌ FAIL +3 pp |

**No alternative reaches paired Wilcoxon significance at any conventional threshold.** Cross-substrate validation: cat F1 + reg `top10_acc_indist` per-task-best transfer between MPS bs=1024 (published) and CUDA bs=1024/2048 within 0.5σ, so verdicts are substrate-robust. Reg MRR is bs-confounded (MPS within 0.42σ of CUDA bs=1024) — not a substrate effect.

**Implication for paper framing:** CH22 + CH22b together make the strongest scale-conditional claim. The paper can defensibly state: *"we ruled out FAMO (NeurIPS 2023), Aligned-MTL (CVPR 2023), and hierarchical-softmax-on-the-reg-head as drop-in alternatives that recover FL Δm; the architectural cost at large region cardinality is robust to head and balancer changes."* Tier 1.5 (cross-attn mechanism probes) and Tier 2 (PLE / Cross-Stitch backbone) further test the structural-incompatibility reading.

**Source:** `research/F50_T1_RESULTS_SYNTHESIS.md` + run dirs `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260429_{0019,0045,0128,0153}/`.
**Test:** F50 Tier 1 — DONE 2026-04-29 on RTX 4090.
**Phase:** F50.
**Status:** `confirmed (Tier A sub-claim of CH22) 2026-04-29`. Strengthens CH22.

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
| **CH16** | **A** | **P1.5b → Phase 3** | **confirmed leakage-free at 5/5 states (AL+AZ+FL+CA+TX, 2026-04-30)** | **Check2HGI > HGI on next-category (PRIMARY SUBSTRATE).** Canonical matched-head STL substrate Δ = +15.50 (AL), +14.52 (AZ), +29.02 (FL), +28.81 (CA), +28.34 (TX), all p=0.0312 (max possible n=5 paired Wilcoxon), 5/5 folds positive each. Δ follows a broad two-band pattern (~15 pp at AL/AZ; ~28-29 pp at FL/CA/TX), not a strict monotone scale law. |
| **CH17** | **A** | **paper** | **pending (audit done 2026-04-23)** | **Check2HGI > POI-RGNN.** Reproduced POI-RGNN FL 34.49 / CA 31.78 / TX 33.03; matched-head STL Check2HGI at FL/CA/TX is 63.43 / 59.94 / 60.24 (+27 to +29 pp), while the MTL row widens the gap to roughly +32 to +34 pp. Protocol audit in `docs/baselines/POI_RGNN_AUDIT.md`. |
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
| **CH15** | **B** | **P1.5 → Phase 3** | **REJECTED at AL/AZ/FL, tied at CA/TX (leak-free, 2026-04-30)** | Phase 3 leak-free reg STL `next_getnext_hard` (per-fold log_T) shows HGI ≥ C2HGI on Acc@10 at all 5 states: Δ AL=−2.71, AZ=−3.13, FL=−2.12, CA=−1.85, TX=−1.59 (sign-flipped vs Phase 2 leaky). TOST δ=2pp non-inf passes only at CA+TX (2/5; bar was ≥4/5). The Phase 2 "tied/non-inf" verdict was an artifact of the F44 substrate-asymmetric leakage (~3 pp differential — c2hgi benefited more from leaky log_T than hgi). |
| **CH18** | **A** | **F21c → F48-H3-alt → F49 → Phase 3** | **cat-side STRENGTHENED 5/5; reg-side REJECTED 0/5 (leak-free, 2026-04-30)** | Phase 3 MTL B9 leak-free: cat F1 5/5 states confirm C2HGI > HGI with paper-grade p=0.0312, all folds positive (CH18-cat solidified). Reg Acc@10 0/5 states show C2HGI > HGI — sign reversed at every state (AL Δ=−7.79, AZ −3.46, FL −1.00, CA −1.09, TX −0.13). The Phase 2 "C2HGI > HGI on reg under MTL" verdict was driven by the F44 leakage advantage. **Net: CH18 reframes to cat-only.** Mechanism: per-visit context (CH19) helps cat (POI-level coarsening erases the per-visit signal needed for category) but doesn't help reg (POI-level HGI embeddings serve the reg coarser label adequately). |
| **CH19** | **A** | **Phase-1 C4** | **confirmed 2026-04-27** | **Per-visit context = ~72% of CH16 cat substrate gap**; training signal residual = ~28%. POI-pooled C2HGI counterfactual (mechanism partial): linear probe AL Δ canonical−pooled=+7.64 (~63%); matched-head AL Δ=+11.19 (~72%). Source: `results/probe/alabama_check2hgi_pooled_last.json` + `results/check2hgi_pooled/alabama/...`. |
| **CH20** | **A** | **F49** | **confirmed 2026-04-27** | **Cat-supervision transfer ≤ |0.75| pp on AL/AZ/FL n=5; H3-alt reg lift is architecture-dominant.** Refutes legacy +14.2 pp transfer claim at ≥9σ on FL alone. Methodological contribution: loss-side `task_weight=0` ablation is unsound under cross-attention MTL (silenced encoder co-adapts via K/V); encoder-frozen isolation gives the only clean architectural decomposition. Layer 1 + Layer 2 paper-grade now; FL absolute architectural Δ vs STL pending F37. |
| **CH21** | **A** | **synthesis (Phase 3 reframed)** | **partially refuted 2026-04-30 — TOP-LINE PAPER CLAIM, NOW CAT-ONLY** | **MTL B-recipe's CAT lift is interactional architecture × substrate.** Substrate (CH16/CH18-cat + CH19) is necessary on cat: HGI substitution loses 15–34 pp of cat F1 (Phase 3 leak-free). Architecture (CH20) is necessary on cat. **Reg-side claim withdrawn (Phase 3, 2026-04-30):** the Phase 1 Leg III "−30 pp reg under HGI" was leakage-amplified; under leak-free MTL B9, MTL+HGI reg ≥ MTL+C2HGI reg at every state. The substrate-specific advantage is **cat-only**, not joint cat+reg. CH21 now reads: "C2HGI's per-visit context is the load-bearing substrate **for next-category prediction**; for next-region prediction the substrate is at parity with HGI." |

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
