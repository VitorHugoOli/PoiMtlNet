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

**Source:** P1.5b measurement (CHECK2HGI vs HGI on next-category, AL, 5f × 50ep). Same head class, same optim, identical splits.
**Test:** P1.5b — launched 2026-04-16 evening.
**Phase:** P1.5b.
**Status:** `pending` (running).

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
**Status:** `confirmed (tied)` — closes C07 with a "tied" outcome and a pivot in paper framing.

---

### CH13 — Spatial enrichment improves next-category F1 over vanilla Check2HGI

**Statement:** Adding continuous geospatial positional encoding from (lat, lon) + distance-to-previous-POI + distance-to-region-centroid as node features improves the P3 champion's next-category macro-F1 by ≥ 1pp on AL.

**Source:** Sphere2Vec (Mai et al. 2023), Space2Vec (Mai et al. 2020). Current Check2HGI has no explicit spatial features — geography enters only via region assignment and graph structure.
**Test:** P6 ablation — enriched vs vanilla at matched MTL config.
**Phase:** P6.
**Status:** `pending` (requires literature review to finalise implementation).

---

## Summary dashboard

| ID | Tier | Phase | Status | Decides |
|----|------|-------|--------|---------|
| **CH16** | **A** | **P1.5b** | **pending (running)** | **Check2HGI > HGI on next-category (PRIMARY SUBSTRATE)** |
| **CH17** | **A** | **paper** | **pending** | **Check2HGI > POI-RGNN + prior HGI-next-category article** |
| **CH01** | A | P3 | pending | Bidirectional MTL lift on both heads |
| **CH02** | A | P3 | pending | No negative transfer on either head (statistical) |
| **CH03** | A | P4 | partial (AL dev) | Per-task input modality > shared / concat — confirmed directionally on AL |
| CH04 | B | P1 | retired | Region head validates — demoted to reported comparison (1.16× Markov-1-region) |
| CH05 | B | P1 | confirmed | Head choice matters for region task (GRU wins, transformer collapses) |
| CH06 | B | P2 | pending | Champion MTL arch × optim |
| CH07 | B | P3 | pending | Seed variance bound (<2pp std across 15 runs) |
| CH08 | C | P4 | pending | State-dependent MTL gain (AL vs FL) |
| CH09 | D | P5 (gated) | pending | Cross-attention > per-task modality |
| CH10 | E | — | declared | External-validity limit (Gowalla ≠ FSQ) |
| CH11 | E | P6 | pending | Enrichment is a research track |
| CH12 | F | P6 | pending | Temporal enrichment (Time2Vec-like) |
| CH13 | F | P6 | pending | Spatial enrichment (Sphere2Vec-like) |
| **CH15** | **B** | **P1.5** | **pending** | **Check2HGI substrate ≥ POI2HGI on region task (preempt reviewer Q)** |
