# Claims and Hypotheses Catalog — Check2HGI Study

Task pair: `{next_category (7 classes), next_region (~1K classes)}` on Check2HGI check-in-level embeddings.

**Core thesis (bidirectional, clarified 2026-04-16):** The two tasks help each other. MTL must improve **both heads** (next-category macro-F1 AND next-region Acc@10) over their respective single-task baselines. A single-direction lift (e.g., category improved, region degraded) does not satisfy the thesis.

**Rule:** no claim enters the paper without `status ∈ {confirmed, partial}` and an evidence pointer.

---

## Tier A — Headline

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

### CH02 — No per-head negative transfer under MTL (symmetric statistical test)

**Statement:** Under the CH01 MTL config, the paired test `MTL_per_head − SingleTask_per_head > 0` rejects `H0: ≤ 0` at α=0.05 for **both** heads on **both** states. This is the statistical teeth of CH01 — no head may silently regress.

**Test:** P3 paired comparison (MTL per-head vs single-task per-head), 15 paired samples per (head, state).
**Phase:** P3.
**Status:** `pending`.

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

**Source:** P1 asymmetric-modality findings; two-encoder MTLnet architecture.
**Test:** P4 — 4-way comparison at matched MTL config.
**Phase:** P4.
**Status:** `pending`.
**Notes:** Supersedes the earlier "dual-stream concat" framing of CH03 (which is now just one of four variants under test). Per-task modality is the architectural hypothesis this study is designed to validate.

---

## Tier B — Methodology & supporting

### CH04 — Region head validates: best head beats simple baselines by ≥ 2×

**Statement:** The best-performing next_region head (from P1 head ablation) achieves Acc@10 ≥ 2× the Markov baseline (AL: 21.3%, FL: 45.9%) in single-task training.

**Source:** Pipeline-correctness floor + validates that the region task is learnable.
**Test:** P1 head ablation.
**Phase:** P1.
**Status:** `not validated` — re-measured 2026-04-16 after correcting the Markov floor (the old `markov_1step` keyed on POI IDs, severely underestimating). The honest region-granularity floor is:

- **AL Markov-1-region: 47.01% Acc@10** (not 21.3%)
- **FL Markov-1-region: 65.05% Acc@10** (not 45.9%)

Best neural head on AL (`next_gru` default, region-emb input, 5f×50ep): **56.94% ± 4.01 Acc@10 → 1.21× Markov-1-region**, not 2.5×. The 2× target is not reachable at this head/data scale; even 1.5× is out.

Longer history (Markov-2, Markov-3) hurts Markov due to transition sparsity (AL: 37.87% / 35.22%; FL: 59.17% / 56.65%), so Markov-1-region is the binding floor.

**Conclusion:** The region task is **learnable-but-not-dramatically-beyond-Markov**. The neural head's ≈ +6pp gain over the binding Markov floor comes from embedding-space generalisation + 9-step sequence context, not from lifting the learnability ceiling. This changes CH04's role from "pipeline-correctness floor" to "sanity check + context" — the paper should report Markov-1-region as the meaningful comparator, not the POI-level 21.3% earlier documents used.

See `results/P1/region_head_alabama_region_1f_30ep_E_region_only.json` + `_E2_scale_gru_384.json`; `results/P0/simple_baselines/{alabama,florida}/next_region.json` (now includes `markov_{1,2,3}step_region`).

**Notes:** CH04 is retired as a **gate**; demoted to a reported comparison in the paper.

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
| **CH01** | A | P3 | pending | Bidirectional MTL lift on both heads (HEADLINE) |
| **CH02** | A | P3 | pending | No negative transfer on either head (statistical) |
| **CH03** | A | P4 | pending | Per-task input modality > shared / concat (ARCH CHOICE) |
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
