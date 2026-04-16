# Claims and Hypotheses Catalog — Check2HGI Study

Task pair: `{next_category (7 classes), next_region (~1K classes)}` on Check2HGI check-in-level embeddings.

**Rule:** no claim enters the paper without `status ∈ {confirmed, partial}` and an evidence pointer.

---

## Tier A — Headline

### CH01 — MTL {next_category, next_region} improves next-category macro-F1 over single-task (HEADLINE)

**Statement:** On Check2HGI, the 2-task MTL with champion architecture + optimiser raises next-category macro-F1 over the single-task next-category baseline, at matched compute, on AL and FL. Multi-seed (n=15).

**Source:** HMT-GRN / MGCL show region-auxiliary helps next-POI tasks. We test on next-category specifically.
**Test:** P3 — MTL champion vs P1 single-task next-category.
**Phase:** P3.
**Status:** `pending`.

### CH02 — No per-head negative transfer under MTL

**Statement:** Under MTL from CH01, both heads' val metrics ≥ their single-task baselines.

**Test:** P3 paired comparison (MTL per-head vs P1 single-task per-head).
**Phase:** P3.
**Status:** `pending`.

### CH03 — Dual-stream region-embedding input improves next-category

**Statement:** Feeding region embeddings alongside check-in embeddings (`[B,9,128]` instead of `[B,9,64]`) improves next-category macro-F1 over the MTL baseline from CH01.

**Source:** Check2HGI produces region_embeddings.parquet — currently unused. Probe showed check-in embs carry variable region signal (AL 3.4×, FL 1.04× majority).
**Test:** P4 — dual-stream vs check-in-only at matched MTL config.
**Phase:** P4.
**Status:** `pending`.

---

## Tier B — Methodology & supporting

### CH04 — Region head validates: best head beats simple baselines by ≥ 2×

**Statement:** The best-performing next_region head (from P1 head ablation) achieves Acc@10 ≥ 2× the Markov baseline (AL: 21.3%, FL: 45.9%) in single-task training.

**Source:** Pipeline-correctness floor + validates that the region task is learnable.
**Test:** P1 head ablation.
**Phase:** P1.
**Status:** `pending`.
**Notes:** FL's 2× target (91.8% Acc@10) may be unrealistic; relax to ≥ 1.5× if needed.

### CH05 — Head choice matters for next_region: literature-aligned heads may outperform the default

**Statement:** Among `{next_mtl, next_gru, next_lstm, next_tcn_residual, next_temporal_cnn}`, the winner on next_region differs from the winner on next_category. Specifically, GRU-family heads (per HMT-GRN's approach) may beat the transformer default on the coarser region task.

**Test:** P1 head ablation — vary region head while keeping category head fixed.
**Phase:** P1.
**Status:** `pending`.

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

## Tier C — Region-input mechanism

### CH08 — Region-input gain is state-dependent

**Statement:** The Δ next-category F1 from dual-stream region input (CH03) differs between AL (1,109 regions) and FL (4,703 regions).

**Source:** Probe showed FL has weaker implicit region signal in check-in embeddings.
**Test:** P4 cross-state comparison.
**Phase:** P4.
**Status:** `pending`.

---

## Tier D — Architecture exploration

### CH09 — Cross-attention between streams > concat (gated on CH03)

**Statement:** `MTLnetCrossAttn` with bidirectional cross-attention between check-in and region streams achieves higher next-category macro-F1 than dual-stream concat.

**Test:** P5 — only runs if CH03 shows ≥ 2pp lift on FL.
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
| **CH01** | A | P3 | pending | MTL lift on next-category (HEADLINE) |
| **CH02** | A | P3 | pending | No negative transfer |
| **CH03** | A | P4 | pending | Dual-stream region input helps |
| CH04 | B | P1 | pending | Region head validates (beats baselines) |
| CH05 | B | P1 | pending | Head choice matters for region task |
| CH06 | B | P2 | pending | Champion MTL arch × optim |
| CH07 | B | P3 | pending | Seed variance bound |
| CH08 | C | P4 | pending | State-dependent gain |
| CH09 | D | P5 (gated) | pending | Cross-attention > concat |
| CH10 | E | — | declared | External-validity limit |
| CH11 | E | P6 | pending | Enrichment is a research track |
| CH12 | F | P6 | pending | Temporal enrichment (Time2Vec-like) |
| CH13 | F | P6 | pending | Spatial enrichment (Sphere2Vec-like) |
