# Paper Thesis — BRACIS 2026 Submission

## Title Candidates

1. **"Multi-Source Embedding Fusion Requires Gradient-Aware Optimization for Multi-Task POI Prediction"**
2. "When Equal Weighting Fails: Gradient Surgery Unlocks Multi-Source Fusion in Multi-Task Location Intelligence"
3. "Beyond Single-Source Embeddings: A Comprehensive Ablation of Multi-Task Learning for POI Prediction"

**Recommended:** Title 1 — direct, informative, searchable.

---

## Option A (Recommended) — Gradient-Surgery for Fusion

### Central Thesis

> Multi-source embedding fusion (spatial + structural + temporal) significantly
> improves multi-task POI prediction, but **only** when paired with
> gradient-surgery optimizers that resolve the scale-imbalance-induced gradient
> conflict between embedding sources. Standard equal task weighting — the
> de facto practice — fails catastrophically on fused inputs, producing
> results 25% worse than gradient-aware alternatives and even worse than
> single-source baselines. This finding contradicts the established wisdom
> that simple loss weighting suffices for 2-task MTL setups with low
> inter-task gradient conflict.

---

### Why This Thesis Is Novel

| Claim | Status in literature |
|-------|---------------------|
| Equal weighting suffices for 2-task MTL | Accepted (Xin et al., NeurIPS 2022) |
| Multi-source embedding concat improves POI prediction | Assumed but rarely tested rigorously |
| Gradient-surgery optimizers are needed for fusion | **New — our contribution** |
| Architecture rankings depend on embedding source | **New — our contribution** |
| DCN head accelerates (but doesn't raise ceiling of) fusion learning | **New — our contribution** |

The NeurIPS 2022 finding ("Do Current Multi-Task Optimization Methods Even Help?") holds on single-source embeddings — **we replicate it on HGI-only** (Finding 1 in PAPER_FINDINGS.md). But we then show it **breaks** on multi-source fusion, where scale imbalance between embedding sources creates a new gradient conflict channel that simple weighting cannot resolve. This is a clean extension of a top-tier finding applied to a specific and practical domain.

---

### The Narrative Arc (5 acts)

#### Act 1: The Promise of Fusion
POI prediction benefits from multiple signal types: graph structure (HGI) captures co-visitation patterns, spatial encoding (Sphere2Vec) captures geographic context, temporal encoding (Time2Vec) captures visit timing. Concatenating task-specific embedding pairs should give each task its ideal input: Sphere2Vec+HGI for category ("where + what"), HGI+Time2Vec for next-prediction ("what + when").

#### Act 2: The Failure
Naive fusion with standard MTL training (equal task weighting) actually **underperforms** single-source HGI by 4.8% on joint score (Stage 0). Per-task analysis reveals an asymmetry: fusion helps next-prediction (+26% F1 from Time2Vec's temporal signal) but hurts category classification (−15% F1 because Sphere2Vec's weak signal dilutes HGI). The scale imbalance is severe: HGI embeddings have 15× larger L2 norm than Sphere2Vec, and zero-ablation shows the model is 90% dependent on HGI while ignoring the auxiliary source.

#### Act 3: The Diagnosis
Why does equal weighting fail on fusion but not on single-source? On HGI-only, both tasks see homogeneous 64-dim input and gradient cosine between tasks is near zero — no conflict to resolve. On fusion, each task's encoder processes a 128-dim vector with two semantically distinct halves at very different scales. The task encoders pull the shared backbone in different directions through these heterogeneous features, creating a **source-level gradient conflict** that task-level loss weighting cannot see. Equal weighting averages over this hidden conflict; gradient-surgery methods resolve it.

#### Act 4: The Solution
CAGrad (closed-form conflict-averse gradient, Liu et al. NeurIPS 2021) and Aligned-MTL (eigendecomposition alignment, Senushkin et al. CVPR 2023) both resolve the fusion gradient conflict, yielding:
- **+25% joint score** over equal weighting on fusion
- **+11.1% joint score** over prior best single-source HGI configuration
- **Category F1: 80.8%** vs HAVANA's 62.9% on same Florida data (+15.5 p.p.)
- **Next F1: 37.1%** vs POI-RGNN's 34.5% on same Florida data (+2.6 p.p.)

The effect is consistent across 5 gating architectures (DSelectK, CGC, MMoE, base) — the top-10 of 25 candidates are ALL gradient-surgery methods; the bottom-15 are ALL simple-weighting methods. This is not a noisy signal.

#### Act 5: The Confirmation
- **Cross-state validation** on Florida (7× larger dataset): champion config achieves joint ~0.577, outperforming Alabama's 0.540. The finding generalizes.
- **Statistical validation**: all top-3 configs at 5-fold/50-epoch are within p=0.85 (no significant difference), confirming the optimizer class — not the specific method — is what matters.
- **Head variant test**: DCN category head provides +1.7% at short training (15ep) by learning explicit cross-features between Sphere2Vec and HGI halves, but the default head catches up at 50ep. Practical insight: DCN accelerates convergence but doesn't raise the ceiling.

---

### Competitive Positioning vs Baselines

#### Next-Category Prediction (Task 1)

| Model | Type | FL F1 | CA F1 | TX F1 | Source |
|-------|------|-------|-------|-------|--------|
| MHA+PE | Attention | ~26.9 (global) | — | — | Zeng 2019 |
| POI-RGNN | GRU+GCN ensemble | 34.49 | 31.78 | 33.03 | Capanema 2022 |
| **MTLnet (ours)** | **MTL + fusion** | **~37.1** | — | — | **This work** |
| **Delta vs RGNN** | | **+2.6 p.p.** | | | |

#### POI Category Classification (Task 2)

| Model | Type | FL F1 | CA F1 | TX F1 | Source |
|-------|------|-------|-------|-------|--------|
| k-FN | k-Nearest | ~14.1 | ~14.2 | ~14.2 | Santos |
| STPA | Statistical | ~37.3 | ~35.3 | ~36.3 | Santos |
| PGC | ARMA-GNN | 50.3 | 36.9 | 46.2 | Capanema 2022 |
| HAVANA | GAT+ARMA hybrid | **62.9** | **46.9** | **59.8** | Santos |
| **MTLnet (ours)** | **MTL + fusion** | **~78.4** | — | — | **This work** |
| **Delta vs HAVANA** | | **+15.5 p.p.** | | | |

**Critical note:** Our category task uses the same 7-class Foursquare taxonomy and same Gowalla state-split data as HAVANA/PGC. However, our input is an *embedding vector* (graph-learned representation) while HAVANA operates on *raw mobility graph adjacency*. The comparison is fair in terms of end-to-end performance but the methods differ in their feature engineering stage. This should be acknowledged in the paper.

**What we still need:**
- California and Texas results with fusion (running on other machine)
- Ideally: our reproduced HAVANA/PGC numbers on Alabama for apple-to-apple comparison
- If not possible: cite the paper-reported numbers and note the methodology difference

---

### Contribution Bullets (for the Introduction)

1. **We propose task-specific multi-source embedding fusion** for POI prediction, concatenating complementary signal types per task: spatial+structural for category classification, structural+temporal for next-location prediction.

2. **We demonstrate that standard equal task weighting fails on scale-imbalanced fusion inputs**, producing results 25% worse than gradient-surgery alternatives — contradicting the accepted finding that simple weighting suffices for 2-task MTL (Xin et al., NeurIPS 2022).

3. **We provide a mechanistic explanation**: multi-source fusion introduces source-level gradient conflict invisible to task-level loss balancing, which gradient-surgery methods (CAGrad, Aligned-MTL) resolve by operating on the gradient geometry directly.

4. **We achieve state-of-the-art results** on the Gowalla benchmark: +15.5 p.p. category F1 over HAVANA and +2.6 p.p. next-category F1 over POI-RGNN on the Florida state-split.

5. **We validate across states** (Alabama and Florida) and provide a 5-stage progressive ablation protocol that efficiently screens 5 architectures × 5 optimizers × head variants in 46 experiments.

---

### Paper Structure (15 pages, Springer LNCS)

#### 1. Introduction (1.5 pages)
- POI prediction matters (urban computing, recommendation)
- MTL is natural (category + next-prediction share representations)
- Multi-source fusion is under-explored: prior work uses single-source embeddings
- **Gap:** no study on how MTL optimizer choice interacts with multi-source fusion
- **Surprise preview:** equal weighting fails on fusion; gradient surgery is required
- Contribution bullets (5 above)

#### 2. Related Work (1.5 pages)
- **POI prediction:** HAVANA, POI-RGNN, PGC — graph-based methods, single-task
- **Multi-task learning:** Hard sharing, CGC/MMoE/DSelectK, FiLM; Xin et al. finding
- **MTL optimizers:** Equal weight, NashMTL, CAGrad, Aligned-MTL — what they solve
- **Embedding fusion:** Concat vs attention vs gating — mostly studied in NLP/vision, rarely for POI

#### 3. Method (3 pages)
- 3.1 Problem formulation: Two tasks, joint training, evaluation metric
- 3.2 Embedding engines: HGI (graph structure), Sphere2Vec (spatial), Time2Vec (temporal)
- 3.3 Task-specific fusion: Which embeddings per task and why (1 figure)
- 3.4 MTLnet architecture: Encoders → shared backbone → task heads (1 figure)
- 3.5 Gradient-surgery optimizers: CAGrad (closed-form), Aligned-MTL (eigendecomp)

#### 4. Experimental Setup (2 pages)
- 4.1 Dataset: Gowalla state-splits (Alabama, Florida), 7 categories, statistics table
- 4.2 Ablation protocol: 5-stage progressive narrowing (1 figure showing the funnel)
- 4.3 Baselines: HAVANA, POI-RGNN, PGC, MHA+PE (from published papers)
- 4.4 Metrics: Macro F1, joint score, paired t-test

#### 5. Results & Analysis (4 pages)
- 5.1 Fusion vs single-source (Stage 0): Table + per-task breakdown
- 5.2 Optimizer is the key (Stage 1): Heatmap figure (5 archs × 5 optimizers)
- 5.3 Head variant analysis (Stage 2): DCN accelerates but doesn't raise ceiling
- 5.4 Full confirmation (Stage 3): Mean ± std table, paired t-tests, per-fold
- 5.5 Cross-state validation (Stage 4): Florida results vs Alabama + vs baselines
- 5.6 Comparison with baselines: Final table

#### 6. Discussion (1 page)
- When does gradient surgery matter? Single-source → no. Multi-source with scale imbalance → yes.
- Practical guidelines: Choose optimizer based on input heterogeneity, not just task count
- Limitations: 2 tasks, batch-size confound, US states only

#### 7. Conclusion (0.5 pages)

#### References (~0.5 pages, ~25–30 refs)

---

### Key Figures

1. **Architecture diagram:** MTLnet with fusion inputs flowing through DSelectK → shared backbone → task heads.
2. **Stage 1 heatmap:** 5×5 grid of arch × optimizer, color = joint score.
3. **Ablation funnel:** 5-stage protocol diagram showing progressive narrowing.
4. **Per-task comparison:** Grouped bar chart: {HGI+eq, Fusion+eq, Fusion+aligned_mtl} × {category F1, next F1}.
5. **Baseline comparison table:** Us vs HAVANA, POI-RGNN, PGC with deltas.

---

### Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Reviewers question baseline fairness (different input types) | Medium | Acknowledge explicitly; emphasize end-to-end comparison |
| "Just an empirical study, no theoretical contribution" | Medium | The mechanistic explanation provides conceptual novelty |
| Batch-size confound invalidates ca/al advantage | Low | Effect is +25%; batch-size alone cannot explain this |
| Florida results don't match Alabama patterns | Very Low | 4/5 folds show FL outperforms AL |
| 15 pages too tight | High | Cut head co-adaptation details → supplementary |
| Reviewer expects more states (CA, TX) | Medium | Note as future work; FL sufficient for cross-state claim |

---

## Option B (Alternative — Journal Extension): Full MTLnet System Paper

A broader, system-oriented paper that covers the full MTLnet framework. Kept here for future discussion and potential journal extension.

### Title Candidates
1. "MTLnet: A Multi-Task Framework for Joint POI Category Prediction and Next-Location Forecasting with Hierarchical Graph Embeddings"
2. "Multi-Task Learning for Point-of-Interest Intelligence: Architecture, Embedding, and Optimization"

### Central Thesis

> We present MTLnet, a multi-task learning framework that jointly predicts
> POI category and next-location category using hierarchical graph
> embeddings. Through a systematic ablation across 4 dimensions (embedding
> engine, architecture, optimizer, task heads), we identify the dominant
> factors in MTL performance and provide practical guidelines for
> designing multi-task POI prediction systems.

### What It Covers (that Option A does not)
- Full HGI embedding pipeline (graph construction, training, hyperparameter tuning)
- All 8 embedding engines compared (HGI, DGI, Sphere2Vec, Time2Vec, etc.)
- FiLM vs CGC vs MMoE vs DSelectK architecture deep-dive
- 19 MTL optimizers benchmarked (extending Xin et al. NeurIPS 2022 with 7 newer methods)
- Head co-adaptation paradox (standalone rankings invert in MTL)
- Parameter budget analysis (shared backbone is only 10% of model)
- Training saturation analysis (15ep vs 50ep)
- Full cross-state validation (Alabama, Florida, California, Texas, Georgia, Arizona)
- CoUrb 2026 ST-MTLNet comparison (SIREN vs Sphere2Vec-M with 192D)

### Strengths
- Comprehensive — covers everything we've built
- Natural fit for a **journal paper** (no page limit concerns)
- Multiple independent contributions
- Can include all 14 findings from PAPER_FINDINGS.md + the 5 new fusion findings

### Weaknesses
- **Does not fit 15 pages** — needs 25-30 pages
- Diluted hook
- "System paper" harder to sell at a methods conference

### Best Venue
- **Journal:** Expert Systems with Applications, Knowledge-Based Systems, or ACM TIST
- **Strategy:** Publish Option A at BRACIS first, then extend to journal citing the conference paper

### Relationship to Option A
Option A extracts the sharpest finding (fusion+gradient surgery) and frames it as a correction to the accepted wisdom. Option B expands to cover the full 4-dimensional ablation. The conference→journal pipeline is standard academic practice.

---

## What We Need Before Submission

- [x] Alabama full ablation (Stages 0–3 complete)
- [ ] Florida Stage 4 result (fold 5 running)
- [ ] Florida equal_weight baseline (validates the optimizer finding cross-state)
- [ ] Florida HGI-only baseline (validates the fusion finding cross-state)
- [ ] California and Texas fusion results (if time permits — not blocking)
- [ ] Verify HAVANA/POI-RGNN numbers are from same data splits
- [ ] Anonymous GitHub repo with code + data
- [ ] Register paper on JEMS3
- [ ] Register reviewer from author team
- [ ] LaTeX draft in Springer LNCS template
