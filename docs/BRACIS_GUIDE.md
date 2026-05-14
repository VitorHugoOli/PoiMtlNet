# BRACIS Conference Guide — Submission Strategy for MTLnet

## 1. Conference Overview

**BRACIS** (Brazilian Conference on Intelligent Systems) is Brazil's premier AI conference, born from merging SBIA (21 editions) and SBRN (12 editions). Organized by the Brazilian Computer Society (SBC).

- **Proceedings:** Springer LNAI (Lecture Notes in Artificial Intelligence)
- **Indexation:** Springer, DBLP, Scopus
- **CORE Ranking:** Not ranked (regional venue)
- **Typical venue size:** 100–150 accepted papers, 400–800 attendees
- **Best papers** may receive journal special-issue invitations

### Acceptance Rates

| Year | Submissions | Accepted | Rate |
|------|------------|----------|------|
| 2025 | 393 | 147 | ~37 % |
| 2024 | 285 | 116 | ~41 % |
| 2023 | 242 | 90 | ~37 % |
| 2022 | 225 | 89 | ~40 % |

**Positioning:** Mid-tier regional conference. More selective than most workshops, less selective than AAAI/IJCAI/NeurIPS (~20–25 %). A well-executed empirical study with clear contributions is competitive.

---

## 2. BRACIS 2026 Key Dates (verified 2026-04-28)

| Milestone | Window |
|-----------|--------|
| **Paper registration** | **April 13–27, 2026** (extended) |
| **Paper submission** | **April 20 – May 4, 2026** (extended) |
| Author notification | **June 1–8, 2026** |
| Camera-ready | June 29, 2026 |
| Conference | October 19–22, 2026, Cuiabá, MT, Brazil (Centro de Eventos do UniSENAI) |

All deadlines: 23:59 UTC-12:00 (Anywhere on Earth).

Source: `bracis.sbc.org.br/2026/bracis/`. The original CFP listed single dates (April 13 / April 20 / June 1) — these were extended in April 2026. Submission system: JEMS3 at `https://jems3.sbc.org.br/bracis2026`.

---

## 3. Submission Requirements

### Format
- **Max 15 pages** (including tables, figures, references, appendices — tight!)
- **Abstract: 70–150 words** (LNCS template binding rule), single paragraph, no footnotes, no inline math, no citations, define every acronym
- **English only**, PDF format
- **Springer LNCS template** (LaTeX via Overleaf) — proceedings publish as **LNAI** (Lecture Notes in Artificial Intelligence)
- **Submission system:** JEMS3 (https://jems3.sbc.org.br/bracis2026)

### Review Process
- **Double-blind** — strict anonymization required
- At least **3 reviewers per paper**
- Desk-rejection triggers: failed anonymization, >15 pages, format violations, plagiarism

### Critical Rules
- **Reviewer commitment:** At least one author **must volunteer to review 3 papers** — non-compliance → desk-rejection
- **Anonymous code:** Use Anonymous GitHub or Dropbox for data/code
- **arXiv policy:** If non-anonymous preprint exists, notify PC chairs; do NOT update it during review
- **LLM policy:** LLMs cannot be authors; usage must be acknowledged in Acknowledgments
- **In-person presentation:** At least one author must attend and present

---

## 4. Tracks & Topics of Interest

### Tracks

| Track | Focus | Best for |
|-------|-------|----------|
| **Main Track** | Novel AI *methods* with sound results | Our paper — method novelty + ablation |
| AI for Social Good | Established AI applied to societal problems | Not us |
| General Applications | Established AI in novel domains, ethics | Fallback if method novelty is questioned |
| Published Papers | 2-page abstracts of already-published work | Not applicable |

**Recommendation: Main Track** — our MTL framework, embedding fusion, and comprehensive ablation qualify as methodological novelty, not just application.

### Topics of Interest (relevant to our work)

**Direct fit:**
- Machine Learning
- Deep Learning
- Graph Neural Networks
- Data Mining and Analysis
- Pattern Recognition and Cluster Analysis
- Neural Networks

**Secondary fit:**
- AI/CI algorithms and models
- Hybrid Systems (our multi-embedding fusion)
- Multidisciplinary AI and CI

---

## 5. Quality Bar & What Reviewers Look For

### What gets accepted

BRACIS is **heavily empirical**. Most accepted papers present:
1. A clearly defined problem
2. A method (novel or novel combination of existing methods)
3. Experiments on real datasets with proper baselines
4. Ablation or sensitivity analysis
5. Statistical validation

**Our paper has all of these plus more.** The comprehensive 5-stage ablation study, cross-state generalization, and per-task analysis exceed the typical BRACIS submission depth.

### Reviewer Priorities

1. **Clear contribution statement** — What is new? What did you find?
2. **Sound experimental methodology** — Cross-validation, proper baselines, statistical tests
3. **Reproducibility** — Anonymous code/data link
4. **Well-written English** — Non-native venue, but English quality matters
5. **Comparison with state-of-the-art** — Don't just compare with yourself
6. **Novelty** — Either in method (Main Track) or application (Applications Track)

### Common Pitfalls to Avoid

- Failing anonymization (instant reject)
- Exceeding 15-page limit (instant reject)
- Not registering a reviewer from the author team (desk-reject risk)
- Weak or missing baselines
- Purely incremental work with no clear "why should I care?"
- Poor statistical analysis (no confidence intervals, no significance tests)
- Overstating claims relative to evidence

### How We Compare to Typical Submissions

| Dimension | Typical BRACIS paper | Our paper |
|-----------|---------------------|-----------|
| Ablation depth | 2–3 configurations | 46+ experiments across 5 stages |
| Statistical rigor | Mean only | Mean ± std, paired t-tests, p-values |
| Cross-validation | 5-fold or hold-out | 5-fold with per-fold reporting |
| Generalization | Single dataset | Cross-state (Alabama → Florida) |
| Embedding analysis | Rarely done | Scale imbalance analysis, zero-ablation |
| Architecture comparison | 1–2 baselines | 5 architectures × 5 optimizers |

**Assessment: Our work significantly exceeds the typical quality bar.** The risk is fitting it into 15 pages.

---

## 6. Recent Accepted Papers (Reference Points)

### BRACIS 2023 Best Papers
1. "Embracing Data Irregularities in Multivariate Time Series with Recurrent and Graph Neural Networks" — **GNN + RNN, time series** (methodologically adjacent to us)
2. "Community Detection for Multi-Label Classification" — graph-based classification

### BRACIS 2024 Main Track (selection)
- "Applying Transformers for Anomaly Detection in Bus Trajectories" — applied DL, transportation
- "One-Class Learning for Data Stream Through Graph Neural Networks" — GNN method (best paper)
- "Semi-periodic Activation for Time Series Classification" — novel activation (best paper)
- "A Contrastive Objective for Training Continuous Generative Flow Networks" — novel ML method
- "Adaptive Client-Dropping in Federated Learning" — FL, medical domain

### Pattern
- **GNN papers win best paper awards** — our graph embedding foundation is a strength
- **Application-domain papers are common** — transportation, medical, legal
- **Deep learning dominates** — transformers, GNNs, federated learning
- **Ablation studies are valued** but rarely as thorough as ours

---

## 7. Paper Strategy for Our MTLnet Submission

### Recommended Title Direction
"Multi-Task Learning with Gradient-Aware Optimization for Multi-Source POI Embedding Fusion"
or
"When Equal Weighting Fails: Gradient Surgery for Scale-Imbalanced Embedding Fusion in Multi-Task POI Prediction"

### Key Selling Points (in order of novelty)

1. **Gradient-surgery optimizers are essential for multi-source fusion** — the biggest finding; prior work uses equal weighting which fails on scale-imbalanced inputs
2. **Task-specific embedding fusion** — Sphere2Vec+HGI for category, HGI+Time2Vec for next — complementary signals per task
3. **Comprehensive ablation** — 5 architectures × 5 optimizers × head variants, with progressive narrowing protocol
4. **Cross-state generalization** — Alabama + Florida validation
5. **Scale imbalance analysis** — publishable insight on why fusion fails without proper gradient handling

### Suggested Paper Structure (15 pages)

| Section | Pages | Content |
|---------|-------|---------|
| Introduction | 1.5 | Problem, motivation, contribution bullets |
| Related Work | 1.5 | MTL methods, POI prediction, embedding fusion |
| Method | 3 | MTLnet architecture, fusion design, optimizer landscape |
| Experimental Setup | 2 | Datasets, ablation protocol, metrics, baselines |
| Results & Analysis | 4 | Stages 0–3 results, per-task analysis, stat tests |
| Cross-State Validation | 1 | Florida results |
| Discussion & Conclusion | 1.5 | Findings, limitations, future work |
| References | ~0.5 | Keep tight — 25–30 refs max |

### What to Emphasize for BRACIS

- **The optimizer finding is the hook.** "We show that gradient-surgery optimizers, previously validated on vision/NLP MTL benchmarks, are *essential* for multi-source embedding fusion in location intelligence — a finding that contradicts the common practice of equal task weighting."
- **The ablation methodology is a selling point.** BRACIS values experimental rigor, and our 5-stage progressive protocol is rare at any venue.
- **Brazilian data context:** If the Foursquare/Gowalla datasets include Brazilian cities, mention it. BRACIS appreciates local relevance.
- **Practical implications:** Emphasize that the findings guide practitioners on choosing MTL configurations for POI systems.

### What NOT to Do

- Don't try to sell this as a "new architecture" — the core MTLnet is a combination of known components (FiLM, residual blocks, transformers)
- Don't oversell the fusion results vs HGI — be honest that the improvement is optimizer-dependent
- Don't include all 46 experiments in the paper — pick the most informative comparisons
- Don't omit the batch-size confound between gradient-surgery and non-gradient-surgery optimizers

---

## 8. Comparison to Other Venues

| Venue | Acceptance | Level | Fit |
|-------|-----------|-------|-----|
| **BRACIS** | ~37–41 % | Regional (Springer LNAI) | **Best fit** — empirical MTL + application |
| IJCNN | ~50 % | Mid-tier (IEEE) | Good fit — neural networks focus |
| ECML-PKDD | ~25 % | Strong (Springer) | Possible — needs stronger novelty claim |
| ACM SIGSPATIAL | ~25 % | Strong (ACM) | Good for spatial computing angle |
| NeurIPS / ICML | ~20–25 % | Top-tier | Would need theoretical contribution |
| KDD | ~20 % | Top-tier | Would need scalability / production angle |

**BRACIS is the sweet spot** for this work: the empirical depth matches what reviewers value, the topic aligns perfectly, and the acceptance rate is realistic for a strong submission.

---

## 9. Checklist Before Submission

- [ ] Anonymize all author info, affiliations, acknowledgments
- [ ] Set up Anonymous GitHub with code + data
- [ ] Register paper on JEMS3 by April 13
- [ ] Verify ≤ 15 pages in Springer LNCS format
- [ ] Nominate at least one author as reviewer
- [ ] Include contribution statement in introduction
- [ ] Report mean ± std with statistical tests
- [ ] Compare against at least 2–3 external baselines (not just our own variants)
- [ ] Acknowledge LLM usage if applicable
- [ ] Proofread English carefully

---

## 10. Empirical Patterns from Recent BRACIS Abstracts (research 2026-04-28)

This section consolidates findings from a sweep of 13 verbatim accepted-paper abstracts spanning BRACIS 2023–2024 plus the official Springer LNCS author-template rules. Source-of-truth for the title/abstract decisions in `studies/check2hgi/PAPER_DRAFT.md`.

### 10.1 LNCS Abstract Rules (binding)

From the LNCS author template (`dokie.li/lncs-splnproc`):

> "The abstract is a mandatory element that should summarize the contents of the paper and should contain at least 70 and at most 150 words."
>
> "Please note that no footnotes may be included in the abstract."

- **70–150 words.** Single paragraph.
- **No footnotes.** No inline math (Greek symbols, equations) — abstracts appear stand-alone on SpringerLink.
- **No citations.** Define every acronym used.
- The JEMS3 plain-text abstract field is separate; it accepts a longer plain-prose summary (~1500–4000 chars) but strips formatting. Same content can be reused.

### 10.2 Style Rules from 13 Sampled Accepted BRACIS Abstracts

1. **Open with domain framing, not claim-first.** 12 of 13 sampled abstracts open with a context sentence (e.g., "Crime prediction is a critical research area…", "Trajectory anomaly detection is essential…"). Claim-first openings are off-style.

2. **State a gap before "we propose".** Common patterns:
   - Enumerated list: *"(i) lack of methods based on GNNs and (ii) lack of interpretable methods"* (OPENCAST)
   - "However, X fails to…" (HAVANA)
   - "Existing solutions fail to utilize…"

3. **Use "we propose" once.** 12 of 13 abstracts use this verb. Variants ("we introduce", "this work proposes") also work; passive ("is proposed") is rare. Always third-person plural.

4. **Headline result is specific.** Strong examples:
   - HAVANA: *"up to 25.05% in F1-Score compared to three state-of-the-art models across three different datasets"*
   - OPENCAST: *"outperforming seven other methods"*
   - Heterogeneous-graph: *"F1-Score of 83.66% overcoming the 60.70%"*
   - Semi-Periodic Activation: *"112 benchmark datasets… best average ranking in all comparative scenarios"*

   Vague qualitative-only claims are below venue norm.

5. **Mechanism is conceptual, not algorithmic.** No equations, no layer counts. HAVANA: *"spatial and spectral filter integrated with a self-attention mechanism"* — good. Don't list hyperparameters or architecture depths.

6. **For GNN/applied papers, justify *why a graph* by listing data modalities fused.**
   - Crime/GNN: *"integrates crime, street map graphs, and urban data"*
   - COVID/GNN: *"captures movement patterns between Brazilian cities and integrate it with time series data"*
   - Heterogeneous-graph: *"process different types of node features, such as text, images, and subgraphs"*

7. **Honest framing is rewarded.** Best papers (BRACIS 2023) included candid framing:
   - *Embracing Data Irregularities* led with **"low computational cost"**, not peak F1.
   - *Community Detection for Multi-label Classification* ended noting the classifier still struggles.
   - *COVID-19 forecasting with GNNs* reported being beaten by Prophet on RMSE, then pivoted to **stability**.

   Secondary virtues (efficiency, interpretability, robustness) can headline if accuracy isn't the cleanest story.

8. **Single paragraph, ~150–220 words for SBC SOL versions; ≤150 binding for the LNAI Springer version.** No paragraph breaks, no citations, no reproducibility note, no formalism.

### 10.3 Title Patterns (from 13 sampled BRACIS papers)

Three dominant shapes:

**A — Method-acronym + colon (HAVANA-style):**
- *"HAVANA: Hybrid Attentional Graph Convolutional Network…"*
- *"MAT-Tree: A Tree-Based Method for Multiple Aspect Trajectory Clustering"*

Use when method has a memorable name. Commits to acronym branding throughout the paper.

**B — Concept-for-Task (most BRACIS-canonical):**
- *"Semi-Periodic Activation for Time Series Classification"*
- *"One-Class Learning for Data Stream Through Graph Neural Networks"*
- *"Hierarchical Graph Convolutional Networks for Image Classification"*

Direct, descriptive, no hook. Reviewer-friendly. Most common shape.

**C — Verb-Object-Domain:**
- *"Modeling and Predicting Crimes in the City of São Paulo Using Graph Neural Networks"*
- *"Detecting Multiple Epidemic Sources in Network Epidemics Using Graph Neural Networks"*
- *"Applying Transformers for Anomaly Detection in Bus Trajectories"*

Action-oriented; common for applied papers.

**D — Phenomenon-and-Method (best paper at BRACIS 2023):**
- *"Embracing Data Irregularities in Multivariate Time Series with Recurrent and Graph Neural Networks"*

Length norm: **6–15 words, most around 8–12.** Avoid math symbols, undefined acronyms, and polemical claims ("Not X" can read aggressive). Two of 13 used a colon-subtitle; the other eleven used a single descriptive clause.

### 10.4 Sample Sources

13 BRACIS 2023–2024 abstracts captured 2026-04-28:

- BRACIS 2023 best papers: *Embracing Data Irregularities…* (GNN+RNN time series); *Community Detection for Multi-label Classification*
- BRACIS 2024 highlights: HAVANA (POI annotation, GNN+attention); OPENCAST (one-class GNN); *Semi-Periodic Activation* (time series); *Contrastive Objective for Continuous GFlowNets* (theory); *Heterogeneous-graph embeddings*
- BRACIS 2023 additional: *Hierarchical GCN for Image Classification*; *Detecting Epidemic Sources with GNNs*; *COVID-19 forecasting with GNN+Mobility*; *MAT-Tree*

Source URLs:
- `https://sol.sbc.org.br/index.php/bracis/issue/view/1247` — 2023 SOL proceedings
- `https://sol.sbc.org.br/index.php/bracis/issue/view/1454` — 2024 SOL proceedings
- `https://link.springer.com/conference/bracis` — LNAI proceedings index
- `https://dblp.org/db/conf/bracis/` — paper listings
- `https://dokie.li/lncs-splnproc` — LNCS template author-instructions mirror
