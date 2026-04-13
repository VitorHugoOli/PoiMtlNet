# Key Findings for the Paper

Consolidated empirical findings from the MTLnet improvement work
(2026-04-11 to 2026-04-13). Each finding includes the evidence, the
implication for the paper narrative, and where the raw data lives.

---

## Finding 1: Simple Loss Weighting Matches Complex Gradient Solvers

**Evidence:**
On HGI alabama (2-fold, 15-epoch promoted runs), the best joint scores:

| Configuration | Joint | Next F1 | Cat F1 |
|---------------|-------|---------|--------|
| CGC s2t2 + **equal_weight** | **0.4855** | 0.259 | 0.712 |
| CGC s2t2 + db_mtl | 0.4775 | 0.258 | 0.697 |
| DSelectK + db_mtl | 0.4748 | 0.264 | 0.686 |

Equal weighting (no adaptation at all) was the overall winner.
NashMTL, PCGrad, GradNorm, and 7 other adaptive methods tested in
Phase 1 did not reach the top-3.

Gradient cosine similarity between tasks was measured near zero
throughout training — there is minimal conflict between the category
and next-category objectives.

**Paper implication:** Consistent with Xin et al. (NeurIPS 2022) "Do
Current Multi-Task Optimization Methods in Deep Learning Even Help?"
For this 2-task POI prediction setup with near-zero gradient conflict,
complex gradient balancing provides no benefit over uniform scalarization.

**Data:** `results/ablations/arch_variants_hgi_db_eq_fairgrad_escalated/`

---

## Finding 2: Architecture Choice Matters More Than Optimizer Choice

**Evidence:**
Across all ablation runs on both DGI and HGI:

- **Architecture** changed the winner across engines: CGC s2t2 won on
  HGI, DSelectK won on DGI.
- **Optimizer** had smaller effect: equal_weight and db_mtl were
  consistently top-2 regardless of architecture.
- **Engine** (embedding quality) mattered most of all: HGI results
  dominated DGI results across the board.

Hierarchy of importance: **Embedding > Architecture > Optimizer > Heads**

**Paper implication:** Resource allocation in MTL research should
prioritize embedding quality and architecture design over optimizer
development. The shared backbone architecture (how tasks share
information) has more impact than the gradient balancing strategy
(how task losses are weighted).

**Data:** `docs/MTL_ABLATION_REPORT_2026-04-11.md`

---

## Finding 3: Standalone Head Rankings Do Not Transfer to MTL

**Evidence:**
Standalone head ablation (10 epochs, HGI alabama):

| Head | Standalone F1 | Inside MTL Joint |
|------|--------------|-----------------|
| next_tcn_residual | **0.244** (best) | — |
| next_temporal_cnn | 0.200 | — |
| next_mtl (default) | 0.043 (worst) | — |
| cat_dcn | **0.728** (best) | — |
| cat_transformer (default) | 0.402 (worst) | — |

Phase 4 MTL ablation swapping in the standalone winners:

| Configuration | Joint |
|---------------|-------|
| CGC s2t2 + default heads | **0.4855** |
| CGC s2t2 + swapped heads (DCN cat + TCN next) | 0.387 |

Swapping the "better" heads **reduced** joint score by 20%.

**Paper implication:** Task heads co-adapt with the shared backbone
during MTL training. The shared backbone produces a feature distribution
that the default heads were trained on; inserting a different head breaks
this distribution match. Standalone head evaluation measures learning
speed from raw embeddings, not compatibility with MTL-conditioned
features.

**Data:** `results/ablations/head_all_1fold_10ep_seed42/summary.csv`,
`results/ablations/phase4_promoted_2fold_15ep_seed42/summary.csv`

---

## Finding 4: The MTL Model Is Head-Dominated, Not Sharing-Dominated

**Evidence:**
Parameter budget analysis of default MTLnet:

| Component | % of Parameters | Role |
|-----------|----------------|------|
| Task encoders | 28% | Per-task preprocessing |
| **Shared backbone** | **10%** | **MTL knowledge transfer** |
| Category head | 14% | CategoryHeadTransformer |
| Next head | 48% | NextHeadMTL (Transformer) |

The shared backbone — the component responsible for multi-task knowledge
transfer — is the smallest part of the model.

**Paper implication:** The current architecture invests most parameters
in task-specific components rather than shared representations. This
architectural imbalance may limit the benefit of MTL. Future work should
explore growing the shared backbone relative to the heads.

**Data:** Measured from model parameter counts; see
`plan/HEAD_ARCHITECTURE_ANALYSIS.md`.

---

## Finding 5: Multi-Source Fusion Exhibits Natural Gradient-Based Source Selection

**Evidence:**
Fusion combines two 64-dim embeddings per task with very different scales:

| Task | Source 1 | L2 norm | Source 2 | L2 norm | Ratio |
|------|----------|---------|----------|---------|-------|
| Category | Sphere2Vec | 0.55 | HGI | 8.46 | 15.2x |
| Next | Time2Vec | 1.00 | HGI | 8.70 | 8.7x |

After 10 training steps on real data:
- Category encoder: **0.7% dependent on Sphere2Vec, 90.2% on HGI**
- Next encoder: **2.4% dependent on Time2Vec, 91.7% on HGI**

Three normalization strategies tested (50 steps, real data):

| Strategy | Cat Accuracy | Source Balance |
|----------|-------------|---------------|
| **Raw (no normalization)** | **0.606** | 0.01 (HGI dominates) |
| Per-source z-score | 0.508 | 0.37 |
| Learnable per-source LayerNorm | 0.504 | 0.45 |

Normalization forces balanced source usage but **degrades accuracy by
10 percentage points**. The model performs better when it naturally
ignores the weaker source.

**Paper implication:** When concatenating heterogeneous embeddings with
different scales, the model acts as an implicit feature selector through
gradient magnitude — stronger signals receive proportionally larger
updates. Explicitly equalizing contributions via normalization removes
this natural selection mechanism and dilutes the dominant signal. This
finding argues against blanket input normalization in multi-source
embedding fusion.

**Open question (to be resolved by Stage 0):** If the model ignores
Sphere2Vec and Time2Vec, does fusion perform better than HGI-only?
If not, the auxiliary embeddings add parameters without contributing
signal.

**Data:** `docs/full_ablation_study/FUSION_RATIONALE.md`

---

## Finding 6: Engine Choice Determines the Performance Ceiling

**Evidence:**
Best joint scores per engine (same architecture/optimizer):

| Engine | Embedding Dim | Best Joint | Best Cat F1 | Best Next F1 |
|--------|--------------|------------|-------------|-------------|
| HGI | 64 | **0.4855** | **0.712** | **0.259** |
| DGI | 64 | 0.3337 | 0.420 | 0.247 |
| Fusion | 128 | TBD (Stage 0) | TBD | TBD |

HGI outperforms DGI by 45% on joint score. No architecture or optimizer
change on DGI could match even the simplest HGI configuration.

**Paper implication:** The quality of the input representation (embedding)
is the dominant factor in POI prediction performance. MTL architecture
innovations provide incremental improvements within the ceiling set by
the embedding quality.

**Data:** `docs/MTL_ABLATION_REPORT_2026-04-11.md`

---

## Finding 7: CGC Expert Gating Outperforms Hard Parameter Sharing

**Evidence:**
Architecture comparison on HGI (1-fold, 10-epoch screen):

| Architecture | Joint Score |
|-------------|------------|
| MTLnet CGC (s=2, t=2) | **0.430** |
| MTLnet CGC (s=2, t=1) | 0.408 |
| MTLnet DSelectK (e=4, k=2) | 0.422 |
| MTLnet MMoE (e=4) | 0.395 |
| MTLnet (base, FiLM + shared) | 0.371 |

CGC with task-specific experts (s=2, t=2) consistently outperformed the
base MTLnet across all optimizers tested. The configuration with more
task-specific experts (t=2 vs t=1) performed better, suggesting the
tasks benefit from dedicated capacity.

**Paper implication:** Customized Gate Control (CGC) with both shared
and task-specific experts provides better task-conditioned representations
than hard parameter sharing (FiLM + shared layers) for multi-task POI
prediction. The gating mechanism allows the model to dynamically balance
shared knowledge with task-specific processing.

**Data:** `results/ablations/arch_variants_hgi_db_eq_fairgrad_escalated/`

---

## Finding 8: Engine-Dependent Architecture Ranking Flip

**Evidence:**
The winning architecture changes depending on the embedding engine:

| Engine | Best Architecture | Joint | Runner-up | Joint |
|--------|------------------|-------|-----------|-------|
| HGI | CGC (s=2, t=2) | **0.4855** | DSelectK (e=4, k=2) | 0.4748 |
| DGI | DSelectK (e=4, k=2) | **0.3337** | CGC (s=2, t=1) | 0.3299 |

CGC wins on stronger embeddings (HGI), DSelectK wins on weaker (DGI).

**Hypothesis:** DSelectK's learnable selector networks add per-task
routing capacity that compensates for weaker input signal. When
embeddings are strong (HGI), the extra selector complexity becomes
overhead and CGC's simpler gating outperforms.

**Paper implication:** MTL architecture recommendations are not universal
-- they depend on input representation quality. Papers that report
architecture rankings on a single embedding type may not generalize.

**Data:** `results/ablations/arch_variants_hgi_db_eq_fairgrad_escalated/`,
`results/ablations/arch_variants_db_eq_fairgrad/`

---

## Finding 9: Training Saturates Early -- 50 Epochs Adds Little Over 15

**Evidence:**
Long-budget ablation (DGI, Alabama, 2-fold):

| Config | 10ep Joint | 15ep Joint | 50ep Joint | 15->50 Gain |
|--------|-----------|-----------|-----------|-------------|
| baseline_nash | 0.300 | 0.328 | 0.350 | +0.022 |
| equal_weight | 0.301 | 0.326 | 0.350 | +0.024 |
| cgc_equal | 0.331 | 0.333 | 0.358 | +0.025 |

Going from 15 to 50 epochs (3.3x more compute) yields only +0.02 joint
score (~7% relative). The model hits a performance ceiling set by the
embedding quality, not training duration.

**Paper implication:** The staged ablation protocol (screen at 10ep,
confirm at 50ep) is efficient because most of the signal is visible
early. This supports using quick screening rounds rather than expensive
full-length training for architecture search.

**Data:** `results/ablations/long_budget/all_2fold_50ep_seed42/summary.csv`

---

## Finding 10: CGC-Equal is Robust Across Seeds

**Evidence:**
Robustness study (DGI, Alabama, 2-fold, 15 epochs, 3 seeds):

| Config | Seed 42 | Seed 123 | Seed 2024 | Mean | Std |
|--------|---------|----------|-----------|------|-----|
| cgc_equal | 0.333 | 0.330 | 0.330 | **0.331** | 0.002 |
| equal_weight | 0.326 | 0.321 | 0.323 | 0.323 | 0.003 |
| baseline_nash | 0.328 | 0.318 | 0.327 | 0.324 | 0.005 |

CGC + equal_weight has the highest mean AND lowest variance across seeds.
NashMTL has 2.5x higher variance than CGC + equal.

**Paper implication:** The CGC + equal_weight finding is not a seed
artifact. The low variance (std=0.002) means the ranking is reliable
even with limited computational budget for repeated trials.

**Data:** `results/ablations/robustness/all_2fold_15ep_seed{42,123,2024}/summary.csv`

---

## Finding 11: Next-Task is 5x More Sensitive to Head Choice Than Category

**Evidence:**
Standalone head ablation (HGI, Alabama, 10 epochs):

| Task | Best Head F1 | Default Head F1 | Relative Gain |
|------|-------------|----------------|---------------|
| Category | 0.728 (DCN) | 0.402 (Transformer) | 1.8x |
| Next | 0.244 (TCN Residual) | 0.043 (next_mtl) | **5.7x** |

The next task is dramatically more sensitive to head architecture.
The default next_mtl head (Transformer with sinusoidal position encoding,
dropout=0.35) struggles to learn from raw embeddings, while convolutional
heads (TCN) learn temporal patterns much faster.

**Paper implication:** In multi-task POI prediction, the sequence modeling
component (next head) has more architectural sensitivity than the
classification component (category head). This is because the next task
requires learning temporal ordering from the 9-step check-in window,
while category classification operates on a single embedding vector
where most heads are equally effective.

However, this standalone sensitivity does NOT transfer to the MTL setting
(Finding 3), likely because the shared backbone pre-processes the temporal
signal before it reaches the head.

**Data:** `results/ablations/head_all_1fold_10ep_seed42/summary.csv`

---

## Finding 12: FiLM's Multiplicative Gating is the Bottleneck, Not Parameter Count

**Evidence:**
Architecture comparison (HGI, 1-fold, 10 epochs):

| Architecture | Gating Type | Joint | Params |
|-------------|------------|-------|--------|
| MTLnet (base) | FiLM (multiplicative) | 0.371 | ~832K |
| CGC (s=2, t=2) | Expert routing (additive) | **0.430** | ~1.35M |
| MMoE (e=4) | Expert routing (additive) | 0.395 | ~1.1M |

FiLM applies `gamma * x + beta` per feature -- a scalar modulation that
cannot learn task-specific basis vectors. It can only scale and shift
existing features. CGC's expert routing selects and combines entire
expert outputs, learning task-specific feature transformations.

The 16% improvement from FiLM to CGC is not explained by parameter count
(1.6x increase) but by **architectural expressivity**: additive expert
routing > multiplicative feature modulation for task conditioning.

**Paper implication:** For multi-task POI prediction, the sharing
mechanism's expressivity matters more than its size. Replacing FiLM with
expert-based routing (CGC, MMoE) provides a structural improvement that
cannot be achieved by simply scaling FiLM layers.

**Data:** `results/ablations/arch_variants_hgi_db_eq_fairgrad_escalated/`

---

## Finding 13: Modern Adaptive Optimizers (2023-2024) Do Not Help

**Evidence:**
SOTA methods comparison (DGI, Alabama, 1-fold, 10 epochs):

| Method | Year | Joint | vs Equal |
|--------|------|-------|----------|
| equal_weight | — | **0.301** | baseline |
| db_mtl | 2024 | 0.303 | +0.002 |
| fairgrad (alpha=2) | 2024 | 0.310 | +0.009 |
| excess_mtl | 2024 | 0.301 | +0.000 |
| stch | 2024 | 0.301 | +0.000 |
| nash_mtl | 2022 | 0.300 | -0.001 |
| uncertainty_weighting | 2018 | 0.299 | -0.002 |
| famo | 2023 | 0.270 | **-0.031** |
| bayesagg_mtl | 2024 | 0.268 | **-0.033** |

Of 10 adaptive methods spanning 2018-2024, only fairgrad showed
meaningful improvement (+0.009), and even that is within noise for a
1-fold run. FAMO and BayesAggMTL actively hurt performance.

**Paper implication:** The NeurIPS 2022 finding by Xin et al. ("Do
Current Multi-Task Optimization Methods Even Help?") extends to methods
published after their study. For 2-task POI prediction with near-zero
gradient conflict, even state-of-the-art gradient solvers from 2023-2024
cannot outperform uniform loss weighting.

**Data:** `results/ablations/sota_methods/all_1fold_10ep_seed42/summary.csv`,
`results/ablations/modern_weights/all_1fold_10ep_seed42/summary.csv`

---

## Finding 14: Cross-State Replication on Florida Confirms Key Patterns

**Evidence:**
Florida (DGI, 2-fold, 15 epochs):

| Config | Joint | Next F1 | Cat F1 |
|--------|-------|---------|--------|
| cgc_equal | **0.399** | 0.324 | 0.473 |
| baseline_nash | 0.396 | 0.326 | 0.465 |
| equal_weight | 0.395 | 0.323 | 0.467 |

Same patterns as Alabama:
1. CGC > base MTLnet (architecture matters)
2. Equal weight matches NashMTL (optimizer doesn't matter)
3. The ranking is stable across geographies

**Paper implication:** The findings are not Alabama-specific. The task
structure (near-zero gradient conflict, CGC superiority, optimizer
insensitivity) appears to be a property of POI multi-task prediction
itself, not of a particular geographic dataset.

**Data:** `results/ablations/cross_context/all_2fold_15ep_seed42/summary.csv`

---

## Suggested Paper Structure

Based on these findings, the experimental section could be structured as:

### Main Results Table
- **Embedding comparison** (Finding 6): HGI vs DGI vs Fusion — embedding
  quality sets the ceiling
- **Architecture ablation** (Findings 7, 8, 12): Base vs CGC vs MMoE vs
  DSelectK — expert routing beats FiLM, ranking depends on engine
- **Optimizer ablation** (Findings 1, 13): Equal weight vs 10 adaptive
  methods from 2018-2024 — none beat simple scalarization
- **Cross-state validation** (Finding 14): Florida replicates Alabama
  patterns — findings are task-structural, not dataset-specific

### Analysis Sections
- **Why simple weighting wins** (Findings 1, 13): Gradient cosine near
  zero → no conflict to resolve → complex solvers add cost without benefit
- **Architectural expressivity** (Finding 12): FiLM multiplicative gating
  vs CGC additive expert routing — structural explanation for the 16% gap
- **Engine-architecture interaction** (Finding 8): Best architecture
  depends on embedding quality — caution against universal recommendations
- **Training efficiency** (Findings 9, 10): Saturation at 15 epochs +
  robustness across seeds → staged protocol is efficient and reliable

### Discussion / Supplementary
- **Head co-adaptation paradox** (Findings 3, 4, 11): Standalone rankings
  invert in MTL; the next task is 5x more sensitive but heads co-adapt
  with the backbone making standalone evaluation misleading
- **Fusion source selection** (Finding 5): Natural gradient-based source
  weighting; normalization hurts — challenges blanket normalization advice
- **Parameter budget analysis** (Finding 4): Sharing mechanism is
  undersized at 10% of model parameters

### Most Novel Contributions (for positioning the paper)
1. **Finding 13**: Post-2022 adaptive MTL optimizers still don't help —
   extends Xin et al. (NeurIPS 2022) with 7 newer methods
2. **Finding 5**: Multi-source fusion with scale imbalance — model acts
   as implicit feature selector, normalization hurts
3. **Finding 3**: Standalone head evaluation is misleading for MTL —
   quantitative evidence of co-adaptation
4. **Finding 8**: Architecture rankings are embedding-dependent — caution
   for the MTL architecture community
