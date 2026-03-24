# MTLNet (`mtl_poi.py`) Critical Analysis and Upgrade Plan

**Last Updated**: 2026-02-09
**Status**: Phase 0 → Week 1-2 Sprint Active

## 1) Executive Summary

Current `mtl_poi` is not the main bottleneck by itself; the biggest blockers are in the data/training interface around it.

**Top P0 Blockers (From Original Analysis)**:
1. Fusion inputs are being silently truncated to 64 dims (the second embedding half is dropped), so fusion is not actually trained as intended.
2. Check2HGI category input is check-in level (113k samples) while next-task input is sequence level (~12.7k), creating severe task imbalance and scheduler mismatch risk.
3. Data splitting for next-task still uses random stratified CV over generated samples; this is not temporal/inductive and is prone to optimistic metrics.
4. `mtl_poi` uses a simple hard-shared trunk + FiLM modulation, but no real expert routing despite `NUM_EXPERTS` existing in config.

**New P0 Issues (From Diagnostic Analysis 2026-02-09)**:
5. **Extreme temporal bias dominance**: 66% attention on single temporal parameter, only 34% on sequence (Check2HGI)
6. **Gradient instability**: Gradients increase (0.17→0.25) as LR decays (epochs 60-100)
7. **Class imbalance**: Food class only 21-36% recall despite 31% of dataset
8. **Training inefficiency**: No F1 improvement after epoch 30-50, but trains to epoch 100

If we want a serious jump, we should first fix data/task integrity AND training dynamics, then move to MMoE/PLE-style routing with strict ablations.

## 2) What Is Being Trained (From Docs + Code)

### Intended tasks

From `docs/DATA_LEAKAGE_ANALYSIS.md`:

- Task A (next): predict **next POI category** from sequence of past check-ins/embeddings.
- Task B (category): classify a POI embedding into category.

### Actual mounted inputs in current pipeline

Input generation flow:

- `pipelines/create_inputs.pipe.py`
- `src/etl/mtl_input/builders.py`
- `src/etl/mtl_input/core.py`

Output files:

- `output/<engine>/<state>/input/next.parquet`
- `output/<engine>/<state>/input/category.parquet`

Observed Alabama shapes (local check):

```text
hgi:
  category.parquet  (11706, 65)   -> 64 features + category
  next.parquet      (12699, 578)  -> 576 features + next_category + userid

fusion:
  category.parquet  (11706, 130)  -> 128 features + placeid + category
  next.parquet      (12773, 1154) -> 1152 features + next_category + userid

check2hgi:
  category.parquet  (113753, 68)  -> check-in-level rows, includes userid/placeid/datetime/category + 64 dims
  next.parquet      (12699, 578)  -> sequence-level rows
```

### Training flow

1. `pipelines/train/mtl.pipe.py` builds folds via `FoldCreator(TaskType.MTL)`.
2. `src/etl/create_fold.py` loads next/category parquet, maps category strings -> class IDs.
3. `src/train/mtlnet/mtl_train.py` runs CV, NashMTL weighting, and trains `MTLnet` from `src/model/mtlnet/mtl_poi.py`.

## 3) Critical Findings (Be Radical)

### [P0] Fusion is currently broken in fold loading

- `src/etl/create_fold.py` uses `InputsConfig.EMBEDDING_DIM` (64) to select feature columns.
- For fusion data (128-dim category, 1152-dim next), loader still returns 64/576 only.
- Local check confirmed:
  - Fusion loaded as `next X (12773, 576)` and `cat X (11706, 64)`.

Impact: fusion experiments are invalid; half of features are discarded.

### [P0] Category task semantics are wrong for Check2HGI

- `generate_category_input()` currently copies embeddings directly.
- For check-in-level embeddings, this creates a check-in classification task (not POI classification).
- Category dataset explodes (`113753`) while next dataset remains (`12699`).

Impact: training objective no longer matches documented task definition; results become hard to interpret.

### [P0] Scheduler/epoch step mismatch risk in MTL loop

- Train loop iterates `max(len(next_loader), len(cat_loader))` using `zip_longest_cycle`.
- OneCycle scheduler uses only `len(next_loader)` as `steps_per_epoch`.
- On Check2HGI folds (local check): next/cat train batches = `5/45`.

Impact: OneCycle step count is inconsistent; training instability or runtime failure risk.

### [P1] Data split strategy is still evaluation-weak for sequential behavior

- Current fold creation is random stratified CV over samples.
- For sequential recommendation behavior, this is weaker than temporal/global split and user-aware inductive protocols.
- Already documented in `docs/DATA_LEAKAGE_ANALYSIS.md` and `docs/IMPLEMENTATION_GUIDE.md`, but not fully applied.

Impact: optimistic validation and uncertain real-world generalization.

### [P1] `mtl_poi` is not expert-based despite config hint

- `ModelParameters` has `NUM_EXPERTS`, `EXPERT_*` settings, but `MTLnet` uses shared MLP blocks + FiLM only.
- FiLM here is task-ID conditioning, not dynamic expert routing.

Impact: limited ability to resolve task conflict/negative transfer.

### [P2] Sequence generation is data-inefficient

- `generate_sequences()` uses non-overlapping windows (`step = window_size`).
- This yields far fewer training samples than stride-1 sliding windows.

Impact: weaker signal for next-task and higher variance.

### [P2] Training loop quality issues

- `gradient_accumulation_steps` arg is unused in MTL train loop.
- Per-batch `classification_report` is expensive and noisy for training-time signal.
- `evaluate_model()` zips loaders by shortest length in validation.

Impact: avoidable compute waste and metric inconsistency.

## 4) `mtl_poi.py` Architecture Assessment

File: `src/model/mtlnet/mtl_poi.py`

Current pattern:

- Two task-specific encoders.
- Task-ID embedding + FiLM modulation.
- Shared residual MLP trunk.
- Task heads: `CategoryHeadMTL` + `NextHeadMTL`.

Strengths:

- Clean separation of shared vs task-specific parameters (works with NashMTL).
- Reasonable modular structure for extension.

Weaknesses:

- Shared trunk is still hard-sharing; FiLM is coarse (2 static task tokens).
- No expert sparsity or conditional routing by sample/context.
- Single `feature_size` parameter limits future task-specific input dims.

Bottom line: good skeleton, but not enough for robust multi-objective learning under heterogeneous data.

## 5) Literature Signals to Use (What We’re Missing)

### Multi-task + experts

- MMoE (KDD 2018): per-task gates over shared experts improves task relationship modeling in recommender settings.
  - [Google publication page](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)
- PLE (RecSys 2020): separates shared and task-specific experts across layers, explicitly reducing negative transfer.
  - [DOI](https://dl.acm.org/doi/10.1145/3383313.3412236)

### MTL optimization under task conflict

- GradNorm (ICML 2018): balances task gradients by relative training rates.
  - [PMLR](https://proceedings.mlr.press/v80/chen18a.html)
- PCGrad (NeurIPS 2020): projects conflicting gradients to reduce interference.
  - [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- CAGrad (NeurIPS 2021): conflict-averse gradient combination.
  - [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
- Nash-MTL (already in repo): game-theoretic balancing; keep as baseline competitor.
  - [NVIDIA page](https://research.nvidia.com/labs/par/publication/navon2022multi/)

### MoE stability/scaling practices

- Switch Transformer: load-balancing and routing regularization are mandatory for stable sparse experts.
  - [arXiv](https://arxiv.org/abs/2101.03961)
- ST-MoE: transfer-stable sparse expert training refinements.
  - [arXiv](https://arxiv.org/abs/2202.08906)
- Expert Choice routing: better expert utilization vs naive token-choice routing.
  - [arXiv](https://arxiv.org/abs/2202.09368)

### Sequential recommendation evaluation

- Time to Split (2025): temporal split choices strongly change offline ranking conclusions.
  - [arXiv](https://arxiv.org/abs/2507.16289)
- Strong sequential baselines remain essential (SASRec, BERT4Rec, STAN, GETNext).
  - [BERT4Rec](https://arxiv.org/abs/1904.06690)
  - [SASRec DOI](https://doi.org/10.1109/ICDM.2018.00035)
  - [STAN DOI](https://doi.org/10.1145/3442381.3449998)
  - [GETNext DOI](https://doi.org/10.1145/3477495.3531983)

## 6) Concrete Plan: Add MoE to `mtl_poi` (and Make It Real)

## Phase 0: Fix Validity Before New Modeling (mandatory)

1. Make fold loading dimension-aware (`CATEGORY_INPUT_DIM`, `NEXT_INPUT_DIM`) instead of fixed `EMBEDDING_DIM`.
2. Enforce POI-level category dataset for category task (deduplicate/aggregate if source is check-in-level).
3. Align scheduler steps with actual MTL loop step count (`max_size_cycle`), or switch to scheduler independent of `steps_per_epoch`.
4. Implement split protocol from leakage docs (at minimum: user-aware/temporal split for next; POI-aware for category).

Acceptance gate: no silent feature truncation; no task-semantics mismatch; reproducible non-leaky eval.

## Phase 1: MMoE Baseline in `mtl_poi`

Replace shared MLP block with:

- `N_shared` experts (MLP experts on encoded features).
- one gate for next-task and one gate for category-task.
- each task gets weighted combination of experts.

Design:

- Keep task encoders.
- For next-task, run expert routing per timestep then pass to next head.
- For category-task, route pooled/category encoding then pass to category head.

Start dense (softmax over all experts) before sparse top-k.

## Phase 2: PLE Upgrade

Add:

- shared experts + task-specific experts per layer.
- stacked extraction layers (PLE style) to progressively disentangle shared/task information.

Reason: category vs next are related but not symmetric; PLE handles this better than single-layer MMoE.

## Phase 3: Sparse MoE + Router Regularization

Move from dense gating to top-2 routing with:

- load-balancing loss,
- router z-loss / entropy regularization,
- minimum expert load constraints.

This is optional until dataset size justifies sparse dispatch overhead.

## Phase 4: MTL objective and metrics hardening

1. Compare NashMTL vs PCGrad/CAGrad on same architecture.
2. Add calibration and class-imbalance strategy consistency (sampler vs weighted CE, not both).
3. Track per-task and joint Pareto behavior (not only average loss).

## 7) Proposed `mtl_poi` MoE Architecture (Practical v1)

```text
category_input -> category_encoder -> z_cat ----\
                                                  -> MMoE/PLE block -> category_head
next_input     -> next_encoder     -> z_next ---/
                                                  -> MMoE/PLE block -> next_head
```

Recommended v1 hyperparameters:

- experts: 4 to 8
- expert hidden: 256
- gate temperature: 1.0 (anneal to 0.7)
- dropout in experts: 0.1 to 0.2
- load-balance coeff (if sparse): 1e-2

Training policy:

- warmup 3 to 5 epochs with dense gates
- then enable sparse top-k (if used)
- gradient clipping global norm 1.0 to 2.0

## 8) Experiment Matrix (Ablation You Should Actually Run)

1. Baseline current `MTLnet` + NashMTL (after Phase 0 fixes).
2. MMoE (dense) + NashMTL.
3. MMoE (dense) + PCGrad.
4. PLE (dense) + NashMTL.
5. PLE (dense) + CAGrad.
6. PLE (sparse top-2) + NashMTL + load-balance loss.

Keep data split and metrics fixed across all runs.

Primary metrics:

- next: macro-F1 + weighted-F1 (+ top-k metrics if target changes to POI-ID)
- category: macro-F1 + per-class F1
- multi-task: Pareto dominance / win-rate across tasks

## 9) Immediate Action List

1. Fix data loader dimensionality and category granularity first.
2. Refactor `MTLnet` constructor to accept `category_feature_size` and `next_feature_size`.
3. Implement `MMoEBlock` and swap current `shared_layers`.
4. Add tight unit tests for:
   - feature dimensions per engine,
   - no truncation in fusion,
   - expert routing shapes and non-degenerate gate distributions.
5. Only then run large ablations.

---

## 10) Research-Backed Solutions for Diagnostic Issues (2026-02-09)

### Problem 1: Temporal Bias Dominance (66% attention on single parameter)

**Research Findings**:
- [Contextual Priority Attention (CPA)](https://www.nature.com/articles/s41598-025-32639-x) (2025): Implicit regularization by routing through shared global context, reducing spurious local correlations
- [Multi-Scale Temporal Self-Attention (MSTSA)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1635588/pdf) (2025): Multi-scale dependencies instead of single-scale temporal bias
- [Weight Decay as Forget Gate](https://arxiv.org/html/2507.19595v1) (2025): Learned forget gates (e.g., Gated DeltaNet, Titans) to limit influence of very old/noisy data

**Proposed Solutions** (Priority Order):

1. **Attention Regularization Loss** (Week 1-2)
   ```python
   # Add to training loss
   temporal_reg = torch.relu(attn_temporal - 0.5).mean()  # Penalize >50% temporal attention
   loss_total = loss_task + 0.1 * temporal_reg
   ```
   Expected: Force model to use sequence positions more (target: 50/50 split)

2. **Learned Forget Gate for Temporal Bias** (Week 3-4)
   ```python
   # Replace fixed temporal_bias with gated version
   self.temporal_gate = nn.Sequential(
       nn.Linear(embed_dim, 1),
       nn.Sigmoid()
   )
   temporal_contrib = self.temporal_gate(pooled_sequence) * temporal_features
   ```
   Expected: Context-dependent temporal weighting (not fixed 66%)

3. **Multi-Scale Temporal Attention** (Week 5-6)
   - Implement MSTSA module with 3 scales (recent 3 POIs, mid 3-6, distant 6-9)
   - Each scale learns its own importance weights
   - Expected: Better temporal granularity, less bias toward single scale

**References**:
- [Scientific Reports: Contextual Priority Attention](https://www.nature.com/articles/s41598-025-32639-x)
- [Frontiers Neuroscience: TFANet MSTSA](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1635588/pdf)
- [arXiv: Efficient Attention Mechanisms Survey](https://arxiv.org/html/2507.19595v1)

---

### Problem 2: Gradient Instability (Gradients increase as LR decays)

**Research Findings**:
- [Training Instability Theory](https://uuujf.github.io/instability/) (2025): Oscillatory training is normal in deep learning; best configs operate in unstable regime
- [Adaptive Learning Rate (AMC)](https://www.mdpi.com/2227-7390/13/4/650) (2025): Dynamically adjust LR based on model complexity
- [Step Decay + Cosine Annealing](https://www.lunartech.ai/blog/mastering-adaptive-learning-rates-in-deep-learning-enhance-training-efficiency-and-model-performance) (2025): Combine warmup + step decay + cosine for late-stage refinement

**Proposed Solutions** (Priority Order):

1. **Replace OneCycleLR with ReduceLROnPlateau** (Week 1)
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='max', factor=0.5, patience=10,
       threshold=1e-3, min_lr=1e-6
   )
   # Call after validation: scheduler.step(val_f1)
   ```
   Expected: Adaptive LR decay when F1 plateaus → stable gradients

2. **Early Stopping Based on Per-Class F1 Plateau** (Week 1)
   ```python
   # Stop if no class improves F1 > 1e-3 for 15 epochs
   if all(class_f1_improvement < 1e-3 for 15 consecutive epochs):
       break
   ```
   Expected: Save 50 epochs (50-100 contribute minimal gains)

3. **Gradient Noise Monitoring** (Week 2)
   - Track batch-wise gradient variance: `var(grad_norms_per_batch)`
   - If variance > 0.5, increase batch size or enable gradient accumulation
   - Expected: Smoother optimization, less late-stage instability

**References**:
- [Theoretical Insights on Training Instability](https://uuujf.github.io/instability/)
- [AMC: Adaptive Learning Rate](https://www.mdpi.com/2227-7390/13/4/650)
- [Learning Rate Schedules Guide](https://www.lunartech.ai/blog/mastering-adaptive-learning-rates-in-deep-learning-enhance-training-efficiency-and-model-performance)

---

### Problem 3: Class Imbalance (Food 21-36% recall, Entertainment 25% recall)

**Research Findings**:
- [Focal Loss](https://www.ultralytics.com/glossary/focal-loss) (Standard): Down-weight easy examples, focus on hard examples
- [Batch-Balanced Focal Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/) (2023): Reweights within each batch for better convergence
- [Dual Focal Loss](https://www.sciencedirect.com/science/article/abs/pii/S0925231221011310) (2021): Separate focal parameters for false positives and false negatives

**Proposed Solutions** (Priority Order):

1. **Focal Loss with Class Weights** (Week 1)
   ```python
   from src.criterion.FocalLoss import FocalLoss

   criterion = FocalLoss(
       gamma=2.0,  # Down-weight easy examples
       alpha=class_weights,  # Balanced weights per class
       reduction='mean'
   )
   ```
   Expected: +5-10% F1 on Entertainment, Outdoors, Nightlife

2. **Oversample Minority Classes** (Week 2)
   ```python
   from torch.utils.data import WeightedRandomSampler

   # Target: 500+ samples per class per fold
   class_counts = Counter(train_targets)
   sample_weights = [1.0 / class_counts[t] for t in train_targets]
   sampler = WeightedRandomSampler(sample_weights, len(train_targets))
   ```
   Expected: +3-5% F1 on rare classes

3. **Investigate Food Class Labels** (Week 3)
   - Manual inspection of Food→Community misclassifications
   - Check if Food/Community/Shopping have annotation errors
   - Consider hierarchical labels (Restaurant vs Cafe) if needed

**References**:
- [Ultralytics: Focal Loss Guide](https://www.ultralytics.com/glossary/focal-loss)
- [PMC: Batch-Balanced Focal Loss](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/)
- [Medium: How Focal Loss Fixes Class Imbalance](https://medium.com/analytics-vidhya/how-focal-loss-fixes-the-class-imbalance-problem-in-object-detection-3d2e1c4da8d7)

---

### Problem 4: Multi-Modal Fusion (Check2HGI 40% F1 vs DGI 30% F1)

**Research Findings**:
- [Spatio-Temporal Adaptive Fusion Transformer (STAFT)](https://www.sciencedirect.com/science/article/pii/S2666307425000506) (2025): Learnable Random Fourier Features (RFF) for spatial encoding + Category-Aware Attention + TCN for temporal dependencies
- [Multi-View Contrastive Fusion (MVHGAT)](https://www.mdpi.com/2227-7390/13/6/998) (2025): Constructs 3 hypergraphs (interaction, trajectory, geographic) capturing high-order dependencies
- [Multi-Graph Multi-Contrastive Learning (MGMCL)](https://link.springer.com/article/10.1007/s44443-025-00325-7) (2025): Geospatial relation graph + trajectory transition graph with location-aware attention

**Proposed Solutions** (Priority Order):

1. **Learnable Fusion Weights** (Week 4-5)
   ```python
   # Replace concat with learned weighted sum
   fusion_gate = nn.Sequential(
       nn.Linear(check2hgi_dim + space2vec_dim + time2vec_dim, 3),
       nn.Softmax(dim=-1)
   )
   weights = fusion_gate(torch.cat([check2hgi, space, time], dim=-1))
   fused = weights[:, 0:1] * check2hgi + weights[:, 1:2] * space + weights[:, 2:3] * time
   ```
   Expected: +2-5% F1 from optimal embedding combination

2. **Category-Aware Attention** (Week 5-6)
   ```python
   # Attend to relevant categories before prediction
   category_context = self.category_attention(
       query=pooled_sequence,  # (batch, embed_dim)
       keys=category_embeddings,  # (7, embed_dim) - one per class
       values=category_embeddings
   )
   logits = self.classifier(torch.cat([pooled_sequence, category_context], dim=-1))
   ```
   Expected: +3-5% F1 by explicitly modeling category semantics

3. **Multi-View Contrastive Learning** (Week 7-8)
   - Implement contrastive loss between positive pairs (same user trajectory)
   - Negative pairs: different users or shuffled trajectories
   - Expected: +5-10% F1 from better representation learning

**References**:
- [ScienceDirect: STAFT Model](https://www.sciencedirect.com/science/article/pii/S2666307425000506)
- [MDPI: Multi-View Contrastive Fusion](https://www.mdpi.com/2227-7390/13/6/998)
- [Springer: Multi-Graph Multi-Contrastive Learning](https://link.springer.com/article/10.1007/s44443-025-00325-7)
- [arXiv: Spatial-Semantic Augmentation with Remote Sensing](https://arxiv.org/html/2404.04271v1)

---

## 11) Weekly Sprint Plan (8-Week Roadmap)

### **Week 1-2: Stabilization & Validation** (Phase 0 Completion)

**Goals**: Fix P0 data issues, stabilize training, implement early stopping

**Tasks**:
1. ✅ Fix fold loading for fusion (use `CATEGORY_INPUT_DIM`, `NEXT_INPUT_DIM`)
2. ✅ Enforce POI-level category data (deduplicate check-in-level)
3. ✅ Align OneCycle scheduler with actual loop iteration count
4. ✅ Replace OneCycleLR with ReduceLROnPlateau
5. ✅ Implement per-class F1 plateau early stopping
6. ✅ Implement Focal Loss (γ=2.0, class weights)
7. ✅ Add attention regularization loss (penalize >50% temporal bias)
8. ✅ Run ablation: learned PE + temporal bias vs baseline (8 layers, sinusoidal PE)

**Deliverables**:
- No silent feature truncation (fusion 128-dim category, 1152-dim next loads correctly)
- Early stopping saves ~50 epochs
- ReduceLROnPlateau prevents late-stage gradient instability
- Focal Loss improves minority class F1 by 5-10%
- Attention regularization reduces temporal bias from 66% → 50%

**Success Metrics**:
- Gradient norms decrease or stabilize (no 0.17→0.25 spike)
- Training stops at epoch 40-60 (not 100)
- Entertainment F1 > 0.35 (vs 0.27 baseline)
- Temporal bias attention < 55%

---

### **Week 3-4: Architecture Experiments** (Phase 1 Start)

**Goals**: Test alternative architectures, implement learned forget gate, validate cross-dataset

**Tasks**:
1. ✅ Implement NextHeadHybrid (GRU + Attention) from `ANALYSIS.md`
2. ✅ Implement learned forget gate for temporal bias (context-dependent)
3. ✅ Oversample minority classes (target: 500+ samples per class)
4. ✅ Cross-dataset validation:
   - Run Check2HGI embeddings on Texas dataset
   - Run DGI embeddings on Alabama dataset
5. ✅ Manual inspection of Food class labels (investigate 40% Food→Community error)
6. ✅ Implement learnable fusion weights (weighted sum of Check2HGI + Space2Vec + Time2Vec)

**Deliverables**:
- NextHeadHybrid baseline: Expected +2-5% F1 for seq=9 (vs Transformer)
- Learned forget gate: Temporal bias becomes context-dependent (not fixed 66%)
- Cross-dataset results isolate embedding effect from dataset effect
- Food class root cause identified (labels? embeddings? both?)
- Fusion v1: Learnable weights instead of naive concat

**Success Metrics**:
- NextHeadHybrid F1 > 0.42 (vs 0.40 Transformer baseline on Check2HGI)
- Temporal bias variance across samples > 0.1 (context-dependent)
- Food class F1 > 0.38 (vs 0.29-0.36 baseline)
- Learnable fusion F1 > 0.42 (vs 0.40 Check2HGI-only)

---

### **Week 5-6: Multi-Scale Attention & Category-Aware Fusion** (Phase 1 Continue)

**Goals**: Implement MSTSA, category-aware attention, refine fusion

**Tasks**:
1. ✅ Implement Multi-Scale Temporal Self-Attention (MSTSA):
   - Recent scale (last 3 POIs)
   - Mid scale (POIs 3-6)
   - Distant scale (POIs 6-9)
   - Each scale learns separate attention weights
2. ✅ Implement Category-Aware Attention module
3. ✅ Add gradient noise monitoring (track batch-wise grad variance)
4. ✅ Test Batch-Balanced Focal Loss (vs standard Focal Loss)
5. ✅ Refactor `MTLnet` constructor for `category_feature_size` ≠ `next_feature_size`

**Deliverables**:
- MSTSA model with 3 temporal scales
- Category-aware fusion improves class separation
- Gradient noise < 0.5 (smoother optimization)
- MTLnet supports variable input dimensions

**Success Metrics**:
- MSTSA F1 > 0.43 (vs 0.40 single-scale baseline)
- Temporal bias distributed across 3 scales (not 66% on one)
- Category-aware attention F1 > 0.43 (vs 0.40 naive pooling)
- Gradient noise variance < 0.5

---

### **Week 7-8: MMoE Implementation & Contrastive Learning** (Phase 1 Complete)

**Goals**: Implement MMoE baseline, multi-view contrastive learning

**Tasks**:
1. ✅ Implement `MMoEBlock` (4-8 experts, dense softmax gating)
2. ✅ Replace shared MLP trunk with MMoE routing
3. ✅ Implement multi-view contrastive loss:
   - Positive pairs: same user trajectory segments
   - Negative pairs: different users or shuffled
4. ✅ Run full ablation matrix:
   - Baseline (current) + NashMTL
   - NextHeadHybrid + Focal Loss + ReduceLROnPlateau
   - MSTSA + Category-Aware Attention + Focal Loss
   - MMoE + NashMTL
   - MMoE + Contrastive Loss + NashMTL
5. ✅ Unit tests for:
   - Feature dimensions per engine (no truncation)
   - Expert routing shapes (non-degenerate gates)
   - Contrastive loss convergence

**Deliverables**:
- MMoE baseline functional and tested
- Contrastive learning improves representation quality
- Complete ablation results table
- Production-ready model (best F1 from ablation)

**Success Metrics**:
- MMoE F1 > 0.44 (vs 0.40 hard-shared baseline)
- Contrastive loss F1 > 0.45 (vs 0.40 baseline)
- Best model: F1 > 0.45, all classes > 0.30 recall
- Food class F1 > 0.40, Entertainment F1 > 0.38

---

### **Week 9-10: PLE Upgrade & Production** (Phase 2 Start)

**Goals**: Implement PLE architecture, deploy best model

**Tasks**:
1. ✅ Implement PLE (Progressive Layered Extraction):
   - Shared experts + task-specific experts per layer
   - Stacked extraction layers
2. ✅ Test PLE + PCGrad (vs NashMTL)
3. ✅ Test PLE + CAGrad
4. ✅ Deploy best model from Weeks 1-10 to inference pipeline
5. ✅ Monitor per-class F1 on held-out test set (not CV folds)

**Deliverables**:
- PLE architecture functional
- Production model deployed
- Test set evaluation (final validation)

**Success Metrics**:
- PLE F1 > 0.46 (vs 0.44 MMoE)
- Test set F1 within ±2pp of CV validation (no overfitting)
- All classes > 0.35 recall on test set

---

## 12) Updated Experiment Matrix (With Research Solutions)

| ID | Architecture | Loss | Scheduler | Expected F1 | Priority |
|----|-------------|------|-----------|-------------|----------|
| 1  | Current Transformer + Phase 0 fixes | Focal | ReduceLROnPlateau | 0.42 | Week 1 |
| 2  | NextHeadHybrid | Focal | ReduceLROnPlateau | 0.43 | Week 3 |
| 3  | MSTSA (Multi-Scale) | Focal | ReduceLROnPlateau | 0.44 | Week 5 |
| 4  | MSTSA + Category-Aware Attn | Focal | ReduceLROnPlateau | 0.45 | Week 6 |
| 5  | MMoE + NashMTL | Focal | ReduceLROnPlateau | 0.45 | Week 7 |
| 6  | MMoE + Contrastive | Focal | ReduceLROnPlateau | 0.46 | Week 8 |
| 7  | PLE + CAGrad | Focal | ReduceLROnPlateau | 0.47 | Week 9 |

**Baseline (Current)**: Check2HGI 40% F1, DGI 30% F1

**Target**: Check2HGI 47% F1 (+7pp), DGI 37% F1 (+7pp) by end of Week 10

---

## 13) Critical Path & Dependencies

```
Week 1-2 (Phase 0) ─────────────────────┐
                                          ├──> Week 3-4 (Architecture) ──> Week 5-6 (MSTSA) ──> Week 7-8 (MMoE) ──> Week 9-10 (PLE)
                                          │
                                          └──> Can parallelize: Focal Loss (W1) || NextHeadHybrid (W3) || MSTSA (W5) || MMoE (W7)
```

**Blockers**:
- Week 1-2 MUST complete before all else (data integrity)
- MSTSA (Week 5) requires NextHeadHybrid baseline (Week 3)
- MMoE (Week 7) requires MTLnet refactor (Week 5)
- PLE (Week 9) requires MMoE implementation (Week 7)

**Parallel Work**:
- Focal Loss (Week 1) + Early Stopping (Week 1) can run together
- Cross-dataset validation (Week 3) independent of architecture work
- Contrastive learning prep (Week 6) independent of MSTSA

---

## 14) Risk Mitigation

**Risk 1**: Focal Loss doesn't improve minority classes
- **Mitigation**: Fallback to Batch-Balanced Focal Loss (Week 5)
- **Backup**: Dual Focal Loss with separate γ for FP/FN

**Risk 2**: MSTSA complexity hurts performance (overfitting)
- **Mitigation**: Ablate number of scales (test 2-scale first)
- **Backup**: Keep NextHeadHybrid as simpler alternative

**Risk 3**: Food class labels are fundamentally wrong
- **Mitigation**: Hierarchical labels (Restaurant vs Cafe) or merge Food→Shopping
- **Backup**: Exclude Food class from macro-F1 calculation

**Risk 4**: MMoE implementation introduces bugs
- **Mitigation**: Comprehensive unit tests (Week 7) + gradual rollout
- **Backup**: Keep hard-shared baseline as fallback

---

If we skip Phase 0 and jump straight to MoE, we risk publishing improvements that come from pipeline artifacts rather than real modeling gains.
