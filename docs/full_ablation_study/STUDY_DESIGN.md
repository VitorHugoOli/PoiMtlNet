# Ablation Study Design (v2)

## Honest Assessment

The full cartesian product of our experimental dimensions is:

    5 architectures x 19 optimizers x 9 category heads x 10 next heads
    = 8,550 experiments

At ~3 minutes per 1-fold/10-epoch run, that's **18 days** of sequential
computation on a single MPS device. This is not feasible or scientifically
useful — most combinations are redundant, and ablation science requires
isolating one variable at a time, not exhaustive grid search.

**What I recommend instead:** A 5-stage progressive narrowing protocol that
tests ~55 total experiments and produces paper-quality results with proper
controls and statistical validation.

The key insight from our prior ablation work is that the four dimensions
are NOT independent — they interact. But the interactions follow a clear
hierarchy of importance:

1. **Architecture** (most impactful): CGC s2t2 dominated on HGI
2. **Optimizer** (second): equal_weight and db_mtl consistently good
3. **Heads** (least in MTL): Phase 4 showed head swaps hurt when backbone
   changes the feature distribution

The protocol exploits this hierarchy: fix low-impact variables first using
prior evidence, then sweep the high-impact ones.

## Fusion Embedding

The fusion engine concatenates task-specific embedding pairs:

| Task | Embeddings | Dim | Signal |
|------|-----------|-----|--------|
| Category | Sphere2Vec (64) + HGI (64) | 128 | Spatial location + graph structure |
| Next | HGI (64) + Time2Vec (64) | 128 | Graph structure + temporal context |

See `FUSION_RATIONALE.md` for detailed justification.

---

## Protocol Overview

```
Stage 0: Baseline Comparison            (4 experiments,  ~15 min)
    Fusion sanity check + HGI reference point

Stage 1: Architecture x Optimizer       (25 experiments, ~75 min)
    Fix: default heads
    Sweep: 5 architectures x 5 top optimizers

Stage 2: Head Variants on Top-3         (9 experiments,  ~90 min)
    Fix: top-3 arch+optimizer from Stage 1
    Sweep: head alternatives for category and next

Stage 3: Confirmation                   (3 experiments,  ~4-6 hours)
    Fix: top-3 configurations from Stages 1-2
    Run: 5-fold CV, 50 epochs, full statistical reporting

Stage 4: Cross-State Validation         (1 experiment,   ~2 hours)
    Fix: top-1 from Stage 3
    Run: 5-fold CV, 50 epochs on florida
```

Total: ~50 screening experiments + 4 confirmation = **~8-10 hours**

---

## Stage 0: Baseline Comparison

**Goal:** Verify fusion works and establish a comparison point against
HGI-only (our previous best engine).

**Candidates (4):**

| # | Engine | Architecture | Optimizer | Rationale |
|---|--------|-------------|-----------|-----------|
| 1 | fusion | mtlnet | equal_weight | Simplest fusion baseline |
| 2 | fusion | mtlnet_cgc (s=2,t=2) | equal_weight | Prior HGI winner on fusion |
| 3 | fusion | mtlnet_dselectk (e=4,k=2) | db_mtl | Prior DGI winner on fusion |
| 4 | **hgi** | mtlnet_cgc (s=2,t=2) | equal_weight | **HGI reference point** |

**Config:** 1 fold, 10 epochs, seed 42.

**Pass criteria:** All 4 complete without errors. Loss decreases.

**Decision point:**
- If fusion > HGI: proceed, fusion adds value.
- If fusion ~ HGI (within 5%): proceed, fusion doesn't hurt.
- If fusion < HGI by >10%: investigate before Stages 1-3. The richer
  embedding may introduce noise that the backbone can't exploit.

---

## Stage 1: Architecture x Optimizer Sweep

**Goal:** Find the best (architecture, optimizer) pair on fusion, using
default heads.

**Why default heads:** Phase 4 showed that head swaps hurt in the MTL
pipeline. The default CategoryHeadTransformer and NextHeadMTL are
co-adapted with the backbone. We test heads separately in Stage 2
only on the winning backbones.

### Architecture Candidates (5)

| Architecture | Config | Prior Evidence |
|-------------|--------|----------------|
| mtlnet | default | Baseline |
| mtlnet_cgc (s=2, t=2) | num_shared=2, num_task=2 | Best HGI joint score |
| mtlnet_cgc (s=2, t=1) | num_shared=2, num_task=1 | Strong on DGI |
| mtlnet_mmoe (e=4) | num_experts=4 | Simpler MoE baseline |
| mtlnet_dselectk (e=4, k=2) | num_experts=4, selectors=2 | Best DGI joint |

**Note:** PLE is excluded from Stage 1. It had the worst Phase 4 result
(joint=0.235) and adds the most parameters. It can be tested in a
follow-up if CGC wins again.

### Optimizer Candidates (5)

| Optimizer | Type | Rationale |
|-----------|------|-----------|
| equal_weight | Static | Best HGI joint winner; simplest baseline |
| db_mtl | Gradient (EMA) | Best DGI joint winner; consistently competitive |
| cagrad (c=0.4) | Gradient (conflict-averse) | New; principled for 2 tasks; cheap |
| aligned_mtl | Gradient (eigendecomp) | New; no hyperparams; addresses conflict + dominance |
| uncertainty_weighting | Learned (log-variance) | Most-cited adaptive MTL baseline (Kendall 2018); reviewers expect it |

**Why uncertainty_weighting instead of DWA:** Uncertainty weighting
(Kendall et al. CVPR 2018) is the most-cited learned task weighting
method in the MTL literature and will be expected by reviewers. DWA
is tested as a supplementary experiment only.

### Experiment Matrix (25 candidates)

5 architectures x 5 optimizers = 25 experiments.

**Config:** 1 fold, 10 epochs, seed 42.

**Ranking metric:** `joint_score = 0.5 * next_macro_f1 + 0.5 * category_macro_f1`

**Promotion:** Top 5 by joint score advance to 2 folds, 15 epochs.
Top 3 from promoted results advance to Stage 2.

---

## Stage 2: Head Variant Test

**Goal:** Test whether alternative heads improve the top-3 configurations
from Stage 1.

### Head Candidates

**Category head options (2):**
- `default` (CategoryHeadTransformer) — MTL co-adapted baseline
- `category_dcn` — standalone HGI winner (F1=0.728); Deep & Cross may
  be particularly relevant for fusion because it learns explicit
  cross-features between the Sphere2Vec and HGI embedding halves

**Next head options (2):**
- `default` (NextHeadMTL) — MTL co-adapted baseline
- `next_tcn_residual` — new standalone winner (F1=0.244); canonical TCN
  with residual blocks

### Per-configuration matrix

For each of the top-3 from Stage 1, run 3 new head combos:

| Category Head | Next Head | Rationale |
|--------------|-----------|-----------|
| category_dcn | default | Test if DCN cross-features help with fusion's dual-source category input |
| default | next_tcn_residual | Test if TCN benefits from Time2Vec's per-step temporal signal |
| category_dcn | next_tcn_residual | Both swapped |

(Default+default = Stage 1 result, no rerun needed.)

3 configs x 3 head combos = **9 new experiments**.

**Config:** 2 folds, 15 epochs, seed 42.

**Decision rule:** A head swap is worth keeping ONLY if it improves joint
score over the default-heads version from Stage 1.

---

## Stage 3: Full Confirmation

**Goal:** Statistically validated results for the paper.

**Candidates:** Top 3 from Stages 1+2 combined.

**Config:** 5 folds, 50 epochs, seed 42.

**Reported metrics (per candidate):**
- Joint score: mean +/- std across folds
- Per-task macro F1: mean +/- std
- Per-task accuracy: mean +/- std
- Per-class F1 from the joint-checkpoint classification report
- Training wall-clock time
- Parameter count and FLOPs
- Gradient cosine similarity (diagnostic)
- Loss weight evolution (for adaptive optimizers)

**Statistical tests:**
- Paired t-test (or Wilcoxon signed-rank) across folds between top-1
  and runner-up
- Report p-values and effect sizes
- If p > 0.05, report as "no significant difference"

---

## Stage 4: Cross-State Validation

**Goal:** Demonstrate that the best configuration generalizes beyond
Alabama.

**Candidate:** Top-1 from Stage 3.

**State:** Florida (largest dataset in the collection).

**Config:** 5 folds, 50 epochs, seed 42.

**Requirement:** Fusion inputs must exist for florida. If not, generate
them via `pipelines/fusion.pipe.py` before running.

**Reported:** Same metrics as Stage 3. Compare Alabama vs Florida
performance to discuss dataset-dependency.

---

## What This Study Can Conclude

1. **Architecture contribution:** "CGC/DSelectK-style expert gating
   improves over shared-backbone hard parameter sharing by X% on
   joint score."

2. **Optimizer contribution:** "Simple equal weighting matches/exceeds
   gradient-based MTL optimizers on this 2-task setup."
   OR: "CAGrad/Aligned-MTL improve joint score by X%, justified by
   gradient cosine analysis."

3. **Head contribution:** "Task heads co-adapted with the MTL backbone
   outperform individually superior standalone heads."
   OR: "DCN cross-features between spatial and structural embeddings
   improve category F1 by X% in the fusion setting."

4. **Embedding contribution:** "128-dim fusion embeddings (Sphere2Vec +
   HGI for category, HGI + Time2Vec for next) improve/match HGI-only,
   confirming that task-specific auxiliary signals [spatial for category,
   temporal for next] add complementary information."

5. **Generalization:** "The winning configuration transfers from Alabama
   to Florida with [comparable/degraded] performance."

---

## Runtime Estimates

| Stage | Experiments | Config | Est. Time |
|-------|------------|--------|-----------|
| 0 | 4 | 1f x 10ep | ~15 min |
| 1 screen | 25 | 1f x 10ep | ~75 min |
| 1 promote | 5 | 2f x 15ep | ~50 min |
| 2 | 9 | 2f x 15ep | ~90 min |
| 3 | 3 | 5f x 50ep | ~4-6 hours |
| 4 | 1 | 5f x 50ep | ~1.5-2 hours |
| **Total** | **~47** | | **~8-10 hours** |

All sequential on a single MPS device.
