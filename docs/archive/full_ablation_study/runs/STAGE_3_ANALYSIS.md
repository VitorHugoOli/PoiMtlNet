# Stage 3 Results — 2026-04-13

## Configuration
- Engine: fusion (128-dim)
- State: alabama
- 3 candidates × 5 folds × 50 epochs, Seed 42

## Results

| Rank | Candidate | Joint (mean±std) | Next F1 | Cat F1 | Time |
|------|-----------|-------------------|---------|--------|------|
| 1 | s3_s1_dsk42_al (default heads) | **0.5481 ± 0.0151** | **0.2726** | 0.8076 | 1329 s |
| 2 | s3_s2_dsk42_al_hd_dcn | 0.5466 ± 0.0052 | 0.2648 | 0.8070 | 1266 s |
| 3 | s3_s2_dsk42_ca_hd_dcn | 0.5455 ± 0.0054 | 0.2655 | 0.8060 | 1270 s |

## Per-Fold Breakdown

| Fold | dsk42_al (default) | dsk42_al + DCN | dsk42_ca + DCN |
|------|-------------------|----------------|----------------|
| 1 | cat=0.821, next=0.269, **j=0.545** | cat=0.825, next=0.261, j=0.543 | cat=0.819, next=0.261, j=0.540 |
| 2 | cat=0.823, next=0.269, j=0.546 | cat=0.826, next=0.275, **j=0.550** | cat=0.825, next=0.274, j=0.549 |
| 3 | cat=0.805, next=0.288, j=0.546 | cat=0.813, next=0.295, **j=0.554** | cat=0.802, next=0.290, j=0.546 |
| 4 | cat=**0.863**, next=0.282, **j=0.572** | cat=0.814, next=0.271, j=0.543 | cat=0.819, next=0.286, j=0.552 |
| 5 | cat=0.784, next=0.277, j=0.531 | cat=0.812, next=0.275, j=0.543 | cat=0.807, next=0.275, j=0.541 |

## Paired t-Tests (5-fold, two-tailed)

| Pair | t-stat | p-value | Conclusion |
|------|--------|---------|------------|
| #1 vs #2 | 0.200 | **0.851** | No significant difference |
| #1 vs #3 | 0.512 | **0.636** | No significant difference |
| #2 vs #3 | 0.357 | **0.739** | No significant difference |

**All three configurations are statistically equivalent at 5-fold resolution.**

## Key Findings

### 1. **Default heads recover at 50 epochs — DCN advantage was transient**

Stage 2 (2f/15ep): DCN boosted joint by +1.7 %. Stage 3 (5f/50ep): DCN and default are within noise (p=0.85).

Interpretation: at short training, DCN's explicit cross-feature learning gives it a head start on the dual-source fusion input. But with 50 epochs, the default CategoryHeadTransformer learns to exploit the Sphere2Vec × HGI interaction implicitly through its multi-head attention. The head "co-adapts" with the backbone given enough time — exactly what Phase 4 predicted.

**For the paper:** report the transient DCN advantage as evidence that fusion-aware heads can accelerate convergence, even though they don't improve the ceiling. This is a practical insight for deployments where training budget is limited.

### 2. **DCN yields lower variance**

Default heads std = 0.0151; DCN variants std = 0.005. The DCN head produces more consistent fold-to-fold results. The default head has a wider range (0.531–0.572), dominated by Fold 4's outlier (cat F1 = 0.863 — unusually good split). DCN regularizes this.

**For the paper:** if fold-to-fold stability matters (e.g. for deployment confidence), DCN may be preferred even without a mean improvement.

### 3. **Aligned-MTL vs CAGrad: no practical difference**

Both gradient-surgery methods land at the same performance. This makes sense: for a 2-task problem, both methods resolve the same conflict. CAGrad uses a closed-form QP (cheaper), Aligned-MTL uses eigendecomposition (slightly more expensive, CPU fallback on MPS). **Recommend CAGrad for production** due to lower compute cost.

### 4. **Comparison to prior benchmarks**

| Configuration | Joint | Source |
|--------------|-------|--------|
| HGI + cgc22 + equal_weight (Phase 1) | 0.4855 | Prior best |
| HGI + cgc22 + equal_weight (Stage 0, 1f/10ep) | 0.3861 | Reference |
| **Fusion + dsk42 + aligned_mtl (Stage 3, 5f/50ep)** | **0.5481** | **This study** |

**Improvement: +12.9 % joint score over prior HGI best**, driven by:
- Fusion embedding: the fusion enriches both tasks (Time2Vec for next, HGI dominance for category)
- Gradient-surgery optimizer: Aligned-MTL resolves scale-imbalance-induced gradient conflict
- DSelectK architecture: soft expert selection lets the model route mixed-source features flexibly

### 5. **Category F1 is the big winner**

Category F1: 0.808 fusion vs 0.712 HGI Phase 1 → **+13.5 %**. This is remarkable given that Stage 0 showed fusion *hurting* category (0.489 vs 0.577 at 10 ep). The combined effect of Aligned-MTL + DSelectK + 50 epochs completely reversed the early-training scale imbalance penalty.

Next F1: 0.273 fusion vs 0.259 HGI Phase 1 → **+5.4 %**. Smaller gain, consistent with Time2Vec providing marginal temporal context beyond HGI.

## Decision

**Champion configuration:** `mtlnet_dselectk(e=4, k=2, temp=0.5) + aligned_mtl + default heads`

For Stage 4: this configuration will be validated on Florida.

**However:** Florida requires regenerating HGI, Sphere2Vec, Time2Vec embeddings + fusion inputs. Only Alabama embeddings exist. See Stage 4 prerequisites in CONTINUE.md.

## Artifacts
- `results/ablations/full_fusion_study/s3_confirm_5f_50ep/summary.csv`
- Per-fold metrics: `results/ablations/full_fusion_study/s3_confirm_5f_50ep/s3_*/fusion/alabama/*/metrics/`
- `docs/full_ablation_study/runs/stage3.log`
