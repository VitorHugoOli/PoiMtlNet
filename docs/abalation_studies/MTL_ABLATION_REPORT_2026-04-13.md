# MTLnet Ablation Report: Phase 3-4 (2026-04-13)

## Scope

This report covers the second round of MTLnet improvements:

- Literature survey of new MTL optimizers and architectures (Phase 3).
- Implementation and ablation of CAGrad, Aligned-MTL, DWA, and PLE.
- Task-head architecture analysis and ablation (Phase 4).
- Head-swap experiments testing whether standalone head rankings transfer
  to the full MTL pipeline.

Builds on the prior report: `docs/MTL_ABLATION_REPORT_2026-04-11.md`.

---

## What Was Implemented

### Phase 3: New Optimizers

| Method | Paper | Venue | Key Idea |
|--------|-------|-------|----------|
| **CAGrad** | Liu et al. | NeurIPS 2021 | Maximizes worst-case task improvement in a conflict-averse ball around the average gradient. Closed-form for 2 tasks. |
| **Aligned-MTL** | Senushkin et al. | CVPR 2023 | Eigendecomposition-based gradient alignment. Makes gradients orthogonal and equal-magnitude. No hyperparameters. |
| **DWA** | Liu et al. | CVPR 2019 | Dynamic Weight Average: weights tasks by loss rate of change. Zero gradient computation overhead. |

### Phase 3: New Architecture

| Method | Paper | Venue | Key Idea |
|--------|-------|-------|----------|
| **PLE** | Tang et al. | RecSys 2020 | Progressive Layered Extraction: stacked CGC layers with inter-level gating. Natural extension of CGC (prior best on HGI). |

### Phase 4: Head-Swap Candidates

9 candidates testing task-head replacements inside the MTL pipeline:
- Category head: `category_single` (2-layer MLP, ~10K params) replacing
  `CategoryHeadTransformer` (~263K params).
- Next head: `next_temporal_cnn` (dilated causal Conv1d, ~442K params)
  or `next_transformer_optimized` (~1.3M params) replacing
  `NextHeadMTL` (~1.3M params).

### Candidates Researched and Rejected

**Optimizers:** SDMGrad (3x gradient cost), MoCo (redundant with DB-MTL),
MGDA (superseded by CAGrad), GradDrop/GradVac (incremental over PCGrad),
IMTL (subsumed by Aligned-MTL), AutoLambda (too expensive).

**Architectures:** HoME (overkill for 2 tasks), Cross-stitch (old,
underperforms MMoE/CGC), MTAN (vision-specific).

**Heads:** Mamba/RetNet (seq_len=9 too short, no MPS support),
NextLocMoE (requires LLM backbone), ATST-LSTM (RNNs consistently worse).

Full justifications in `plan/NEW_CANDIDATES_ANALYSIS.md` and
`plan/HEAD_ARCHITECTURE_ANALYSIS.md`.

---

## Phase 4: Head-Swap Ablation Results

### Protocol

- Engine: `hgi`, State: `alabama`, Seed: `42`.
- Stage A: 9 candidates, 1 fold x 10 epochs.
- Stage B: Top 3 promoted to 2 folds x 15 epochs.

### Stage A Results (1 fold x 10 epochs)

| Rank | Candidate | Joint | Next F1 | Cat F1 |
|------|-----------|-------|---------|--------|
| 1 | head_cgc_s2t2_both_tcnn | 0.345 | 0.185 | 0.506 |
| 2 | head_cgc_s2t2_both_tcnn_db | 0.341 | 0.185 | 0.497 |
| 3 | head_next_tcnn | 0.318 | 0.185 | 0.451 |
| 4 | head_dselectk_both_tcnn | 0.315 | 0.198 | 0.432 |
| 5 | head_both_tcnn | 0.311 | 0.203 | 0.420 |
| 6 | head_cat_single | 0.303 | 0.195 | 0.411 |
| 7 | head_both_topt | 0.276 | 0.119 | 0.433 |
| 8 | head_next_topt | 0.267 | 0.130 | 0.405 |
| 9 | head_ple_l2_both_tcnn | 0.235 | 0.176 | 0.294 |

### Stage B Results (promoted top 3 -> 2 folds x 15 epochs)

| Rank | Candidate | Joint | Next F1 | Cat F1 |
|------|-----------|-------|---------|--------|
| 1 | head_cgc_s2t2_both_tcnn_db | 0.389 | 0.194 | 0.584 |
| 2 | head_cgc_s2t2_both_tcnn | 0.387 | 0.193 | 0.581 |
| 3 | head_next_tcnn | 0.377 | 0.191 | 0.564 |

### Comparison Against Prior Best

| Configuration | Joint | Next F1 | Cat F1 |
|---------------|-------|---------|--------|
| **Prior best: cgc_s2t2 + equal (default heads)** | **0.4855** | **0.259** | **0.712** |
| Head swap: cgc_s2t2 + both_tcnn + db_mtl | 0.389 | 0.194 | 0.584 |
| Head swap: cgc_s2t2 + both_tcnn + equal | 0.387 | 0.193 | 0.581 |
| Head swap: base mtlnet + next_tcnn + equal | 0.377 | 0.191 | 0.564 |

---

## High-Impact Findings

### 1. Head swaps hurt performance in the MTL pipeline

The swapped heads reduced joint score from 0.4855 to 0.389 (a 20% drop).
Both category F1 (0.712 -> 0.584) and next F1 (0.259 -> 0.194) degraded.

### 2. Standalone head benchmarks do NOT predict MTL performance

The standalone head ablation (1 epoch, raw embeddings, no shared backbone)
showed:
- `category_single` (F1=0.426) matching `category_transformer` (F1=0.280).
- `next_temporal_cnn` (F1=0.242) dominating `next_mtl` (F1=0.026).

In the full MTL pipeline with 10+ epochs and a shared backbone, these
rankings reversed. The shared backbone produces richer, task-conditioned
representations that the default heads are already well-tuned to process.

**Lesson:** Standalone head rankings measure learning speed from raw
embeddings. MTL rankings measure end-to-end feature extraction where the
head and backbone co-adapt during training.

### 3. Temporal CNN consistently outperforms transformer_optimized for next

Within the Phase 4 results, temporal CNN next heads consistently beat
transformer_optimized variants (0.185-0.203 vs 0.119-0.130 next F1).
This relative ranking does hold across MTL settings.

### 4. PLE + swapped heads was worst

`head_ple_l2_both_tcnn` scored 0.235 joint -- worst in the field. Too
many architectural changes at once prevented effective co-adaptation.

### 5. CGC s2t2 remains the best backbone architecture

Even with suboptimal heads, CGC s2t2 candidates took the top 2 spots.
The architecture contribution dominates the head contribution.

---

## Parameter Budget Analysis

The head-swap analysis revealed a structural imbalance in the current
MTLnet architecture:

| Component | Default Heads | Swapped Heads |
|-----------|--------------|---------------|
| Task encoders | 28% | 30% |
| Shared backbone | 10% | 14% |
| Category head | 14% | 2% |
| Next head | 48% | 54% (TCN) |

The shared backbone -- where MTL knowledge transfer occurs -- is only
10-14% of parameters regardless of head choice. Despite the head swaps
not improving results, this analysis suggests that **growing the shared
backbone** (more layers, wider hidden size) could be more impactful than
any head architecture change.

---

## Recommendations

### Do Not Change

1. **Default category head:** Keep `CategoryHeadTransformer`. Despite
   being larger than needed for standalone classification, it co-adapts
   well with the shared backbone in MTL training.

2. **Default next head:** Keep `NextHeadMTL`. The Transformer with causal
   masking and attention pooling outperforms alternatives when receiving
   MTL-conditioned features.

3. **Best configuration:** `mtlnet_cgc(s=2, t=2) + equal_weight` with
   default heads remains the HGI winner (joint=0.4855).

### Worth Investigating Next

1. **Backbone scaling:** Increase `shared_layer_size` from 256 to 384 or
   512, or increase `num_shared_layers` from 4 to 6. The shared backbone
   is undersized relative to the heads.

2. **Phase 3 optimizer ablation:** CAGrad, Aligned-MTL, and DWA should
   be tested with the default heads (not swapped heads). They were
   implemented and smoke-tested but not yet ablated on real data.

3. **5-fold confirmation** of `cgc_s2t2 + equal_weight` as recommended
   in the prior ablation report.

---

## Reproducibility Artifacts

### Phase 4 Head-Swap

- Stage A: `results/ablations/phase4_1fold_10ep_seed42/summary.csv`
- Stage B: `results/ablations/phase4_promoted_2fold_15ep_seed42/summary.csv`

### Analysis Documents

- `plan/NEW_CANDIDATES_ANALYSIS.md` — optimizer/architecture research
- `plan/HEAD_ARCHITECTURE_ANALYSIS.md` — head architecture research

### Implementations Added

- `src/losses/cagrad/` — CAGrad (NeurIPS 2021)
- `src/losses/aligned_mtl/` — Aligned-MTL (CVPR 2023)
- `src/losses/dwa/` — DWA (CVPR 2019)
- `src/models/mtl/mtlnet_ple/` — PLE (RecSys 2020)
- `src/models/mtl/_components.py` — `PLELiteLayer`
- `src/ablation/candidates.py` — 22 new candidates (phases 3-4)
- Tests: 687 passed, 0 failed, 18 skipped
