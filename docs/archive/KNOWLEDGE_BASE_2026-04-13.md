# MTLnet Knowledge Base (2026-04-13)

Consolidated findings from the Phase 3-4 improvement work. This document
captures empirical results, architectural insights, and design decisions
that should inform all future experimentation.

---

## 1. Empirical Results Summary

### Prior Ablation (Phase 1-2, HGI+DGI, Alabama)

Source: `docs/MTL_ABLATION_REPORT_2026-04-11.md`

| Engine | Best Config | Joint | Next F1 | Cat F1 |
|--------|------------|-------|---------|--------|
| HGI | cgc_s2t2 + equal_weight | **0.4855** | 0.259 | 0.712 |
| DGI | dselectk_e4k2 + db_mtl | 0.3337 | 0.247 | 0.420 |

Key finding: **Engine choice changes the winner more than arch/loss choice.**

### Phase 4 Head-Swap Ablation (HGI, Alabama)

Source: `docs/MTL_ABLATION_REPORT_2026-04-13.md`

Best head-swap (cgc_s2t2 + both_tcnn + db_mtl): joint=0.389 --
**20% worse than default heads** (0.4855).

Key finding: **Standalone head rankings do NOT predict MTL performance.**
Heads co-adapt with the shared backbone; swapping in "better" standalone
heads breaks this co-adaptation.

### Standalone Head Rankings (HGI, Alabama, 10 epochs)

**Category heads:**

| Rank | Head | F1 | Notable |
|------|------|-----|---------|
| 1 | cat_dcn | 0.728 | Deep & Cross Network — best by a wide margin |
| 2 | cat_residual | 0.690 | |
| 3 | cat_gated | 0.679 | |
| 8 | cat_linear | 0.545 | Single nn.Linear — backbone does most work |
| 9 | cat_transformer | 0.402 | Current MTL default — worst standalone |

**Next heads:**

| Rank | Head | F1 | Notable |
|------|------|-----|---------|
| 1 | **next_tcn_residual** | **0.244** | New — canonical TCN with residual blocks |
| 2 | next_temporal_cnn | 0.200 | |
| 3 | next_single | 0.198 | |
| 9 | next_mtl | 0.043 | Current MTL default — worst standalone |
| 10 | next_transformer_relpos | 0.043 | New — relative pos bias didn't help |

---

## 2. Architectural Insights

### Parameter Budget Imbalance

The MTLnet model is **head-dominated**:

| Component | % of Params | Role |
|-----------|------------|------|
| Task encoders | 28% | Per-task MLP preprocessing |
| **Shared backbone** | **10%** | **MTL knowledge transfer (the whole point)** |
| Category head | 14% | CategoryHeadTransformer |
| Next head | 48% | NextHeadMTL (Transformer) |

The shared backbone — where MTL actually works — is the smallest
component. Future work should consider growing the backbone
(wider `shared_layer_size` or more `num_shared_layers`).

### Architecture Hierarchy of Importance

From all ablation data, the dimensions rank:

1. **Embedding engine** (HGI >> DGI; fusion TBD)
2. **MTL architecture** (CGC/DSelectK >> base MTLnet >> MMoE)
3. **Optimizer** (equal_weight and db_mtl consistently good)
4. **Task heads** (least impactful in MTL context)

### Gradient Dynamics

- Gradient cosine between tasks is **near zero** — minimal sustained
  conflict between category and next.
- This explains why simple equal weighting matches complex gradient
  solvers: there's little conflict to resolve.
- CAGrad and Aligned-MTL are the right methods to test whether this
  finding holds on fusion embeddings.

---

## 3. What Was Implemented

### New MTL Optimizers (3)

| Name | Registry Key | Paper | Type |
|------|-------------|-------|------|
| CAGrad | `cagrad` | Liu et al., NeurIPS 2021 | Conflict-averse gradient (closed-form for 2 tasks) |
| Aligned-MTL | `aligned_mtl` | Senushkin et al., CVPR 2023 | Eigendecomposition gradient alignment (no hyperparams) |
| DWA | `dwa` | Liu et al., CVPR 2019 | Loss-rate-based dynamic weighting |

### New MTL Architecture (1)

| Name | Registry Key | Paper |
|------|-------------|-------|
| PLE | `mtlnet_ple` | Tang et al., RecSys 2020 |

Stacked CGC layers with inter-level gating. Natural extension of CGC.
Had poor Phase 4 results (joint=0.235) — likely needs its own tuning
rather than being tested with simultaneous head swaps.

### New Task Heads (4)

| Name | Registry Key | Task | Result |
|------|-------------|------|--------|
| TCN Residual | `next_tcn_residual` | next | **Best standalone next head** (F1=0.244) |
| Conv-Attention | `next_conv_attn` | next | Mid-pack (F1=0.161) |
| Transformer RelPos | `next_transformer_relpos` | next | Poor (F1=0.043) |
| Linear Probe | `category_linear` | category | Diagnostic (F1=0.545) |

### Infrastructure

- Loss registry: 20 canonical losses + 3 aliases
- Model registry: 5 MTL architectures, 9 category heads, 10 next heads
- Ablation runner: `--embedding-dim` support for fusion (128-dim)
- Phase 3/4/5 stage support in CLI
- Full fusion ablation runner: `experiments/full_fusion_ablation.py`

---

## 4. Fusion Embedding Design

Preset: `space_hgi_time`

| Task | Embeddings | Dim | Signal |
|------|-----------|-----|--------|
| Category | Sphere2Vec(64) + HGI(64) | 128 | Where (spherical coords) + What (graph structure) |
| Next | HGI(64) + Time2Vec(64) | 128 | What (graph structure) + When (temporal context) |

**Rationale:**
- Category: spatial location correlates with POI type (restaurants
  cluster downtown). Sphere2Vec preserves spherical distance. HGI
  captures graph neighborhood. Complementary signals.
- Next: Time2Vec transforms the check-in window from a POI sequence
  into a spatio-temporal trajectory. Same POI at 8am vs 8pm gets
  different vectors, encoding temporal patterns (morning routines,
  evening entertainment).

**Scale imbalance — experimentally validated (2026-04-13):**

Sphere2Vec L2 norm: 0.55, HGI L2 norm: 8.46 (**15.2x ratio**).
Gradient analysis on real data confirmed the model receives 12x larger
gradients from HGI than Sphere2Vec features. Zero-ablation showed the
encoder is 90% dependent on HGI and only 0.7% on Sphere2Vec after 10
training steps.

**Normalization tested and rejected:** Per-source z-score standardization
and learnable per-source LayerNorm both **hurt** performance (accuracy
dropped from 0.606 to 0.504). The model naturally down-weights the
weaker source through gradient magnitude — forcing balanced contributions
dilutes the stronger HGI signal.

**Implication:** The fusion may effectively be HGI + noise. Stage 0 of
the ablation study (fusion vs HGI-only) will determine whether the
auxiliary embeddings contribute anything at full training scale.

See `docs/full_ablation_study/FUSION_RATIONALE.md` for full experimental
details.

**DCN relevance:** The Deep & Cross category head may still excel on
fusion if it can learn useful cross-features between the spatial
(Sphere2Vec) and structural (HGI) dimensions that concatenation misses.

---

## 5. Methods Researched and Rejected

### Optimizers (not implemented)

| Method | Reason |
|--------|--------|
| SDMGrad (NeurIPS 2023) | 3x gradient cost for marginal benefit |
| MoCo (ICLR 2023) | Redundant with DB-MTL (both use EMA gradients) |
| MGDA (NeurIPS 2018) | Superseded by CAGrad |
| GradDrop, GradVac | Incremental over PCGrad |
| IMTL (ICLR 2021) | Subsumed by Aligned-MTL |
| AutoLambda (TMLR 2022) | Requires validation loss every step |

### Architectures (not implemented)

| Method | Reason |
|--------|--------|
| HoME (Kuaishou 2024) | Designed for 6+ tasks; overkill for 2 |
| Cross-stitch (CVPR 2016) | Old; underperforms MMoE/CGC |
| MTAN (CVPR 2019) | Vision-specific (pixel-level attention) |

### Heads (not implemented)

| Method | Reason |
|--------|--------|
| Mamba / S4 | seq_len=9 too short; requires CUDA kernels |
| RetNet | seq_len=9 too short for linear complexity benefit |
| NextLocMoE | Requires pretrained LLM backbone |
| Capsule Networks | Over-engineered for flat vector classification |

Full justifications in `plan/NEW_CANDIDATES_ANALYSIS.md` and
`plan/HEAD_ARCHITECTURE_ANALYSIS.md`.

---

## 6. Test Suite Status

As of 2026-04-13: **692 passed, 0 failed, 17 skipped.**

All new implementations have:
- Registry integration
- metadata.yaml + README.md
- Parametrized forward-pass tests
- Smoke tests (20-step training convergence on synthetic data)
- Ablation candidate entries
