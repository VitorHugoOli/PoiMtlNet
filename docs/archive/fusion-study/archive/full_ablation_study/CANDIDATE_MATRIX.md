# Candidate Matrix (v2)

All candidates for the full fusion+alabama ablation study.

Engine: `fusion` (Sphere2Vec+HGI for category, HGI+Time2Vec for next)
State: `alabama` (Stages 0-3), `florida` (Stage 4)
Embedding dim: `128`

---

## Stage 0: Baseline Comparison

| Name | Engine | Architecture | Optimizer |
|------|--------|-------------|-----------|
| s0_fusion_base_equal | fusion (128D) | mtlnet | equal_weight |
| s0_fusion_cgc22_equal | fusion (128D) | mtlnet_cgc(s2,t2) | equal_weight |
| s0_fusion_dselectk_db | fusion (128D) | mtlnet_dselectk(e4,k2) | db_mtl |
| **s0_hgi_cgc22_equal** | **hgi (64D)** | mtlnet_cgc(s2,t2) | equal_weight |

The HGI reference enables direct comparison: does the richer fusion
embedding improve over HGI-only?

## Stage 1: Architecture x Optimizer (5 x 5 = 25)

### Architectures

| Short | Model | Config |
|-------|-------|--------|
| base | mtlnet | — |
| cgc22 | mtlnet_cgc | num_shared_experts=2, num_task_experts=2 |
| cgc21 | mtlnet_cgc | num_shared_experts=2, num_task_experts=1 |
| mmoe4 | mtlnet_mmoe | num_experts=4 |
| dsk42 | mtlnet_dselectk | num_experts=4, num_selectors=2, temperature=0.5 |

### Optimizers

| Short | Loss | Params | Type |
|-------|------|--------|------|
| eq | equal_weight | — | Static |
| db | db_mtl | beta=0.9, beta_sigma=0.5 | Gradient (EMA) |
| ca | cagrad | c=0.4 | Gradient (conflict-averse) |
| al | aligned_mtl | — | Gradient (eigendecomp) |
| uw | uncertainty_weighting | — | Learned (log-variance) |

### Full Matrix (25)

| | eq | db | ca | al | uw |
|---|---|---|---|---|---|
| base | s1_base_eq | s1_base_db | s1_base_ca | s1_base_al | s1_base_uw |
| cgc22 | s1_cgc22_eq | s1_cgc22_db | s1_cgc22_ca | s1_cgc22_al | s1_cgc22_uw |
| cgc21 | s1_cgc21_eq | s1_cgc21_db | s1_cgc21_ca | s1_cgc21_al | s1_cgc21_uw |
| mmoe4 | s1_mmoe4_eq | s1_mmoe4_db | s1_mmoe4_ca | s1_mmoe4_al | s1_mmoe4_uw |
| dsk42 | s1_dsk42_eq | s1_dsk42_db | s1_dsk42_ca | s1_dsk42_al | s1_dsk42_uw |

Screen: 1 fold x 10 epochs. Top 5 promoted to 2 folds x 15 epochs.

## Stage 2: Head Variants (on top-3 from Stage 1)

For each of the top-3 arch+optimizer from Stage 1 promoted results:

| Suffix | Cat Head | Next Head | Rationale |
|--------|----------|-----------|-----------|
| _hd_dcn | category_dcn | default | DCN cross-features between Sphere2Vec + HGI halves |
| _hd_tcnr | default | next_tcn_residual | TCN may leverage Time2Vec per-step signal |
| _hd_both | category_dcn | next_tcn_residual | Both swapped |

3 winners x 3 variants = 9 experiments at 2 folds x 15 epochs.

## Stage 3: Confirmation (top-3 overall)

Top 3 from Stages 1+2 combined. 5 folds x 50 epochs.

## Stage 4: Cross-State Validation

Top 1 from Stage 3. 5 folds x 50 epochs on **florida**.

---

## Summary

| Stage | Experiments | Config | Purpose |
|-------|------------|--------|---------|
| 0 | 4 | 1f x 10ep | Fusion sanity + HGI reference |
| 1 screen | 25 | 1f x 10ep | Arch x Optimizer sweep |
| 1 promote | 5 | 2f x 15ep | Top-5 confirmation |
| 2 | 9 | 2f x 15ep | Head variants |
| 3 | 3 | 5f x 50ep | Full confirmation |
| 4 | 1 | 5f x 50ep | Florida generalization |
| **Total** | **47** | | **~8-10 hours** |
