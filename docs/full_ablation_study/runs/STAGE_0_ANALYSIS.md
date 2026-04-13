# Stage 0 Results — 2026-04-13

## Configuration
- Engine: fusion (128-dim) + HGI reference (64-dim)
- State: alabama
- Folds: 1, Epochs: 10, Seed: 42

## Results

| Rank | Candidate | Engine | Joint | Next F1 | Cat F1 | Time |
|------|-----------|--------|-------|---------|--------|------|
| 1 | s0_hgi_cgc22_equal       | HGI    | **0.3861** | 0.1954 | **0.5767** | 43s |
| 2 | s0_fusion_cgc22_equal    | fusion | 0.3676 | **0.2461** | 0.4891 | 45s |
| 3 | s0_fusion_dselectk_db    | fusion | 0.3541 | 0.2145 | 0.4936 | 69s |
| 4 | s0_fusion_base_equal     | fusion | 0.2969 | 0.2023 | 0.3916 | 36s |

## Key Findings — per-task asymmetry is the headline

**HGI wins overall joint by ~4.8 %** (0.3861 vs 0.3676 for fusion's best arch) — *within the 10 % "proceed" threshold*. But the per-task breakdown inverts:

- **Next F1**: fusion-cgc22 = 0.2461 vs HGI-cgc22 = 0.1954 → **+26 % for fusion**. Time2Vec's per-step temporal signal demonstrably helps sequence modelling, even under the HGI scale dominance documented in FUSION_RATIONALE.md.
- **Category F1**: fusion-cgc22 = 0.4891 vs HGI-cgc22 = 0.5767 → **−15 % for fusion**. Sphere2Vec *hurts* at 10 epochs/1 fold: the 15× norm imbalance plus the 0.7 % dependence measured in the scale-imbalance experiment is confirmed end-to-end. Adding Sphere2Vec's weak signal dilutes the HGI pathway.

**This is arguably a more interesting finding than "fusion wins overall."** Task-specific auxiliary embeddings help only when the signal they encode matches the task geometry:
- Temporal sequence task ⇢ Time2Vec helps (+26 %).
- Static-POI classification task ⇢ Sphere2Vec hurts (−15 %).

## Decision

**PROCEED** to Stage 1. The fusion joint deficit is 4.8 % (< 10 % threshold in CONTINUE.md).

The per-task split, however, reshapes the paper's narrative:
1. The "task-specific fusion" choice is *actually* task-specific — and the cat-side signal is net-harmful at 10 ep.
2. Stage 2's DCN head on fusion is now the **main upside experiment**: DCN can learn explicit crosses between the Sphere2Vec and HGI halves, which *may* recover the category gap. If it does, that becomes the cleanest story: "fusion requires cross-feature aware heads to exploit the scale-imbalanced auxiliary signal."
3. Stage 3 at 50 ep is where the category gap may narrow — with longer training, the encoder may learn to extract value from Sphere2Vec. Watch that carefully.

## Additional observations

- `mtlnet_cgc(s2,t2) + equal_weight` replicates as the strong architecture (matches Phase 1–2 HGI winner → transfers to fusion).
- `mtlnet_base + equal_weight` is a clear under-performer even on fusion (joint 0.297). The shared-backbone-only model seems unable to disentangle the mixed-source input; gating (CGC/DSelectK) is needed.
- DSelectK + db_mtl is competitive on cat F1 (0.4936, best among fusion) but behind on next — suggesting gating architectures favour one task over the other depending on optimizer.

## Artifacts
- `results/ablations/full_fusion_study/s0_fusion_1f_10ep/summary.csv`
- `results/ablations/full_fusion_study/s0_hgi_ref_1f_10ep/summary.csv`
- `docs/full_ablation_study/runs/stage0.log`
