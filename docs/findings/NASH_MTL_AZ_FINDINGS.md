# Nash-MTL on Arizona — findings (2026-04-19)

**Setup:** cross-attn backbone, Nash-MTL optimizer, AZ 5f × 50ep, max_lr=0.003, per-task modality, GRU region head (hd=256).

**Motivation:** user's hypothesis — since we have a Pareto-bidirectional problem (cat vs reg), Nash-MTL (designed for Nash-equilibrium gradient balancing) might outperform PCGrad as the MTL loss.

## Results

| Config | cat F1 | reg Acc@10 (indist) | reg MRR (indist) | cat Top3 | reg Top5 |
|---|---:|---:|---:|---:|---:|
| STL (AZ) | 42.08 ± 0.89 | 48.88 ± 2.48 | — | — | — |
| cross-attn + **PCGrad** (baseline) | 43.13 ± 0.55 | 41.07 ± 3.46 | — | — | — |
| **cross-attn + Nash-MTL** | **43.35 ± 0.84** | **40.14 ± 3.11** | 21.31 ± 1.87 | 85.51 ± 0.91 | 31.00 ± 2.98 |
| Δ (Nash − PCGrad) | +0.22 pp | −0.93 pp | — | — | — |

## Verdict: Nash-MTL is statistically tied with PCGrad on AZ

Both deltas fall well within the per-fold σ (cat ±0.55–0.84, reg ±3.1–3.5 on this task pair). Neither task side shows a meaningful preference for Nash-MTL's solution concept over PCGrad's gradient-projection on this data.

**Interpretation:** the Pareto-bidirectional framing captures the *problem structure* but at AZ scale the two optimizers reach essentially the same stationary point. This suggests that on cross-attn the gradient-conflict is already mild — content-based cross-attention lets the task streams share without competing for shared parameters, so PCGrad's projections find nothing material to project away. Nash-MTL's more principled game-theoretic weighting therefore has no dominant lever to pull.

## What this means for the paper

- **CH-Mx (draft claim):** "Nash-MTL's Pareto-equilibrium framing gives cross-attn an additional lift over PCGrad." → **NOT supported on AZ**. Mark hypothesis as tested and rejected.
- **Scale-curve narrative still holds:** the paper's headline stays +3.29 pp FL cat via cross-attn+pcgrad. Nash doesn't change the story on either head.
- **Worth testing on FL?** If Pareto conflict magnifies with scale (more region classes = more gradient interference), Nash might open up a gap on FL. Estimated cost ~4 h on Mac; low expected lift given AZ null result.

## Cost

- 42.87 min wall, 5 folds, max RSS ~2 GB, no instability.
- Clean run with the `--no-checkpoints` fix from commit `10889ba` — no torch.save crashes, no SSD flakes.

## Next actions

1. **Keep cross-attn + pcgrad as champion** (marginal lead + more conservative narrative).
2. **Region hparam investigation** (H-R1 hd=512 OOM'd on AZ — retry at **hd=384**, or skip to H-R4 cat_weight=0.3).
3. **Deferred**: Nash on FL (only if region hparams show region lift — otherwise no reason to rerun loss family on FL).
