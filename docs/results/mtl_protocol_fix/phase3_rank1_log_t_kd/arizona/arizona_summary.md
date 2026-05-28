# log_T-KD sweep — arizona

Single-seed=42, 5 folds, 50 epochs. Baseline (w=0.0) vs supervisory log_T KD at w ∈ {0.05, 0.1, 0.2}.

| log_T-KD weight | MTL @ disjoint reg | MTL @ geom_simple reg | MTL @ b9 reg | disjoint cat | geom cat |
|---|---:|---:|---:|---:|---:|
| 0.0 | 41.33 ± 2.73 | 39.60 ± 3.11 | 38.43 ± 2.31 | 48.87 ± 1.80 | 47.30 ± 1.75 |
| 0.05 | 45.20 ± 2.87 | 42.87 ± 2.96 | 39.35 ± 2.54 | 48.61 ± 1.80 | 45.35 ± 1.25 |
| 0.1 | 46.19 ± 3.09 | 43.51 ± 2.72 | 40.28 ± 2.38 | 48.67 ± 1.85 | 46.05 ± 1.50 |
| 0.2 | 46.39 ± 2.83 | 44.97 ± 3.28 | 39.83 ± 3.18 | 48.87 ± 1.39 | 45.86 ± 1.12 |

**Wilcoxon (one-sided, w>baseline) on disjoint reg:**
- w=0.05: Δ=+3.88 pp, p=0.0312
- w=0.1: Δ=+4.86 pp, p=0.0312
- w=0.2: Δ=+5.06 pp, p=0.0312
