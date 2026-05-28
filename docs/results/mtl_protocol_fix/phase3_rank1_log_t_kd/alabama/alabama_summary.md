# log_T-KD sweep — alabama

Single-seed=42, 5 folds, 50 epochs. Baseline (w=0.0) vs supervisory log_T KD at w ∈ {0.05, 0.1, 0.2}.

| log_T-KD weight | MTL @ disjoint reg | MTL @ geom_simple reg | MTL @ b9 reg | disjoint cat | geom cat |
|---|---:|---:|---:|---:|---:|
| 0.0 | 50.82 ± 3.21 | 48.56 ± 3.13 | 47.60 ± 4.14 | 45.76 ± 1.34 | 45.18 ± 1.89 |
| 0.05 | 52.34 ± 3.15 | 49.94 ± 3.40 | 48.56 ± 2.91 | 45.68 ± 1.75 | 44.80 ± 1.70 |
| 0.1 | 52.92 ± 3.48 | 50.84 ± 3.16 | 48.99 ± 3.81 | 45.78 ± 1.90 | 45.02 ± 1.64 |
| 0.2 | 53.22 ± 3.30 | 51.48 ± 2.89 | 49.87 ± 3.86 | 45.74 ± 1.48 | 45.14 ± 2.57 |

**Wilcoxon (one-sided, w>baseline) on disjoint reg:**
- w=0.05: Δ=+1.52 pp, p=0.0312
- w=0.1: Δ=+2.10 pp, p=0.0312
- w=0.2: Δ=+2.40 pp, p=0.0312
