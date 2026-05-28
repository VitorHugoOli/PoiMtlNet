# log_T-KD sweep — florida

Single-seed=42, 5 folds, 50 epochs. Baseline (w=0.0) vs supervisory log_T KD at w ∈ {0.05, 0.1, 0.2}.

| log_T-KD weight | MTL @ disjoint reg | MTL @ geom_simple reg | MTL @ b9 reg | disjoint cat | geom cat |
|---|---:|---:|---:|---:|---:|
| 0.0 | 63.98 ± 0.76 | 61.14 ± 0.95 | 53.73 ± 9.22 | 70.49 ± 0.92 | 66.98 ± 0.80 |
| 0.05 | 66.18 ± 0.77 | 64.27 ± 1.15 | 57.96 ± 1.02 | 70.40 ± 0.86 | 66.36 ± 1.16 |
| 0.1 | 66.21 ± 0.69 | 64.74 ± 1.31 | 58.79 ± 1.20 | 70.51 ± 0.79 | 66.85 ± 1.37 |
| 0.2 | 66.30 ± 0.80 | 65.35 ± 1.15 | 59.52 ± 0.70 | 70.52 ± 0.99 | 66.93 ± 1.56 |

**Wilcoxon (one-sided, w>baseline) on disjoint reg:**
- w=0.05: Δ=+2.20 pp, p=0.0312
- w=0.1: Δ=+2.23 pp, p=0.0312
- w=0.2: Δ=+2.32 pp, p=0.0312
