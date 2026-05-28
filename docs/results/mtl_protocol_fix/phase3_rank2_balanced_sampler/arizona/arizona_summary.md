# Rank 2 balanced-sampler compare — arizona

Single-seed=42, 5 folds, 50 epochs. baseline (weighted-CE) vs --reg-balanced-sampler (WeightedRandomSampler on reg only).

| arm | disjoint reg | geom_simple reg | b9 reg | disjoint cat | geom cat |
|---|---:|---:|---:|---:|---:|
| baseline | 41.33 ± 2.73 | 39.60 ± 3.11 | 38.43 ± 2.31 | 48.87 ± 1.80 | 47.30 ± 1.75 |
| balanced | 22.83 ± 9.72 | 22.83 ± 9.72 | 15.61 ± 2.13 | 48.71 ± 1.60 | 47.58 ± 2.07 |

**Wilcoxon (one-sided, balanced > baseline) on disjoint reg**: Δ=-18.49 pp, p=1.0000
