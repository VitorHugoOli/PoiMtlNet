# Rank 2 balanced-sampler compare — alabama

Single-seed=42, 5 folds, 50 epochs. baseline (weighted-CE) vs --reg-balanced-sampler (WeightedRandomSampler on reg only).

| arm | disjoint reg | geom_simple reg | b9 reg | disjoint cat | geom cat |
|---|---:|---:|---:|---:|---:|
| baseline | 50.82 ± 3.21 | 48.56 ± 3.13 | 47.60 ± 4.14 | 45.76 ± 1.34 | 45.18 ± 1.89 |
| balanced | 20.36 ± 5.72 | 20.33 ± 5.78 | 19.43 ± 5.43 | 45.54 ± 1.85 | 44.73 ± 1.87 |

**Wilcoxon (one-sided, balanced > baseline) on disjoint reg**: Δ=-30.46 pp, p=1.0000
