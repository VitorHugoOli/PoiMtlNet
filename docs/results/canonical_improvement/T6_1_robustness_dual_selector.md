## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| shipping | 35.0 ± 5.7 | 70.49 ± 0.86 | 4.2 ± 0.4 | 76.12 ± 0.33 |
| T6.1 λ=0.2 ORIGINAL (B=1024, τ=0.1, dedup, asym) | 40.2 ± 2.4 | 70.41 ± 0.82 | 4.8 ± 0.4 | 76.23 ± 0.43 |
| T6.1 λ=0.2 ROBUST (B=4096, τ=0.3, no-dedup, sym) | 37.6 ± 5.2 | 70.50 ± 0.98 | 4.4 ± 0.5 | 76.29 ± 0.22 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping | 14.0 ± 8.5 | 67.93 ± 1.74 | 72.38 ± 2.20 |
| T6.1 λ=0.2 ORIGINAL (B=1024, τ=0.1, dedup, asym) | 9.0 ± 1.7 | 66.89 ± 1.17 | 73.72 ± 1.28 |
| T6.1 λ=0.2 ROBUST (B=4096, τ=0.3, no-dedup, sym) | 12.2 ± 9.5 | 67.28 ± 2.25 | 73.31 ± 2.36 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping | 29.2 ± 10.8 | 69.99 ± 1.13 | 65.38 ± 9.10 |
| T6.1 λ=0.2 ORIGINAL (B=1024, τ=0.1, dedup, asym) | 34.6 ± 12.3 | 69.90 ± 0.96 | 61.05 ± 11.95 |
| T6.1 λ=0.2 ROBUST (B=4096, τ=0.3, no-dedup, sym) | 26.6 ± 8.8 | 70.12 ± 1.18 | 61.31 ± 11.39 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
