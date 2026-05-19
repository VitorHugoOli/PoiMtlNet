## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| shipping (ep=50, seed=42) | 35.0 ± 5.7 | 70.49 ± 0.86 | 4.2 ± 0.4 | 76.12 ± 0.33 |
| T6.1 λ_p2p=0.05 | 38.0 ± 5.6 | 70.50 ± 0.95 | 4.4 ± 0.5 | 76.32 ± 0.19 |
| T6.1 λ_p2p=0.1 | 36.6 ± 5.3 | 70.44 ± 0.82 | 4.2 ± 0.4 | 76.17 ± 0.28 |
| T6.1 λ_p2p=0.2 | 40.2 ± 2.4 | 70.41 ± 0.82 | 4.8 ± 0.4 | 76.23 ± 0.43 |
| T6.1 λ_p2p=0.3 | 38.6 ± 6.2 | 70.40 ± 0.75 | 4.4 ± 0.5 | 76.28 ± 0.32 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping (ep=50, seed=42) | 14.0 ± 8.5 | 67.93 ± 1.74 | 72.38 ± 2.20 |
| T6.1 λ_p2p=0.05 | 12.2 ± 9.5 | 67.20 ± 2.42 | 73.25 ± 2.28 |
| T6.1 λ_p2p=0.1 | 12.8 ± 9.2 | 67.19 ± 2.23 | 73.22 ± 2.27 |
| T6.1 λ_p2p=0.2 | 9.0 ± 1.7 | 66.89 ± 1.17 | 73.72 ± 1.28 |
| T6.1 λ_p2p=0.3 | 15.4 ± 12.6 | 67.95 ± 1.22 | 72.39 ± 2.12 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping (ep=50, seed=42) | 29.2 ± 10.8 | 69.99 ± 1.13 | 65.38 ± 9.10 |
| T6.1 λ_p2p=0.05 | 28.0 ± 8.9 | 70.23 ± 0.67 | 65.38 ± 9.16 |
| T6.1 λ_p2p=0.1 | 28.0 ± 11.5 | 70.08 ± 0.75 | 61.30 ± 11.76 |
| T6.1 λ_p2p=0.2 | 34.6 ± 12.3 | 69.90 ± 0.96 | 61.05 ± 11.95 |
| T6.1 λ_p2p=0.3 | 34.8 ± 4.9 | 70.11 ± 1.10 | 65.09 ± 8.90 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
