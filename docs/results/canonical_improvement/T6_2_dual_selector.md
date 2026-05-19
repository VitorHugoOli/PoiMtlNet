## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| shipping | 35.0 ± 5.7 | 70.49 ± 0.86 | 4.2 ± 0.4 | 76.12 ± 0.33 |
| T6.2 α=1.5 w_r=0.3 | 37.6 ± 5.2 | 67.82 ± 0.85 | 4.2 ± 0.4 | 76.49 ± 0.33 |
| T6.2 α=1.5 w_r=0.5 | 37.6 ± 5.2 | 69.21 ± 0.84 | 4.4 ± 0.5 | 76.35 ± 0.29 |
| T6.2 α=2.0 w_r=0.3 | 42.8 ± 3.8 | 66.94 ± 0.77 | 6.2 ± 4.4 | 76.88 ± 0.82 |
| T6.2 α=2.0 w_r=0.5 | 38.0 ± 5.6 | 68.40 ± 0.74 | 4.4 ± 0.5 | 76.39 ± 0.23 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping | 14.0 ± 8.5 | 67.93 ± 1.74 | 72.38 ± 2.20 |
| T6.2 α=1.5 w_r=0.3 | 22.2 ± 12.1 | 65.81 ± 2.33 | 70.34 ± 1.64 |
| T6.2 α=1.5 w_r=0.5 | 14.6 ± 8.2 | 66.49 ± 2.00 | 71.79 ± 1.91 |
| T6.2 α=2.0 w_r=0.3 | 21.8 ± 8.1 | 65.08 ± 1.51 | 72.01 ± 3.76 |
| T6.2 α=2.0 w_r=0.5 | 23.6 ± 10.4 | 67.05 ± 1.10 | 70.09 ± 1.14 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping | 29.2 ± 10.8 | 69.99 ± 1.13 | 65.38 ± 9.10 |
| T6.2 α=1.5 w_r=0.3 | 33.2 ± 4.3 | 67.59 ± 0.94 | 49.48 ± 2.78 |
| T6.2 α=1.5 w_r=0.5 | 33.2 ± 6.3 | 69.01 ± 0.61 | 56.58 ± 11.57 |
| T6.2 α=2.0 w_r=0.3 | 33.8 ± 7.1 | 66.32 ± 0.91 | 65.56 ± 9.58 |
| T6.2 α=2.0 w_r=0.5 | 31.0 ± 3.5 | 68.15 ± 0.88 | 56.77 ± 11.41 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
