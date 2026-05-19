## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| shipping (ep=50, seed=42) | 35.0 ± 5.7 | 70.49 ± 0.86 | 4.2 ± 0.4 | 76.12 ± 0.33 |
| T6.4 two_pass (ep=50, seed=42) | 38.0 ± 5.6 | 70.55 ± 0.85 | 4.8 ± 0.4 | 76.20 ± 0.27 |
| T6.4 infonce τ=0.5 (ep=50, seed=42) | 37.2 ± 6.1 | 70.49 ± 0.95 | 4.4 ± 0.5 | 76.29 ± 0.29 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping (ep=50, seed=42) | 14.0 ± 8.5 | 67.93 ± 1.74 | 72.38 ± 2.20 |
| T6.4 two_pass (ep=50, seed=42) | 12.2 ± 9.5 | 67.33 ± 2.06 | 73.33 ± 2.28 |
| T6.4 infonce τ=0.5 (ep=50, seed=42) | 12.2 ± 9.6 | 67.12 ± 2.45 | 73.48 ± 2.48 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| shipping (ep=50, seed=42) | 29.2 ± 10.8 | 69.99 ± 1.13 | 65.38 ± 9.10 |
| T6.4 two_pass (ep=50, seed=42) | 30.2 ± 11.3 | 70.13 ± 1.06 | 61.19 ± 11.86 |
| T6.4 infonce τ=0.5 (ep=50, seed=42) | 31.6 ± 10.6 | 70.28 ± 0.82 | 56.78 ± 11.79 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
