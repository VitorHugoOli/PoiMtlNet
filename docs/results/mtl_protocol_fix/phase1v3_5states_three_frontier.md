## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| AL_H3alt | 38.6 ± 12.8 | 45.76 ± 1.34 | 12.8 ± 1.9 | 50.82 ± 3.21 |
| AZ_H3alt | 47.2 ± 3.7 | 48.87 ± 1.80 | 6.2 ± 1.9 | 41.33 ± 2.73 |
| FL_B9 | 36.6 ± 5.3 | 70.45 ± 0.86 | 4.4 ± 0.5 | 76.47 ± 0.35 |
| CA_B9 | 31.2 ± 9.1 | 64.78 ± 0.38 | 2.0 ± 0.0 | 50.61 ± 1.23 |
| TX_B9 | 39.8 ± 8.2 | 64.97 ± 0.17 | 1.0 ± 0.0 | 50.83 ± 1.89 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| AL_H3alt | 28.8 ± 12.3 | 45.18 ± 1.89 | 48.56 ± 3.13 |
| AZ_H3alt | 15.6 ± 3.7 | 47.30 ± 1.75 | 39.60 ± 3.11 |
| FL_B9 | 10.2 ± 1.6 | 67.38 ± 1.06 | 72.88 ± 1.49 |
| CA_B9 | 6.2 ± 9.4 | 58.44 ± 3.56 | 49.24 ± 3.55 |
| TX_B9 | 2.2 ± 2.7 | 56.65 ± 3.79 | 49.30 ± 4.47 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| AL_H3alt | 38.6 ± 12.8 | 45.76 ± 1.34 | 47.60 ± 4.14 |
| AZ_H3alt | 22.4 ± 6.1 | 48.20 ± 1.63 | 38.43 ± 2.31 |
| FL_B9 | 28.8 ± 9.3 | 69.78 ± 1.19 | 61.47 ± 11.48 |
| CA_B9 | 26.4 ± 6.9 | 64.52 ± 0.59 | 42.31 ± 2.75 |
| TX_B9 | 36.8 ± 7.6 | 64.87 ± 0.19 | 40.39 ± 0.51 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
