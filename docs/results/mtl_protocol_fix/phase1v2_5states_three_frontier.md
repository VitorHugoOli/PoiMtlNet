## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| AL_shipping | 38.0 ± 5.4 | 42.05 ± 0.54 | 36.0 ± 8.8 | 50.15 ± 3.73 |
| AZ_shipping | 35.0 ± 10.4 | 46.98 ± 1.76 | 20.2 ± 4.1 | 41.07 ± 2.69 |
| FL_shipping | 36.6 ± 5.3 | 70.45 ± 0.86 | 4.4 ± 0.5 | 76.47 ± 0.35 |
| CA_shipping | 31.2 ± 9.1 | 64.78 ± 0.38 | 2.0 ± 0.0 | 50.61 ± 1.23 |
| TX_shipping | 39.8 ± 8.2 | 64.97 ± 0.17 | 1.0 ± 0.0 | 50.83 ± 1.89 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| AL_shipping | 39.6 ± 7.0 | 41.97 ± 0.56 | 49.88 ± 3.79 |
| AZ_shipping | 37.4 ± 5.0 | 46.96 ± 1.77 | 40.37 ± 2.63 |
| FL_shipping | 10.2 ± 1.6 | 67.38 ± 1.06 | 72.88 ± 1.49 |
| CA_shipping | 6.2 ± 9.4 | 58.44 ± 3.56 | 49.24 ± 3.55 |
| TX_shipping | 2.2 ± 2.7 | 56.65 ± 3.79 | 49.30 ± 4.47 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| AL_shipping | 41.4 ± 4.6 | 41.98 ± 0.58 | 49.81 ± 3.67 |
| AZ_shipping | 41.4 ± 6.4 | 46.95 ± 1.78 | 40.35 ± 2.65 |
| FL_shipping | 28.8 ± 9.3 | 69.78 ± 1.19 | 61.47 ± 11.48 |
| CA_shipping | 26.4 ± 6.9 | 64.52 ± 0.59 | 42.31 ± 2.75 |
| TX_shipping | 36.8 ± 7.6 | 64.87 ± 0.19 | 40.39 ± 0.51 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
