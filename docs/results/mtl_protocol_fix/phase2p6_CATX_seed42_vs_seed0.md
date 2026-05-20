## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| CA_seed42 | 31.2 ± 9.1 | 64.78 ± 0.38 | 2.0 ± 0.0 | 50.61 ± 1.23 |
| CA_seed0 | 31.2 ± 7.6 | 64.92 ± 0.15 | 2.0 ± 0.0 | 50.36 ± 0.93 |
| TX_seed42 | 39.8 ± 8.2 | 64.97 ± 0.17 | 1.0 ± 0.0 | 50.83 ± 1.89 |
| TX_seed0 | 37.4 ± 7.7 | 65.21 ± 0.42 | 1.0 ± 0.0 | 49.71 ± 2.01 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| CA_seed42 | 6.2 ± 9.4 | 58.44 ± 3.56 | 49.24 ± 3.55 |
| CA_seed0 | 7.4 ± 11.5 | 59.55 ± 3.16 | 48.37 ± 3.41 |
| TX_seed42 | 2.2 ± 2.7 | 56.65 ± 3.79 | 49.30 ± 4.47 |
| TX_seed0 | 6.8 ± 9.3 | 60.00 ± 3.76 | 46.60 ± 4.87 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| CA_seed42 | 26.4 ± 6.9 | 64.52 ± 0.59 | 42.31 ± 2.75 |
| CA_seed0 | 27.6 ± 9.8 | 64.90 ± 0.14 | 42.06 ± 2.84 |
| TX_seed42 | 36.8 ± 7.6 | 64.87 ± 0.19 | 40.39 ± 0.51 |
| TX_seed0 | 38.6 ± 9.1 | 65.20 ± 0.43 | 40.16 ± 0.49 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
