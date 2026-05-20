## Dual-selector analysis (single-seed=42, n=5 folds)

_Each variant trained for the full ep=50 horizon. Three selection rules applied per fold:_

- **Per-task disjoint best**: cat from its cat-best epoch, reg from its reg-best epoch (two checkpoints; substrate-capacity framing).
- **joint_geom_simple**: single epoch maximising `sqrt(cat_f1 * reg_top10_indist)` (one deployable checkpoint).
- **joint_canonical_b9**: single epoch maximising `0.5*(cat_f1 + reg_macro_f1)` — the current canonical selector. Shown for reference.

### Per-task disjoint best (substrate capacity)

| Variant | cat-best ep | cat F1 | reg-best ep | reg top10 |
|---|---:|---:|---:|---:|
| FL_seed42_STALE | 36.6 ± 5.3 | 70.45 ± 0.86 | 4.4 ± 0.5 | 76.47 ± 0.35 |
| FL_seed42_FRESH | 38.6 ± 5.6 | 70.49 ± 0.92 | 3.8 ± 0.4 | 63.98 ± 0.76 |
| FL_seed0 | 34.8 ± 7.0 | 70.40 ± 0.78 | 4.2 ± 0.4 | 63.70 ± 1.03 |
| FL_seed1 | 32.6 ± 8.4 | 70.26 ± 1.03 | 4.0 ± 0.0 | 64.15 ± 1.18 |
| FL_seed7 | 34.6 ± 10.5 | 70.40 ± 0.31 | 4.2 ± 0.8 | 63.89 ± 0.52 |
| FL_seed100 | 38.6 ± 4.0 | 70.30 ± 0.75 | 4.4 ± 0.5 | 63.90 ± 0.83 |

### joint_geom_simple = sqrt(cat_f1 * reg_top10_indist) (single deployable checkpoint)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| FL_seed42_STALE | 10.2 ± 1.6 | 67.38 ± 1.06 | 72.88 ± 1.49 |
| FL_seed42_FRESH | 9.0 ± 1.2 | 66.98 ± 0.80 | 61.14 ± 0.95 |
| FL_seed0 | 8.6 ± 1.3 | 66.93 ± 0.99 | 61.52 ± 1.22 |
| FL_seed1 | 9.4 ± 0.9 | 66.90 ± 1.04 | 61.34 ± 1.34 |
| FL_seed7 | 8.8 ± 1.9 | 66.58 ± 1.07 | 61.67 ± 0.87 |
| FL_seed100 | 8.8 ± 2.2 | 66.65 ± 1.78 | 61.64 ± 1.53 |

### joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) (current production selector)

| Variant | selected ep | cat F1 | reg top10 |
|---|---:|---:|---:|
| FL_seed42_STALE | 28.8 ± 9.3 | 69.78 ± 1.19 | 61.47 ± 11.48 |
| FL_seed42_FRESH | 27.0 ± 10.1 | 69.68 ± 1.08 | 53.73 ± 9.22 |
| FL_seed0 | 25.0 ± 9.9 | 69.75 ± 0.83 | 51.97 ± 9.70 |
| FL_seed1 | 14.0 ± 1.6 | 68.88 ± 1.08 | 58.81 ± 0.80 |
| FL_seed7 | 18.0 ± 6.0 | 69.41 ± 1.01 | 54.38 ± 9.69 |
| FL_seed100 | 16.6 ± 4.3 | 69.30 ± 0.42 | 58.51 ± 1.08 |

### Reference: shipping FL canonical §0.1 (multi-seed n=20)
- cat F1 = 68.56 ± 0.79
- reg top10 = 63.27 ± 0.10
