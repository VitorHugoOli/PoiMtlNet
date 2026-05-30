# Tier A1 large-state pilot — analysis tables

**Date**: 2026-05-28  ·  **Seed**: 42 (development seed) — sign-and-magnitude pilot, NOT paper-grade.

## Florida (analysed folds: 5)

| W | disjoint reg (top10_acc_indist) | geom_simple reg | disjoint cat F1 | geom cat F1 |
|---|---:|---:|---:|---:|
| W0.0 | 63.98 ± 0.76 (n=5) | 61.14 ± 0.95 | 70.49 ± 0.92 | 66.98 ± 0.80 |
| W0.2 | 66.38 ± 0.58 (n=5) | 65.20 ± 0.74 | 70.50 ± 0.92 | 67.15 ± 1.14 |

**Per-fold Δ disjoint reg (W=0.2 − W=0.0):**

| fold | W=0.0 | W=0.2 | Δ pp |
|---:|---:|---:|---:|
| 1 | 64.30 | 66.64 | +2.34 |
| 2 | 62.74 | 65.37 | +2.63 |
| 3 | 64.76 | 66.84 | +2.09 |
| 4 | 64.19 | 66.47 | +2.28 |
| 5 | 63.90 | 66.55 | +2.65 |

**Wilcoxon (one-sided, W=0.2>W=0.0, n=5, raw per-fold):** mean Δ = +2.40 pp, median Δ = +2.34 pp, 5/5 folds positive, p = 0.03125.

## California (analysed folds: 1)

| W | disjoint reg (top10_acc_indist) | geom_simple reg | disjoint cat F1 | geom cat F1 |
|---|---:|---:|---:|---:|
| W0.0 | 50.06 (fold1, n=1) | 50.06 | 64.66 | 56.85 |
| W0.2 | 51.48 (fold1, n=1) | 51.48 | 64.55 | 56.64 |

**Per-fold Δ disjoint reg (W=0.2 − W=0.0):**

| fold | W=0.0 | W=0.2 | Δ pp |
|---:|---:|---:|---:|
| 1 | 50.06 | 51.48 | +1.42 |

**n=1 fold — sign-and-magnitude only (no significance test):** Δ disjoint reg = +1.42 pp.

## Texas (analysed folds: 1)

| W | disjoint reg (top10_acc_indist) | geom_simple reg | disjoint cat F1 | geom cat F1 |
|---|---:|---:|---:|---:|
| W0.0 | 50.38 (fold1, n=1) | 45.38 | 64.96 | 62.04 |
| W0.2 | 52.09 (fold1, n=1) | 48.05 | 65.00 | 63.75 |

**Per-fold Δ disjoint reg (W=0.2 − W=0.0):**

| fold | W=0.0 | W=0.2 | Δ pp |
|---:|---:|---:|---:|
| 1 | 50.38 | 52.09 | +1.71 |

**n=1 fold — sign-and-magnitude only (no significance test):** Δ disjoint reg = +1.71 pp.

