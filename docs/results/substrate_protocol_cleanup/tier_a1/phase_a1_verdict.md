# Tier A1 — log_T-KD multi-seed promotion sweep summary

**Date**: 2026-05-28
**Scope**: ['alabama', 'arizona'] × seeds [0, 1, 7, 100] × W ∈ ['W0.0', 'W0.2'] × 5 folds → n=20 per cell per state

## Three-frontier table per state

### Alabama

| W | disjoint reg (top10_acc) | geom_simple reg | disjoint cat F1 | geom cat F1 |
|---|---:|---:|---:|---:|
| W0.0 | 50.59 ± 3.53 (n=20) | 48.00 ± 3.33 | 45.96 ± 1.96 | 45.29 ± 2.28 |
| W0.2 | 52.85 ± 3.48 (n=20) | 51.21 ± 3.15 | 45.76 ± 2.16 | 45.16 ± 2.27 |

### Arizona

| W | disjoint reg (top10_acc) | geom_simple reg | disjoint cat F1 | geom cat F1 |
|---|---:|---:|---:|---:|
| W0.0 | 41.30 ± 2.60 (n=20) | 38.79 ± 1.99 | 48.86 ± 1.64 | 47.81 ± 1.90 |
| W0.2 | 46.22 ± 2.75 (n=20) | 44.05 ± 2.33 | 48.94 ± 1.58 | 46.34 ± 1.95 |

## Wilcoxon (one-sided, paired by seed×fold, W=0.2 > W=0.0) on disjoint reg top10_acc

| State | n | mean Δ pp | median Δ pp | folds positive | p-value | verdict @ α=0.05 |
|---|---:|---:|---:|---:|---:|---|
| alabama | 20 | +2.27 | +2.15 | 20/20 | 9.537e-07 | PROMOTE |
| arizona | 20 | +4.91 | +4.95 | 20/20 | 9.537e-07 | PROMOTE |

> **Reproducibility note (scipy ≥ 1.16 dispatch):** with raw per-fold precision (no ties), `scipy.stats.wilcoxon` auto-selects `method='exact'` → p ≈ 9.537e-07 (the 2⁻²⁰ asymptotic minimum for n=20 / 20-positive). When re-running on the 2-decimal-place tabular Δ values, ties force `method='approx'` → p ≈ 4.42e-05. Same data, different p by ~2 orders of magnitude. Always re-run on the raw CSV values, never on rounded tabular Δs.

## Per-fold deltas (W=0.2 − W=0.0) on disjoint reg top10_acc

### Alabama

| seed | fold | W=0.0 | W=0.2 | Δ pp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 53.55 | 56.34 | +2.79 |
| 0 | 2 | 49.09 | 52.17 | +3.08 |
| 0 | 3 | 52.11 | 54.15 | +2.04 |
| 0 | 4 | 52.30 | 54.37 | +2.07 |
| 0 | 5 | 43.29 | 45.51 | +2.22 |
| 1 | 1 | 54.05 | 55.71 | +1.66 |
| 1 | 2 | 51.47 | 54.50 | +3.03 |
| 1 | 3 | 53.01 | 54.59 | +1.58 |
| 1 | 4 | 50.10 | 51.68 | +1.58 |
| 1 | 5 | 44.53 | 47.13 | +2.60 |
| 7 | 1 | 53.81 | 56.68 | +2.86 |
| 7 | 2 | 49.55 | 51.47 | +1.92 |
| 7 | 3 | 52.12 | 54.24 | +2.12 |
| 7 | 4 | 54.28 | 57.06 | +2.78 |
| 7 | 5 | 45.56 | 46.89 | +1.33 |
| 100 | 1 | 52.90 | 55.48 | +2.58 |
| 100 | 2 | 50.74 | 52.92 | +2.17 |
| 100 | 3 | 53.15 | 54.86 | +1.71 |
| 100 | 4 | 52.16 | 53.75 | +1.59 |
| 100 | 5 | 43.94 | 47.60 | +3.66 |

### Arizona

| seed | fold | W=0.0 | W=0.2 | Δ pp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 40.02 | 45.22 | +5.21 |
| 0 | 2 | 40.96 | 46.14 | +5.18 |
| 0 | 3 | 45.28 | 50.80 | +5.52 |
| 0 | 4 | 38.04 | 41.97 | +3.94 |
| 0 | 5 | 42.21 | 47.52 | +5.31 |
| 1 | 1 | 41.65 | 46.08 | +4.43 |
| 1 | 2 | 40.77 | 45.52 | +4.75 |
| 1 | 3 | 44.00 | 49.16 | +5.16 |
| 1 | 4 | 37.35 | 42.06 | +4.72 |
| 1 | 5 | 42.64 | 48.09 | +5.45 |
| 7 | 1 | 41.59 | 46.33 | +4.74 |
| 7 | 2 | 40.64 | 45.78 | +5.14 |
| 7 | 3 | 45.06 | 49.82 | +4.76 |
| 7 | 4 | 36.34 | 40.93 | +4.59 |
| 7 | 5 | 43.33 | 48.63 | +5.30 |
| 100 | 1 | 40.28 | 44.54 | +4.26 |
| 100 | 2 | 41.41 | 46.72 | +5.31 |
| 100 | 3 | 44.89 | 49.53 | +4.65 |
| 100 | 4 | 37.30 | 42.87 | +5.58 |
| 100 | 5 | 42.33 | 46.63 | +4.30 |
