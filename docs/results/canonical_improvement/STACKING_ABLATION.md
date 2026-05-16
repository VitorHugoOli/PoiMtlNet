# STACKING ABLATION — canonical → +v3c → +T3.2 ResLN
**Date**: 2026-05-16 (Tier-4 paper closeout)
**Source**: per-seed JSONs under `docs/results/canonical_improvement/`. n=5 fold-pairs × 5 seeds = 25 paired observations per cell. Canonical reference from `RESULTS_TABLE.md §0.1` (n=20 fold-pairs over 4 seeds, seed=42 single).

## §1 — FL multi-seed (5 seeds: 42, 0, 1, 7, 100)

| Stage | cat F1 (mean ± σ_seed) | reg Acc@10 | leak F1 | Δcat vs canonical | Δreg vs canonical | Δleak vs canonical |
|---|---:|---:|---:|---:|---:|---:|
| canonical (T1.1 / §0.1) | 68.56 ± 0.79 | 63.27 ± 0.10 | 40.85 ± 0.39 | — | — | — |
| + v3c (AdamW WD=5e-2) | 68.23 ± 0.18 | 63.90 ± 0.05 | 41.19 ± 0.25 | -0.33 | **+0.63** | +0.34 |
| + v3c + T3.2 ResLN (SHIPPING) | 69.42 ± 0.25 | 63.98 ± 0.09 | 43.09 ± 0.12 | **+0.86** | +0.71 | +2.24 |
| **T3.2 marginal contribution** | **+1.19** (cat axis) | +0.08 (reg null) | +1.90 | | | |

## §2 — AL multi-seed (5 seeds: 42, 0, 1, 7, 100)

| Stage | cat F1 | reg Acc@10 | leak F1 |
|---|---:|---:|---:|
| canonical | 40.57 ± 0.24 | 50.17 ± 0.24 | 31.04 ± 1.33 |
| + v3c + T3.2 ResLN | 42.05 ± 0.29 | 49.88 ± 0.55 | 32.73 ± 0.49 |
| Δ vs canonical | **+1.48** | -0.29 | +1.69 |

## §3 — AZ multi-seed (5 seeds: 42, 0, 1, 7, 100)

| Stage | cat F1 | reg Acc@10 | leak F1 |
|---|---:|---:|---:|
| canonical | 45.10 ± 0.19 | 40.78 ± 0.07 | 34.57 ± 0.78 |
| + v3c + T3.2 ResLN | 46.80 ± 0.39 | 40.63 ± 0.40 | 37.03 ± 0.34 |
| Δ vs canonical | **+1.70** | -0.15 | +2.46 |

## §4 — Paired sign test (5/5 directional per state)

| state | axis | signs (T3.2 stack > canonical per seed) | one-sided p |
|---|---|---|---:|
| FL | cat | 5/5 | 0.03125 |
| FL | reg | 5/5 | 0.03125 |
| AL | cat | 5/5 | 0.03125 |
| AZ | cat | 5/5 | 0.03125 |

## §5 — Shipping stack final claim

**canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResidualLN encoder**:
- **FL**: cat **+0.86 pp** (5/5 seeds positive, sign-test p=0.03125); reg **+0.71 pp** (5/5 seeds positive, p=0.03125). Leak Δ +2.24 pp, well below +5 pp red flag.
- **AL**: cat **+1.48 pp** (5/5 seeds positive); reg Δ -0.29 (null at small states, matching v3c precedent).
- **AZ**: cat **+1.70 pp** (5/5 seeds positive); reg Δ -0.15 (null).

Two paper-grade micro-improvements on disjoint axes (v3c→reg, T3.2→cat), additive by construction, leak-budgeted at 44% of the red-flag allowance.
