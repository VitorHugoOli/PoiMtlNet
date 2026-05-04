# FL H3-alt — perf-variant comparison

_Generated from `/workspace/PoiMtlNet/results/check2hgi/florida`_

Reference: F48-H3-alt FL 5f × 50ep, seed 42, cat F1 67.92 ± 0.72, reg Acc@10_indist 71.96 ± 0.68 (from `docs/studies/check2hgi/NORTH_STAR.md`).

Equivalence marker: ✓ ≤ 1σ from NORTH_STAR mean | ~ ≤ 2σ | ✗ > 2σ.

| Variant | Cat F1 (5f) | Δ vs NORTH_STAR | Reg Acc@10 indist (5f) | Δ vs NORTH_STAR | Run dir |
|---|---:|---:|---:|---:|---|
| **baseline** | 67.35 ± 0.83 | -0.57 pp (0.79σ) ✓ | 63.68 ± 12.07 | -8.28 pp (12.17σ) ✗ | `mtlnet_lr1.0e-04_bs1024_ep50_20260428_2149` |
| **tf32** | 67.35 ± 0.83 | -0.57 pp (0.79σ) ✓ | 63.68 ± 12.07 | -8.28 pp (12.17σ) ✗ | `mtlnet_lr1.0e-04_bs1024_ep50_20260428_2226` |
| **compile** | — | — | — | — | _missing_ |
| **bs2048** | — | — | — | — | _missing_ |

## Verdict template

Both perf variants count as a publishable "preview mode" iff Δ ≤ 1σ on BOTH metrics for that variant. If a variant lands in 1–2σ band, it is exploratory only. > 2σ → quality is hurt; do not use.
