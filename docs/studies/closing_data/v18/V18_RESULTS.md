# v18 — RESULTS

> Generated 2026-08-07T20:31:52.834008+00:00 from [`data/v18_results.json`](data/v18_results.json) by `make_results.py`. Every number here is traceable to that JSON, which [`score_all.py`](score_all.py) regenerates from the rundirs.

> Commit `da179081` · seeds run: see the n column of each table.

**v18 = the frozen v17 recipe on a leak-free substrate**: the consecutive-visit graph is forward-only in training and at readout, plus 4 elapsed-time node columns (`in_channels` 15). Not an architecture change. See [`METHODOLOGY.md`](METHODOLOGY.md).

## Conventions

- **diag-best** (`db_*`) — per-task diagnostic-best epochs. The Table-3 convention.
- **joint-best** (`jb_*`) — both heads at the single `geom_simple`-selected epoch, `min_best_epoch` 0. What the served checkpoint delivers.
- cat = macro-F1. reg = `top10_acc_indist · (1 − ood_fraction) · 100`, i.e. **Acc@10**.
- "beats" = paired one-sided superiority test. "matches" = TOST non-inferiority within ±2 pp. A non-inferior result is never upgraded to a win.

## 1 · MTL vs its own dedicated ceiling (same substrate, same protocol)

This is the citable contrast: both arms measured on v18, so the comparison is within-protocol.

| state | n | dedicated cat | MTL cat | **Δcat** | verdict | dedicated reg | MTL reg | **Δreg** | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| istanbul | 10 | 32.82 | 32.65 | **-0.17** | matches (TOST +/-2 pp, p=0.000) | 75.17 | 75.40 | **+0.23** | **beats** (paired one-sided p=0.000) |
| alabama | 10 | 28.29 | 27.25 | **-1.04** | matches (TOST +/-2 pp, p=0.010) | 70.09 | 69.77 | **-0.33** | matches (TOST +/-2 pp, p=0.000) |
| arizona | 10 | 31.91 | 31.58 | **-0.33** | matches (TOST +/-2 pp, p=0.002) | 59.49 | 59.49 | **-0.00** | matches (TOST +/-2 pp, p=0.000) |
| florida | 5 | 35.97 | 35.88 | **-0.09** | matches (TOST +/-2 pp, p=0.000) | 76.70 | 77.26 | **+0.56** | **beats** (paired one-sided p=0.000) |
| texas | 5 | 34.12 | 35.08 | **+0.96** | **beats** (paired one-sided p=0.015) | 64.95 | 67.00 | **+2.05** | **beats** (paired one-sided p=0.000) |
| california | 5 | 33.50 | 33.24 | **-0.26** | matches (TOST +/-2 pp, p=0.000) | 63.45 | 65.57 | **+2.12** | **beats** (paired one-sided p=0.000) |

## 2 · Joint model — both epoch-selection conventions

| state | n | cat diag-best | cat joint-best | reg diag-best | reg joint-best |
|---|---:|---:|---:|---:|---:|
| istanbul | 10 | 32.65 ± 0.07 | — | 75.40 ± 0.05 | — |
| alabama | 10 | 27.25 ± 0.19 | — | 69.77 ± 0.12 | — |
| arizona | 10 | 31.58 ± 0.15 | — | 59.49 ± 0.07 | — |
| florida | 5 | 35.88 ± 0.00 | — | 77.26 ± 0.00 | — |
| texas | 5 | 35.08 ± 0.00 | — | 67.00 ± 0.00 | — |
| california | 5 | 33.24 ± 0.00 | — | 65.57 ± 0.00 | — |

## 3 · Against the v17 published board — CROSS-SUBSTRATE, descriptive only

> ⚠ v17 and v18 are **different substrates**. These differences are reported to show the size of the leak's contribution; they are **not** superiority tests and must not be written as one.

| state | Δcat v18 | Δcat v17 | shift | Δreg v18 | Δreg v17 | shift |
|---|---:|---:|---:|---:|---:|---:|
| istanbul | -0.17 | +8.59 | **-8.76** | +0.23 | +0.28 | -0.05 |
| alabama | -1.04 | +7.72 | **-8.76** | -0.33 | -0.31 | -0.02 |
| arizona | -0.33 | +9.40 | **-9.73** | -0.00 | +0.10 | -0.10 |
| florida | -0.09 | +5.34 | **-5.43** | +0.56 | +0.72 | -0.16 |
| texas | +0.96 | +7.45 | **-6.49** | +2.05 | +2.11 | -0.06 |
| california | -0.26 | +6.45 | **-6.71** | +2.12 | +2.20 | -0.08 |

## 4 · Pooled across states

- **Δcat** pooled over 45 (state, seed, fold) pairs: mean **-0.275** — matches (TOST +/-2 pp, p=0.000)
- **Δreg** pooled over 45 pairs: mean **+0.503** — **beats** (paired one-sided p=0.000)

Pooling across states is reported for a single headline figure; the per-state rows in §1 are the primary result, since the states differ in size and in their v17 deltas.

## 5 · Related findings

- [`LOSS_WEIGHT_PROBE.md`](LOSS_WEIGHT_PROBE.md) — the 0.75/0.25 split is **not** a leak artifact and the heads are **orthogonal, not competing** (cos ≈ +0.001). Rebalancing and gradient surgery are both null; the equal split is significantly *harmful* to region at Florida. Keep 0.75/0.25.

- [`READOUT_EQUIVALENCE.md`](READOUT_EQUIVALENCE.md) — the per-window readout is an identity on a forward-only graph; the engine is materialized from the one-shot export.

- [`AUDIT.md`](AUDIT.md) — the §6 self-checks with measured values.

