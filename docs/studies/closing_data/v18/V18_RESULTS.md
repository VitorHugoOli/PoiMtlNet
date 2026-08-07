# v18 — RESULTS

> Generated 2026-08-06T23:04:46.730403+00:00 from [`data/v18_results.json`](data/v18_results.json) by `make_results.py`. Every number here is traceable to that JSON, which [`score_all.py`](score_all.py) regenerates from the rundirs.

> Commit `9240da4f` · seeds run: see the n column of each table.

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
| istanbul | 5 | 32.77 | 32.70 | **-0.07** | matches (TOST +/-2 pp, p=0.001) | 75.16 | 75.36 | **+0.21** | **beats** (paired one-sided p=0.000) |
| alabama | 5 | 28.03 | 27.38 | **-0.65** | matches (TOST +/-2 pp, p=0.021) | 70.00 | 69.68 | **-0.31** | matches (TOST +/-2 pp, p=0.000) |
| arizona | 5 | 31.35 | 31.69 | **+0.34** | matches (TOST +/-2 pp, p=0.013) | 59.48 | 59.54 | **+0.06** | matches (TOST +/-2 pp, p=0.000) |
| florida | 5 | 35.97 | 35.88 | **-0.09** | matches (TOST +/-2 pp, p=0.000) | 76.70 | 77.26 | **+0.56** | **beats** (paired one-sided p=0.000) |

## 2 · Joint model — both epoch-selection conventions

| state | n | cat diag-best | cat joint-best | reg diag-best | reg joint-best |
|---|---:|---:|---:|---:|---:|
| istanbul | 5 | 32.70 ± 0.00 | — | 75.36 ± 0.00 | — |
| alabama | 5 | 27.38 ± 0.00 | — | 69.68 ± 0.00 | — |
| arizona | 5 | 31.69 ± 0.00 | — | 59.54 ± 0.00 | — |
| florida | 5 | 35.88 ± 0.00 | — | 77.26 ± 0.00 | — |

## 3 · Against the v17 published board — CROSS-SUBSTRATE, descriptive only

> ⚠ v17 and v18 are **different substrates**. These differences are reported to show the size of the leak's contribution; they are **not** superiority tests and must not be written as one.

| state | Δcat v18 | Δcat v17 | shift | Δreg v18 | Δreg v17 | shift |
|---|---:|---:|---:|---:|---:|---:|
| istanbul | -0.07 | +8.59 | **-8.66** | +0.21 | +0.28 | -0.07 |
| alabama | -0.65 | +7.72 | **-8.37** | -0.31 | -0.31 | -0.00 |
| arizona | +0.34 | +9.40 | **-9.06** | +0.06 | +0.10 | -0.04 |
| florida | -0.09 | +5.34 | **-5.43** | +0.56 | +0.72 | -0.16 |

## 4 · Pooled across states

- **Δcat** pooled over 20 (state, seed, fold) pairs: mean **-0.119** — matches (TOST +/-2 pp, p=0.000)
- **Δreg** pooled over 20 pairs: mean **+0.127** — matches (TOST +/-2 pp, p=0.000)

Pooling across states is reported for a single headline figure; the per-state rows in §1 are the primary result, since the states differ in size and in their v17 deltas.

## 5 · Related findings

- [`LOSS_WEIGHT_PROBE.md`](LOSS_WEIGHT_PROBE.md) — the 0.75/0.25 split is **not** a leak artifact and the heads are **orthogonal, not competing** (cos ≈ +0.001). Rebalancing and gradient surgery are both null; the equal split is significantly *harmful* to region at Florida. Keep 0.75/0.25.

- [`READOUT_EQUIVALENCE.md`](READOUT_EQUIVALENCE.md) — the per-window readout is an identity on a forward-only graph; the engine is materialized from the one-shot export.

- [`AUDIT.md`](AUDIT.md) — the §6 self-checks with measured values.

