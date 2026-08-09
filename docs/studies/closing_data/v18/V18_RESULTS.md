# v18 — RESULTS

> Generated 2026-08-09T06:30:14.493296+00:00 from [`data/v18_results.json`](data/v18_results.json) by `make_results.py`. Every number here is traceable to that JSON, which [`score_all.py`](score_all.py) regenerates from the rundirs.

> Commit `e351d4b0` · seeds run: see the n column of each table.

**v18 = the frozen v17 recipe on a leak-free substrate**: the consecutive-visit graph is forward-only in training and at readout, plus 4 elapsed-time node columns (`in_channels` 15). Not an architecture change. See [`METHODOLOGY.md`](METHODOLOGY.md).

## Conventions

- **diag-best** (`db_*`) — per-task diagnostic-best epochs. The Table-3 convention.
- **joint-best** (`jb_*`) — both heads at the single `geom_simple`-selected epoch, `min_best_epoch` 0. What the served checkpoint delivers.
- cat = macro-F1. reg = `top10_acc_indist · (1 − ood_fraction) · 100`, i.e. **Acc@10**.
- "beats" = paired one-sided superiority test. "matches" = TOST non-inferiority within ±2 pp. A non-inferior result is never upgraded to a win.


> ⚠ **Two disclosures that qualify every p-value below.**
>
> **1. The tests pool (seed, fold) pairs as if independent.** `paired_diffs` concatenates the 5 per-fold differences from each seed, so n = seeds × 5. The 5 folds within one seed share ~80 % of their training data, so these are **not** independent replicates and the reported p-values are anti-conservative. At n=10 (2 seeds) the honest independent unit is the **seed**, giving n=2 — underpowered. Read the p-values as descriptive, and do not promote a marginal result on their strength alone.
>
> **2. The dedicated-cat comparator was tuned on this same CV.** Its per-state `max_lr` and the joint `category_weight` were selected at seed 0 on the same 5-fold splits reported here. The dedicated arm got a 4-point per-state LR search while the MTL cat-lr axis was measured null, so the dedicated ceiling is optimistically biased relative to MTL — which makes Δcat a **conservative** estimate of any MTL advantage, and an optimistic one of any MTL deficit. See METHODOLOGY.md.

## 1 · MTL vs its own dedicated ceiling (same substrate, same protocol)

This is the citable contrast: both arms measured on v18, so the comparison is within-protocol.

| state | n | dedicated cat | MTL cat | **Δcat** | verdict | dedicated reg | MTL reg | **Δreg** | verdict |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| istanbul | 0 | 35.35 | — | — | — | 75.17 | — | — | — |
| alabama | 0 | 30.77 | — | — | — | 70.09 | — | — | — |
| arizona | 0 | — | — | — | — | 59.49 | — | — | — |
| florida | 0 | — | — | — | — | 76.70 | — | — | — |
| texas | 0 | — | — | — | — | 64.95 | — | — | — |
| california | 0 | — | — | — | — | 63.45 | — | — | — |

## 2 · Joint model — both epoch-selection conventions

| state | n | cat diag-best | cat joint-best | reg diag-best | reg joint-best |
|---|---:|---:|---:|---:|---:|
| istanbul | 0 | — | — | — | — |
| alabama | 0 | — | — | — | — |
| arizona | 0 | — | — | — | — |
| florida | 0 | — | — | — | — |
| texas | 0 | — | — | — | — |
| california | 0 | — | — | — | — |

## 3 · Against the v17 published board — CROSS-SUBSTRATE, descriptive only

> ⚠ v17 and v18 are **different substrates**. These differences are reported to show the size of the leak's contribution; they are **not** superiority tests and must not be written as one.

| state | Δcat v18 | Δcat v17 | shift | Δreg v18 | Δreg v17 | shift |
|---|---:|---:|---:|---:|---:|---:|

## 5 · Related findings

- [`LOSS_WEIGHT_PROBE.md`](LOSS_WEIGHT_PROBE.md) — the 0.75/0.25 split is **not** a leak artifact and the heads are **orthogonal, not competing** (cos ≈ +0.001). Rebalancing and gradient surgery are both null; the equal split is significantly *harmful* to region at Florida. Keep 0.75/0.25.

- [`READOUT_EQUIVALENCE.md`](READOUT_EQUIVALENCE.md) — the per-window readout is an identity on a forward-only graph; the engine is materialized from the one-shot export.

- [`AUDIT.md`](AUDIT.md) — the §6 self-checks with measured values.

