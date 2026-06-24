# HANDOFF — closing-data (current index, 2026-06-24)

> The MTL board is **done + merged**; we are in the **baseline phase** for the MobiWac paper. This is the top
> index — go straight to the doc you need.

## Executing a machine? Read your ONE self-contained handoff:
| Machine | File | Job |
|---|---|---|
| **H100** (CUDA) | [`BASELINE_H100.md`](BASELINE_H100.md) | FL representation block (Check2HGI-SC · CTLE-SC · CTLE-E2E · feature-concat) + CSLSL@FL |
| **A40** (CUDA) | [`BASELINE_A40.md`](BASELINE_A40.md) | CSLSL cascade @ AL/AZ (the published MTL alternative) |
| **M4 Pro** (MPS) | [`BASELINE_M4.md`](BASELINE_M4.md) | CTLE frozen-below-floor diagnosis (gates H100) + TX HMT-GRN |

Cross-machine overview / drop-list / done-list → [`BASELINE_DISTRIBUTION.md`](BASELINE_DISTRIBUTION.md).
Locked decisions + the 3 baseline roles → [`../../../articles/[mobiwac]/BASELINE_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/BASELINE_HANDOFF.md).

## Results & protocol (the sources of truth)
- **Board results (the paper's Part-2 numbers):** [`RESULTS_BOARD.md`](RESULTS_BOARD.md) — MTL champion-G vs STL
  ceilings, per state, with provenance + honesty caveats. **Headline:** MTL beats category everywhere (+4.7…+8.1),
  beats region at the large states (FL +0.57, CA +2.18 5f, TX +2.17 2/5f), matches at the small.
- **Protocol:** [`RUN_MATRIX_REDUCE.md`](RUN_MATRIX_REDUCE.md) (reduced 1-seed board) + [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md) (Wilcoxon/TOST).
- **Baseline tables (paper read-from):** [`../../baselines/`](../../baselines/) (`next_category/` + `next_region/`).
- **The fp16 root cause (why MTL was re-run):** [`CA_MTL_DIVERGENCE.md`](CA_MTL_DIVERGENCE.md).

## Standing traps (verify, don't trust — burned this project before)
Stale log_T (per-fold seed-tagged, fresher than `next_region.parquet`); dev-seed 42 (report {0,1,7,100}, deadline
uses seed 0); matched metric+seeds+folds+fp32 on BOTH sides of every Δ; commit with explicit pathspec (the repo
pre-stages `articles/*` WIP); pin `--canon`. Full list: CLAUDE.md + `docs/CONCERNS.md` C25–C28.

## Per-cell records (history, not execution)
`CA_CELL.md` · `TX_CELL.md` · `TX_A40_BF16_NAN.md` · `AL_PRECISION_GATE.md` · `AZ_CELL.md` · `FL_PRECISION_GATE.md`
· `EP100_ABLATION_AND_TX_RAM.md` · `BOARD_H100_FINDINGS.md` · `MACS_BOARD_RESULTS.md` (baselines + engineering
knowledge) · `BASELINES_IMPL_AUDIT.md` · `SUBSTRATE_VERSION_MAP.md`.

> The original full-board (n=20, all-seeds) study design lives in `PLAN.md` / `AGENT_PROMPT.md` / `M0_P3_PLAN.md`
> — that is the **post-deadline** target; the deadline runs the reduced board above.
