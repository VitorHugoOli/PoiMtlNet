# HANDOFF — closing-data (current index, 2026-06-26)

> The MTL board, the substrate (Tbl 2), and the baseline phase are **done + merged** (incl. PRs #50/#51/#52/#53).
> The reduced-board per-machine handoffs + the spent Istanbul baseline handoff were one-shot and **removed as spent**
> — their results live in the records below. **Remaining GPU work is small and secondary** (region externals +
> one corroboration cell) → [`HANDOFF_A40.md`](HANDOFF_A40.md).

## Active work (the few remaining MAIN-DATA cells)
- **A40 — Blocker 4** [`HANDOFF_A40.md`](HANDOFF_A40.md): **finish FL faithful-STAN** (in-flight, fold-0 v6 ckpt
  Acc@10 0.7307; fills the Table-3 FL STAN cell), then **ReHDM-faithful** (AL/AZ/FL exist; Istanbul via the
  FSQ→mahalle adapter or footnote; CA/TX footnote infeasible). STAN-`stl_hgi` is reclassified a future-headroom
  signal, NOT a baseline.
- **H100 — Blocker 1 (corroboration, not blocking):** FL CTLE-SC 5f (currently 2/5; W3 already closed at AL/AZ/Istanbul).
- Close-gate + paper status: [`../../../articles/[mobiwac]/CLOSE_BLOCKERS_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/CLOSE_BLOCKERS_HANDOFF.md)
  · region-externals brief [`../../../articles/[mobiwac]/STAN_REFOOTING_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/STAN_REFOOTING_HANDOFF.md).

## Results & protocol (the sources of truth)
- **Board results (Part-2):** [`RESULTS_BOARD.md`](RESULTS_BOARD.md) — MTL beats category everywhere (+4.7…+7.7);
  beats region at the large states (FL +0.57, CA +2.18, TX +2.06 — all 5f), matches at the small. §1b CSLSL
  cascade = dead tie (AL/AZ/FL). §1c W6 = trunk, not transfer. §4 baselines (HMT-GRN-style, faithful STAN, CTLE, …).
- **Per-cell board detail:** [`BOARD_CELLS.md`](BOARD_CELLS.md) (per-state MTL cells + AL/FL precision gates, consolidated).
- **Protocol:** [`RUN_MATRIX_REDUCE.md`](RUN_MATRIX_REDUCE.md) + [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md) (Wilcoxon/TOST + the n=5 power statement).
- **Baseline tables (paper read-from):** [`../../baselines/`](../../baselines/).

## Records (history, not execution)
Board cells (consolidated): [`BOARD_CELLS.md`](BOARD_CELLS.md). Mechanics: `EP100_ABLATION_AND_TX_RAM` ·
`BOARD_H100_FINDINGS`. Findings: `CA_MTL_DIVERGENCE` (fp16 root cause) · `TX_A40_BF16_NAN` (Ampere bf16 grad-NaN) ·
`CSLSL_CASCADE` · `W6_ENCODER_ISOLATION` · `FAITHFUL_STAN_FINDINGS` (PR #53) · `ISTANBUL_BASELINES_RESULTS` ·
`MACS_BOARD_RESULTS` (baselines + engineering) · `BASELINES_IMPL_AUDIT` · `SUBSTRATE_VERSION_MAP`. Decisions trail: `log.md`.

## Standing traps (verify, don't trust)
Stale log_T (seed-tagged, fresher than `next_region.parquet`); fp32 on large-state Ampere (bf16 grad-NaN);
matched metric+seeds+folds+precision on BOTH sides of every Δ; commit with explicit pathspec; pin `--canon`
(or `--canon none` on dk_ovl). Full list: CLAUDE.md + `docs/CONCERNS.md` C25–C28.

> The original full-board (n=20) study design is in `PLAN.md` (current-status banner at top) / `AGENT_PROMPT.md` /
> `M0_P3_PLAN.md` (the **post-deadline** target). Phase verdicts: `C1_VERDICT` / `FREEZE_READINESS` / `PHASE1_VERDICT` (historical).
