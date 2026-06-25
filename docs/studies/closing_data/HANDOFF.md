# HANDOFF — closing-data (current index, 2026-06-25)

> The MTL board AND the baseline phase are **done + merged**. The reduced-board per-machine handoffs
> (BASELINE_H100/A40/M4/DISTRIBUTION + the W6 probe run-spec) were one-shot and have been **removed as spent** —
> their results live in the records below. **The only continuing GPU work is Blocker 2** (HGI-category-STL under
> overlap) → [`HANDOFF_A40.md`](HANDOFF_A40.md).

## Active work
- **A40 — Blocker 2** (the last item needing new runs): [`HANDOFF_A40.md`](HANDOFF_A40.md) (HGI-cat-STL under
  overlap; needs the canonical CA/TX HGI build first). Close-gate + status: [`../../../articles/[mobiwac]/CLOSE_BLOCKERS_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/CLOSE_BLOCKERS_HANDOFF.md).

## Results & protocol (the sources of truth)
- **Board results (Part-2):** [`RESULTS_BOARD.md`](RESULTS_BOARD.md) — MTL beats category everywhere (+4.7…+8.1);
  beats region at the large states (FL +0.57, CA +2.18, TX +2.06 — all 5f), matches at the small. §1b CSLSL
  cascade = dead tie (AL/AZ/FL). §1c W6 = trunk, not transfer.
- **Protocol:** [`RUN_MATRIX_REDUCE.md`](RUN_MATRIX_REDUCE.md) + [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md) (Wilcoxon/TOST + the n=5 power statement).
- **Baseline tables (paper read-from):** [`../../baselines/`](../../baselines/).

## Records (history, not execution)
Board cells: `CA_CELL` · `TX_CELL` · `TX_A40_BF16_NAN` · `AL_PRECISION_GATE` · `AZ_CELL` · `FL_PRECISION_GATE`
· `EP100_ABLATION_AND_TX_RAM` · `BOARD_H100_FINDINGS`. Findings: `CA_MTL_DIVERGENCE` (fp16 root cause) ·
`CSLSL_CASCADE` · `W6_ENCODER_ISOLATION` · `MACS_BOARD_RESULTS` (baselines + engineering) · `BASELINES_IMPL_AUDIT`
· `SUBSTRATE_VERSION_MAP`. Decisions trail: `log.md`.

## Standing traps (verify, don't trust)
Stale log_T (seed-tagged, fresher than `next_region.parquet`); fp32 on large-state Ampere (bf16 grad-NaN);
matched metric+seeds+folds+precision on BOTH sides of every Δ; commit with explicit pathspec; pin `--canon`
(or `--canon none` on dk_ovl). Full list: CLAUDE.md + `docs/CONCERNS.md` C25–C28.

> The original full-board (n=20) study design is in `PLAN.md` / `AGENT_PROMPT.md` / `M0_P3_PLAN.md` (the
> **post-deadline** target). Phase verdicts: `C1_VERDICT` / `FREEZE_READINESS` / `PHASE1_VERDICT` (historical).
