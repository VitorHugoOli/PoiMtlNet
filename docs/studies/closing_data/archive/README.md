# closing_data/archive — spent docs (provenance only)

> Archived 2026-07-01 during the v17 close-out pass. These are **done / superseded** — kept for provenance, **not
> live state**. The live board is [`../RESULTS_BOARD.md`](../RESULTS_BOARD.md); the remaining-run track is
> [`../v17_completion/`](../v17_completion/README.md); the outcomes log is [`../log.md`](../log.md).

| Folder | What | Why archived |
|---|---|---|
| `plans/` | `AGENT_PROMPT.md`, `PLAN.md`, `M0_P3_PLAN.md`, `RUN_MATRIX_REDUCE.md` | the origin/phased design + pre-launch scaffold + the deadline-grade reduced matrix — the heavy spend is DONE, we're in close-out; each self-declares superseded. (The live recipe/scope reference `RUN_MATRIX.md` stays at top level.) |
| `verdicts/` | `PHASE1_VERDICT.md`, `C1_VERDICT.md`, `FREEZE_READINESS.md`, `BASELINES_IMPL_AUDIT.md` | closed-track gate verdicts (P1a cross-study re-eval, C1 promote→supportive, pre-freeze checklist, baseline impl-audit) — all resolved. |
| `provenance/` | `*_HASH_MANIFEST.json`, `M2PRO_MANIFEST.json`, `V14_REBUILD_H100_PROVENANCE.json`, `SUBSTRATE_VERSION_MAP.md`, `CATEGORY_DISTRIBUTION.md` | one-shot build/provenance manifests + a version snapshot + a factual distribution computation; ground truth is `docs/results/CANONICAL_VERSIONS.md`. |
| `run_logs/` | `catx_v17_runs/`, `bf16_island_runs/`, `istanbul_build/`, `PART1_QUALITY/`, `run_bf16_island.sh`, `monitor_catx_ram.sh` | spent run/build workdirs + helpers (logs only; results graduated into RESULTS_BOARD / the KEEP findings docs). |
| `HANDOFF.md` | the old closing_data index | superseded as the index by `RESULTS_BOARD.md` + `v17_completion/` + `HANDOFF_A40.md`; its own banner declares the board/substrate/baseline phase done. |

**Do not treat anything here as current.** If a number is needed, trace it via `RESULTS_BOARD.md §3`.

**Added 2026-07-08 (post-PR #58):** `HANDOFF_A40.md` (the old root A40 worklist — superseded by `../v17_completion/A40.md`), `CATX_V17_N20_H100_HANDOFF.md` + `run_logs/run_catx_v17_n20_h100.sh` (the H100 lane is gone; the A40 driver `../run_catx_v17_n20.sh` + `../v17_completion/A40.md §A1` supersede).

**Added 2026-07-13 (post-PR #63 — the v17 board is COMPLETE):** `handoffs/` = the three per-machine v17-completion
handoffs `A40.md` / `M2PRO.md` / `H100.md` (A1/M1-full/T6 all DONE; H100 was decommissioned 2026-07-08). The single
live board is now [`../v17_completion/README.md`](../v17_completion/README.md); the only still-running item (A2
ReHDM CA/TX) + the optional coverage/doc chores are tracked in its task table.

**Added 2026-07-13 (essentials-only compaction):** `findings/` = the four closed per-result findings
(`W6_ENCODER_ISOLATION.md`, `CSLSL_CASCADE.md`, `ISTANBUL_BASELINES_RESULTS.md`, `FAITHFUL_STAN_FINDINGS.md` — their
headline reads live in `RESULTS_BOARD §1b/§1c/§4`); the three 1-line lesson stubs were deleted (inbound links now
point straight at `lessons/`); `STATISTICAL_PROTOCOL.md` + `PRECISION_LESSONS.md` moved to `../v17_completion/`;
`run_logs/` gains the spent drivers `run_catx_v17_n20.sh` + `run_catx_v17_audit_1fold.sh`; the sibling study
`closing_data_v2/` was folded in as [`../joint_best/`](../joint_best/README.md) (with `JOINT_BEST_SCORING.md`).
Top level is now README + RESULTS_BOARD + log + RUN_MATRIX + perhead_lr_n20 only.

**Added 2026-07-08 (compaction pass):** `lessons/{CA_MTL_DIVERGENCE,TX_A40_BF16_NAN,EP100_ABLATION_AND_TX_RAM}.md`
(the 3 precision/schedule forensics — merged into the compact `../PRECISION_LESSONS.md`; 1-line breadcrumbs left at
the old paths so inbound links keep resolving), `BOARD_CELLS.md` + `BOARD_H100_FINDINGS.md` (board-production
per-cell/per-session provenance — the board is settled, `../RESULTS_BOARD.md §3` repointed),
`run_logs/run_catx_v17_seed0_5f.sh` (produced the committed seed-0 CA/TX cells; the live n=20 driver is
`../run_catx_v17_n20.sh`), `run_logs/{h2_runs,a40_5_cascade}/` + `run_logs/finalize_all.sh` (spent v17_completion
run dirs/drivers — results graduated into CEILINGS_N20_FINAL / RESULTS_BOARD §1b).
