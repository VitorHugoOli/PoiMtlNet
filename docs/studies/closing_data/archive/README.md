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
