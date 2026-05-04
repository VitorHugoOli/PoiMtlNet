# Archive — Post Paper-Closure 2026-05-01

This subdirectory holds study-side docs that were either (a) **superseded** by the paper-closure canonical sources, (b) **workflow-time** artefacts (handoffs, prompts, trackers) whose work has landed, or (c) **chronological audit** records (session handoffs) preserved for traceability but no longer current.

**They are kept for audit, not for navigation.** New readers should start at `../../README.md` and `../../CHANGELOG.md`.

## What's here, why it's here

### Superseded by RESULTS_TABLE.md §0 (v7/v8) as canonical numerical source

| File | Why archived |
|---|---|
| `PAPER_DRAFT.md` | The study-side paper draft (committed title + 130-word abstract). Now lives in `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md`. The committed title and several numbers in this file were superseded by the post-Codex audit reframe and the v8 cat-Δ Wilcoxon. |
| `PAPER_STRUCTURE.md` | The study-side paper-structure doc (table layout, baselines, scope). Now lives in `articles/[BRACIS]_Beyond_Cross_Task/PAPER_STRUCTURE.md`. |
| `PAPER_CLOSURE_RESULTS_2026-05-01.md` | Background provenance — records the lab-trail of how paper-closure numbers were computed. Some numbers there were superseded by RESULTS_TABLE v7/v8 (e.g., FL Δ_reg simple mean-diff −7.28 vs paired Δ −7.99; AL/AZ STL cat means refreshed to multi-seed). Useful as audit; **do not cite as paper canon**. |
| `OBJECTIVES_STATUS_TABLE.md` | Per-objective scorecard (v5). Superseded by `../../CHANGELOG.md` + `../../results/RESULTS_TABLE.md §0` as the current state-of-evidence sources. |

### Workflow / orchestration artefacts (work landed)

| File | Why archived |
|---|---|
| `PAPER_CLOSURE_PHASES.md` | Phase plan for the paper-closure 28-run matrix. Work landed (commit `03af55c`). |
| `PAPER_PREP_TRACKER.md` | Paper-deliverable tracker. Superseded by `../../CHANGELOG.md` + the article-side files. |
| `F50_NORTH_STAR_DEEP_EXPLORATION_PROMPT.md` | Prompt artefact for the F50 audit. Audit landed. |
| `GAP_A_AUDIT_SNAPSHOT_20260430.md` | Snapshot of Gap A (faithful baselines) state on 2026-04-30. Closed via `GAP_A_CLOSURE_20260430.md` (also archived). |
| `GAP_A_CLOSURE_20260430.md` | Closure note for Gap A. Work landed. |
| `GAP_A_GEORGIA_EXTENSION.md` | Extension proposal for Gap A. Scoped out of paper. |
| `GAP_A_RUNPOD_HANDOFF_PROMPT.md` | Operational handoff prompt for Gap A. Used; work landed. |
| `H100_CAMERA_READY_GAPS_PROMPT.md` | Prompt for the H100 camera-ready gap-fill (Gap 1 + Gap 2). Both gaps landed (commit `bd707e8`). |
| `HANDOVER.md` | Operational handover note. Superseded by CHANGELOG. |
| `FOLLOWUPS_TRACKER.md` | Live work queue (F-numbers). Used during active study; audit-only now. |

### Phase handoffs (chronological)

| File | Phase |
|---|---|
| `PHASE2_CA_HANDOFF.md`, `PHASE2_FL_HANDOFF.md`, `PHASE2_FL_STATUS.md`, `PHASE2_HANDOFF_PROMPT.md`, `PHASE2_TRACKER.md` | Phase 2 (substrate replication FL/CA/TX) |
| `PHASE3_HANDOFF_PROMPT.md`, `PHASE3_LIGHTNING_HANDOFF.md`, `PHASE3_TRACKER.md` | Phase 3 (leak-free closure on Lightning Studio H100) |

### Session handoffs (operational chronology)

| File | Date |
|---|---|
| `SESSION_HANDOFF_2026-04-22.md` | First weekend session |
| `SESSION_HANDOFF_2026-04-24.md`, `_PM.md` | F21c gap discovery + F27 cat-head refinement |
| `SESSION_HANDOFF_2026-04-26.md` | H3-alt champion discovered |
| `SESSION_HANDOFF_2026-04-27.md` | F49 attribution + Phase-1 substrate validation |
| `SESSION_HANDOFF_2026-05-01.md`, `_PM.md` | Paper closure + H100 camera-ready gaps |

## How to use this archive

- **Don't cite these as paper canon.** Cite `results/RESULTS_TABLE.md §0` (v8) for paper numbers.
- **Use them as audit / re-derivation.** If you need to know *why* a number is what it is, the per-phase trackers and per-session handoffs hold the lab-trail.
- **The `F49 +6.48 pp MTL > STL on AL` story is preserved here in audit form.** That number was a leak artefact (asymmetric C4 leak inflated MTL more than STL); the leak-free closure says MTL trails STL on reg at every state by 7-17 pp. This is documented in detail in `../../CHANGELOG.md` (entry: 2026-05-01 paper closure) and `../../research/F50_T4_C4_LEAK_DIAGNOSIS.md`.

## Rule for adding to this archive

When work lands, move the workflow / handoff / prompt artefacts here. Do not delete — preserve the lab-trail. Update `../../CHANGELOG.md` to log the move under that day's row.
