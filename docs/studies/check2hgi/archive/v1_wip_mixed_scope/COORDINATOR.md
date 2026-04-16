# Check2HGI Track — Coordinator Pointer

This track reuses the legacy coordinator/worker skill definitions at `docs/studies/COORDINATOR.md`. The only differences are:

- **Study root:** `docs/studies/check2hgi/` (not `docs/studies/`).
- **State file:** `docs/studies/check2hgi/state.json`.
- **Claims file:** `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`.
- **Phases:** `docs/studies/check2hgi/phases/P{-1,0,1,2,3,4}_*.md`.
- **Results:** `docs/studies/check2hgi/results/P{...}/<run_id>/summary.json`.

When invoking `/coordinator` or `/worker` on this branch, point them at this subtree. Legacy study is frozen on this branch — do not let the coordinator wander into `docs/studies/*.md` at the top level.

## State machine

Same as legacy: `pending → running → completed`. See legacy `coordinator/state_schema.md` for the JSON schema.

## Runbook handoff

Each phase doc contains its runbook section. The coordinator reads the active phase from `state.json`, validates its gates, and executes the runbook.
