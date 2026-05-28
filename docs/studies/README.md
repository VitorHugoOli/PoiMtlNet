# docs/studies/ — Active follow-up studies on check2hgi

This folder hosts ongoing research tracks layered on top of the primary check2hgi study (`docs/` root). Each subdir is a self-contained study with its own briefing, design, log, and findings.

## Active studies

| Study | Status | Branch (post-merge) | Read first |
|---|---|---|---|
| [`canonical_improvement/`](canonical_improvement/) | **CLOSED** 2026-05-19 — Tier 1-6 substrate axis exhausted at ±0.8 pp ceiling | (closed) | [`canonical_improvement/AGENT_PROMPT.md`](canonical_improvement/AGENT_PROMPT.md) |
| [`mtl-protocol-fix/`](mtl-protocol-fix/) | **CLOSED** 2026-05-24 — F1 selector fix + P4 mechanism identification (residual is architectural) | (closed) | [`mtl-protocol-fix/AGENT_PROMPT.md`](mtl-protocol-fix/AGENT_PROMPT.md) + [`mtl-protocol-fix/DEFERRED_WORK.md`](mtl-protocol-fix/DEFERRED_WORK.md) |
| [`mtl_improvement/`](mtl_improvement/) | **ACTIVE** — T0-T8 chain (backbones, loss, batch, LR, α, heads, multi-seed champion) | `mtl-improve` | [`mtl_improvement/AGENT_PROMPT.md`](mtl_improvement/AGENT_PROMPT.md) |
| [`substrate-protocol-cleanup/`](substrate-protocol-cleanup/) | **ACTIVE** — Tier A-D substrate + protocol cleanup; small states only; independent of `mtl_improvement` | `main` worktree (this branch) | [`substrate-protocol-cleanup/AGENT_PROMPT.md`](substrate-protocol-cleanup/AGENT_PROMPT.md) |
| [`merge_design/`](merge_design/) | **ACTIVE-CLOSING** — Designs A-M and Levers 1-6 mostly falsified or saturated; Phase 11 plan in flight | (no dedicated branch) | [`merge_design/STATE.md`](merge_design/STATE.md) |
| [`hgi_category_injection/`](hgi_category_injection/) | **CLOSED** (AZ falsified 2026-05-04) — kept here pending decision to revisit on FL/CA/TX. **Do NOT treat as active without an explicit re-open commit.** | (no dedicated branch) | [`hgi_category_injection/INDEX.md`](hgi_category_injection/INDEX.md) + [`hgi_category_injection/STATUS.md`](hgi_category_injection/STATUS.md) |

## Folder semantics — `studies/` vs `findings/` vs `archive/`

> **`studies/`** = active or pending-decision research tracks (still being worked on, or recently closed but kept here pending re-open).
> **`findings/`** = closed per-experiment findings supporting the BRACIS paper (read-only history).
> **`archive/`** = fully closed studies and snapshots, unlikely to be re-opened.

## Archive policy

Studies in this folder are *active or pending decision*. When a study is fully closed and unlikely to be re-opened, `git mv` it to `docs/archive/<study>-closed-YYYY-MM-DD/`.

## Where do paper-supporting findings go?

Closed per-experiment findings that support the BRACIS paper (the "F-trail") live at [`docs/findings/`](../findings/), not here.

## Adding a new study

1. Create the subdir: `docs/studies/<study_name>/`.
2. Author a study-onboarding doc at the top — `AGENT_PROMPT.md` (if agent-driven), `README.md`, or `STATE.md` (per the existing studies' conventions).
3. Update this `README.md` with a row in the Active studies table.
4. If the study is large enough to warrant its own branch, branch from `main` (or the latest target branch) and document the branching point in the study's onboarding doc.
5. Optional: add a `state.json` for run-tracking (see fusion archive's pattern).

## Layered-study pattern

Studies in this folder are *layered on check2hgi*: they assume the canonical Check2HGI substrate, the B9 MTL recipe (or its small-state H3-alt variant), and the matched-head GRU evaluation protocol. They don't redo the substrate work; they extend it.

Read [`docs/AGENT_CONTEXT.md`](../AGENT_CONTEXT.md), [`docs/NORTH_STAR.md`](../NORTH_STAR.md), and [`docs/CLAIMS_AND_HYPOTHESES.md`](../CLAIMS_AND_HYPOTHESES.md) before starting a new study — they are the project-wide scientific baseline.
