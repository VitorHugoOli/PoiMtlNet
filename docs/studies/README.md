# docs/studies/ — Active follow-up studies on check2hgi

This folder hosts ongoing research tracks layered on top of the primary check2hgi study (`docs/` root). Each subdir is a self-contained study with its own briefing, design, log, and findings.

## Active studies

| Study | Status | Branch (post-merge) | Read first |
|---|---|---|---|
| [`canonical_improvement/`](canonical_improvement/) | **CLOSED** 2026-05-19 — Tier 1-6 substrate axis exhausted at ±0.8 pp ceiling | (closed) | [`canonical_improvement/AGENT_PROMPT.md`](canonical_improvement/AGENT_PROMPT.md) |
| [`mtl-protocol-fix/`](mtl-protocol-fix/) | **CLOSED** 2026-05-24 — F1 selector fix + P4 mechanism identification (residual is architectural) | (closed) | [`mtl-protocol-fix/AGENT_PROMPT.md`](mtl-protocol-fix/AGENT_PROMPT.md) + [`mtl-protocol-fix/DEFERRED_WORK.md`](mtl-protocol-fix/DEFERRED_WORK.md) |
| [`mtl_improvement/`](mtl_improvement/) | **CLOSED** 2026-06-12 — the C25 class-weighting confound dissolved the "MTL sacrifices reg" gap; champion **G (= canon v16, the train.py MTL default)** matches the STL reg ceiling + beats the cat ceiling +2.6…+4.1 (4 states × 4 seeds); mechanism = gradient orthogonality; no MTL optimizer helps. CA/TX + the aligned-pairing pre-freeze gate → `closing_data/` | `mtl-improve` | [`mtl_improvement/FINAL_SYNTHESIS.md`](mtl_improvement/FINAL_SYNTHESIS.md) |
| [`closing_data/`](closing_data/) | **SCAFFOLDED (not launched; re-scoped 2026-06-12)** — the experimental engine for the NEW paper: cross-study re-eval + BRACIS-suite inventory (RUN_MATRIX) → pre-freeze gates → recipe+substrate FREEZE → full base regeneration ONCE (STL baselines re-run + champion + suite cells, ALL states × 4 seeds × 5 folds) → hand-off to the story effort. Launch pending user sign-off on the plan | (TBD) | [`closing_data/HANDOFF.md`](closing_data/HANDOFF.md) → [`closing_data/AGENT_PROMPT.md`](closing_data/AGENT_PROMPT.md) + [`closing_data/PLAN.md`](closing_data/PLAN.md) |
| [`substrate-protocol-cleanup/`](substrate-protocol-cleanup/) | **CLOSED** 2026-05-29 — log_T-KD PROMOTED (now **v12 default**); substrate axis NULL in MTL at AL/AZ/FL (regime-limited, even HGI ≈ canonical in MTL); ResLN encoder = STL-best (v12 default). v11→v12 default flip + v13 base: [`results/CANONICAL_VERSIONS.md`](../results/CANONICAL_VERSIONS.md) | `main` | [`substrate-protocol-cleanup/CLOSURE.md`](substrate-protocol-cleanup/CLOSURE.md) |
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
