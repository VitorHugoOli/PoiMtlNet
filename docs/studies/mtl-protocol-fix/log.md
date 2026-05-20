# mtl-protocol-fix — Progress Log

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-20`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Next** what the next agent should pick up.

---

## 2026-05-20 — Study launched

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/mtl-protocol-fix/` created alongside `canonical_improvement/` and `merge_design/`.
- `AGENT_PROMPT.md`, `considerations.md`, `INDEX.html` written. The study is direct successor to the closed `canonical_improvement` (Tier 1-6, 26 mechanism families, ceiling ±0.8 pp on substrate axis) and supersedes the F1/F2/F3 split of `mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`.
- 5 future-work memos created under `docs/future_works/` for items deliberately scoped OUT of this study (paper-canon re-evaluation, substrate-adaptive MTL balancing, MTL architecture revisit, head/window/batch audit, reg-head architecture sweep).
- Doc cross-references updated: `docs/CHANGELOG.md`, `docs/README.md`, `docs/CONCERNS.md` C21, `docs/NORTH_STAR.md`, `docs/AGENT_CONTEXT.md`, `docs/future_works/README.md`, `docs/studies/canonical_improvement/log.md`, `docs/studies/mtl-exploration/README.md` — all now point to this study as the active protocol-axis track.
- User-directed scope clarified (2026-05-20): IN = Rank 1 (F1 selector fix) + Rank 3 (Tier 5/6 candidate re-eval under F1) + three-frontier MTL evaluation protocol. OUT = Rank 2 / Rank 4 / Rank 6 / Rank 7 / §4.3 — all documented in `docs/future_works/`.

**Decision** — Branch `mtl-protocol-fix` will be the dedicated worktree for execution. Do not contaminate `check2hgi-canonical-improve` or any other branch.

**Decision** — Three-frontier evaluation (best joint + best disjoint + STL ceiling) is the study's primary methodological deliverable; replaces the implicit single-selector reporting that obscured C21.

**Decision** — Phase-1 single-seed n=5 first; multi-seed only in Phase 3 if residual gap characterisation requires statistical power. Full §0.1 n=20 multi-seed re-evaluation is deferred to `docs/future_works/paper_canon_reevaluation.md` (sequenced after `mtl_architecture_revisit.md`).

**Next**

1. Implementing agent must read in order: this log, `INDEX.html` (top-down), `considerations.md`, `docs/CONCERNS.md` C21, `docs/studies/canonical_improvement/log.md` 2026-05-19 final entry, `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md`.
2. Create the dedicated worktree before running anything.
3. Start with Phase 0 (pre-flight gate: reproduce `T6_4_dual_selector_final.json` shipping arm under F1-fixed code path).
4. Use `TaskCreate` to break down every Phase into validate → code-change → unit-test → re-evaluate → analyze sub-tasks.
5. After each Phase completes, call advisor with the Phase's results before proceeding.
6. After the full study completes, run a final advisor pass on the whole INDEX.html + log + the three-frontier table before declaring done.

---

## (template — copy and date for next entry)

## YYYY-MM-DD — <Short title>

**Phase**: <Phase 0 / 1 / 2 / 3 / Final synthesis>

**What happened**

- <bullet>
- <bullet>

**Decision** (only if changed direction):
- <what changed and why>

**Blocker** (only if stuck):
- <what's blocked>
- <what you tried>

**Findings** (only if results landed):
- per state, three-frontier numbers (best joint / best disjoint / STL)
- statistical significance / Wilcoxon p / fold-by-fold deltas
- **Verdict**: <"fix validated" | "partial" | "regression" | "inconclusive at n=5">
- Updated INDEX.html `#PhaseN-*` Results placeholder: yes/no

**Next**:
- <experiment ID> next, or <decision needed>

---
