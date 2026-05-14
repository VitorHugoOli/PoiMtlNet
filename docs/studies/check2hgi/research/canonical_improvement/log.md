# Canonical Check2HGI Improvement Track — Progress Log

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-14`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Next** what the next agent should pick up.

---

## 2026-05-14 — Track designed, awaiting execution

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/check2hgi/research/canonical_improvement/` created alongside `merge_design/`.
- `INDEX.html` written: 5-tier slate, 18 experiments, audit of user considerations, falsified-history table, evaluation framework, integration appendix.
- 5 breadth-search sub-agents run in parallel (feature engineering, architecture, data flow, training/loss, external literature). Synthesis lives in INDEX.html.
- Critical finding lifted to the design rules: **`fclass` IS the linear-probe label** — using it as a feature/supervision is tautological leak. This killed user consideration C7 and reframed Tier 4 to use unsupervised proxies (SwAV prototypes K=64, popularity, opening-hours, co-visit mix) rather than fclass-labelled side-tasks.
- User alignment captured (`AskUserQuestion` answers): all 5 tiers shipped, T5.2a + T5.2b both included, branch name `check2hgi-canonical-improve`, falsified-history table in front matter.

**Decision** — Branch `check2hgi-canonical-improve` is the dedicated worktree for execution. Do not contaminate `check2hgi-up`.

**Housekeeping note (resolved 2026-05-14)** — The user's considerations file was originally captured at `docs/studies/check2hgi/research/merge_design/considerartions.md` (typo: extra 'r'). It has now been moved + renamed to `docs/studies/check2hgi/research/canonical_improvement/considerations.md` since this track is the one that addresses it. No further action needed.

**Late additions (post-advisor)** — Two final updates made before handoff:

1. **Richer baseline table in INDEX.html.** The original 3-row pinned baseline was extended into three tables grounded in `results/BASELINES_AND_BEST_MTL.md` and `results/RESULTS_TABLE.md §0.1`:
   - Table 1 — simple floors (Random / Majority / Markov-1-region) for sanity gates.
   - Table 2 — STL matched-head ceilings (leak-free per-fold, seed=42, n=5) — **the primary comparison target for Tier 1-4**.
   - Table 3 — MTL B9 paper-canonical multi-seed v11 (n=20) — **only for Tier-4 final-winner shipping comparison**.
   Tier-1 to Tier-4 iteration uses Table 2 (cheap, paired Wilcoxon at n=5). Final shipping candidates are promoted to Table 3 (4 seeds × 5 folds, n=20 pooled Wilcoxon).

2. **`AGENT_PROMPT.md` written.** Standalone onboarding prompt for the implementing agent. Contains: required reading list (10 files in order), hard rules (no merge-family, no fclass-as-feature, mandatory pre-flight, multi-seed for stat claims, unit-test gate, falsified-history off-limits), required workflow (worktree → TaskCreate → /goal → advisor at tier boundaries → log.md), execution order (T1.1 first, T1.2 last in Tier 1; advisor between tiers; final advisor pass), baseline-table comparison protocol, results-block template for filling in INDEX.html results placeholders, file-path quick reference, and an explicit reminder that deviation from the design is authorized.

**Next**

1. Implementing agent must read in order: this log, `INDEX.html` (top-down, including Execution Guidelines), `../merge_design/STUDY_BRIEFING.html`, `../merge_design/STATE.md`, `../merge_design/AUDIT_HGI_GAP.md`.
2. Create the dedicated worktree before running anything.
3. Start with Tier 1 (T1.1 leak audit first — it gates everything else). Use `TaskCreate` to break down every experiment into validate → launch → import → analyze sub-tasks. Use the `/goal` command for autonomous execution.
4. After each tier completes, call advisor with the tier's results before proceeding.
5. After the full track completes, run a final advisor pass on the whole HTML + log before declaring done.

---

## (template — copy and date for next entry)

## YYYY-MM-DD — <Short title>

**Phase**: <Tier 1 hygiene / Tier 2 negatives / ... / Final synthesis>

**What happened**

- <bullet>
- <bullet>

**Decision** (only if changed direction):
- <what changed and why>

**Blocker** (only if stuck):
- <what's blocked>
- <what you tried>

**Findings** (only if results landed):
- <state>: cat F1 = X.XX (Δ vs canonical Y.Y), reg Acc@10 = X.XX (Δ Y.Y), fclass probe = X.X%
- statistical significance / Wilcoxon p / fold-by-fold deltas
- **Verdict**: <"strict dominance pass" | "non-inferior" | "falsified" | "inconclusive at n=5">
- Updated INDEX.html `#TX-Y` Results placeholder: yes/no

**Next**:
- <experiment ID> next, or <decision needed>

---
