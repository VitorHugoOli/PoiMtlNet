# review/ — Germano-review close-out workspace

Working folder for finishing the co-author review of the MobiWac paper. **Read order:**

1. [`../REVIEW_GERMANO.md`](../REVIEW_GERMANO.md) — the 70 comments. The **⭐ Final decision pass (v2)** at the top
   is authoritative; the v1 Response/Edit blocks below it are provenance.
2. [`review_v2.md`](review_v2.md) — the broad "other points" as a task catalog (OP1–OP5 text sweeps, INV1–INV6
   investigations).
3. [`PLAN.md`](PLAN.md) — how to attack them: phasing (investigate → decide → edit) + dynamic-workflow shapes.
4. [`LOG.md`](LOG.md) — the running cross-reference ledger. **Update it whenever a change ripples.**

## State (2026-06-30) — Germano edits applied + building clean; the OP phases still owed

- **Phases 0, 1, 3, 5 — ✅ DONE.** Decisions locked; investigations run (workflow `germano-author-notes-eval` + the
  Istanbul / Fig-3 / distribution agents); the **prose pass applied** across the abstract + §1–§8 + Tables 1/3 +
  Fig 1/2 captions, plus the cross-cutting sweeps **CC1–CC7**; Istanbul corrected; Fig 3 regenerated; per-state
  distribution documented. **Build: 9 pages, 0 undefined refs, 0 non-font warnings.** CC3 applied (global
  "outperforms"). As-built record: [`PLAN.md`](PLAN.md) (Execution status) + [`LOG.md`](LOG.md) (Timeline).
- **Phase 2 (OP2/OP3/OP4 sweeps) and Phase 4 (OP5 English) — ✗ NOT RUN as designed.** The
  `germano-lens-chapter-audit` covered **§4–§8 only** with a pitfall lens; the dedicated full-paper acronym / gloss /
  redundancy matrices and the full-paper English pass (incl. abstract + §1–§3) were never built, and OP1 was never
  pursued. So the broad **`other points to review` (OP1–OP5) still owe their proper, separate phases** — see
  [`PLAN.md`](PLAN.md) "What remains" and [`review_v2.md`](review_v2.md) (OP1–OP5 are correctly still TODO there).

## What's done vs. what's left

- **Done:** all accepted Germano edits + CC1–CC7; the resolved investigations (#67 Option A "+5% was FALSE", #66.1
  empirical loss rewrite, #50 method-contrast, #64/#26 MTL+trunk glosses, #69 Fig 3 regen, #31 trunk, #38 author
  incorrect); the **Istanbul correction** (it IS stride-1 overlap n=20 — the article's "not rebuilt / cross-setting"
  claim was wrong, now fixed); the **per-state category distribution** (`docs/studies/closing_data/CATEGORY_DISTRIBUTION.md`)
  + a §5.2 stratification sentence; the **7%-floor audit** (kept "about 7%" — Gowalla mean 6.80%, my earlier "6%"
  suggestion withdrawn).
- **Owed — the OP phases (NOT optional):** OP1 (orthogonal-transfer sentence), OP2/OP3/OP4 (full-paper acronym /
  gloss / redundancy matrices, `WF-CONSISTENCY`), OP5 (full-paper English pass, `WF-ENGLISH`), then a final build +
  GLOSSARY §6 checklist. See [`PLAN.md`](PLAN.md) "What remains".
- **Smaller / author-acknowledged (leave unless asked):** the literal Chrome screenshot of Fig 3; the R21
  scaling-confound hedge in abstract/§1/§8; reconciling the internal `CLAUDE.md`/`PAPER_PLAN` Istanbul wording. Out
  of scope: **P1 (n=20 multi-seed, Gowalla)**. Future work (author): the mobility-aware-service citation (#57).

## Guardrails

The honesty rules don't bend: `GLOSSARY.md` (beats/matches vocabulary, no "Pareto"/"ties", no repo codenames, no
em-dash, American English) and the claim-discipline whitelist in `PAPER_PLAN.md §3`. Every number traces to a board
JSON (`docs/studies/closing_data/RESULTS_BOARD.md §3`), never to prose.
