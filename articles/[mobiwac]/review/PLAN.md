# review — attack plan

> **Goal.** Close out the Germano review (the per-comment edits in [`../REVIEW_GERMANO.md`](../REVIEW_GERMANO.md))
> **and** the broad "other points" ([`review_v2.md`](review_v2.md)) without one change quietly breaking another.
> **Principle:** investigate → decide → edit, in that order; never edit `src/` from a number we haven't verified.
>
> **STATUS: EXECUTED (2026-06-28 → 06-30).** All phases below have run, the prose pass is applied, and the paper
> builds clean (9 pages, 0 undefined refs, 0 non-font warnings). This file is now the record of what was designed
> **and** what actually happened: see **Execution status** right after the phasing diagram. The workflow-shape
> designs further down are kept as the as-built record.

## Phasing

```
Phase 0  Decisions locked            ✅ DONE — REVIEW_GERMANO v2 final-decision pass (2026-06-28)
Phase 1  INV investigations (facts)  ✅ DONE — workflow germano-author-notes-eval (INV1/2/3/5) + INV4 fig agent
Phase 2  OP discovery sweeps         ✗ NOT RUN as designed — only INCIDENTAL coverage (the §4–§8 Germano-lens audit
                                        + the CC1–CC7 sweeps). The dedicated full-paper OP2/OP3/OP4 matrices
                                        (acronyms, gloss coverage, redundancy, incl. abstract + §1–§3) were NOT built
Phase 3  Apply per-comment edits     ✅ DONE — prose pass: accepted edits + CC1–CC7 (2026-06-29)
Phase 4  OP5 English pass            ✗ NOT RUN as a dedicated pass — only CC1 (em-dash) + the run-ons/connectives the
                                        §4–§8 audit happened to surface; no full-paper English sweep
Phase 5  Build + checklist           ✅ DONE — pdflatex→bibtex→pdflatex×2 clean, 9 pages, 0 undefined / 0 non-font warnings
```

## Execution status (2026-06-30)

What designed-vs-actual mapped to:
- **Phase 1 (facts)** ran as the **`germano-author-notes-eval`** workflow (investigator + adversarial advisor per
  author note) plus the re-run of the two that hit the schema cap (AN66.1 loss, AN54). It closed INV1 (precedent —
  web), INV2 (+5% measured = FALSE), INV3 (Acc@10 frequency-weighted; loss reason backwards; stratification), INV5
  (MTL never defined). INV4 (Fig 3) ran later as a focused agent.
- **Phase 2 (OP sweeps) was NOT run as designed (corrected 2026-06-30).** The `germano-lens-chapter-audit` workflow
  applied the Germano pitfall lens to **§4–§8 only**, which overlaps the OP themes but is **not** the dedicated
  WF-CONSISTENCY: there was no full-paper (abstract + §1–§3 included) OP2 acronym-occurrence map, OP3 gloss-coverage
  matrix, or OP4 redundancy matrix. The two final advisors audited the *review*, not the paper's prose
  systematically. So OP2/OP3/OP4 are only **incidentally** addressed (the instances those passes + the CC1–CC7
  sweeps happened to surface), not systematically. **They still owe their proper phase (see "What remains").**
- **Phase 3 (prose pass)** applied across the abstract + §1–§8 + Tables 1/3 + Fig 1/2 captions, plus the
  cross-cutting sweeps CC1–CC7. Then the **Istanbul correction + Fig 3 regeneration + per-state distribution +
  7%-floor audit** round (2026-06-30).
- **Phase 4 (OP5 English) was NOT run.** Only CC1 (em-dash) and the run-ons / missing-connectives the §4–§8 audit
  surfaced were fixed inline. A dedicated full-paper English/phrasing sweep (`WF-ENGLISH`) has not run.

## Gated edits — all resolved

These edits were gated on a fact; each is now settled and applied (except OP1, which was not pursued):

| Edit | Was gated on | Outcome |
|---|---|---|
| §4.2 "+5% params/compute" (#67) | INV2 measurement | **+5% FALSE** (vs one model +38–90% params); rewrote as Option A ("cheaper than two models") ✅ |
| §4.2 loss justification (#66.1) | INV3 | macro-F1-alignment reason is backwards; rewrote as the empirical (qualitative) version ✅ |
| §2.2 "little precedent" (#50) | INV1 (web) | coarse-cell-as-target IS precedented; softened to "to our knowledge … underexplored", no new bib ✅ |
| §2.3/§6.2 orthogonality sentence (OP1) | OP1 doc scrape | **NOT pursued** — left out (optional; would need the docs/ regime finding) |
| Fig 3 (#69) | INV4 | not a clustering bug; regenerated with neutral axis + "kNN purity (k=10)" + clustering-free caption ✅ |

## Workflow shapes (dynamic workflows — author invokes when ready)

> These are the **designs**; the **Execution status** section above records which ran and as what. Each was a single
> well-scoped fan-out returning a findings report; the discovery workflows did **not** edit the paper, the prose pass
> (Phase 3) did. (As-built: WF-CODEBASE ran as `germano-author-notes-eval`; WF-CONSISTENCY ran as
> `germano-lens-chapter-audit`; WF-ENGLISH was folded into the prose pass; INV4 ran as a focused agent.)

### WF-CODEBASE  (Phase 1 — INV1, INV2, INV3, OP1)
- **Shape:** fan-out one agent per question, each scoped to its files, structured output, then a synthesis.
  - INV2 → `src/models/*`, `src/models/registry.py`, `src/utils/flops.py` → {param Δ%, FLOPs Δ%, basis}.
  - INV3 → `src/data/folds.py`, `src/data/inputs/*`, `src/losses/*` + memory `mtl_category_loss_unweighted.md` →
    {per-state 7-class distribution, stratification key, what Acc@10 weights, class-weighting-vs-macro-F1 evidence}.
  - OP1 → `docs/` (the MTL regime / orthogonality finding, W6 probe) → {can we claim it, exact sentence, source}.
  - INV1 → **web** via the `deep-research` skill (separate, not a code agent) → {precedent verdict + citations}.
- **Verify:** each numeric claim adversarially re-checked against the source file/JSON before it's reported.
- **Returns:** one report per INV; the author approves the resulting sentences.

### WF-CONSISTENCY  (Phase 2 — OP2, OP3, OP4)
- **Shape:** one reader per section (`01`–`08` + abstract + tables + captions) extracting, with structured output:
  (a) acronym occurrences + first-use, (b) technical terms + glossed?/compliant?, (c) concept→sentence index.
  **Barrier**, then cross-section reducers build three matrices: acronym re-expansion map (OP2), gloss-coverage
  table (OP3), redundancy matrix (OP4). The barrier is justified here — the reducers need *all* sections at once to
  spot cross-section duplication.
- **Guard:** OP4 must separate harmful duplication from deliberate thesis reinforcement (the single-model property is
  meant to recur) — flag, don't auto-cut.
- **Returns:** three findings tables + a proposed edit list (deletions/moves), author-approved before Phase 3.

### WF-ENGLISH  (Phase 4 — OP5)
- **Shape:** per-section language reviewers (grammar, connectives, American English, em-dash/CC1, confusing
  phrases) → each flagged phrase adversarially re-checked (real defect vs over-correction) → consolidated,
  line-anchored edit list.
- **Runs last** so it polishes the final wording, not soon-to-change sentences.

### INV4 (figure) — single focused agent, not a workflow
- Read `figs/fig3_embquality.py`, render the PDF, return a regeneration spec (axis/title/labels) + caption/prose.
  Only edits the committed figure on explicit author go.

## Guardrails (every phase)
- **No paper edits in Phase 1–2.** Findings only.
- **Honesty rules are non-negotiable:** GLOSSARY honesty rule (beats/matches, never "Pareto"/"ties"/"beats region
  everywhere"), the claim-discipline whitelist (`PAPER_PLAN.md §3`), and the CC3 verb decision (one superiority verb).
- **Every number traces to a JSON / a measured run** (the board file-map is `RESULTS_BOARD.md §3`); never lift a
  number from prose.
- **Log every cross-cutting change** in [`LOG.md`](LOG.md) the moment it's decided, so a later edit can see it.
- **Author reviews each phase's findings** before the next phase edits `src/`.

## What remains — the OP phases still owed (NOT optional)

The Germano per-comment edits + CC1–CC7 are applied, but the broad **`other points to review` (OP1–OP5) still owe
their proper, separate phases over the WHOLE paper**. The prose has now settled, so this is the right time to run
them (each is discovery → a findings + edit list; no auto-edit, author approves before any prose change):

- **OP1 (Phase 1 tail)** — orthogonal-transfer / why-no-balancer sentence: a `docs/` scrape (the MTL regime finding,
  the W6 probe, C25) → a plain, honest one-liner for §2.3/§6.2, or a documented decision NOT to claim it. NEVER run.
- **OP2 / OP3 / OP4 (Phase 2, `WF-CONSISTENCY`)** — the full-paper matrices the §4–§8 audit never built: every
  acronym's first-use vs re-expansion (OP2), every technical term's gloss coverage (OP3), and a concept→section
  redundancy matrix (OP4), **including the abstract + §1–§3** the audit never touched.
- **OP5 (Phase 4, `WF-ENGLISH`)** — a dedicated full-paper grammar / connectives / American-English / clarity pass,
  run LAST (now valid, the prose has settled).
- Then a final **build + GLOSSARY §6 checklist**.

**Smaller / author-acknowledged (leave unless asked):** the literal Chrome screenshot of Fig 3 (extension was
offline; audited via PNG); the R21 scaling-confound hedge in abstract/§1/§8 (§6.2 has it; author left the rest);
reconciling the internal `CLAUDE.md`/`PAPER_PLAN` Istanbul wording; the mobility-aware-service citation (#57, future
work). Out of scope: the science item **P1 (n=20 multi-seed, Gowalla)**.
