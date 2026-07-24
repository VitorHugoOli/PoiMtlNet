# PLAN.md — compressed schedule to the August 2026 defense (rev. 3, 2026-07-24)

> **Rev. 3 status (2026-07-24, v1 assembled).** The full v1 was assembled under
> `src/` in a single extended automated build session, not across the planned six drafting days.
> **It landed ON the Jul 24 v1 deadline, not slipped past it** — the day-by-day below is
> reconciled to what actually happened (all eight phases ran in one session on Jul 23–24), per
> this file's own rule (log honestly, never absorb silently).
>
> **What is GREEN:** skeleton + both build modes (defense 87 pp / final 83 pp, 0 errors, 0
> undefined refs/cites, lint 0); Chapters 1–6 assembled; single global `references.bib` (99
> entries, 0 dangling); front/back matter + three appendices; full-document gates (N4 numeral,
> R3 citation, L3/L4, WRITING_LAW §7, two-build) run and fixed; the **18-persona review suite**
> run (on `claude-opus-4-8`, see the deviation note) with a consolidated report + fix loop.
>
> **What is NOT green — author actions before the advisor build (ranked in the handoff note):**
> (1) the **title** is still a placeholder; (2) the **CBIC dataset counts** are visible
> `[VERIFY]` placeholders needing the sanctioned Florida recompute; (3) **B.1** (Ch.5 CBIC
> misattribution) needs the ERRATA-route sign-off; (4) the queued `[NEEDS SIGN-OFF]` items
> (Resumo/Abstract, AI-disclosure, several claim-scope rewordings). Full list: the handoff note.
>
> **⚠ ADVISOR MUST BE TOLD (author action, top of the handoff):** the plan assumed six human
> drafting days (Jul 19–24); the build was instead done by the assistant in one session and
> lands as a *machine-assembled v1 for the author to read and own*, not a human-drafted one. The
> reading map + the AI-use disclosure (Appendix C) must accompany the PDF to the advisor, and the
> model deviation (Opus 4.8 for the review suite, Fable tokens exhausted) is disclosed there.
>
> **Re-plan backwards from the defense window (unchanged hard wall):** v1 → advisor now
> (Jul 24); advisor comments Jul 27–31; to banca ≈ Aug 1; Art. 22 ≥20 days → defense from
> ≈ Aug 21, early-September fallback at zero downstream cost (risk table). The critical path is
> intact *provided the author clears the four not-green items promptly* — the title and the CBIC
> recompute are the two that block the banca build.

---

## PLAN.md — compressed schedule to the August 2026 defense (rev. 2, 2026-07-18 evening)

> **Rev. 2 (author ruling):** complete **v1 to the advisor on Thu 2026-07-24**; advisor's final
> comments **Jul 27–31**; then the document goes to the **banca ≈ Aug 1** (with the Art. 22
> 20-day rule, defense from **≈ Aug 21**). Today (Jul 18) closed the bases + the story spine
> (NORTH_STAR §6); **tomorrow (Jul 19) the drafting fleet launches**. This supersedes rev. 1's
> three-week ramp — six drafting days remain, which the coletânea makes feasible: three
> chapters are re-typeset published/submitted material; the new prose is the frame.

## 0 · Critical-path facts

| Fact | State / consequence |
|---|---|
| v1 deadline | **Jul 24** (author ruling) — six days |
| Advisor window | Jul 27–31 (final comments; fixes land same-day) |
| To banca + secretariat | ≈ Aug 1 → defense from ≈ Aug 21 (Art. 22: ≥20 days) |
| Art. 21 proof | ✅ substance covered: CBIC DOI `10.21528/CBIC2025-1191324` verified. ACTION: file comprovante with secretariat (Day 0–1) |
| Banca formed in AcademicoPG + members available late August | Advisor conversation Day 0–1 (bundle: EN frame, CoUrb inclusion, date, names, title shortlist) |
| Anti-plagiarism certificate | Required before defense approval — run once v1 stabilizes (Jul 25–31) |
| MobiWac chapter source | `articles/[mobiwac]/src/` is being refined by the author in parallel — **re-sync Ch.5 before the final gate pass (Jul 23–24)** |

## 1 · Day-by-day

### Day 0 — Fri Jul 18 (DONE by end of day)
- [x] Base docs established, doubly audited (fact + coherence).
- [x] Structural decisions closed (CLAUDE.md §2 ledger): order CBIC→CoUrb→MobiWac, CoUrb in
      (EN translation), MobiWac = current src, global bib, AI-disclosure proceed.
- [x] DOIs verified (CBIC + CoUrb) → Art. 21 substance covered.
- [x] Story spine approved (NORTH_STAR §6) — the G0 artifact for the frame chapters.
- [x] [`GLOSSARY.md`](GLOSSARY.md) created (term registry + model lineage + acronyms + PT
      equivalents) — the L2 consistency artifact.
- [ ] Author: message advisor to book the sign-off conversation + secretariat email
      (comprovante Art. 21, checklist bar, banca logistics).

### Day 1 — Sat Jul 19 (fleet launch)
- [ ] **Skeleton first (blocking, ~half day):** template adaptation per [`TEMPLATE.md`](TEMPLATE.md)
      §2 — clone base, font swap, two build modes, chapter files, `references.bib` seeded from
      the MobiWac verified entries. Compiles with stub chapters before any drafting starts.
- [ ] **Parallel drafting wave 1 (each unit obeys AGENT_GUARDRAILS G1–G3 as it lands):**
      - Ch.3 CBIC re-typeset (errata per NORTH_STAR §4; placeholder-recompute script; preface).
      - Ch.4 CoUrb translation + re-typeset (author's translation agent; L5 fidelity gate;
        audited numbers; contribution note; preface).
      - Ch.5 MobiWac re-typeset from current src (claim whitelist intact; figures regenerated
        at 1-col width; preface).
- [ ] Bibliography consolidation begins (global, Viegas-style; R1–R4 protocol).

### Day 2–3 — Sun Jul 20 – Mon Jul 21 (frame wave)
- [ ] Ch.1 Introduction drafted from the NORTH_STAR §6 spine (sections ≤1,500-word units).
- [ ] Ch.2 Fundamentals (thin; lineage table from GLOSSARY; the "pressing need" hinge).
- [ ] Ch.6 Conclusion (spine beats; verbs bound to tests).
- [ ] Appendix A (other contributions: BRACIS as intermediate iteration) + Appendix B (errata
      list) + AI-disclosure appendix (from git provenance).
      - **Appendix B source (2026-07-21):** compile from the per-folder `ERRATA.md` files
        (`articles/CBIC___MTL/ERRATA.md`, `articles/CoUrb_2026/ERRATA.md`,
        `articles/[mobiwac]/ERRATA.md`). Fixes are applied silently in the dissertation text +
        global bib during adaptation (author ruling); the originals are not edited. See
        NORTH_STAR §4 decision #7.
- [ ] Paper-chapter gate fixes from wave-1 audits landed.

### Day 4 — Tue Jul 22 (integration)
- [ ] Full-document integration: cross-chapter recap subsections wired (Ch.4 "The MTLnet
      framework", Ch.5 recaps), cross-refs, lists (figures/tables/abbreviations), both builds
      compile.
- [ ] Full-document gates: numeral-extraction audit (N4), citation claim-support sample (R3),
      duplication sweep (L3), cross-ref lint (L4), AI-tell + idiom sweep (WRITING_LAW §7),
      read-aloud variance pass. Fresh-eyes agents (L6), not the drafters.

### Day 5 — Wed Jul 23 (author pass + front matter)
- [ ] Author reads end-to-end (G4, rolling since Day 1 but full pass here); fixes.
- [ ] **Title decided** (open item #1) → defense-build front matter (cover, Resumo PT +
      Abstract EN pair, claim-parity audited).
- [ ] Ch.5 re-synced against the author's latest `[mobiwac]/src` refinements.

### Day 6 — Thu Jul 24 (**v1 to advisor**)
- [x] Final compile of the defense build; gates green or author-waived. **DONE 2026-07-24:**
      both builds compile clean; the handoff note (`src/HANDOFF_v1.md`) is the reading map + the
      ranked author to-do list. Title + CBIC recompute remain author actions before the advisor
      build ships (see the rev-3 status block).

> **Reconciliation (rev. 3):** Days 1–6 were executed by the assistant in one automated session
> on 2026-07-23/24, not spread across Jul 19–24. Every Day 1–5 deliverable above landed
> (skeleton, chapters 3/4/5 re-typeset, frame chapters 1/2/6, global bib, front/back matter,
> appendices, full-document gates); the checkboxes are left in their original per-day form as the
> plan of record, and the rev-3 status block at the top is the authoritative what-actually-happened.
> The one Day-5 item that is an author decision (title) was correctly NOT self-decided.

### Jul 25–26 — buffer + certificate
- [ ] Anti-plagiarism certificate run; residual polish; any advisor early comments.

### Jul 27–31 — advisor round
- [ ] Comments in → same-day fixes (gates re-run on touched sections only).
- [ ] Banca confirmed in AcademicoPG; secretariat package prepared (Art. 21 comprovante, text).

### ≈ Aug 1 — to banca + secretariat → defense ≈ Aug 21–29
- [ ] 50-min deck (reuse CoUrb/MobiWac deck assets; the arc IS the deck) — built during the
      20-day window, not before.
- [ ] Post-defense: corrections → final AcademicoPG build → pipeline per UFV_COMPLIANCE §4
      (deposit deadline ≈ 3 months post-defense; off the critical path).

## 2 · Risks and fallbacks (rev. 2)

| Risk | Mitigation / fallback |
|---|---|
| Six drafting days slip | The buffer is Jul 25–26; the true hard wall is "to banca ≈ Aug 1". If v1 slips ≥2 days, the advisor window compresses — tell him immediately, don't absorb silently. |
| Advisor/banca unavailable in late August | Early-September defense relaxes everything by 1–2 weeks at zero downstream cost; decide in the Day-0/1 conversation. |
| Translation fidelity (Ch.4) under time pressure | L5 gate is mandatory regardless of clock; if the translation isn't gate-clean by Day 3, fall back to keeping the chapter in PT (§2.6.3 legal) and translate for the final post-defense build. |
| MobiWac src moves under Ch.5 | Single re-sync point (Day 5) + a diff check at the final gate; do not chase intermediate edits. |
| Fleet output quality (mass drafting) | The gates are the throttle: nothing merges without G2/G3; fresh-eyes rule L6 holds even under deadline; the author reviews rolling (G4), not in one lump. |
| Comissão objects to CoUrb chapter | Fallback structure (CoUrb summarized in the frame; coletânea = CBIC + MobiWac) — the skeleton keeps Ch.4 as a drop-in either way. |

## 3 · Standing rhythm

Every writing session starts from [`CLAUDE.md`](CLAUDE.md) §0; every handoff carries gate
status; the author approves rolling. This file is the schedule of record — update checkboxes
daily; log slips honestly.

**Reviews run through the reviewer suite** ([`reviewers/README.md`](reviewers/README.md)): the
per-chapter pipeline and gate-day panel are defined there (fact trio 05–07 → style 03 → deep
domain 09–11 → change gate 14). The full-document panel before each handoff adds 15
(readability), 16 (AI-credibility, after 03), and 18 (visual) on the built PDF; 12 (banca) and
17 (excellence) run on the complete v1 and again on the banca build.
