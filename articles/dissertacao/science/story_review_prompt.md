# story_review_prompt.md — narrative / storytelling review of the dissertation arc

> **Purpose.** A second prompt for the SAME Claude Science project (the agent context from
> `science.md §3` stays installed and still applies: the papers, the arc, the writing law, the
> fail-closed honesty protocol, the repo map). This prompt re-tasks the project for ONE job:
> review the **storytelling** of the dissertation — the unified CBIC -> CoUrb -> MobiWac arc —
> for cohesion, completeness (missing or worth-adding beats), and craft (is it a pleasure to
> read). Paste **§THE PROMPT** as a new message in the project.
>
> **How this relates to the reviewer suite.** The dissertation already has story-adjacent
> personas: `reviewers/17_excellence_assessor.md` (unity + the intro-conclusion loop test +
> missed-connection hunting), `reviewers/01_cold_reader.md` (does the frame read as one argument
> or as summaries), `reviewers/15_readability_editor.md` (is the text a pleasure to read, one
> voice across three papers). This prompt is the **narrative specialist** that goes deeper than
> persona 17's ten-dimension scorecard on the STORY itself, and it tells Claude Science to read
> those personas so it complements them instead of repeating their defect-level work.

---

## THE PROMPT

```
TASK
Review the STORYTELLING of my master's dissertation: the single unified narrative that runs
across the three article chapters, CBIC -> CoUrb -> MobiWac. This is a coletânea (three papers
framed as one document); the whole point of the frame is that it must read as ONE argument with a
through-line, not three stapled papers. Obey the project custom instructions already installed
(the honest-arc, the claim whitelist, the writing law, canonical names, fail-closed discipline).
This is a narrative/craft review: you assess the story and propose narrative moves; you never
fabricate a fact, number, or claim, and you never propose overclaiming as a "better story" (the
claim whitelist and WRITING_LAW verb-test binding win over narrative appeal, always).

Answer three questions, in order of importance:
  1. Is the story COHESIVE — does it hold together as one argument end to end?
  2. Is there a beat we are MISSING or that would be good to ADD (a gap, an unmotivated jump, a
     thread opened and never closed, a "why" left unanswered)?
  3. Is it ENJOYABLE and WELL-CRAFTED — a text a banca member wants to keep reading?

READ FIRST (the intended story, then the story as actually written, then the bounds)
- The INTENDED spine (the object of review): articles/dissertacao/NORTH_STAR.md — §1 (the research
  question), §2 (the arc + the honest-arc paragraph + the settled CBIC->CoUrb->MobiWac order),
  §3 (chapter map + the mandatory bridging devices + the time-capsule rule), §6 (the settled
  story spine: Ch.1 beats, Ch.2 beats, the Ch.3/4/5 prefaces, Ch.6 beats). This is what the
  dissertation is TRYING to tell.
- The story AS ACTUALLY WRITTEN (read each paper's intro, its background/related-work, and its
  conclusion — that is where each paper's own story lives):
    CBIC:    articles/CBIC___MTL/  (main.tex + sections/, esp. the intro, basis.tex, conclusion)
    CoUrb:   articles/CoUrb_2026/src_en/  (main.tex + sections/, esp. the intro, related.tex,
             conclusion) — the English mirror; the chapter will be the translated reproduction
    MobiWac: articles/[mobiwac]/src/  (sections/ 01_intro, 02_related, 03_problem, ... conclusion)
             — the version of record
- The FUNDAMENTALS just built (does Chapter 2 set up the payoff?): articles/dissertacao/fundamentals/
  — README.md, the 2.1–2.5 folders (esp. 2.5_relevance, the "pressing need" hinge that must
  pre-motivate Ch.3/4/5), model_lineage_table.md, OPEN_QUESTIONS.md. These are plans/citations,
  not final prose, so judge whether they SET UP the arc, not their sentence craft.
- The HONESTY BOUNDS (a compelling story may not break these): articles/dissertacao/WRITING_LAW.md
  §3 (every number carries its reference point; verbs bound to tests — "outperforms" only with a
  paired superiority test, "matches" only with TOST, never upgrade AZ; time-index CBIC/CoUrb
  conclusions) and §5 (Viegas structure devices); GLOSSARY.md (one canonical name per concept);
  AGENT_GUARDRAILS.md §3 (claim registry: any NEW connective claim in the frame needs author
  sign-off — flag it, do not assert it) and §7 (the biases to catch: overclaiming, fake cohesion
  from template transitions).
- COMPLEMENT, do not duplicate, these personas (read them, then go deeper on narrative):
  reviewers/17_excellence_assessor.md (the unity / intro-conclusion-loop / chapter-2 / missed-
  connection tests, and its hard rule: excellence through synthesis and delivery, never through
  inflation), reviewers/01_cold_reader.md, reviewers/15_readability_editor.md,
  reviewers/12_banca_simulator.md. Quality bar + bridging patterns to imitate:
  exemples/viegas/VIEGAS_ANALYSIS.md and the defended same-advisor example exemples/germano/.
  Award/excellence evidence: docs/research/dissertation_excellence_2026-07-20.md.

USE THE INTERNET to calibrate the narrative (verify sources; do not cite from memory):
- How the strongest publication-based / coletânea (compilation) theses achieve narrative UNITY
  across separately-published papers — the documented failure mode is "stapled papers."
- How to frame a NEGATIVE RESULT as a contribution (this dissertation's spine: a published null
  result -> its diagnosis -> its resolution). Find the conventions that make this land as a
  strength, not an apology.
- How award-grade dissertations in ML / mobility structure a problem -> journey -> payoff arc
  (Lovitts "outstanding vs very good", SBC CTD, and comparable ML/mobility theses if findable).
Bring back only what changes a concrete recommendation; keep it to a short calibration note.

THE EVALUATION (run all seven lenses)

1. THE ARC, RECONSTRUCTED. First, read the artifacts and write the story as it ACTUALLY reads
   right now (one tight paragraph + a beat-by-beat map). Then DIFF it against the intended spine
   (NORTH_STAR §6): where does the delivered arc drift from, under-deliver, or contradict the
   intended one? The drift is the finding.

2. THE LOGLINE TEST. State the whole dissertation as ONE compelling sentence (problem -> journey
   -> payoff). Then check: does each of the three paper-chapters earn its clause in that sentence?
   If a chapter does not move the logline forward, say why and what would fix it.

3. COHESION / CONNECTIVE TISSUE. Chapter by chapter, audit the seams:
   - the general Introduction's arc narrative (does it actually own the through-line?);
   - the mandatory bridging subsections (each later paper-chapter recapping the previous artifact
     by name — Ch.4 recaps MTLnet, Ch.5 recaps both) — present, and do they carry the reader?
   - the time-capsule prefaces (each chapter's one italic paragraph: venue, status, what later
     chapters revise) — do they keep superseded conclusions from reading as current?
   - the INTRO-CONCLUSION LOOP: list every promise/question the Introduction and the Fundamentals
     open, and check each is paid off in the Conclusion. Report a THREAD LEDGER: opened -> closed
     / opened -> dropped / closed -> never-opened.
   - does the Fundamentals (Ch.2), and especially its 2.5 "relevance/pressing-need" hinge, set up
     exactly the three questions Ch.3/4/5 answer?

4. MISSING OR WORTH-ADDING BEATS. Name the beats the story needs but lacks, and the specific
   unmotivated jumps. Probe the load-bearing bridges explicitly:
   - the MECHANISM: the arc claims "the representation, not the architecture, is the bottleneck" —
     is the reader ever shown WHY a place-level vector is the limit (the same POI, different
     visit, same vector), or is it asserted?
   - the pivotal jump: CoUrb diagnoses "enrich the representation" -> MobiWac answers with a
     CHECK-IN-LEVEL representation. Is "why check-in level, specifically" motivated, or does it
     arrive as a rabbit from a hat?
   - the STAKES: is the reader given a concrete reason to care (what a mobility-aware service can
     do with next category + next region) early enough to sustain a 100-page read?
   - the negative result: is CBIC's null framed so the reader accepts it as a finding worth
     building on, not a failure to explain away?
   Separate "MISSING (the story breaks without it)" from "WORTH ADDING (an enrichment)".

5. CRAFT, PACING, ENJOYMENT. Map the reader's momentum start to finish: where curiosity is
   created vs satisfied, where the narrative stalls or sags, where the "aha" of the resolution
   lands (or is muffled). Judge the PAGE-TURNER spine and the emotional shape of the honest arc
   (setup -> frustration -> insight -> payoff). Judge the ONE-VOICE seam: do three separately-
   written papers plus the new frame read as one author, or do the joins show?

6. HONESTY UNDER NARRATIVE PRESSURE. Flag every place where making the story cleaner or more
   dramatic would tempt a violation: a verb upgraded past its test ("matches" -> "beats", AZ
   upgraded), a superseded CBIC/CoUrb number allowed to read as current, a scope quietly widened,
   a mechanism asserted more strongly than the evidence licenses, or fake cohesion from templated
   transitions. The story must be compelling AND true; where they pull apart, truth wins and you
   name the tension.

7. PRIORITIZED, LOCATED RECOMMENDATIONS. Every recommendation: what to ADD / CUT / REORDER /
   REFRAME, WHERE (file + section/beat), WHY it strengthens the story, the rough cost (a sentence
   / a paragraph / a subsection / a reorder), and whether it introduces a NEW claim that needs
   author sign-off (mark [NEEDS SIGN-OFF]). Rank by narrative leverage per hour; note that in a
   coletânea the frame chapters (1, 2, 6) usually dominate the leverage.

OUTPUT CONTRACT
  A. The reconstructed arc + the diff against the intended spine (lens 1).
  B. The logline + the per-chapter earn-its-clause verdict (lens 2).
  C. The cohesion audit + the thread ledger (lens 3).
  D. The missing/worth-adding beats, split into MISSING vs WORTH-ADDING (lens 4).
  E. The craft/pacing/enjoyment read, including the momentum map and the one-voice seam verdict
     (lens 5).
  F. The honesty-under-narrative flags (lens 6).
  G. The ranked recommendation table (lens 7), each row located and cost-tagged.
  H. A PROTECT LIST: the parts of the story that already work and must not be diluted by edits.
  I. One honest closing paragraph answering the three questions directly: is it cohesive? is it
     enjoyable? is it well-crafted? — and the single highest-leverage narrative move to make next.

HARD LIMITS
Read-only assessment; propose narrative moves, never apply them and never draft replacement
chapter prose. Never fabricate a fact, number, citation, or claim; anything you assert about the
results traces to the sources named in the project instructions, and any external claim traces to
a page you opened this session (flag the rest [VERIFY]). Never recommend a change that upgrades a
claim beyond its statistical test or the whitelist — if a stronger story would require a stronger
claim, say so and stop. When a proposed connective sentence is a new frame claim, mark it
[NEEDS SIGN-OFF] for the author rather than writing it as settled.
```

---

## Notes for the author (not part of the prompt)

- **What is new here vs the personas.** Persona 17 scores excellence across ten dimensions and
  hunts missed connections; personas 01/15 catch stumbles and judge pleasure-to-read. This prompt
  makes Claude Science a **narrative specialist**: it reconstructs the arc from the artifacts,
  diffs it against `NORTH_STAR §6`, runs a **thread ledger** (every promise opened vs closed),
  and pressure-tests the two load-bearing bridges — *why representation dominates* and *why
  check-in-level specifically* — which are where a "very good" arc either becomes outstanding or
  stays a sequence of summaries.
- **It runs on plans, not final prose.** The frame chapters (1, 2, 6) are not fully drafted yet,
  so treat the output as a **blueprint review**: it tells you which beats to write and where the
  seams must be welded, before the drafting fleet expands `NORTH_STAR §6`. Re-run it on the
  compiled v1 for the craft/enjoyment lenses to bite on real prose.
- **Feed the output back through your gates.** Any beat it proposes adding is a claim: route new
  connective sentences through `AGENT_GUARDRAILS §3 (C2)` sign-off, and let
  `reviewers/07_claim_honesty_auditor.md` + `reviewers/14_adversarial_advisor.md` gate anything
  marked `[NEEDS SIGN-OFF]` before it enters the frame.
- **Same project, same rules.** Because the agent context from `science.md §3` is already
  installed, the honesty protocol, canonical names, and claim whitelist carry over automatically,
  so the prompt only re-tasks the session.
