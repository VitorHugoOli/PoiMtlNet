# 14 · Adversarial advisor — change-gate report (fix-loop run)

> Persona: the adversarial second signature (`reviewers/14_adversarial_advisor.md`).
> Two lenses on every item: **Lens 1 (the law)** — WRITING_LAW / GLOSSARY / AGENT_GUARDRAILS
> / NORTH_STAR decisions / settled author rulings; **Lens 2 (information loss)** — what a
> reader or examiner loses, and whether it is recoverable elsewhere.
> Verdicts: **APPROVE** / **APPROVE-WITH-EDIT** (exact corrected text supplied) / **VETO**
> (rule or unrecoverable loss named + legal alternative) / **NEEDS-AUTHOR** (surface a
> conflict; the author owns the law).
> Read-only. I gate; the applier applies. Written incrementally so a restart cannot lose it.

**Run scope (from the invocation):** the change-gate run for the fix loop. The proposed-edit
batch is (A) the MECHANICAL fixes already landed this session — recorded in `src/_gates/`
reports and the git `phase 6` commit (`49a67996`) — audited for whether any broke a rule or
lost a disclosure; and (B) the top recommended fixes still pending across the 15 landed persona
reports in `src/_review_v1/`, each ruled.

Built PDFs: `src/main_defense.pdf` (87pp), `src/main_final.pdf` (83pp). Sources:
`src/chapters/*.tex` + `src/0_main.tex`.

---

## GATE VERDICT (summary)

**Part A — the 6 mechanical fixes already LANDED (phase-6 commit `49a67996`): ALL 6 CERTIFIED.**
None broke a rule; none lost a disclosure. Re-derived firsthand: the D-1 margin fix moved the
bottom text margin from 1.52 cm (pre-fix, every full page) to a measured **1.92 cm minimum**
across all 64 full body pages (median 2.30 cm), page counts unchanged (87 / 83), 0 undefined
refs/cites in both logs, `check.sh` exits 0. Two residues that are NOT regressions are noted
under A.7 (30 benign `Overfull \vbox 14.5pt` warnings; the page-67 "only floats" layout warning
that the L4 pointer did not clear because the figure legitimately fills its own page).

**Part B — the top PENDING recommendations across the 15 persona reports: ruled below.**
2 items are **NEEDS-AUTHOR** (both pre-existing BLOCKERs: Ch.3 dataset placeholders; the Ch.5
CBIC misattribution, which is a claim-level departure from a paper still under review). 6 are
**APPROVE-WITH-EDIT** with exact text supplied (64.54→64.51 convention; Ch.2 `unlocks`; Ch.4
Gowalla mis-cite F-1; Ch.2 Song 93% scope; Ch.4 title `\:`; frame `percent` style). 1 is
**APPROVE-IN-PRINCIPLE + VETO-BLANKET** (the "at [dataset]" preposition campaign must not be
run mechanically — it collides with the verdict-scope law). 1 figure item is NEEDS-AUTHOR
(Portuguese labels in Figure 2, a regen not a text edit).

**Net size effect:** ≈ neutral. Every pending text fix is an in-place reword; the two landed
dedup edits slightly shortened. No page-budget risk (Fundamentals stays within its 8–12 pp;
document stable at 87/83).

**Interaction flags:** (I1) the pending 64.54→64.51 edit lands in the SAME Ch.6 paragraph the
landed C-1 idiom fix touched — apply without disturbing the C-1 wording; (I2) the A-1/A-2 dedup
pair did NOT jointly delete the weekday-lunch image (it survives once in Ch.1 L119-121, the
signed-off mechanism beat) — safe; (I3) the "at [dataset]" campaign and the region-verdict law
overlap — see V-1.

---

# PART A — the 6 LANDED mechanical fixes (audit for rule-break / disclosure-loss)

Source: git `49a67996` diff (chapters + `0_main.tex` + `check.sh`), each site read in context and
re-verified against the built PDFs.

## A.1 — D-1 layout fix (`0_main.tex`): re-derive the block under 1.5 spacing. **CERTIFIED.**

- **Lens 1 (law):** UFV_COMPLIANCE §7 mandates 3 cm / 2 cm margins AND 1.5 line spacing; the fix
  serves both. It sets `\OnehalfSpacing` before `\setul/lrmarginsandblock{3cm}{2cm}{*}` +
  `\checkandfixthelayout[fixed]`, so the block is fixed under the spacing the manual mandates
  rather than under single spacing. No rule broken; this is the corrective the D-1 finding asked
  for. Note: `\OnehalfSpacing` is now issued twice (`0_main.tex:35` in the fix block and `:144`
  in the body) — harmless (idempotent), but the applier may drop the `:144` duplicate for
  tidiness (NIT, not required).
- **Lens 2 (loss):** nothing lost. Re-measured firsthand on the current `main_defense.pdf` at
  200 dpi across ALL 64 full body pages: bottom margin **min 1.92 cm, median 2.30 cm, max 3.44
  cm**; 0 pages below 1.70 cm (pre-fix: 1.52 cm on every full page). Left 2.96–3.00 cm, right
  1.96–2.02 cm, top unchanged. Page counts **unchanged (87 / 83)** — the fix did not push content
  onto new pages. **1.92 cm is within measurement tolerance of the 2 cm nominal** (ink bounding
  box + last-baseline-to-edge, ±0.05 cm; the glyph descender sits a few pt above nominal). Verdict
  APPROVE. (If the author wants dead-on 2 cm, add ~1.5 mm `\textheight` trim, but this is now a
  NIT, no longer the MAJOR D-1 was.)

## A.2 — Ch.1 L76 dedup (A-1): paper-chapter phrase reworded out of the frame. **CERTIFIED.**

- Landed text: *"a single model to maintain, and a single forward pass that returns both
  predictions together, instead of two dedicated single-task models running side by side."*
- **Lens 1:** claim unchanged (one artifact, one forward pass, two predictions); canonical names
  intact ("dedicated single-task models"); no banned idiom introduced; carries a `[NEEDS
  SIGN-OFF]` comment naming the revert path. The L3 defect (frame echoing Ch.5's version-of-record
  wording) is removed and the paper chapter keeps its wording — correct direction (the paper is
  the record and cannot move). **Crucially, it respects the F3 guard (NORTH_STAR §6 beat 2): it
  does NOT promise lower compute cost** — "a single model to maintain" is operational, not a cost
  claim. Clean.
- **Lens 2:** no disclosure lost; the operational-simplicity point is fully preserved. APPROVE.

## A.3 — Ch.2 L502 dedup (A-2): "weekday lunch / Saturday night out" image de-duplicated. **CERTIFIED.**

- Landed text: *"A vector that stays the same across visits carries nothing about the visit being
  predicted, and the check-in level is the response to that limit."*
- **Lens 1:** states the same static-vector limitation without the image; no rule broken.
- **Lens 2 (the interaction that matters):** I verified the image is NOT orphaned — it survives
  once, in Ch.1 L119-121 (*"a representation that cannot tell a weekday lunch from a Saturday
  night out at the same place is working against both tasks at once"*), which is the **signed-off
  mechanism beat** (NORTH_STAR §6 Ch.1 beat 4(e)). So the A-1 and A-2 edits together did NOT
  delete both copies of a load-bearing image (the classic dedup trap this persona exists to
  catch). The hinge role of §2.5 is preserved. APPROVE.

## A.4 — Ch.5 L420 (B-1/L4): pointer sentence added for the restored embquality figure. **CERTIFIED.**

- Landed text: *"Figure~\ref{fig:mobiwac:embquality} shows the same separability contrast
  graphically."* placed inside the restored-block, above the float.
- **Lens 1:** resolves the L4 defect (a float must be referenced in prose); the new `\ref`
  compiles with **0 undefined references** (verified in both logs). The comment correctly binds
  the pointer's lifetime to the restored block ("leaves with it") so a later revert-to-submitted
  removes both together — good hygiene.
- **Lens 2:** adds a reference, loses nothing. Note the "Text page 67 contains only floats"
  warning PERSISTS (verified: page 67 is the figure on its own page). That warning is about
  page-breaking, not about the missing `\ref`; the fix correctly targeted the `\ref` defect. The
  residual warning is benign (a full-page figure). APPROVE. (D-2 in the style report predicted the
  warning would "fix itself"; it did not, because the figure legitimately fills the page — this is
  cosmetic, not a defect.)

## A.5 — Ch.6 L80 (C-1): "buys nothing"→"yields nothing"; "the win lives in"→"the gain resides in". **CERTIFIED.**

- **Lens 1:** removes two WRITING_LAW §4 idiom-law violations (money-metaphor "buys"; phrasal
  metaphor "the win lives in"; and "win" as a result noun brushing the §3 banned verdict-verb
  family). "yields" and "the gain resides in" are within the safe-verb register. Claim strength
  unchanged (parameter count alone recovers nothing; the effect is in the shared trunk). Correct.
- **Lens 2:** no meaning lost. APPROVE. **Interaction (I1):** the pending 64.54→64.51 fix (B.2
  below) lands in THIS SAME paragraph — the applier must change only the numeral and leave this
  C-1 wording intact.

## A.6 — Ch.6 L126 (C-1): "the size of the win"→"the size of the improvement". **CERTIFIED.**

- **Lens 1:** same idiom-law cleanup ("win" result-noun) inside limitation 6 (the task-pair
  confound concession, a signed-off addition NORTH_STAR §6 Ch.6). "improvement" is neutral and
  preserves the concession's hedged force ("a possible contributor to the size of the
  improvement"). No weakening of the confound disclosure — the whole limitation sentence is
  intact. APPROVE.
- **Lens 2:** the concession (no single ablation separates representation+topology from task-pair
  homogeneity) is fully preserved. Nothing lost.

## A.7 — `check.sh` lint hardening. **CERTIFIED.**

- **Lens 1:** three changes, all correct: (1) em-dash grep now uses an explicit UTF-8 byte
  sequence `\xe2\x80\x94` with a proper comment-line exclusion `^[^:]*:[0-9]*: *%` (the old
  `$'\u2014'` was shell-fragile and the old `^\s*%` filter missed `file:line:  %` comment
  form); (2) the banned-word check now excludes `apx_b_errata` (which legitimately QUOTES the
  published banned words in its wording-substitution table — a real exemption, not a loophole);
  (3) all filters use the tightened comment pattern. I RAN it: **exits 0**, all checks OK. The
  "Pareto/wins" line it prints is a non-failing review sweep; its two hits are Ch.3 published
  method text (MGDA/Nash-MTL "Pareto") and apx_b quoting the published "wins" text — both legal.
- **Lens 2:** the apx_b exemption does not hide anything — apx_b's function IS to quote the
  published defects; the banned words there are the evidence, not a violation. No disclosure
  suppressed. APPROVE.

### A — verdict: all 6 landed fixes hold. No rule broken, no disclosure lost, no interaction damage.

---
# PART B — top PENDING recommendations, gated

Each item: the source finding, my two-lens read, and the verdict. APPROVE-WITH-EDIT items carry
**exact final text**, ready to apply verbatim. Ordered by severity.

## B.1 — Ch.5 CBIC misattribution (MTL-expert BLOCKER Finding 1). **VERDICT: NEEDS-AUTHOR.**

**Re-derived firsthand (not echoed):** Ch.3 L37-38 lists CBIC's two tasks as *POI Category
Classification* (static) + *Next-POI Prediction* (= next category); **there is no region task in
Ch.3.** Ch.3 L358-360 **hypothesizes** "Subtle Negative Transfer" ("We hypothesize"), and L356
reports the result as *"did not consistently demonstrate superior performance"* — a parity null,
not an observed negative transfer. Yet:
- Ch.5 L44: *"Prior work observed exactly this for next-category and next-region~\cite{silva2025mtlnet}"*
  (the "this" = one task helped while the other is hurt).
- Ch.5 L140: *"Our earlier work~\cite{silva2025mtlnet} established this two-task setup and observed
  negative transfer (sharing hurt one task)"*.

Both are false on two counts (region task; "observed" vs "hypothesized"), and both self-contradict
Ch.5 L58 (*"the first work to treat fine-grained region as an end target"*) and the correct
framing stated four other places (Ch.1 §1.2, Ch.3, Ch.4 recap, Ch.6 limitation 6). This is a real
banca kill-shot and it **violates signed-off addition NORTH_STAR §6(a)** ("the task pair evolved
... named plainly, never narrated as one experiment on a constant pair").

**Why NEEDS-AUTHOR, not APPROVE-WITH-EDIT:** the offending sentences are **inherited verbatim from
the version of record** (`articles/[mobiwac]/src/sections/01_introduction.tex` L17 and
`02_related.tex` L48-49) — a paper still **submitted / under review**. Rewriting them is a
claim-level departure from a published-of-record source, which routes through the errata policy
(NORTH_STAR §4/§5.7 → ERRATA.md + Appendix B) AND touches the MobiWac claim whitelist
(AGENT_GUARDRAILS C1). Per my hard limits, I cannot silently rewrite a claim the author has not
ruled on; I surface it. **This is the batch's single most important item — it should not ship to
the advisor unresolved.**

**Proposed legal text (for the author to approve, NOT to apply unilaterally),** repairing both the
task-pair and the observed/hypothesized errors while staying inside what Ch.3 actually supports:

- L44 (repair): *"Sharing can converge to a compromise optimal for neither task, helping one while
  hurting the other~\cite{caruana1997multitask}. Our earlier work reported no consistent
  multi-task advantage for the paired category tasks and attributed it, in part, to this
  effect~\cite{silva2025mtlnet}; the useful question is where sharing helps, what it costs, and how
  to share so the gains hold and the cost stays small."*
- L140 (repair): *"Our earlier work~\cite{silva2025mtlnet} paired static category classification
  with next-category prediction and found no consistent multi-task gain; this chapter introduces
  the next-region task and the check-in-level representation, on which sharing helps instead of
  hurting (Section~\ref{sec:mobiwac:results-part2})."*

Both keep the arc's honesty (the null is genuine, of its time) and remove the region-task error.
If approved: log in `articles/[mobiwac]/ERRATA.md` + Appendix B. **Escalation, not obedience: the
author must rule because it edits an under-review paper's claims.**

## B.2 — Ch.6 64.54 vs Ch.5 64.51 convention blur (MTL Finding 2 / Number N-2 / Banca A-1). **VERDICT: APPROVE-WITH-EDIT.**

**Re-derived firsthand:** Ch.5 Table 3 (5_mobiwac.tex:479) reports AL joint next-category =
**\textbf{64.51}\sd{0.09}** — the **joint-best** value, and the dissertation's reported convention
(author ruling 2026-07-18; `JOINT_BEST_RESULTS.md` L32: `AL | 56.82 | 64.54 diag | 64.51 jb`).
Ch.6 L78 uses **64.54** (the diagnostic-best value, from
`storyline/audit/capacity_baseline_experiment.md` L92/L113). Same cell, two conventions across
chapters — exactly the joint-best/diagnostic-best blur AGENT_GUARDRAILS N5 forbids.

- **Lens 1:** N5 (never blur joint-best vs diagnostic-best); the flagship table is the reference,
  so Ch.6 must match it. Verdict-neutral (capacity gap is +7.72 at 64.54 or +8.35 at 64.51; no
  conclusion moves), so this is a consistency fix, not a result change.
- **Lens 2:** I checked the OTHER two numerals in the sentence — 56.16 (capacity-baseline best
  arm) and 56.82 (dedicated ceiling) — these are capacity-experiment quantities that exist only at
  one convention; only the **joint** value has the dual-convention ambiguity. So editing just the
  64.54→64.51 does NOT create a new intra-paragraph mismatch (the 56.x numbers are not joint-model
  cells). Safe.
- **Exact edit** (6_conclusion.tex:78), preserving the landed C-1 wording around it:
  - FROM: `dedicated model at its own tuned width and 64.54 for the joint model.`
  - TO:   `dedicated model at its own tuned width and 64.51 for the joint model.`
- **Interaction (I1):** this is the same paragraph as landed fix A.5 — change only the numeral;
  do not touch "yields nothing" / "the gain resides in". **Number auditor (06) should re-run the
  N4 numeral sweep after this one-character change** to confirm the capacity gap prose (if any
  states a delta) still traces. (No delta is stated in prose here — the gap is left implicit — so
  no cascade.)

## B.3 — Ch.2 L532 hard-banned "unlocks" in the climax sentence (Style-auditor TOP-2). **VERDICT: APPROVE-WITH-EDIT.**

**Re-derived firsthand:** `2_fundamentals.tex:532` *"what a representation built for check-ins
unlocks for a redesigned joint model"*. **"unlock" IS on the inherited MobiWac GLOSSARY §8 ban
list** (line 222: `leverage, harness, unlock, foster ... → use, apply, enable, obtain`), which
WRITING_LAW §4 inherits wholesale. This is frame prose (full-force zone) and it is the chapter's
hinge sentence, so the tell is maximally visible.

- **Lens 1:** clear §4 idiom/AI-tell violation.
- **Lens 2:** the sentence must keep its forward-looking force (what the check-in representation
  makes possible for the joint model). "enables ... for" preserves it exactly.
- **Exact edit** (2_fundamentals.tex:531-532):
  - FROM: `It finally asks what a representation\nbuilt for check-ins unlocks for a redesigned joint model, one that, by paired`
  - TO:   `It finally asks what a representation\nbuilt for check-ins enables in a redesigned joint model, one that, by paired`
  - ("enables in" reads better than "enables for"; the glossary maps unlock→enable. Claim
    unchanged.)

## B.4 — Ch.4 L226 Gowalla dataset mis-cite (Citation F-1). **VERDICT: APPROVE-WITH-EDIT.**

**Re-derived firsthand:** `4_courb.tex:226` cites `\cite{liu2014geographical}` for "the Gowalla
dataset". That key is Liu et al., CIKM 2014, a location-recommendation *method* paper, not the
dataset source. The canonical dataset citations already appear correctly in the SAME chapter
(`cho2011gowalla` at L18, `jure2014snap` at L33), so L226 is both a mis-source and internally
inconsistent. Inherited from the CoUrb original (`src_en/sections/results.tex:7`).

- **Lens 1:** AGENT_GUARDRAILS R2 (attribute fidelity / cite the right source). It is inherited
  published text, so under the errata policy it is a fix-in-dissertation + Appendix B item (author
  ruling 2026-07-21 already makes such fixes silent in the text + listed once in Appendix B — this
  fits that standing ruling, so it does NOT need fresh author sign-off; it needs the Appendix B
  line).
- **Lens 2:** I verified `liu2014geographical` is cited ONLY at L226, so replacing it orphans the
  key — benign in a numeric single-list bib (an uncited entry simply is not numbered), but the
  applier should DROP the now-unused entry from `references.bib` to keep the list clean (or leave
  it; harmless). No information lost — the dataset is still cited, correctly.
- **Exact edit** (4_courb.tex:226), matching the chapter's own L18 usage:
  - FROM: `conducted with the Gowalla \textit{dataset} \cite{liu2014geographical} in the states`
  - TO:   `conducted with the Gowalla \textit{dataset} \cite{cho2011gowalla,jure2014snap} in the states`
  - Then remove `liu2014geographical` from `references.bib` if it is now uncited anywhere.
  - Appendix B: add one line ("CoUrb chapter: Gowalla dataset citation corrected from a
    recommendation-method paper to the dataset sources").

## B.5 — Ch.2 §2.1 Song 93% presented as a universal "ceiling" (POI-expert MAJOR M1/M2). **VERDICT: APPROVE-WITH-EDIT.**

**Re-derived firsthand:** §2.1 (L35-37) says *"a potential predictability of about 93\% on where
an individual goes next~\cite{song2010limits}. That ceiling is the reference point against which
any predictive model should be read."* §2.4 (L428-433) then correctly scopes it: *"it is not,
however, a ceiling on seven-class category macro-F1 or on region ranking ... The operative ceiling
... is the dedicated single-task model."* The §2.1 wording ("any predictive model") overstates a
next-location bound as a ceiling for tasks with different label spaces; §2.4 disowns it. The two
passages are ~400 lines apart, so a reader hits the universal claim first.

- **Lens 1:** WRITING_LAW §3 (scope every universal; every number carries its correct reference
  point). "any predictive model" is an unscoped universal contradicted later in the same chapter.
  This is a genuine honesty-law item, not a nit.
- **Lens 2 — the trap to avoid:** do NOT delete the 93% figure or its motivational role (it frames
  *why mobility is learnable at all*, a real and useful point that §2.4 also relies on). The fix
  is to SCOPE the §2.1 claim to next-location and point forward, not to cut it. Deleting it would
  lose the "learnable at all" motivation.
- **Exact edit** (2_fundamentals.tex:35-37):
  - FROM: `93\% on where an individual goes next \cite{song2010limits}. That ceiling is the\nreference point against which any predictive model should be read. A learned model\nthat trails it by a wide margin has room to improve; one that approaches it is near\nthe limit the data allows.`
  - TO:   `93\% on where an individual goes next \cite{song2010limits}. This bound is specific to
    next-location prediction at coarse spatial resolution; it shows that mobility is far from
    random and is learnable at all, and Section~\ref{sec:fund:eval} states the reference points
    that actually bound the category and region tasks studied here.`
  - This keeps the figure and its "learnable at all" role, removes the false universal ceiling,
    and forward-references §2.4 (`sec:fund:eval`, verified as the correct label) where the true
    reference points (majority-class floor, Markov floor, dedicated ceiling) live. No number lost;
    scope corrected; consistent with §2.4.

## B.6 — Ch.4 chapter title `\:` renders as thin space not colon (Line-editor TOP-2 / N4 out-of-scope). **VERDICT: APPROVE-WITH-EDIT.**

**Re-derived firsthand:** `4_courb.tex:8`:
`\chapter{Article 2: ST-MTLNet\: Spatio-Temporal Point-of-Interest Representations for Multi-Task
Learning}`. The `\:` is a math-mode medium-space command; in a title it prints a thin space where
a colon is intended ("ST-MTLNet Spatio-Temporal..." with an odd gap). The first `Article 2:` uses
a correct literal colon, so the intent is unambiguous.

- **Lens 1:** not a law item per se, but a visible typographic defect in a heading; the fix is
  mechanical and unambiguous.
- **Lens 2:** nothing lost; purely corrective.
- **Exact edit** (4_courb.tex:8):
  - FROM: `\chapter{Article 2: ST-MTLNet\: Spatio-Temporal`
  - TO:   `\chapter{Article 2: ST-MTLNet: Spatio-Temporal`
  - Applier confidence (verified firsthand): a literal `:` in a `\chapter{}` title is already
    proven safe in this exact template — **Ch.3 L8 (`Article 1: An Investigation...`) and Ch.5 L14
    (`Article 3: Predicting...: A Check-in-Level...`, two literal colons) compile cleanly**, and
    babel's main language is `english` (`\selectlanguage{english}`, `0_main.tex:72`; `brazil` is
    loaded as a secondary option only, and neither makes the colon active). Ch.4's `\:` is the
    lone outlier. Rebuild and eyeball the running header + TOC after applying.

## B.7 — Frame number style: "93 percent" vs "93\%" (Line-editor TOP-3). **VERDICT: APPROVE-WITH-EDIT (Ch.2 is the outlier).**

**Re-derived firsthand:** the SAME 93% figure prints as **"93 percent"** in Ch.1 L38 and as
**"93\%"** in Ch.2 L35 and L429. Ch.5 uses the spelled form consistently ("25 percent", "34
percent", "67 to 87 percent", "13 to 27 points"...). So the document convention is **spelled-out
"percent"**, and **Ch.2's two "93\%" are the outliers**, not Ch.1.

- **Lens 1:** WRITING_LAW §1 asks for consistency (digits for data quantities, but the running
  choice for percentages here is the spelled word, set by Ch.1 and Ch.5). Internal consistency is
  the law; the majority form wins.
- **Lens 2:** nothing lost; purely stylistic alignment.
- **Exact edits** (align Ch.2 to the document's spelled-out form):
  - `2_fundamentals.tex:35`: `93\%` → `93 percent`
  - `2_fundamentals.tex:429`: `93\%` → `93 percent`
  - (Do NOT touch Ch.5's `\%`-free prose or any table cell; tables and math stay numeric. Only
    these two body-prose instances are inconsistent with the rest of the frame.)
  - Interaction: B.5 rewrites L35's sentence entirely — if B.5 is applied, its replacement already
    reads "93\%"; apply the "percent" spelling INSIDE the B.5 replacement text too (i.e. the B.5
    final text should read "a potential predictability of about 93 percent"). L429 is independent.

## B.8 — "at [dataset/state]" as the reported-performance preposition (Line-editor TOP-1, systematic). **VERDICT: APPROVE-IN-PRINCIPLE, but VETO any blanket/mechanical find-replace.**

The line editor flags "at Alabama", "at four of six datasets", "at Istanbul" etc. as non-idiomatic
and inconsistent, proposing a systematic sweep to "on"/"for". **I APPROVE fixing genuine
readability instances but VETO running this as a mechanical campaign,** for one reason grounded in
the law:

- **Lens 1 — the collision:** the region-verdict law (WRITING_LAW §3; GLOSSARY §4) fixes exact
  verdict phrasings: *"outperforms ... at four of six datasets"*, *"matches ... at AL/AZ"*. These
  are **law-mandated verbatim scopes**, echoed identically in the Abstract, Resumo, Ch.2 §2.5,
  Ch.5, and Ch.6 (the L3/style report confirmed the parity). A blanket "at→on/for" replace would
  (a) desynchronize the Abstract↔Resumo↔body verdict wording that persona 03/08 certified as
  clause-for-clause parallel, and (b) risk changing a scope preposition inside a frozen verdict
  sentence. That is exactly the "compression that drops/alters a scope qualifier" trap in my
  known-trap list.
- **Lens 2:** the "at four of six datasets" / "at AL/AZ" phrasings carry the verdict SCOPE; they
  are not free stylistic choices. Their information (which datasets, how many) is load-bearing.
- **Legal alternative:** the line editor may fix ONLY the non-verdict, non-parallel instances
  (e.g. a stray "the model performs well at Texas" in descriptive prose), leaving every
  verdict-bearing "at N of six" / "at AL/AZ" / "at Istanbul/FL/TX/CA" UNTOUCHED, and must re-run
  the Abstract↔Resumo↔body parity check (persona 03/08) after any change. **This item goes back to
  the author/line-editor with that constraint; it is not a clean APPROVE.** If the author wants the
  verdict preposition itself changed, that is a whitelist-wording decision (AGENT_GUARDRAILS C1),
  changed everywhere at once or nowhere — NEEDS-AUTHOR for that sub-part.

---

# PENDING items I did NOT rule (correctly out of this gate's scope)

These are pre-existing BLOCKERs / regen tasks that are NOT "proposed edit texts" — they are author
decisions or asset regenerations, so they are NEEDS-AUTHOR by nature, not gateable edits:

- **Ch.3 dataset placeholders** (`3_cbic.tex:235`, the `[$N_{users}$; VERIFY]` triplet rendered in
  the built PDF — N4-1 / Style BLOCKER-1 / POI BLOCKER-B1 / Banca A-4 / Cold-reader ②).
  **NEEDS-AUTHOR:** the sanctioned fix is a repo-committed recompute over the CBIC-era Florida
  pipeline, author-approved before insertion (NORTH_STAR §4). This is not an edit I can supply
  text for — no number may be invented. It is the document's top pre-existing blocker; it must be
  resolved (run the script, or the author explicitly accepts visible placeholders for the advisor
  draft) before the advisor build. Flagged, not gated.
- **Dissertation title** still `[TITLE — ...]` on cover/Resumo/Abstract (Banca A-2, Cold-reader ③)
  — open decision NORTH_STAR §5.8; author-only.
- **Front-matter placeholders** (approval sheet, banca, date) — open decisions; author-only. NOTE
  the C-4 trap from the style report: these placeholders currently contain `---` that DO render as
  em-dashes in the PDF; when the real values land, confirm em-dash count returns to 0 (the
  chapter prose is already clean). A cheap hardening: switch the placeholder separators to colons
  now. (Recommend, not gate.)
- **Figure 2 Portuguese labels** (Visual-18 MAJOR-1) and **Figure 3 color-only Food/Shopping
  encoding** (Visual-18 MAJOR-2) — these are figure REGENERATIONS (matplotlib re-render with EN
  labels / hatch dual-encoding per WRITING_LAW §5), not text edits. NEEDS-AUTHOR/asset work; I
  cannot supply "exact replacement text" for a figure. Both are real and should be done before the
  advisor build; neither is in this gate's edit-text scope.
- **Ch.4 italicized loanwords** ("embedding"/"folds"/"dataset" in `\textit{}`, Readability-15
  MAJOR / Translation-08 handoff) — a fidelity-vs-style call on translated published prose (L5
  zone). NEEDS-AUTHOR: if the PT original italicized them, fidelity (L5) may require keeping them;
  a blanket de-italic is a style choice the author/persona-08 must rule. Not a clean gate item.

---

# INTERACTION FINDINGS (cross-item)

- **I1 (numeral vs idiom, same paragraph):** B.2 (64.54→64.51) edits the same Ch.6 paragraph that
  landed fix A.5 (C-1 idiom) already touched. Apply B.2 as a single-numeral change; do not
  re-touch the A.5 wording. No conflict, but same-site — sequence them in one edit pass.
- **I2 (dedup pair, image survival):** A.2 (Ch.1) + A.3 (Ch.2) are the two dedup edits. Verified
  they did NOT jointly orphan the "weekday lunch/Saturday night" image (survives in Ch.1 L119-121,
  the signed-off beat). No further action — recorded because this is the exact trap (delete both
  copies of a disclosure/motif) the persona guards.
- **I3 (preposition campaign vs verdict law):** B.8's "at→on/for" sweep overlaps the frozen
  verdict scopes. See B.8 VETO-blanket. The two must not be run together mechanically.
- **I4 (B.5 vs B.7, same line):** both touch 2_fundamentals.tex:35. If B.5 is applied, fold the
  B.7 "percent" spelling into the B.5 replacement text (do not apply B.7 to L35 separately). L429
  (B.7) is independent.
- **I5 (F-1 orphan):** B.4 removes the only use of `liu2014geographical`; the applier should drop
  the bib entry to avoid an uncited-but-present record (benign either way in numeric style).

---

# NET-SIZE ESTIMATE vs the page budget

- Active budget: Fundamentals is spec'd thin (8–12 pp, NORTH_STAR §5.6); document is 87 pp
  defense / 83 pp final, both stable post-phase-6.
- Every Part B text edit is an in-place reword or a single-token change; none adds or removes a
  paragraph. B.1 (if the author approves it) slightly SHORTENS two Ch.5 sentences. B.5 replaces
  four short sentences with two (slightly shorter). **Net effect: neutral-to-slightly-shorter.**
  No page-budget risk; re-confirm 87/83 after application (the margin fix already proved the
  layout has ~0.4 cm/page of reclaimed vertical space, so minor prose shortening will not reflow
  chapter boundaries).

---

# FINAL APPROVED LIST (ready to apply verbatim)

Apply in this order; rebuild + `check.sh` + persona-06 numeral re-sweep after.

1. **B.3** — `2_fundamentals.tex:531-532` — `unlocks for` → `enables in` (exact text in B.3).
2. **B.5** — `2_fundamentals.tex:35-37` — Song ceiling scoped to next-location + forward-ref to
   `sec:fund:eval` (exact text in B.5; write "93 percent" per I4).
3. **B.7** — `2_fundamentals.tex:429` — `93\%` → `93 percent` (L35 handled inside B.5 per I4).
4. **B.4** — `4_courb.tex:226` — `\cite{liu2014geographical}` → `\cite{cho2011gowalla,jure2014snap}`;
   drop the now-uncited bib entry; add the Appendix B line.
5. **B.6** — `4_courb.tex:8` — `ST-MTLNet\:` → `ST-MTLNet:` in the chapter title.
6. **B.2** — `6_conclusion.tex:78` — `64.54` → `64.51` (single numeral; leave A.5 wording intact).

**Held for the AUTHOR before the advisor build (NOT applied, NOT gateable as edit-text):**
- **B.1** (Ch.5 CBIC misattribution) — the batch's most important item; proposed repair text in
  B.1 for the author to approve, then ERRATA.md + Appendix B. Do not ship to advisor unresolved.
- **B.8** (preposition campaign) — approve targeted fixes only; VETO blanket; re-run parity check.
- Ch.3 dataset placeholders; title; front-matter placeholders; Figure 2/3 regen; Ch.4 loanword
  italics — pre-existing author decisions / asset work, listed above.

_End of gate report. Read-only pass; I gated, I applied nothing._
