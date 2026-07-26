# 01 · Cold reader — first-pass comprehension review

**Read:** `src/dissertacao.pdf` (94 pp), rendered pages, front matter through Appendix.
**Date:** 2026-07-26. **Persona:** `reviewers/01_cold_reader.md`.

## Method disclosure — read this first

This persona's instruction is explicit: *"Do NOT read NORTH_STAR, the papers, or the repo docs:
fresh eyes are the point,"* and *"a cold reader who has seen the text before is no longer cold."*

**I am not eligible to be this reader.** By the time this pass came up in the suite I had read
`CLAUDE.md`, `WRITING_LAW.md`, `GLOSSARY.md`, `AGENT_GUARDRAILS.md`, the reviewer protocol, and — in
the course of the other eight passes — the sources of truth, the statistical protocol, and every
chapter source. Running the stumble log myself would have produced a fiction: I would have "not
stumbled" over things only because I already knew the answers.

**What I did instead.** I dispatched the first pass to seven genuinely uncontaminated readers: seven
independent model instances, each given the cold-reader brief and **one span of the rendered PDF
text only** — no project documents, no other chapters, no prior reports, no knowledge of the
research. Each logged its stumbles blind. I then **verified every quoted stumble against the
document** before admitting it to this report, and discarded the ones that were wrong.

This is a deviation from the persona's letter and I am flagging it rather than concealing it. Two
consequences the author should weigh:

- The readers are not human, and their friction is not identical to a banca member's. Treat the log
  as *evidence of where the text is hard*, not as proof it will trip a specific examiner.
- Because I verified each one, three claimed stumbles turned out to be **reader error, not text
  defect**. Those are listed separately below rather than passed through — reporting them as
  findings would waste the author's time, which this persona's hard limits forbid.

Spans read: front matter + abstract (pp. 3–6); Ch.1 (13–19); Ch.2 (20–28); Ch.3 (29–42);
Ch.4 (43–57); Ch.5 (58–79); Ch.6 (74–79).

## Verdict

**The document is followable.** Every reader reached the end of its span understanding what was
claimed. Not one reported losing the argument outright. The friction is concentrated, and it
concentrates in three places: the abstract's task vocabulary, the four-grounds paragraph, and the
places where a correction was added inside an existing sentence rather than after it.

**Top 3 stumbles: C-01, C-02, C-03.**

---

## The stumble log

### ★ C-01 · The abstract uses a task name the reader has not met, and Figure 1 uses two more

**Where:** p. 5 (Abstract) and p. 6 (List of Figures). Reported independently by the front-matter
reader.

The abstract's task vocabulary is *category* and *region* throughout, until:

> "It is also the study that introduced the next-region task; the first two paired category
> classification with next-category prediction, so the task pair itself changed across the
> collection."

**The stumble, in the reader's words:** "Suddenly a third term, 'category classification,' appears
as distinct from 'next-category prediction,' with no explanation of how these two differ. I couldn't
tell whether 'category classification' is a third task, a synonym, or what got replaced."

Then the List of Figures, on the facing page, supplies a fourth and fifth label:

> "Figure 1 – The proposed Multi-Task Learning (MTL) architecture. Task-specific encoders process
> inputs for POI Category Classification and **Next-POI Prediction**."

**The reader's second stumble:** "Now a *fourth* task label appears — 'Next-POI Prediction' — never
used in the abstract at all... I can't tell if Figure 1 depicts Study 1, Study 2, or something
separate from all three studies."

**Verified.** Both quotes are exact. And the distinction *is* explained — on **p. 13**, eight pages
later: "A fourth task also appears in this dissertation: the first two studies paired next category
prediction with the static classification of a place's category, category classification, and
Section 1.2 explains why the final study replaced it." Chapter 3's preface (p. 29) separately
explains that the published "Next-POI Prediction" means the frame's next category.

So the document handles both terms well — just after the reader has already met them. The abstract
and the List of Figures are read first and gloss neither.

**Why it breaks:** a reader's model of "how many tasks are there?" is built in the abstract. Meeting
four labels for three tasks there, with the reconciliation eight pages downstream, means the reader
spends Chapter 1 unsure whether they have miscounted.

**Suggested direction:** three or four words in the abstract clause ("category classification, a
static task the third study drops") would close it. The Figure 1 caption is reproduced from
Chapter 3 and may be constrained by the errata policy.

### ★ C-02 · The four-grounds paragraph loses the reader partway through

**Where:** pp. 65–66, `5_mobiwac.tex:367`. Reported independently by **two** readers (Ch.5 reader
and, in the equivalent passage, the Ch.2 reader on a different long block).

> "The 'Integrity of the representation' paragraph is a wall of four numbered 'grounds' before you
> learn what's actually being defended against... by the time you reach 'Fourth, we probe the
> channel...' — several hundred words later — it's easy to lose track of which of the four grounds
> you're on and why. **I had to scroll back to re-anchor.**"

And on the audit inside it:

> "The paragraph... buries its actual finding (an attention-based encoder leaked and was disqualified)
> three sentences after introducing the ceiling metric — by the time the number 0.4976 appears, I'd
> half-forgotten what 'the ceiling' meant."

**Verified.** The paragraph is a single unbroken block of **546 words** — by a wide margin the
longest in the document (next longest: 330). The topic sentence is good and the reader said so; the
problem is that nothing after it is navigable.

This is the passage that defends the work against the leakage accusation, so a reader losing the
thread here is expensive.

**Suggested direction:** paragraph breaks at the ground boundaries. No words need changing.
(Persona 15 reached the same conclusion from the editor's side; persona 09 independently rates the
paragraph's *content* as the round's strongest work. The content is not the problem.)

### ★ C-03 · Four `(??)` markers stop the reader hard

**Where:** pp. 21, 45, 49, 50.

> "**'(??)'** — literal double-question-mark citation: *'A related encoder uses spherical harmonics
> together with sinusoidal representation networks... (??)'*. This is a broken reference... and it
> **stopped me hard** — I assumed a missing citation number... but it reads like an error was never
> caught."

**Verified**, all four. This is the only defect in the document that a reader described as stopping
them. It is also the cheapest to fix. (Personas 05 and 18 own the diagnosis and the fix.)

### C-04 · The 93 percent figure is given weight, then walked back

**Where:** p. 13 (Ch.1) and p. 21 (Ch.2).

> "Section 1.1 says an entropy analysis found 'potential predictability of an individual's next
> location at about 93 percent,' presented as motivating context. Then in 2.1 it's repeated, and
> immediately hedged... Reads as if the number was used for motivational weight in Ch1, then
> partially disowned in Ch2 as not actually applicable to the tasks studied — **the walk-back is
> subtle enough that a skim would miss it and just retain '93%.'**"

**Verified.** Chapter 2 reads: "This bound is specific to next-location prediction at coarse spatial
resolution; it shows that mobility is far from random and is learnable at all... it is not, however,
a ceiling on seven-class category macro-F1 or on region ranking, which are different label spaces."

The hedge is correct, thorough, and exactly right. The reader's point is about *sequence*: the
number lands as motivation in Chapter 1 and is qualified in Chapter 2, so a reader who skims
Chapter 2 carries an unqualified 93% into the results.

**Suggested direction:** a short qualifier at the Chapter 1 first use, so the number never travels
unaccompanied.

### C-05 · Corrections added inside sentences make those sentences hard

Three readers independently flagged sentences that this round lengthened:

- **Ch.6's opening** (p. 74): "one sentence carrying three separate comparative claims with different
  quantifiers ('all six,' 'four of the six,' 'the other two'); I had to parse it twice to keep the
  three groups straight." (110 words — the longest sentence in the document.)
- **Ch.1's contributions** (p. 19): "This is an important limitation, but it's tacked onto the end of
  the 'contributions' list rather than flagged where the twenty-models claim is first made." The
  reader is describing the fixed-partition caveat, added this round.
- **Ch.6's fairness detail** (p. 75): "'One detail supports the fairness of the comparison: at both
  datasets the widened model's best training configuration uses a lower learning rate than the
  narrow width's, so the sweep found the wide model's own better setting rather than rerunning the
  narrow model's.' — dense, **had to re-read twice**."

**All three verified.** Each correction is substantively right; each was inserted into the existing
sentence. The pattern is worth naming because it will recur in the next correction round.

### C-06 · Chapter 6's numbers need a table

**Where:** p. 75.

> "The Alabama widened-model paragraph packs five numbers in two sentences (4.2M vs 0.6M params;
> 56.16 vs 56.82 vs 64.51 macro-F1) with no table — I had to reread to keep track of which number
> belonged to 'widened,' 'dedicated at own width,' and 'joint.'"

and

> "'69.88... standard deviation 0.26... against 70.60, standard deviation 0.07' — then '0.72 points
> below at California and 0.66 at Alabama' requires **recomputing/cross-checking** against the
> numbers just given rather than being stated as an obvious readout."

**Verified.** The capacity-matched paragraph carries eleven numbers in prose. The reader is not
disputing them; they cannot hold them.

**Suggested direction:** a four-row table would make the comparison instant. (Note for the author:
this trades against Chapter 6 being a discursive conclusion rather than a results chapter.)

### C-07 · Two "didn't it just say the opposite?" moments, both resolvable

**Chapter 2, HGI** (p. 21): the reader met "HGI is the place-level baseline representation the later
chapters measure against, and it is the direct base of the representation the dissertation
contributes," then two paragraphs later "A place embedding, however it is trained, shares one
property: it assigns each place a single fixed vector... This is the limitation the rest of the
dissertation responds to."

> "on first pass it reads as: 'this is our great baseline and direct foundation' immediately
> followed by 'and here's why it's fundamentally broken for our purposes' — the pivot is real but
> **arrives with no transitional signal**."

**Chapter 5, region gain** (pp. 58 vs 70): the contributions claim "outperforms or matches" and
§5.6.2 later reports "−0.41 at the smallest count".

> "Taking these together cold, 'outperforms or matches' reads like it's papering over an actual loss
> until you notice −0.41 is within the ±2pp non-inferiority margin. **The text never flags this
> reconciliation explicitly at first mention.**"

**Both verified.** Neither is a contradiction — the second in particular is exactly what
non-inferiority means, and Chapter 5 states the Alabama deficit plainly when it gets there ("a small
but statistically significant deficit, still well within the two-point margin"). But a cold reader
meets the summary before the explanation, and the summary's "matches" does cover a small negative
number without saying so at that point.

### C-08 · Terms met before they mean anything

Consistently reported across readers, in order of how often:

| Term | First met | Explained | Reader comment |
|---|---|---|---|
| **TOST** | p. 3 (Resumo), p. 5 (Abstract) | expanded in the abbreviations list, p. 9; mechanics at p. 67 | "used with a gloss ('a test of equivalence') but the mechanics aren't spelled out until Section 5.5.3, **twenty pages later**" |
| **Next-POI Prediction / category classification** | pp. 5–6 | p. 13, p. 29 | see C-01 |
| **hard parameter sharing** | p. 13 | p. 22 | "used before any definition of what it structurally means" |
| **HMT-GRN, STAN, ReHDM, POI-RGNN** | p. 73 | Ch.5 baselines, p. 67 | "thrown in as a parenthetical list with zero introduction" — but the reader met p. 73 before p. 67 in its span ordering; **partially reader-order artifact** |
| **"frame-level analysis"** | p. 75 | never | "unclear how this differs from the main Chapter 5 analysis or what 'frame' refers to here" — **verified: the term is not glossed anywhere in the body.** This is repo/structural vocabulary that leaked into prose |
| **"their Equation 2"** (HGI) | p. 21 | not shown | "refers to an equation not shown anywhere... I don't know if 0.4→0.7 is a big or small change in mechanism terms" |

**On TOST specifically:** the readers' claim that it is "never spelled out" is **wrong** — it is in
the List of Abbreviations on p. 9 as "TOST Two One-Sided Tests", and Chapter 5 gives "the two
one-sided tests (TOST) procedure" at p. 67. What is true is that it first appears on **p. 3**, in
the Resumo, six pages before the abbreviation list and sixty-four before the mechanics. That is
compliant with the law and still surprising to a reader.

**"frame-level analysis" is the one genuine gap** — it appears in Chapter 6 and is glossed nowhere.
A reader has no way to know "frame" means the non-article chapters.

### C-09 · Captions and floats: mostly interpretable alone

Readers could interpret Table 9, Table 10, Figure 5, Figure 6 and Figure 7 from their captions.
Two friction points:

- Figure 1 (p. 35) — see C-01: the caption's task labels do not match the frame's vocabulary.
- Table 10 (p. 71) — interpretable, but the reader noted the page is nothing but floats: the
  argument stops on p. 70 and resumes on p. 72.

---

## Claimed stumbles I checked and rejected

Reporting these so the author does not chase them:

1. **"TOST is never spelled out."** False — List of Abbreviations, p. 9. Reframed in C-08 as a
   distance problem, not an absence.
2. **"'mahalle' is never glossed."** False — "the *mahalle*, a municipal neighborhood" at p. 62,
   parenthetically at first use in Chapter 5. (A different reader independently praised this same
   gloss as the model of how to do it.) The Chapter 1 uses do precede it, so the *order* is the
   issue, not the absence.
3. **"The keyword list runs together: 'ponto de interesseprevisão da próxima categoria'."** False —
   an artifact of PDF text extraction across a page break. I rendered p. 4 at 3× and the keywords
   are correctly on separate lines.

---

## Chapter-by-chapter "did I follow it?"

| Chapter | Verdict from the cold read |
|---|---|
| Front matter / Abstract | **Followed**, with the task-vocabulary confusion of C-01 unresolved at the end of the span |
| 1 Introduction | **Followed well.** "By the end I understood the high-level claim" |
| 2 Fundamentals | **Followed**, effortfully. Long paragraphs and acronym density; the reader reported "flipping back to Table 1 to keep them straight" — which means the lineage table is doing its job |
| 3 CBIC | **Followed.** No comprehension failure reported |
| 4 CoUrb | **Followed.** Translation register noted, clarity not affected |
| 5 MobiWac | **Followed**, with one re-anchor at the four-grounds paragraph (C-02) |
| 6 Conclusion | **Followed.** "By the end I understood the high-level claim (representation + sharing topology change explains the shift from negative to positive)" |

**No reader checked out of any chapter.** That is a genuinely good result for a 94-page document.

## What read effortlessly — do not "improve" these

Named unprompted by the readers:

1. **Chapter 6's consolidated answer** (p. 74): *"Does multi-task learning help... With a place-level
   embedding and naive hard sharing, no... With a check-in-level representation... yes"* — "clean
   parallel structure, **immediately graspable**."
2. **The closing line** (p. 79): *"The negative result was not an obstacle on the way to the
   contribution; worked through, it was the contribution's first half."* — "rhetorically clear and
   memorable, **no rereading needed**."
3. **The limitations list** (§6.3): "each item is a single crisp sentence, numbered, **no
   ambiguity**."
4. **The user-disjoint split sentence** (p. 26): *"the folds are formed so that no user spans a
   split: a grouped, stratified splitter keeps all of a user's check-ins on one side of every fold,
   so that measured accuracy reflects generalization to new users rather than memorization of
   familiar ones"* — the Ch.2 reader singled this out as "the one that's actually **clear on first
   pass**" in a dense section.
5. **The mahalle gloss** (p. 62) — "this one actually works, flagging it only because most other
   jargon isn't handled this cleanly."

## Open questions for the author

1. C-01: can the abstract afford three words to gloss "category classification"? It is the single
   highest-value clarity fix in the document, and the abstract is claim-parity-locked to the Resumo,
   so any change must be made in both.
2. C-08: "frame-level analysis" — is there a plainer phrase, or should it be glossed at first use?
3. Method: is the delegated cold read acceptable as evidence, or does the author want a human first
   pass before the banca build? Given the persona's fresh-eyes rule, a human reader who has not seen
   the document would be strictly better and I would recommend it if anyone is available.

## Out-of-scope handoffs

- Persona 05/18: the four `(??)` markers.
- Persona 15: C-02, C-05 and C-06 are all readability findings; we converged independently.
- Persona 04: C-01's task-vocabulary gap is also a concordance seam (the abstract uses a term the
  body defines later).
