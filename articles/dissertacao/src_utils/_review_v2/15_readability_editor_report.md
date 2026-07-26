# 15 · Readability editor — professional-editor quality review

**Read:** `src/dissertacao.pdf` (94 pp) for the reading experience; `src/chapters/*.tex` for quoting.
**Date:** 2026-07-26. **Persona:** `reviewers/15_readability_editor.md`. Read-only.
Science, numbers and citations assumed correct — other personas own them. I judge the writing as
writing, including what is technically legal but reads poorly.

**Special charge this round:** WRITING_LAW §4.3 — an edit pass that only smooths is a regression.
The corrected sentences were assistant-written; I read them as a reader, and separately measured
whether they flattened the prose.

## Verdict on the §4.3 charge

**The round did not smooth.** Measured sentence-length dispersion by chapter (coefficient of
variation):

| Chapter | CV | edited this round? |
|---|---:|---|
| 3 CBIC | 0.414 | barely |
| 4 CoUrb | 0.424 | barely |
| 1 Introduction | 0.495 | yes |
| 5 MobiWac | 0.497 | **heavily** |
| 2 Fundamentals | 0.568 | yes |
| **6 Conclusion** | **0.640** | **heavily** |

The most-edited chapters are the burstiest, not the flattest. Chapter 6 runs 6-word sentences
against a 110-word one. If an assistant had homogenized this text, the numbers would run the other
way. **No variance compression.**

What the round *did* introduce is a different problem, and it is the opposite of smoothing:
**several corrected passages grew by accretion.** Each clause was added to answer a specific
objection, honestly, and the result is paragraphs that a reader has to work at. That is the finding
below.

## Scores (1–10)

| Dimension | Score | One-line justification |
|---|:--:|---|
| **Readability** | **7** | Sentences are clear individually; several paragraphs are too long to hold in one pass |
| **Coherence and flow** | **8** | The arc is genuinely felt; transitions between chapters are among the best I have read in a coletânea |
| **Clarity** | **8** | Almost no ambiguity; the two-conventions issue (Table 9 vs Table 10) is the one place a careful reader must stop and reconstruct |
| **Conciseness** | **6** | The weakest dimension. Chapter 2 averages 154-word paragraphs; Chapter 5 has a 546-word one |
| **Consistency of voice** | **7** | Frame chapters and Chapter 5 read as one author; Chapters 3 and 4 audibly do not (see the seam verdict) |
| **Overall writing quality** | **7.5** | A document whose honesty is its literary strength, carrying more density than it needs |

## Top findings

1. **R-01 (Critical)** — the 546-word representation-integrity paragraph.
2. **R-02 (Major)** — the 110-word opening sentence of Chapter 6.
3. **R-03 (Major)** — Chapter 2 averages 154 words per paragraph across 26 paragraphs.

---

## Critical

### R-01 · The four-grounds paragraph is 546 words in one block

`src/chapters/5_mobiwac.tex:367`, rendered pp. 65–66. It opens:

> "We train the representation once on the whole dataset and feed it to every model, dedicated and
> joint. A representation built this way could pass information about a test visit along more than
> one channel. We therefore bound the channels that we can measure, and state for each what the
> measurement covers, on four grounds."

and then runs, unbroken, through all four grounds, three stated limits, a counter-example, and a
sentence about the baselines — 546 words, roughly one and a half pages of continuous text with no
paragraph break.

**Why it affects the reader.** This is the most important defensive passage in the dissertation and
the one an examiner is most likely to interrogate line by line. As a single block, its structure is
invisible: the reader cannot see at a glance that there are four grounds, cannot navigate back to
the second one, and cannot tell where ground three ends and ground four begins without re-reading.
The "First… Second… Third… Fourth…" markers are doing structural work that white space should be
doing.

The irony is that the content is excellent — the honesty of "The measurement bounds this channel
rather than closing it" and the volunteered counter-example ("one encoder that passed it leaked
under a downstream sequence model") are the strongest sentences in the chapter. They are buried at
word 400.

**Suggested direction:** break at the ground boundaries. The opening three sentences become the
frame paragraph; each ground becomes its own paragraph; the three limits and the counter-example
close. Five paragraphs, not one, with no words changed. The "First/Second/Third/Fourth" markers then
become paragraph openers, which is what they are already trying to be. **No rewrite needed — this is
purely a break-insertion.**

---

## Major

### R-02 · Chapter 6 opens with a 110-word sentence

`src/chapters/6_conclusion.tex:14-22`:

> "This dissertation asked one question, whether multi-task learning helps point-of-interest
> prediction for the next category and next region tasks and what the answer depends on, and
> answered it across three studies: a joint model that did not outperform its dedicated counterparts
> (Chapter~\ref{ch:cbic}), the diagnosis that located the bottleneck in the input representation
> (Chapter~\ref{ch:courb}), and a check-in-level representation with a redesigned sharing topology
> under which one joint model finally outperforms the dedicated single-task models on the category
> task at all six datasets and on the region task at four of the six, and is statistically
> non-inferior to them, within a two-point margin (TOST), at the other two
> (Chapter~\ref{ch:mobiwac})."

**Why it affects the reader.** This is the first sentence of the conclusion — the sentence a busy
examiner reads first and quotes in the arguição. It contains the research question, the three-study
structure, three chapter cross-references, two verdicts with their scopes, and a statistical test
name. The reader reaches "and is statistically non-inferior to them" 90 words in and has to
reconstruct what "them" refers to.

The round *created* this length: the source comment at `:23-27` records that the previously
unqualified "outperforms both dedicated models" was given its scope here, which was the right call
substantively. The scope was added inside the existing sentence rather than after it.

**Suggested direction:** split after "three studies", then let the three studies run as a colon list
or three short clauses, and give the Chapter 5 verdict its own sentence. The correction survives
intact; only the container changes.

### R-03 · Chapter 2 averages 154 words per paragraph

26 paragraphs, mean 154 words, **10 of them over 150 words**, longest 324. For comparison: Chapter 6
averages 109, Chapter 1 averages 92.

The three heaviest:

- **324 words** — the validation-protocol paragraph (`:483-510`), which runs from stratified k-fold
  through the per-chapter split differences, the Wilcoxon floor, Holm, TOST, and the verb-test
  binding rule. Five distinct topics.
- **278 words** — the infomax/representation-learning paragraph.
- **277 words** — the class-imbalance paragraph.
- **268 words** — the gradient-balancer family paragraph.

**Why it affects the reader.** Chapter 2 is the chapter a future student reads to learn the field.
Its job is didactic, and WRITING_LAW §1 explicitly grants it "didactic room". But didactic room means
worked examples and definitions, not longer paragraphs — a 300-word block is harder to learn from
than three 100-word blocks, because the reader cannot tell which sentence is the topic sentence.

The 324-word one is the clearest case: it contains the protocol *and* the per-chapter comparison
*and* the statistical apparatus. Those are three paragraphs' worth of material and the reader has no
resting point.

**Suggested direction:** break the 324-word paragraph at "The statistical treatment is scoped the
same way" (`:493`) — there is a natural seam there, since everything before is splitting and
everything after is testing. Same for the other three: each has a visible internal seam. Again,
break-insertion rather than rewriting.

### R-04 · Chapter 5's statistics paragraph is 315 words and carries seven distinct moves

`5_mobiwac.tex:394`. It contains: why gains and matches need different tests; the analysis plan's
existence and timing; what the plan assigned; what it did not cover; the Wilcoxon floor and the *t*
substitution; the definition of a seed; the pairing and n; the Holm families; the TOST definition;
the margin's operational justification; and the power claim. Eleven, on a recount.

Every one of those is *necessary* — persona 09 would object to removing any of them, and I agree.
The problem is purely that they arrive as one block. A reader looking for "what test did they use
and on how many observations" must scan 315 words.

**Suggested direction:** three paragraphs — the test-assignment logic and the plan; the repetition
structure and the tests actually run; the margin and its justification. Nothing cut.

---

## Minor

### R-05 · "Table 9's column is not Table 10's column" costs the reader a stop

`5_mobiwac.tex:411`: "The check-in-level column keeps one fixed configuration, not the
per-dataset-tuned dedicated model of Table~\ref{tab:mobiwac:results}."

The sentence is correct and necessary. But a reader meeting Table 9's "Check-in level" column of
55.87 and then Table 10's "Dedicated" column of 56.82, thirty pages later, will not remember it and
will wonder whether the document contradicts itself. The freeze-control paragraph then quotes deltas
computed against Table 9 (`:662`), so the reader who tries to verify the arithmetic against Table 10
gets the wrong answer.

**Suggested direction:** the disambiguation would work harder as part of Table 9's column heading or
caption than as a sentence in the body. Persona 04 raised the same seam from the concordance side.

### R-06 · Chapter 5's contributions list forward-references three floats by number

`5_mobiwac.tex:60`, p. 59: "(Table~\ref{tab:mobiwac:representation})" and Figures 4, 5, 7 referenced
on the same page. Tables 9 and 10 render on pp. 69 and 71; Figure 7 on p. 71. A first-time reader
who follows the pointer flips twelve pages, finds a table they cannot yet interpret, and flips back.

This is standard practice in papers with an itemized contributions list, and Chapter 5 is a
reproduced paper. Recording it as reader friction, not as a defect.

### R-07 · Two long baseline sentences in §5.5.4

`:398` runs 237 words describing five baselines with their adaptations and caveats; `:400` runs 69
words on two representation controls with a semicolon-joined pair. Both are dense but well
organized ("The first role… The second role… The third role"), and the density is proportionate to
the content. Lower priority than R-01 to R-04.

---

## Strengths — protect these

1. **Chapter 5's honesty sentences are the best writing in the document.** Short, flat, declarative,
   and they land because everything around them is measured:

   > "We report this attribution as a finding, not a hypothesis." (`:668`)
   > "It does not follow that the bias cancels exactly." (`:736`)
   > "This remains motivation, not a measured service result." (`:733`)
   > "At Arizona, the interval is centered on zero, so we report a match, not a gain." (`:648`)

   Each is under 20 words, each refuses an overclaim, and each arrives at exactly the moment a
   skeptical reader is forming the objection. That is craft, not accident. **Do not lengthen or
   soften any of them.**

2. **The Chapter 3 preface** (`3_cbic.tex:20-31`). It tells the reader when the chapter is from, what
   it concluded, what later chapters revise, and — critically — that the paper's "Next-POI
   Prediction" means the frame's *next category*. A reader arriving at a five-year-old paper inside
   a dissertation needs exactly these four things, and gets them in twelve lines.

3. **The Chapter 4 protocol declaration** (`4_courb.tex:238`). A reproduced paper that states its own
   split is weaker than the next chapter's, names why, and points forward — without defensiveness
   and without undermining the chapter it introduces. This is difficult to write and it reads
   effortlessly.

4. **The Table 9 coincidence footnote** (`:457-458`): "The matching place-level value at Alabama and
   Istanbul ($26.56$) is a coincidence of two independent runs; their per-fold values differ." One
   sentence that answers a question the reader was about to ask. More documents should do this.

5. **Chapter 1's funnel.** Three paragraphs from "location-based social networks record check-ins" to
   the research question, with the arc laid out and the three studies previewed. It does not
   over-explain and it does not rush.

6. **Chapter 6's capacity-matched paragraph** (`:96-119`), despite R-02's neighbor. It alternates
   short and long sentences, gives the parameter counts, and closes on the methodological
   observation about learning rates — a detail that strengthens the fairness argument and that most
   authors would have left out. It reads like someone thinking, not someone summarizing.

7. **Section 5.6.2's opening** (`:490`): "One model outperforms or matches the dedicated models on
   both tasks." Nine words, the whole result, no throat-clearing. Every results section should open
   like this.

---

## Chapter-seam verdict: do the papers and the frame read as one voice?

**Mostly, with one audible seam.**

- **Chapters 1, 2, 5, 6 read as one author.** Same register, same preference for concrete nouns and
  plain verbs, same habit of stating a limit immediately after a claim. Moving from Chapter 2's
  evaluation section into Chapter 5's protocol section is seamless — the reader does not notice a
  boundary. This is a real achievement in a coletânea.

- **Chapters 3 and 4 are audibly different**, and differently from each other:
  - **Chapter 3** carries conference-paper register: "For a comprehensive evaluation, we selected
    two state-of-the-art approaches as baselines", "This architecture strategically balances model
    capacity with knowledge transfer". Bolded run-in headings ("**Computational Efficiency:**",
    "**Implicit Regularization:**") that appear nowhere else. Adverb density 1.69% against the frame's
    0.5%.
  - **Chapter 4** carries translation register: "the two ST-MTLNet variants reach higher mean F1 than
    the original MTLNet in every category and state", "This diversity allows evaluating how
    different geometric and architectural assumptions impact performance". Grammatical, precise, and
    recognizably rendered from another language — the noun-heavy constructions and the
    "allows evaluating" pattern give it away.

  Both are **correct**: these are reproduced published articles under an errata policy that governs
  what may change, and the prefaces tell the reader they are reproductions. A reader who has read
  the prefaces will attribute the shift to the source, not to the author.

  The seam is at its most audible on **p. 43**, moving from Chapter 3's future-work paragraph
  straight into Chapter 4's translated preface. The preface absorbs most of the shock. Nothing needs
  fixing here; I record it so the author knows the seam is perceptible and can decide whether the
  prefaces do enough work. My judgment: they do.

- **Appendices A and B** read in the frame voice, which is right — they are the author speaking about
  the papers, not the papers speaking.

---

## Reader-experience map

| Where | Experience |
|---|---|
| pp. 13–19 (Ch.1) | **Effortless.** The funnel works; I knew what the dissertation claimed by p. 15 |
| pp. 20–28 (Ch.2) | **Dense but navigable.** The section purpose statements save it; the long paragraphs slow it |
| pp. 29–42 (Ch.3) | **Different voice, clearly flagged.** Readable |
| pp. 43–57 (Ch.4) | **Translation register.** Occasionally effortful, never unclear |
| pp. 58–66 (Ch.5 setup) | **Excellent until p. 65**, where the 546-word paragraph arrives |
| pp. 67–73 (Ch.5 results) | **Strong**, apart from the page-71 floats interrupting the argument (persona 18's finding, but I felt it as a reader: I lost the thread between the Holm result and the Wilcoxon confirmation) |
| pp. 74–79 (Ch.6) | **Strong after the opening sentence** |
| pp. 86–94 (appendices) | **Clear.** Appendix B in particular is unusually readable for an errata document |

## Where a reader would get confused

1. **p. 69 vs p. 71** — the two "single-task" columns (R-05).
2. **pp. 70→72** — the argument resumes after a full page of floats; I had to page back.
3. **p. 65** — the four grounds; I re-read the second and third to keep them straight.

## Open questions for the author

1. R-01 to R-04 are all paragraph *breaks*, not rewrites. Is break-insertion inside reproduced
   Chapter 5 prose within the errata policy, or does changing paragraphing count as a departure that
   Appendix B must record? Chapter 5 is a submitted paper, so this matters.
2. R-05: is the two-conventions disambiguation better placed in Table 9's heading than in the body?

## Out-of-scope handoffs

- Persona 18: the page-71 float block, which I felt as a reading break.
- Persona 03: the 110-word sentence is also a §4 metric (it measured it independently).
- Persona 04: the two-conventions seam is also a concordance finding.
