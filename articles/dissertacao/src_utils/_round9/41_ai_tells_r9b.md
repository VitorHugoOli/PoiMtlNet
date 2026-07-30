# 41 · AI-credibility (does the changed prose read as machine-written) — round 9, wave B

**Persona:** `reviewers/16_ai_credibility.md` — two channels, the detector screener and the
suspicious expert. Read-only; I edited no `.tex`.
**Scope, and only this:** the diff `c94d1f19..HEAD` (`37b37aaf`), four files —
`src/chapters/2_fundamentals.tex` (+106), `src/chapters/apx_f_cosine.tex` (+205/−92),
`src/chapters/1_introduction.tex` (+37/−14), `src/tables/frame/cosine.tex` (+74/−33).
**Read this session:** `reviewers/16_ai_credibility.md`, `reviewers/README.md` §Common protocol,
`WRITING_LAW.md` (full, including the §1 process-narration ban added today), `AGENT_GUARDRAILS.md`
(full; §4b V1–V15c), `GLOSSARY.md` (full), the MobiWac `GLOSSARY.md` §7–§8 (the inherited ban
tables), `src_utils/check_process_narration.py`, `src_utils/check_negative_parallelism.py`,
`src_utils/check_audit_claims.py` (`live_text`), and `_round9/reviews/16_ai_credibility.md` (this
persona's earlier run against `901a0408`, so I say where I agree and where I differ rather than
re-issuing it).
**Built and read the render, not the source:** `make defense` → `build/main.pdf`,
**102 pages, `tex_errors=0`**. Every quote below is located by page in that PDF.
**Severity mapping** (persona scale → this report's labels): BLOCKER/MAJOR → **REQUIRED**,
MINOR → **RECOMMENDED**, NIT → **OPTIONAL**.

---

## Verdict

**The changed prose is defensible as it stands except for five sentences, all of one class — the
class the author caught by eye, and none of which any gate can see.** Screener risk **LOW**;
expert-suspicion risk **LOW-to-MEDIUM**. The five are findings 1, 2 (two sentences) and 3 (two
sentences); a sixth, finding 4, is the same habit at RECOMMENDED. They sit inside a document-wide
family of **fifteen** matches, ten of which predate this round and are listed separately so they
do not inflate this round's count.

**Screener channel: LOW.** The added prose is neither vocabulary-flattened nor low-variety, which
is the L2 false-positive trap this persona exists to avoid pushing the text into. Sentence-length
spread in the new §2.3 block is cv = 0.585 (n = 17, min 7 words, max 66) and in the new Appendix F
prose cv = 0.599 (n = 26, min 5, max 61), both **above** the two published article chapters
(Ch.3 cv = 0.491, Ch.4 cv = 0.463) — this edit wave did not compress variance, which is
WRITING_LAW §4.3's deepest tell. Any score a committee obtains on this document is a hybrid-text
score and is window-size-sensitive by measurement, so it is evidence about the window and not
about authorship; no detector is installed in this checkout and I claim no score.

**Expert channel: LOW-to-MEDIUM.** Word-level discipline is clean (zero banned-list hits, zero
em-dashes in prose, zero contractions, −ly density 0.29–0.40 %). What a suspicious examiner would
key on is not vocabulary but a habit: the text repeatedly explains why it contains what it
contains. "Texas illustrates why the positive-fold count is reported next to the mean" is a
sentence about a table's design, in a table caption, reproduced verbatim in the List of Tables
(p. 7) and again under the table (p. 100). That is the same reflex as the paragraph deleted today,
one turn milder.

---

## The class the author caught, swept for its siblings

His instance was *"the machine that would have run them was out of disk."* **Confirmed removed:**
the string `out of disk` occurs **zero times in the 102-page render**, and the gate written for it
(`check_process_narration.py`) passes on 51 files. The four gate patterns, run against every
candidate below, catch **none of them** (verified in both directions: the same patterns still fire
on the deleted sentence). So this section is the part of the family the instrument is blind to.

I swept six detectors for it — "why the document reports X", "why an item entered or stayed",
"pointer to a neighbouring paragraph", "document as an agent with a policy", "the text's own
rhetorical duty", "what this passage is" — each validated to fire on a positive control, over all
51 files the gate covers, comments stripped with the repo's own `(?<!\\)%` rule. **Fifteen hits
document-wide, five of them new this round.** Findings 1–4 are the five new ones; §"Pre-existing"
lists the other ten so the author sees the whole family without them inflating this round's count.

---

## Findings

### 1 · REQUIRED — the prose tells the reader where to look next, and the pointer is imprecise

**File:** `src/chapters/2_fundamentals.tex` · **PDF p. 22** (new §2.3, added `beebd33b`)

> over 𝐾 tasks at shared parameters 𝜽, and each method named in the next paragraph is a different
> answer to how the weights 𝑤𝑘, or the update direction they imply, should be set.

**Why.** WRITING_LAW §1 sub-class 4 bans *"Self-reference to the writing"*, with the instances
*"as noted above"*, *"the paragraph above draws"*. This is that class pointing forward instead of
back, and I state plainly that the law's listed instances are all backward-looking, so this is an
extension of the named class rather than a verbatim match — which is precisely the gap the author
asked me to sweep. The clause and the quote sit together without contradiction: a sentence whose
subject is *which paragraph* something appears in is a sentence about the document, not about
multi-task learning.

It is also imprecise as a pointer, and I measured this rather than assuming it. The immediately
following paragraph is about those methods' *guarantees*; its **single** occurrence of "weights" is
"for task weights fixed in advance", which states Aligned-MTL's assumption. The paragraph that
actually enumerates how weights are set is **two paragraphs later**, "A family of methods tries to
manage the conflict…" (p. 23), which holds six of the page's weight words. I do not claim the
sentence is false — the named methods are answers to the question — only that the reader who turns
to the next paragraph does not find the promised content there.

**Fix.** Drop the pointer and let the equation hand off to the concept: "…over 𝐾 tasks at shared
parameters 𝜽. The methods that set the weights 𝑤𝑘, or the update direction they imply, differ in
what they optimize." Same information, no reference to the layout.

---

### 2 · REQUIRED — a table caption explains why the table has a column

**File:** `src/tables/frame/cosine.tex` · **PDF p. 100**, and reproduced verbatim in the
**List of Tables, p. 7**

> Texas illustrates why the positive-fold count is reported next to the mean: four of its five
> folds are positive and its mean is negative, because one fold at −0.0032 outweighs four smaller
> positive ones.

And its sibling in the body, same page, same round:

**File:** `src/chapters/apx_f_cosine.tex` · **PDF p. 100**

> California makes the same point from the other side and is the reason the table carries a
> sign-test column at all: its 𝑡-test returns 0.048, below the conventional threshold, while its
> exact sign test returns 0.375 on four of five positive folds, so what looks like a finding under
> one test is not even a leaning under the other.

**Why.** WRITING_LAW §1: *"The prose states what is true of the work, never how the work came to
be done or what the writing went through"*, and the test it gives is *"if the sentence would be
false or pointless once the circumstance changes … cut it."* Both sentences are about the table's
design history: why a column is present, why a count sits where it sits. Change the table and both
become pointless; the statistics they carry do not change at all. A banca member cannot act on the
fact that a column exists for a reason. The caption instance is the more expensive of the two
because abnTeX2 reproduces the full 185-word caption in the List of Tables, so the reader receives
it **twice**, on p. 7 before any of the appendix and again on p. 100. (The "118 words" in the
caption's own provenance comment measured the round-7 caption; this round widened it and the comment
was not re-measured. Counted from the rendered p. 100 caption: 185.)

This is also the one place in the round where the idiom budget moved: metaphorical
*carries* in Appendix F went **3 → 4** across this diff (N of N read, all four), against
WRITING_LAW §4's *"metaphor budget for carry/carries ≤3 per chapter"*. The fourth instance is
"the reason the table **carries** a sign-test column" — the same sentence. One edit closes both.

**Fix.** State the statistics; delete the design rationale. Caption: "At Texas four of five folds
are positive while the mean is negative, because one fold at −0.0032 outweighs four smaller
positive ones." Body: "California is the opposite case: its 𝑡-test returns 0.048, below the
conventional threshold, while its exact sign test returns 0.375 on four of five positive folds, so
what looks like a finding under one test is not even a leaning under the other."

---

### 3 · REQUIRED — two paragraph openers name the text's own rhetorical obligation

**File:** `src/chapters/2_fundamentals.tex` · **PDF pp. 22 and 23**

> Casting it that way obliges one to say what an optimum would be.

> Reaching that front is not what the balancing methods promise, and the distance between the
> promise and the front deserves one statement.

**Why.** Neither sentence is about multi-task learning; each is about what this text is now obliged
to do. That is the same reflex as §1's *"Self-reference to the writing"*, and it is the residual
after persona 03's word-level sweep, which the block passes. I agree with the earlier run of this
persona (`_round9/reviews/16_ai_credibility.md` finding 3), which flagged these two plus
"Conflict has a standard measure" as three consecutive openers of one shape; **both remain in the
current build**, so this is a re-issue with the §1 angle added, which did not exist when that
report was written.

I differ from it on the third opener. "Conflict has a standard measure, and it is the quantity the
gradient methods act on" (p. 23) opens on a fact about the field and is **fine** — I would not
touch it. I also note that WRITING_LAW §5 mandates a purpose statement at each *section* opening;
these two are mid-section paragraph openers, so that sanction does not cover them. §2.3's actual
section opener ("Multi-task learning (MTL) trains one model on several related tasks at once") is
content, and correct.

**Fix.** Lead with the asymmetry, which is already the next sentence in each case. First: "A single
loss orders any two parameter settings; several losses do not." Second: "The balancing methods do
not promise to reach that front." Both keep every claim and lose the meta-layer.

---

### 4 · RECOMMENDED — an appendix is given the agency of reading its own result

**File:** `src/chapters/2_fundamentals.tex` · **PDF p. 23**

> Appendix F measures the cosine on the joint model of Chapter 5 and finds the two tasks' gradients
> statistically indistinguishable from orthogonal on the datasets measured there; it reads that
> result as the reason no balancing method improved on a fixed weighting in this work, and as part
> of the reason the argument moves to the input representation in Chapters 4 and 5.

**Why.** Vague attribution, which this persona lists as a gestalt trigger: a reading is attributed
to a document rather than to the author or the measurement. The device is doing honest work here —
the provenance comment at `2_fundamentals.tex` shows it was chosen so the mechanism claim stays the
appendix's own, and the claim carries `[NEEDS SIGN-OFF]` in both places — so this is style, not
integrity. Same call as the earlier run (its finding 5, NIT); I raise it one notch only because the
sentence survived a round and sits three lines from finding 3's openers, so the paragraph now
carries two meta-moves in four sentences.

**Fix.** "…and argues that this is why no balancing method improved on a fixed weighting in this
work." Same hedge, an agent that can hold a position.

---

### 5 · RECOMMENDED — negative-parallelism density is inside its gate and outside it per page

**Files:** `src/chapters/apx_f_cosine.tex`, `src/tables/frame/cosine.tex` · **PDF pp. 98, 100, 101**

**Document-wide the guard held, and this is worth saying first:** across this round the count went
**120 → 119** hits and the density **3.345 → 3.263 per 1,000** prose words against the gate's 3.60
ceiling, on 33 files, comments stripped — i.e. the round *added* 594 prose words and *reduced* the
density. The instruction frozen into `check_negative_parallelism.py` on 2026-07-20 is doing its job.

The gate is document-wide, so it cannot see concentration. Measured on the rendered pages with the
gate's own four patterns:

| page | prose words | hits | per 1k | the instances |
|---|--:|--:|--:|---|
| 22 | 468 | 3 | **6.41** | "rather than" ×2, ", not" |
| 23 | 525 | 1 | 1.90 | "rather than" |
| 97 | 413 | 1 | 2.42 | "rather than" |
| 98 | 463 | 4 | **8.64** | ", not" ×3, "rather than" |
| 100 | 459 | 3 | **6.54** | "rather than" ×3 |
| 101 | 428 | 3 | **7.01** | "rather than" ×2, ", not" |
| 102 | 186 | 0 | 0.00 | — |

Per file, Appendix F is 5.34/1k, **7th of 27** files over 300 prose words. Of the eleven instances
in it (N of N read), **nine are pre-existing** and two arrived with the widening, so this is a
concentration the round preserved rather than one it created. Three "rather than" on one page is
the density a 2026 examiner is primed to see.

**Fix.** Two edits on p. 100 clear it without touching a claim: "the reason is the design rather
than the data" → "the reason is the design, not the data" would only trade one pattern for another,
so instead → "the design, and not the data, is what prevents it"; and "consistent tendencies rather
than established effects" → "consistent tendencies, short of established effects". Leave p. 98
alone: all three ", not" there scope a claim (unit vs sample size, tasks vs power, mean vs
observation) and are the honesty device the law explicitly protects.

---

### 6 · OPTIONAL — the new §2.3 block runs a metaphor register the published chapters do not

**File:** `src/chapters/2_fundamentals.tex` · **PDF pp. 22–23**

> What the sum conceals is that its minimum need not be the right target.

> Reaching that front is not what the balancing methods promise, and the distance between the
> promise and the front…

**Why.** Not banned, and not idiom of the phrasal-verb kind §8 forbids. It is a density
observation: figurative constructions run at 10.6 per 1,000 words in the added §2.3 block against
1.5 in Ch.2 as a whole and **0.0** in both published article chapters. A sum that conceals and a
promise with a measurable distance from a front are three metaphors in five sentences, in the one
chapter a banca reads as the author's own voice. One decorative instance is noise; the cluster is
what a reader notices.

**Fix.** Optional, and one is enough: "What the sum conceals is" → "The minimum of that sum need
not be the right target." Keep the promise/front figure — it is load-bearing, since the gap between
what the methods prove and what Pareto optimality would require is the paragraph's actual point.

---

## Pre-existing, in the family, outside this round's diff

Listed for completeness because the author asked for the family, not for the diff. **None of these
is a round-9 regression**; each was in the tree at `c94d1f19`, verified by searching all 54 `.tex`
files of that commit (instrument checked in both directions).

**The count reconciles as follows, because a headline that does not match its rows discredits the
rows** (AGENT_GUARDRAILS §4b V13): fifteen matches document-wide, five new, ten pre-existing. The
five new ones, verbatim as the detectors matched them: "the next paragraph" (finding 1); "obliges
one to" and "deserves one statement" (finding 3); "is the reason the table" and "illustrates why the
positive-fold count is reported" (finding 2). The ten pre-existing matches appear below as
**eight rows** — P1 holds two matches ("it enters because" and "stays because" are one sentence) —
plus one match I exclude on the law's own authority, the Appendix F roadmap sentence discussed
immediately after the table. 5 + 8 rows (9 matches) + 1 excluded = 15.

| # | quote (verbatim) | page | file | note |
|---|---|---|---|---|
| P1 | "it enters because the diagnostic ran on it cheaply and stays because dropping a measured dataset would be a choice about the evidence" | 97 | `apx_f_cosine.tex` | why a dataset is in the document; also the only infrastructure word left in scope ("cheaply") |
| P2 | "this appendix supersedes nothing there" | 98 | `apx_f_cosine.tex` | document as agent; here it is doing real work (it prevents a reader from reading Ch.5's numbers as replaced) |
| P3 | "one feature needs saying plainly" | 98 | `apx_f_cosine.tex` | the text's own duty |
| P4 | "both are worth reporting rather than smoothing" | 98 | `apx_f_cosine.tex` | states the writing's virtue; the *fact* (two departures) would survive the cut |
| P5 | "The distance between that result and a null result is the point of this appendix." | 98 | `apx_f_cosine.tex` | what this passage is |
| P6 | "this appendix will not accept for one claim a basis it rejects for another" | 100 | `apx_f_cosine.tex` | document as moral agent; the underlying standard is right and could be stated as a property of the tests |
| P7 | "it is worth stating here because it is what the fourth level costs" | 20 | `2_fundamentals.tex` | the text's own duty |
| P8 | "This chapter carries that diagnosis one step further" | 61 | `5_mobiwac/02_related.tex` | re-typeset paper prose; out of frame-chapter scope |

**Explicitly not a finding:** "The sections below give the measurement, the two findings that
qualify it, and the boundary beyond which none of it is claimed" (p. 97). That is a roadmap
paragraph, which WRITING_LAW §5 *requires* ("Every chapter Introduction ends with a roadmap
paragraph"). A reviewer flagging it would be reading the law against itself.

---

## Categories swept that produced NOTHING (stated, not omitted)

- **Banned vocabulary** (WRITING_LAW §4 + the inherited MobiWac §7 table + the Claude-family tics
  "genuine/genuinely", "comprehensive", "robust", "crucially", "notably"): **zero hits** in all four
  files' added prose. Detector validated on a positive control that fired 8 of its patterns.
- **Banned templates** ("not only X but also Y", "plays a crucial role", "in today's world",
  Firstly/Secondly scaffolds, sentence-initial Moreover/Furthermore/Additionally/Notably,
  "it is worth noting", participial significance tails): **zero**. Control fired 3 patterns.
- **Em-dash:** **zero** in all 51 live-prose files. The render contains two, both in the
  bibliography (p. 85, inside the Massive-STEPS paper's own title; p. 88, abnTeX2's thesis-entry
  separator). Neither is authored prose.
- **Contractions:** **zero**. The ten apostrophes in the added prose are all possessives
  ("Florida's", "task's").
- **−ly adverb density:** new §2.3 0.40 %, new Appendix F 0.29 %, both under the ≈0.8 % band; the
  whole appendix 0.41 %, Ch.2 0.53 %. **No sentence anywhere in the added prose carries two −ly
  adverbs.**
- **Semicolon braids:** no sentence in the added prose has more than one semicolon except the
  Nash-MTL/CAGrad/Aligned-MTL guarantee list on p. 23, which has two — and that one is a
  three-item parallel list of distinct citations, which is what a semicolon is for. Not a braid.
- **Sections ending by restating themselves:** read the closing two sentences of all four Appendix
  F sections and all three new §2.3 paragraphs. **None restates its section.** The p. 102 close
  ("Equivalence there makes orthogonality the task pair's property; otherwise it belongs to this
  architecture") states a condition rather than a summary. The sentence that *did* commit this
  offence, "the boundary the paragraph above draws", went out with the deleted paragraph.
- **Rule-of-three cascades:** zero triadic constructions in the new §2.3. The two in the new
  Appendix F prose are a six-item dataset list and a coordinate pair, not cascades.
- **Copula avoidance** ("serves as", "acts as", "stands as"): **zero** in all added prose and zero
  in the whole appendix.
- **Uniform rhythm / variance compression:** ruled out, numbers in the verdict above.
- **Nominal-style creep:** determiner-plus-"of" density 18.5 % (new §2.3), 17.4 % (new Appendix F),
  against 18.2 % for Ch.2 and 16.6 % for the published Ch.4. In band.

---

## What reads credibly human, and should be protected

1. **The California paragraph is the best thing in this diff** (p. 100): a 𝑡-test at 0.048 set
   against a sign test at 0.375 on the same five folds, reported as a reason to distrust the
   𝑡-test the author could have quoted as a win. No generator produces a result that argues
   against its own significance, and no examiner mistakes it for one.
2. **Texas's negative mean over four positive folds**, reported with the arithmetic that explains
   it (one fold at −0.0032). Friction of exactly the kind LLM filler never contains.
3. **The refusal to upgrade the sign-test floor**, with the floor's reason stated twice (p. 100
   body and footnote): "no five-fold row can reach 0.05 by that test."
4. **"This dissertation therefore claims no Pareto property of any kind for its models"** (p. 23) —
   a definitional passage that ends by disclaiming, against a literature where every cited method
   would have licensed a stronger sentence.
5. **The advisor's seven wording items are the right kind of edit.** FAB-16 in particular removed a
   semicolon he read as a tell, and the replacement is two sentences, not a comma splice: "…is a
   third and different problem, not addressed in this dissertation. Chapter 2 keeps the three tasks
   formally distinct." That is a human editing pass, and its trail (the PT-BR item numbers in the
   provenance comments) is provenance evidence no generator produces. Preserve it.
6. **The specificity audit, where I differ from the earlier run of this persona.** Its finding 4
   held that the new §2.3 block "could have been written without access to the experiments". Half
   true: the Pareto paragraphs are definitional, but the third paragraph of the same addition
   (p. 23, "Conflict has a standard measure…") points at Appendix F's measurement and at the
   fixed-weighting result, so the block as committed is less abstract than that report describes.
   I would not add the extra clause it proposed. The §2.3 material is doing what a fundamentals
   chapter should: defining a quantity that Chapter 5 and Appendix F then measure.

---

## Provenance-shield status (process, not prose)

| defense | status |
|---|---|
| Layered AI-use disclosure | **Present** — Appendix C (p. 93), 313 prose words, names the tool, three model versions, and the scope per part of the work |
| Disclosure is task-precise (generation disclosed as generation) | **Yes** — "The frame chapters … were drafted by the assistant from author-approved outlines"; drafting is not laundered as editing |
| Git AI/author commit separation (AGENT_GUARDRAILS §5) | **Holding** — today's eight source commits are all `feat(...)`/`fix(...)` with the defect named in the subject line |
| PT-BR thinking trail preserved | **Yes** — the advisor's items survive as `[FAB-12]`…`[FAB-24]` provenance comments carrying his Portuguese wording |
| Author can defend each passage orally | **At risk in one place only** — the §2.3 mechanism sentence carries `[NEEDS SIGN-OFF]`; that is the one passage where the author has not yet ruled |
| Local detector run | **Not available** in this checkout. Screener risk is estimated from the distributional numbers above, never from a score |

**Refresh mandate (this persona's unique duty): NOT EXERCISED.** I ran no web pass for
tell-catalogue or venue-policy changes since the two evidence files' dates (2026-07-18 / 07-20),
because the track was scoped to a four-file diff. The evidence base is now ten to twelve days old
and §4.1 of the law says the list rots. **Recommendation for the author:** run one bounded refresh
against Wikipedia's "Signs of AI writing" before the banca build, not before the advisor handoff.

---

## MEASURED — every count above, with the command that produced it

Working directory `articles/dissertacao/` unless stated. `git` invoked with
`GIT_CONFIG_GLOBAL=/dev/null` (the sandbox cannot read `~/.gitconfig`; without this every `git`
call exits 128 — which is how my first diff parse returned zero live lines for all four files and
looked like a clean result).

    # the build I read
    cd src && make defense          # -> build/main.pdf pages=102 tex_errors=0
    python3 -c 'import pypdfium2 …'  # per-page text of all 102 pages

    # live prose = comments stripped with the repo's own rule
    python3 -c 'from check_process_narration import live_text; …'   # (?<!\\)% , imported, not re-implemented

    # the added prose, isolated
    git diff -U0 c94d1f19..HEAD -- <file>   # + lines, then live_line() per line
    # -> live added words: ch2 494, apxF 686, intro 129, table 419

    # gates, run as they ship
    python3 src_utils/check_process_narration.py     # OK, 51 files, 3 exempt; rc=0
    python3 src_utils/check_negative_parallelism.py  # 119 / 36465 = 3.26 per 1k (ceiling 3.60); rc=0

    # document-wide density, before vs after, on the gate's own 33-file scope
    # pre  c94d1f19: 120 hits / 35,871 words = 3.345/1k
    # head 37b37aaf: 119 hits / 36,465 words = 3.263/1k   (delta -1 hit, +594 words, -0.082/1k)

    # per-page density: gate patterns over rendered page text, running head dropped
    # p22 3/468, p23 1/525, p97 1/413, p98 4/463, p99 0/89, p100 3/459, p101 3/428, p102 0/186

    # the six document-narration detectors, 51 files, each validated on a positive control first
    # -> 15 hits: apx_f_cosine 9, 2_fundamentals 4, 5_mobiwac/02_related 1, tables/frame/cosine 1
    # -> dated against all 54 .tex files of c94d1f19: 5 NEW THIS ROUND, 10 pre-existing

    # instrument checks I ran before believing any zero (AGENT_GUARDRAILS §4b V3, V13)
    # - banned-word sweep: control sentence fired 8 patterns, template sweep fired 3
    # - the four gate patterns vs my 10 candidates: 0 caught; vs the deleted sentence: 1 caught
    # - the dating instrument: a known-pre-existing string found, a known-new string absent
    # - "out of disk" in the 102-page render: 0 occurrences (the author's fix, confirmed in the PDF)
    # - triad, negative-parallelism, metadiscourse, copula, -ly, contraction detectors: each
    #   asserted against a positive control before use

    # N of N reads, not samples
    # - carry/carries: apx_f 4 of 4 read (3 pre-round -> 4), ch2 5 of 5 read (5 -> 5)
    # - negative parallelism in apx_f: 11 of 11 read with context
    # - "The first/second/..." scaffolds: apx_f 6 of 6, ch2 0
    # - Appendix F sentences: 79 now vs 79 before, 21 changed/new, all 21 read
    # - Appendix F paragraph openers: 23 of 23 listed and dated

**Two zeros I distrusted and re-took.** (1) The first diff parse returned 0 live added lines for
all four files; the cause was `git` exiting 128 on an unreadable global config, not clean files.
(2) My first quote-location pass reported all sixteen quotes "NOT IN RENDER"; the cause was
newline-wrapped page text, not absent prose. Both were caught by asserting a control I had already
seen on screen. Every page number in this report comes from the second, validated pass.

**Quote fidelity, self-audited.** I re-checked all 19 blockquote lines and all 8 table quotes in
this report as substrings of the extracted render. **All match** except two, both artifacts of my
own comparison and neither a misquote: (a) the extractor emits "𝑤𝑘 ," with a space before the comma
where the subscript ends, so my "𝑤𝑘," differs from the text layer but not from the printed page;
(b) finding 6's second quote ends in an ellipsis, which is my truncation and marked as one. Two page
numbers in an earlier draft of this report were wrong and are corrected here (P7 is p. 20, not 21),
and one word count I had copied from a source comment ("118 words") was stale and was re-measured
at 185.

**What I could not verify.** No local AI detector exists in this checkout, so the screener verdict
is an estimate from distributional measurements and not a score. The web refresh mandate was not
run. Whether finding 4's attribution device should change at all depends on the `[NEEDS SIGN-OFF]`
ruling on the appendix's mechanism claim, which is the author's to make and not mine.
