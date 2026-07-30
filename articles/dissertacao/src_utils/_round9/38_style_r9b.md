# 38 · Style auditor (G3), round 9 wave B — the four changed files, read in the rendered PDF

> Persona: `reviewers/03_style_auditor.md`. Remit: the word-level law — register, canonical names,
> synonym-cycling, banned vocabulary, em-dashes, contractions, American English — plus the
> distributional checks. Scope: the measured diff `c94d1f19..HEAD` over
> `src/chapters/2_fundamentals.tex` (+106), `src/chapters/apx_f_cosine.tex` (+205/−92),
> `src/chapters/1_introduction.tex` (+37/−14), `src/tables/frame/cosine.tex` (+74/−33).
> Read in `build/main.pdf` (102 pp, `tex_errors=0`, built this session with `make defense`), not
> in the source. Every page number below is a page of that PDF.
> I edited no `.tex` file. This report is the only file I wrote.

## VERDICT

**GATE PASS, with two REQUIRED items that are cheap to fix.**

The hard-ban counters are all zero in the changed prose: em-dash 0, contractions 0, British
spellings 0, banned words and templates 0, repo codenames 0. The four new §2.3 terms are all
registered in `GLOSSARY.md` §4 before the prose landed, and three of the four are defined once and
then reused without re-explanation. The two REQUIRED findings are both §1 register defects that the
ban tables cannot catch: **three sentences narrate the document's own composition** (a
forward-pointer to "the next paragraph", a sentence announcing that a statement is coming, and a
clause explaining why a table has a column), and **the fourth registered term, `gradient conflict`,
is defined in §2.3 without ever being named there**, so the registry entry that Appendix F's result
is supposed to point back to has no anchor in the reader's text.

Nothing in the changed prose reintroduces the process-narration class the author deleted. I looked
for it with six patterns wider than the gate's four, over 79 of 79 live sentences in Appendix F, and
the residue is a different, milder class (findings F-04, F-05, F-06), all three of which sit in
sentences that pre-date this round or restate a design choice rather than an infrastructure fact.

---

## 1 · The counted report (this gate's output is quantitative)

Measured on **live prose only**: comments stripped with the `(?<!\\)%` rule, via
`from check_audit_claims import live_text` (V4). The instrument was validated in both directions
before any verdict: `"Pareto dominance" in live_text(2_fundamentals.tex)` is True (it can see live
prose), `"out of disk" not in live_text(apx_f_cosine.tex)` while `"out of disk" in raw text` is True
(it cannot see comment text). Diff extraction needed `GIT_CONFIG_GLOBAL=/dev/null` in this sandbox;
the first attempt returned **zero added lines for all four files** and I treated that zero as a
broken instrument rather than as a clean result — `assert len(live_added[...]) > 1000` is what caught
it.

### 1.1 The hard bans (a nonzero here is a GATE FAIL)

| Counter | Changed prose | Whole §2.3 | Whole Appendix F | Rendered PDF |
|---|--:|--:|--:|--:|
| em-dash (`—`, `–`, `---`, `--`) | **0** | **0** | **0** | 2, both inside bibliography titles (pp. 85, 88) |
| contractions | **0** | **0** | **0** | **0** document-wide |
| British spellings | **0** | — | — | 3 pre-existing, none in scope (§1.5) |
| banned words + templates (55 patterns) | **0** | **0** | **0** | — |
| repo codenames (17 patterns) | **0** | — | — | 3 pre-existing, none in scope (§1.5) |
| `MTLNet` wrong case | **0** | — | — | 23, all in Ch.4, all registered as the published `ST-MTLNet` |

The two rendered em-dashes are **not prose** and not defects: `Massive-STEPS: … Check-ins — Dataset
and Benchmarks` (p. 85) is a title as published, and `Dissertação (Master's Dissertation) —
Universidade Federal de Viçosa` (p. 88) is `abntex2`'s own `@mastersthesis` separator, emitted by the
style, not typed by anyone. WRITING_LAW §1 bans the em-dash *in prose*; a reproduced title and a
bibliography style's punctuation are neither. Reported here so the number is not mistaken for a
clean-sweep claim: **prose em-dashes are 0, rendered em-dashes are 2, and both are outside the law's
reach.**

### 1.2 Density metrics

| Metric | New §2.3 passage | New Appendix F prose | New table caption | Law's band |
|---|--:|--:|--:|---|
| words / sentences | 508 / 17 | 688 / 28 | 415 / — | — |
| -ly adverb density | **0.59%** (imply, strictly, statistically) | **0.29%** (cheaply, entirely) | 0% | ≈0.8% max |
| two -ly in one sentence | **0** | **0** | 0 | banned |
| intensifiers | **0** | 1 (`entirely`, in a pre-existing sentence) | 0 | ≤1 per claim |
| semicolon braids (2+ per prose sentence) | **1** (F-03) | 0 | 0 | banned |
| rule-of-three noun triads | 0 | 1 (a dataset list, legitimate) | 0 | vary |
| negative parallelism | 1.97/1k | 5.81/1k | 2.41/1k | gate ceiling 3.60/1k |
| sentence length: mean / sd / range | 29.9 / **18.0** / 7–66 | 24.6 / **16.1** / 3–62 | — | preserve burstiness |

`bash src_utils/check_negative_parallelism.py` → **`3.26 per 1k (ceiling 3.60)`, rc=0**, document-wide
[`rather than` 71, `, not` 39, `instead of` 8, `not … but` 1]. The new Appendix F prose runs above the
document ceiling **locally** at 5.81/1k, but the file as a whole **improved**: measured old-vs-new on
`apx_f_cosine.tex`, 6.63/1k → **5.29/1k**. The four local instances are all claim-scoping (`worth
reporting rather than smoothing`, `consistent tendencies rather than established effects`, `describes
the data, not a test's sample size`, `about the mean, not about every observation`) — the honesty
device the law protects, not decoration. **No action.**

`bash src_utils/check_process_narration.py` → **rc=0**, `no process narration in 51 files`. Its
self-test fires on the deleted paragraph (I re-ran that control: 3 of its 4 patterns match the deleted
text), so the zero is a measurement and not a silent pass.

**Burstiness is genuinely preserved and this is the strongest thing about the new passage.** sd 18.0
words with a 7-word sentence (`Not every method claims even that much.`) sitting between a 66-word one
and a 44-word one is the opposite of a smoothing pass. Seventeen sentences, seventeen distinct openers,
no repeated opening move. This is what §4.3 asks for.

### 1.3 The four new terms, checked fail-closed against `GLOSSARY.md` (N of N)

All four are registered in `GLOSSARY.md` §4 (the block dated 2026-07-30 under PENDENCIAS 2.12) and §6.
The registry landed **before** the prose, which is the order L2 requires. Occurrence counts are over
the whole live §2.3, not a sample.

| Term | Registered | In §2.3 | Defined once, then reused? |
|---|---|--:|---|
| Pareto dominance | yes, §4 | 1 | yes — defined at first and only use (p. 22) |
| Pareto optimality / Pareto front | yes, §4 | 5 (`Pareto optimal` ×2, `Pareto optimality` ×2, `Pareto front` ×1) | yes — defined at first use, then used bare |
| Pareto-stationary point | yes, §4 | 3 | yes — defined at first use (p. 23), then used bare twice |
| **gradient conflict** | yes, §4 | **0** | **NO — the concept is defined, the registered NAME never appears (F-02)** |

Two supporting measurements the table depends on:

- **The registry's hyphenation rule holds.** `GLOSSARY.md` §4 says "**Hyphenated, as here.**" All 3 of
  §2.3's occurrences are `Pareto-stationary`; document-wide, 16 live `Pareto*` occurrences, and the
  only unhyphenated one is in `tables/courb/errata.tex`, which the registry explicitly leaves as it
  stands, and which **does not render in the defense build** (searched all 102 pages: zero hits for
  `Pareto stationary` unhyphenated).
- **The registry's "not claimed here" mandate is satisfied in substance.** §4 asks for "Pareto
  optimality is not claimed here" once in §2.3. The prose does not use that string; it says
  *"This dissertation therefore claims no Pareto property of any kind for its models."* (p. 23) —
  which is **stronger and covers more** (all Pareto properties, not just optimality). The gate probe
  `R9-pareto` watches exactly this string. Not a finding; recorded so nobody "fixes" it toward the
  registry's literal wording, which would weaken it.

### 1.4 Term-registry lint on the changed prose

Canonical names in the four changed files: `check-in` (never "event"), `place`/`POI` (never "venue"),
`next category` / `next region` / `next place` kept distinct, `dedicated single-task models` (never
bare "baseline"), `fold`, `seed`, `the shared trunk`, `TOST`. **Zero violations.** The p. 12 edit
keeps the registered gloss intact: *"a check-in records that users visited a given place, a point of
interest (POI), at a given time"* matches the registry's "One visit record (user, POI, timestamp)".
The `\emph{next place}` sentence still delimits the out-of-scope task and p. 12 still states it early,
so the §2 requirement survives the semicolon split.

**Synonym-cycling for the method family: measured, and it is at the edge but defensible.** Nine labels
name the balancer family across §2.3: `the balancing methods` ×2, `the gradient methods`, `these
methods`, `A family of methods`, `this family`, `the family`, `MTL optimizers`, `a balancer`. Three of
those labels are new this round (`the gradient methods`, `these methods`, `this family`; `the
balancing methods` went 1→2). I do **not** call this a defect: each label is doing different work
(`this family` scopes the guarantee claims, `the gradient methods` scopes the ones acting on
directions, `MTL optimizers` is the cited study's own term), and §2's rule targets rotating synonyms
for **one** concept, which these are not. Flagged as OPTIONAL only (F-07), because the density is now
high enough that one more label would tip it.

### 1.5 What is out of scope but worth one line each (NOT findings against this round)

Verified pre-existing at `c94d1f19` in every case, so none belongs to this round's diff, and I am not
reporting them as such:

- `neighbourhood` (p. 34) and `towards` (p. 43) are British spellings in live prose. **p. 43's is
  published CBIC text** (`biased towards the features` is verbatim in
  `articles/CBIC___MTL/sections/conclusion.tex`), so §6's paper-chapter rule protects it. **p. 34's
  is not**: the published `method.tex` writes `neighborhood`, and the dissertation's footnote writes
  `neighbourhood`. That one is the dissertation's own prose and would be a real §1 defect on its own
  ticket.
- `fclass` renders at p. 52 in Ch.4's methodology. `GLOSSARY.md` §3 says **"NEVER write `fclass` in
  prose"**, with `fine class` as the registered name. Three live occurrences, all in Ch.4.
- `embedding-engine suite` (p. 90) uses "engine", which WRITING_LAW §2 lists as a repo codename.
- `frozen` renders at pp. 69, 71, 79. §2 permits it "except frozen weights, glossed" — pp. 69 and 71
  are `with frozen weights (no fine-tuning)`, which is the sanctioned glossed form; p. 79's `with the
  region pathway frozen` is not weights and is the one that would need `fixed`.

---

## 2 · Findings

Severity per the persona's scale: **REQUIRED** = a law violation with no mandated exemption, or a
registry violation; **RECOMMENDED** = a register defect a reader would notice; **OPTIONAL** = a
density or taste item where the current state is defensible.

---

### F-01 · REQUIRED · §2.3 tells the reader what the next paragraph will do, and then that a statement is coming

**File** `src/chapters/2_fundamentals.tex` · **PDF pp. 22–23** · new this round.

Quote, p. 22 (rendered):

> over 𝐾 tasks at shared parameters 𝜽, and **each method named in the next paragraph is a different
> answer** to how the weights 𝑤𝑘 , or the update direction they imply, should be set.

Quote, p. 23 (rendered):

> Reaching that front is not what the balancing methods promise, and **the distance between the
> promise and the front deserves one statement.**

Quote, p. 23 (rendered):

> California makes the same point from the other side and **is the reason the table carries a
> sign-test column at all**  *(this one is F-06; listed there)*

**Why.** WRITING_LAW §1: *"No process narration, and this is a hard ban. The prose states what is
true of the work, never how the work came to be done or what the writing went through."* The fourth
sub-class is **self-reference to the writing**, whose banned instances are *"this appendix originally
reported"*, *"as noted above"*, *"the boundary the paragraph above draws"*. A forward pointer to "the
next paragraph" is the same class pointed the other way: it is a fact about the document's layout, not
about multi-task learning, and it becomes false the moment a paragraph is inserted, split, or
reordered. `deserves one statement` is the second shape — the sentence's subject is the act of stating
rather than the thing stated.

The law's own test convicts both: *"if the sentence would be false or pointless once the circumstance
changes … cut it."* Move the equation next to the methods and the first clause is false; delete the
following sentence and the second clause is pointless.

There is a second, independent problem with the forward pointer: **it does not describe the paragraph
it points at.** The clause promises that the next paragraph gives each method's answer to *how the
weights $w_k$ should be set*. The paragraph immediately after the equation is the Pareto-stationary
guarantees paragraph: it names Nash-MTL, CAGrad, Aligned-MTL and PCGrad, and it states what each
*guarantees*, not how any sets $w_k$. Measured over that paragraph's text: `w_k` 0 occurrences,
`update direction` 0, the verb `set` 0, and the single occurrence of `weight` is *"for task weights
fixed in advance"* — a precondition on Aligned-MTL's theorem, not a weighting scheme. The paragraph
that answers the question is **two later** (`A family of methods tries to manage the conflict…`,
p. 23; 7 occurrences of `weight`, 2 of them `loss weight`), and it adds GradNorm, FAMO, uncertainty
weighting and dynamic weight averaging. So the pointer sends the reader one paragraph short of its
own referent, which is the second reason to cut it rather than repair it.

**Fix.** Cut both meta-clauses; the sentences survive intact and stronger.

- p. 22 → `over $K$ tasks at shared parameters $\boldsymbol{\theta}$. The methods of this section
  differ in how they set the weights $w_k$, or the update direction those weights imply.`
- p. 23 → `Reaching that front is not what the balancing methods promise.` (full stop; the next
  sentence already delivers the statement).

---

### F-02 · REQUIRED · `gradient conflict`, the fourth registered term, is defined in §2.3 but never named there

**File** `src/chapters/2_fundamentals.tex` · **PDF p. 23** · new this round.

Quote, p. 23 (rendered):

> **Conflict has a standard measure**, and it is the quantity the gradient methods act on: the cosine
> of the angle between two tasks' gradients at the shared parameters, negative when the tasks disagree
> and near zero when their updates are close to orthogonal [50].

**Measured.** `gradient conflict` appears **0 times** in the whole live Chapter 2 (whole file, not the
section: `len(re.findall(r"gradient conflict", live_text(2_fundamentals.tex), re.I)) == 0`). It appears
**3 times** elsewhere in live prose, all in Chapter 3 and all pre-existing published CBIC text:
`\textbf{Gradient Conflict}` (p. 32), `mitigated overt gradient conflicts` (p. 43), `beyond gradient
conflict mitigation` (p. 43). The regex was validated against `GLOSSARY.md` itself, where it returns 2
hits, so the zero is a measurement and not a broken pattern.

**Why.** Two clauses bind here at once. `GLOSSARY.md` §4, the `gradient conflict` row: *"The quantity
the gradient-surgery balancers act on, and the quantity Appendix~F measures on the joint model.
**Define it in §2.3.**"* And WRITING_LAW §1: *"every term still gets ONE definition at first use
(Fundamentals is where most of them live) and is then used consistently."* The passage defines the
quantity correctly and completely, and then declines to attach the registered name to it — so the
term's one definitional site does not introduce the term. The reader who meets `Gradient Conflict` as
a bold list item on p. 32 has not been told it is the same thing as p. 23's cosine, and the registry
row's stated purpose (give Appendix F's result "something to point back to") is only half served: the
*measure* points back, the *name* does not.

This is the failure mode of a bare-noun opener. `Conflict has a standard measure` reads well and is
why the omission is easy to miss.

**Fix.** Name it in the defining sentence. Minimum edit, one word inserted:

> `Gradient conflict has a standard measure, and it is the quantity the gradient methods act on: the
> cosine of the angle between two tasks' gradients at the shared parameters, negative when the tasks
> disagree and near zero when their updates are close to orthogonal.`

If the author prefers to keep the shorter opener, the alternative is to name the term in the following
clause (`… close to orthogonal. This quantity is the gradient conflict.`), but the one-word version
costs nothing and puts the name at the definition.

---

### F-03 · RECOMMENDED · a 66-word, two-semicolon sentence in the new §2.3 passage

**File** `src/chapters/2_fundamentals.tex` · **PDF p. 23** · new this round.

Quote, p. 23 (rendered):

> Guarantees in this family are stated at that weaker level: Nash-MTL proves that its updates have a
> subsequence converging to a Pareto-stationary point, and reaches Pareto optimality only under an
> added convexity assumption on the losses that a deep network does not satisfy [47]**;** the fixed
> points of CAGrad are Pareto-stationary [48]**;** and Aligned-MTL converges to such a point for task
> weights fixed in advance [49].

**Why.** WRITING_LAW §4 inherits the density rules, including *"no semicolon braids"* — the inherited
form is explicit: *"a sentence that needs two semicolons is two sentences."* This is the only
2-semicolon prose sentence in the new passage (measured: 3 sentences carry 1 semicolon, 1 carries 2),
and at 66 words it is also the longest. It additionally carries a colon before the first semicolon, so
the reader parses four levels of punctuation before the first citation.

I record the counter-argument because it is real: the three clauses are deliberately parallel (three
methods, three guarantee strengths), and the parallel structure is what makes the paragraph's point
that the methods do not claim the same thing. That is a reason to keep the parallelism, not a reason to
keep it in one sentence.

**Fix.** Split at the first semicolon, keeping the parallel frame:

> `Guarantees in this family are stated at that weaker level. Nash-MTL proves that its updates have a
> subsequence converging to a Pareto-stationary point, and reaches Pareto optimality only under an
> added convexity assumption on the losses that a deep network does not satisfy [47]. The fixed points
> of CAGrad are Pareto-stationary [48], and Aligned-MTL converges to such a point for task weights
> fixed in advance [49].`

---

### F-04 · RECOMMENDED · "obliges one to say" is a register outlier: the only impersonal *one* in the document

**File** `src/chapters/2_fundamentals.tex` · **PDF p. 22** · new this round.

Quote, p. 22 (rendered):

> **Casting it that way obliges one to say** what an optimum would be.

**Measured.** I swept the whole live document for impersonal-*one* constructions (`obliges/requires/
allows/forces one to`, `one must/should/can/may/would`, `if one`). **Exactly one hit, this sentence.**
The detector was validated on a planted positive.

**Why.** WRITING_LAW §1's register bar: *"standard academic English a Brazilian author would defend
aloud … The test: would the author say it at the defense, and would the community write it?"* The
impersonal *one* as a grammatical subject is a formal-literary English construction, and the paper
GLOSSARY §8 names exactly this pole — *"native-literary idiom … that a non-native author would not
produce"*. A Brazilian author defending in English says "this requires us to say" or, better, drops
the frame. And at n=1 in 102 pages, it is a visible seam: every other agentive sentence in the
document uses *we*, *this dissertation*, or an inanimate subject.

Not REQUIRED, because it is not on a ban list and the sentence is grammatical and clear.

**Fix.** `That framing requires a definition of the optimum.` — or keep the author's own voice:
`Casting it that way requires us to say what an optimum would be.`

---

### F-05 · RECOMMENDED · Appendix F still explains why Georgia is in the evidence, in the sentence that survived the deletion

**File** `src/chapters/apx_f_cosine.tex` · **PDF p. 97** · **pre-existing at `c94d1f19`**, and it sits
inside the changed block (the added lines rewrote the sentence before it, from `Three of the four…` to
`Six of the seven…`).

Quote, p. 97 (rendered):

> The seventh, Georgia, is a further Gowalla state the dissertation does not otherwise use; **it enters
> because the diagnostic ran on it cheaply and stays because dropping a measured dataset would be a
> choice about the evidence.**

**Why.** This is the closest surviving relative of the deleted paragraph, and it is the answer to the
question the author asked me to check. WRITING_LAW §1's third sub-class is **scheduling and provenance
of the agent's own effort**, and the second is the document's version history; this clause is neither
of those exactly, which is why the four gate patterns do not catch it. What it is: **an explanation of
why a dataset is present rather than a statement of what the evidence is.** `ran on it cheaply` is a
fact about a run's cost, one category away from the banned `a matter of computational resource`, and
`stays because dropping a measured dataset would be a choice about the evidence` narrates an editorial
decision about the document.

The law's disposal rule applies verbatim: *"A limitation the reader must know ('this appendix covers
one architecture family') is a LIMITATION and stays, stated as a property of the evidence. The reason
the limitation exists goes in the provenance comment."* Georgia's non-membership in the six **is** a
property of the evidence and must stay; **why it was run** is provenance.

I am deliberately not calling this REQUIRED. It pre-dates this round, the "stays because" half is a
defensible statement of research ethics (it tells the reader no measured dataset was dropped, which is
a real anti-cherry-picking disclosure), and cutting it wholesale would lose that. It is on this list
because the author asked whether more of the class survived, and this is the answer: **yes, one clause,
milder, and here it is.**

**Fix.** Keep the disclosure, drop the cost narration:

> `The seventh, Georgia, is a further Gowalla state the dissertation does not otherwise use. It is
> reported because it was measured under the same protocol, and no measured dataset is omitted here.`

---

### F-06 · RECOMMENDED · the new California sentence, and the table caption, explain why a column exists

**File** `src/chapters/apx_f_cosine.tex` (p. 100) and `src/tables/frame/cosine.tex` (pp. 7, 100) ·
new this round.

Quote, p. 100 (rendered, body prose):

> California makes the same point from the other side and **is the reason the table carries a
> sign-test column at all**: its 𝑡-test returns 0.048, below the conventional threshold, while its
> exact sign test returns 0.375 on four of five positive folds, so what looks like a finding under one
> test is not even a leaning under the other.

Quote, pp. 7 and 100 (rendered, table caption):

> **Texas illustrates why the positive-fold count is reported next to the mean**: four of its five
> folds are positive and its mean is negative, because one fold at −0.0032 outweighs four smaller
> positive ones.

**Why.** Same §1 sub-class as F-01 (self-reference to the writing), one step removed: the subject of
each clause is a **design decision about the table**, not a property of the data. And the first one is
**false as history**, which is what moves it from taste to defect: I checked the table at `c94d1f19`
and the `$t$ / sign` column, the `\dagger` marker and the `0.0625`-floor footnote were **all already
there**, on a four-dataset table, before California was measured. So `is the reason the table carries a
sign-test column at all` asserts a causal history the repository contradicts. The statistical content
of the sentence is sound and valuable — California is the cleanest case where the two tests disagree —
and it survives the cut untouched.

The Texas clause is not false, only self-referential; and it lands in the **List of Tables** as well as
the body, where a reader scanning a contents page is told why a column exists in a table twelve pages
later.

**Fix.** State the fact, not the rationale.

- p. 100 → `California makes the same point from the other side, and it is the case where the two
  tests disagree most plainly: its $t$-test returns $0.048$, below the conventional threshold, while
  its exact sign test returns $0.375$ on four of five positive folds.`
- caption → `At Texas, four of five folds are positive while the mean is negative, because one fold at
  $-0.0032$ outweighs four smaller positive ones.`

---

### F-07 · OPTIONAL · the table caption is now 194 words and reproduces in full in the List of Tables

**File** `src/tables/frame/cosine.tex` · **PDF pp. 7 and 100** · new this round.

**Measured.** Table 11's caption: **127 → 194 words** this round (brace-matched `\caption{}`, comments
stripped). Its List-of-Tables entry is **196 words**, against a median of **31** across the 11 entries
and a previous maximum of 85 (Table 10). No `\caption[short]{long}` form is used anywhere in the
document (28 `\caption` calls, 0 with the optional argument). The figure caption grew **54 → 97 words**
and is 4 sentences, which is inside the law's 2–4 band.

**Why.** WRITING_LAW §5 requires *"every results table introduced by a lead takeaway sentence"* and
sets no caption word limit, so this is **not** a violation — hence OPTIONAL. The reader-facing harm is
in the front matter: pp. 6–7's List of Tables gives one entry six times the median length, including a
sub-clause about why a column exists (F-06) and a `†` footnote gloss, which a contents page cannot use.

Also note, on the same page, a **spacing defect in the LoT copy only**: p. 7 renders
`California's𝑡-test` with no space before the math italic, while p. 100's body copy renders
`California's 𝑡-test` correctly. Same source string, different line-breaking. That one is a LaTeX
matter rather than a style matter, so I am flagging it for persona 19 rather than proposing an edit.

**Fix (if the author wants it).** Add the short form so the LoT carries the takeaway only:
`\caption[The cosine between the next-region and next-category gradients, seven datasets]{…full text…}`.
One edit, no prose loss, and it fixes the front-matter spacing artifact as a side effect.

---

## 3 · What is good (the calibration half of this report)

1. **Burstiness is real, not simulated.** sd 18.0 words over 17 sentences, range 7–66, with the 7-word
   sentence (`Not every method claims even that much.`) placed for effect between two long ones. Seventeen
   distinct sentence openers. WRITING_LAW §4.3 is the deepest tell in the law and this passage passes it
   on measurement, not on impression.
2. **The paragraph openers do not template.** Across §2.3's fourteen paragraphs: `Multi-task learning
   (MTL) trains…`, `Deep MTL is organized…`, `Sharing is not free.`, `Casting it that way…`, `Reaching
   that front…`, `Conflict has a standard measure…`, `A family of methods…`, `In mobility, MTL has been
   used…`. Three of those are new this round and none reuses the shape of its neighbours (§4.4).
3. **Zero numbers in the new §2.3 passage.** One numeral in 508 words, the `k=1` of the summation index.
   The provenance comment says this is by design (`CONSIDERATIONS` item 28: define the measure in Ch.2,
   report its value where it was measured) and the prose delivers it. That discipline is why F-02 is a
   naming fix and not a claim problem.
4. **The methods are described as their own authors describe them, at differentiated strength.**
   `PCGrad … makes no Pareto claim at all`, `the fixed points of CAGrad are Pareto-stationary`,
   `Nash-MTL … only under an added convexity assumption`. Four methods, four different guarantee
   strengths, no flattening. This is the opposite of the AI failure mode where every cited system gets
   the same verb.
5. **The strongest sentence in the passage earns its place.** *"Orthogonality is not a conflict resolved
   but a conflict absent, which puts a limit on what any of these methods can contribute."* (p. 23) It is
   negative parallelism, which is a tell — and it is scoping a claim, which is the exemption the law
   grants. Keep it verbatim.
6. **Appendix F's parallelism density went down while the file grew**, 6.63/1k → 5.29/1k. An edit pass
   that adds 205 lines and *reduces* the tell density is the opposite of the regression §4.3 warns about.
7. **The introduction edits are all improvements and all clean.** `\section{Research question}` is
   shorter and more accurate than the old heading; the FAB-16 semicolon split removed a braid rather than
   creating one (`is a third and different problem, not addressed in this dissertation.` + `Chapter 2
   keeps the three tasks formally distinct.`); dropping `, Fundamentals,` and `, Conclusion,` from the
   chapter bullets removes a redundancy the `\ref` already covers. Zero new banned tokens; the file's
   parallelism moved 2.11 → 2.65/1k, one instance, inside the ceiling.
8. **The registry was updated before the prose landed**, and the four rows carry their source page. That
   is L2's order, and it is the reason F-02 is the only registry finding rather than four.

---

## 4 · Proposed law update (author approval, not applied)

WRITING_LAW §1's process-narration ban lists four sub-classes, all of them backward-looking (what the
work went through, what an earlier version said, when a measurement was taken, what a previous paragraph
did). Three findings here (F-01 twice, F-06 twice) are the **same offense pointed at the document's own
structure** rather than at its history, and the four gate patterns cannot see them because they contain
no infrastructure noun, no version word and no past-tense effort verb:

- forward pointers to layout: `the next paragraph`, `the section below`, `the paragraph that follows`;
- announcements that a statement is coming: `deserves one statement`, `needs saying plainly`, `is worth
  one sentence`;
- rationales for the document's own apparatus: `is the reason the table carries … at all`, `illustrates
  why X is reported`, `is why this appendix stops at`.

**Proposed fifth sub-class, "the document's own apparatus":** *the prose does not explain why the
document is arranged as it is. A table's columns, a section's ordering, and a sentence's own necessity
are facts about the writing, not about mobility data. State the content; let the arrangement be
invisible.* The test is the same one §1 already gives: reorder the material and the sentence becomes
false.

If the author accepts it, `check_process_narration.py` takes a fifth rule tuple with the three pattern
families above, and its self-test gains F-01's and F-06's quotes as positives — both are real instances
from this repository, which is the standard that file already holds itself to. I did not touch the
checker; that is the applying agent's work.

---

## 5 · Measured — every count with the command that produced it

Working directory `articles/dissertacao` unless stated. Comment stripping is
`from src_utils.check_audit_claims import live_text` in every sweep (V4). This sandbox needs
`GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null` on every `git` call, or `git diff` exits 128
with `unable to access '/Users/vitor/.gitconfig'` **and prints nothing**, which reads exactly like an
empty diff.

```bash
# the build I read (this is the artifact every page number refers to)
cd src && make defense
#   -> latexbuild main -> build/main.pdf  pages=102  tex_errors=0

# the changed-line sets (added lines only, comments stripped afterward)
git diff c94d1f19..HEAD -- articles/dissertacao/src/chapters/2_fundamentals.tex
#   -> +106 added lines -> 3,343 chars of live prose
#      apx_f_cosine.tex +148 -> 4,201 | 1_introduction.tex +25 -> 930 | cosine.tex +51 -> 2,453
#   INSTRUMENT CHECK (V13): assert len(live_added["2_fundamentals.tex"]) > 1000
#      -- the first run returned 0 for all four files (git exit 128) and I did not report that zero

# the gates, run last, exit codes read per gate (V11/V13)
python3 src_utils/check_process_narration.py   #  rc=0  "no process narration in 51 files"
python3 src_utils/check_negative_parallelism.py #  rc=0  3.26/1k vs ceiling 3.60
```

Sweeps run in-kernel over `live_text()` of all 51 `.tex` files under `src/` (`SRC.rglob("*.tex")`,
`"build" not in p.parts`):

| Measurement | Command / method | Result |
|---|---|---|
| em-dash, changed prose | `re.findall(r"[\u2014\u2013]|---", live_added[f])` | 0 / 0 / 0 / 0 |
| em-dash, rendered | `t.count("\u2014")` per page over 102 pages | 2 (pp. 85, 88), both bibliography |
| contractions | 20-pattern regex over all live prose **and** all 102 rendered pages | 0 and 0 |
| banned words + templates | 55 patterns (WRITING_LAW §4 + paper GLOSSARY §7 tables) | 0 in changed prose, 0 in whole §2.3, 0 in whole Appendix F |
| idiom sweep | 12 patterns (paper GLOSSARY §8) | `sits` ×1 (F-07 note), `carries` ×2 new |
| `carry/carries` budget | `re.findall(r"\bcarr(?:y\|ies\|ied)\b", …)` per file, old vs new | apx_f 3→4, Ch.2 5, cosine.tex 0 — over the ≤3 per chapter guidance in Appendix F; 2 of its 4 are new |
| codenames | 17 patterns (WRITING_LAW §2) | 0 in changed prose; 3 pre-existing document-wide (§1.5) |
| new-term occurrences | `re.finditer` per term over the whole live §2.3, **N of N read** | dominance 1, optimality/front 5, stationary 3, `gradient conflict` **0** |
| `gradient conflict` control | same regex over `GLOSSARY.md` | 2 hits — the zero above is real, not a broken pattern |
| `Pareto*` census | all live `.tex` | 16 occurrences, 11 in Ch.2, 1 unhyphenated (errata table, does not render) |
| -ly density | `\b\w+ly\b` minus a 12-word stoplist | §2.3 new 0.59%, apxF new 0.29% |
| sentence stats | `re.split(r"(?<=[.!?])\s+")` after de-TeX | §2.3 new: 17 sents, sd 18.0, 7–66; apxF new: 28 sents, sd 16.1, 3–62 |
| parallelism, per file old vs new | 4 patterns over `live_text` at `c94d1f19` and at HEAD | apx_f 6.63→5.29/1k, cosine.tex 0→2.30, Ch.2 4.39→4.15, Ch.1 2.11→2.65 |
| process narration, wide net | 6 pattern families over **79 of 79** live Appendix F sentences | 3 sentences flagged (F-05, F-06, and one pre-existing `is the point of this appendix`) |
| gate-pattern control | the 4 `RULES` from `check_process_narration.py` against the deleted paragraph's text | 3 of 4 fire — the gate's zero is a measurement |
| caption lengths | brace-matched `\caption{}`, de-TeX'd word count | table 127→194 words; figure 54→97 words, 4 sentences |
| List of Tables | parsed `Table N –` entries from rendered pp. 5–9, `assert rows` | 11 entries, median 31 words, Table 11 = 196 |
| quote↔page pairing | substring match of each quoted string against each of the 102 rendered pages, `assert pg in got` for all 16 | all 16 verified |

**What I did not measure, stated so it is not read as covered.** **The gate suite, at the time this
report was drafted.** The two gate runs above are individual scripts; I did not run
`bash src_utils/check.sh` until later in the session (§5b's correction and §5c), and when I did it came
back **rc=1** on a condition unrelated to my remit. Every style verdict here rests on the in-kernel
sweeps and on the two scripts named, none of which depends on the suite's state — but no sentence in
this report may be read as certifying the tree. Also not measured: whether each citation supports its
sentence (personas 05/07); whether the numbers in the table match the parquet (persona 06); the LaTeX
cause of the p. 7 `California's𝑡-test` spacing (persona 19). The `gradient conflict` verdict in F-02 is
about the **name's absence in Chapter 2**; I did not audit whether Chapter 3's three uses of it are
consistent with the registry, because they are outside this round's diff.

---

## 5b · Why no probe catches F-02 (a gate defect, not a prose one)

> **CORRECTION, and it is mine.** This section originally opened *"Twenty-two gates are green on this
> tree"*, and that sentence was false twice over. I never ran the suite: the only gate runs behind this
> report are the two individual scripts in §5 (`check_process_narration.py`,
> `check_negative_parallelism.py`). The "22" came from this repository's commit conventions, which I had
> read in `AGENT_GUARDRAILS.md` §4b, not from any output — the exact defect V1 names, a number about the
> work with no command behind it. Worse, the tree was **not** green when I wrote it: the first whole-suite
> run in this session, taken later, printed `check.sh RC=1` with `ORPHANED item 2.8 left the tracker
> without reaching the archive`, and that condition already held at the time (the `PENDENCIAS.md` edit's
> mtime, 11:05:48, precedes this report file's 11:15:31). So the heading's own framing was wrong: F-02 did
> not survive a *green* suite. It survived **two probes that cannot see it**, which is the finding, and it
> stands independent of any suite state.

`check_audit_claims.py` carries **two** round-9 probes for this passage. Neither can see F-02, and the
reason is worth recording because it is the same shape as V13's proxy problem.

```
("R9-conflict",  "Ch.2 defines gradient conflict as the cosine between per-task gradients, so "
                 "Appendix F's orthogonality result has a definition to point back to",
                 "chapters/2_fundamentals.tex", r"cosine of the angle between two tasks", True)
("R9-glossary",  "the four Pareto/conflict terms are registered before the prose uses them",
                 "../GLOSSARY.md", r"\*\*Pareto-stationary point\*\*", True)
```

- **`R9-conflict` watches the definition, not the name.** Its pattern is `cosine of the angle between
  two tasks`. That string is present, so the probe passes — and it would pass identically if the words
  `gradient conflict` never appeared anywhere in the document, which is the state the tree is in. The
  probe's own label says the purpose is that Appendix F "has a definition to point back to"; what
  Appendix F actually needs to point back to is a **named** term, and the probe cannot tell the two
  apart.
- **`R9-glossary` is labelled for four terms and checks one row.** Its pattern is
  `\*\*Pareto-stationary point\*\*` (1 possible match). Measured: deleting the `**Pareto dominance**`,
  `**Pareto optimality**` or `**gradient conflict**` row from `GLOSSARY.md` leaves this probe green,
  because the pattern does not reference them. The label claims a coverage the pattern does not have.

Neither is a finding against the prose, and I am not proposing the gate edit (that is the applying
agent's work). It is here because the author's standing question after V14 is *what should have
surfaced this unprompted*, and the answer is a third probe of the inverted kind the registry rule
actually implies: **for each of the four registered terms, the term's own name must appear in live
prose in the chapter that defines it.** That probe fires today on `gradient conflict` and passes on the
other three, which is the both-directions validation §7 requires, available for free.

## 5c · `PENDENCIAS.md`: restored on the author's instruction, gates green, and one risk he needs to know

**Done, and verified.** `git checkout -- articles/dissertacao/src_utils/PENDENCIAS.md`. Item 2.8 is back
(1 occurrence of `^### 2.8`), the file is clean against HEAD, and `bash src_utils/check.sh` → **rc=0**,
read directly as the last action in that shell (V11). The three named gates now read:

```
ORPHANED …            -> gone (grep for ORPHAN over the suite log returns nothing)
holds       R9-pend28 the old 2.8 no longer asks for a decision -- it records what was done
holds       R9-agree  PENDENCIAS 2.8 carries the CORRECTED stale count … (9 of 41, not the superseded 10)
```

Before restoring I preserved the working-tree version, because `git checkout --` on an uncommitted file
is unrecoverable and 385 insertions had no other copy:

- `_round9/_UNCOMMITTED_pendencias_worktree_1105.md` — the full file, **verified byte-identical**
  by `cmp` and md5 `cedb699a54a587eec3edebd10c4477db` before the checkout, not by a token guess.
  (My first attempt asserted on a string I *expected* to be in it; that assert fired and stopped the
  script, which is V15b working as intended — the token was in `check_audit_claims.py`, not in
  `PENDENCIAS.md`. The backup was then verified with the instrument that matches the question.)
- `_round9/_UNCOMMITTED_pendencias_worktree_1105.diff` — the 1,084-line diff, 33 hunks.

Both are left **untracked and uncommitted**, since the instruction is to commit only this report.

**One diagnostic appeared beside that green exit code. I first blamed the tree; it was mine, and it
damaged a commit message.** The shell that made the previous commit also printed
`make: *** No rule to make target 'defense'. Stop.` A green rc next to a failing `make` is the
tolerant-tool bias §7 lists, so I did not let the 0 stand: re-run with the streams separated, `check.sh`
gives **rc=0 with 0 lines of stderr**, so the message is not the suite's. That much held. I then wrote
that it "belongs to something else in the tree" and was "consistent with one of the four concurrent
tracks building from the wrong working directory" — **and that was wrong.** The reviewer traced it to my
own shell, and the mechanism reproduces on demand:

- That cell begins `cd /Users/vitor/Desktop/mestrado/ingred` (the repo root), and its
  `git commit -m "…"` message is **double-quoted** and contained the span `` `make defense` ``.
- Bash executes backticks inside double quotes as command substitution. From the repo root there is no
  `defense` target, so `make` failed — printing to stderr — and substituted **empty output**.
- The consequence is not cosmetic. The committed message of `1da890cf` now reads
  *"that one IS a side effect of my  and is deliberately unstaged"*: the phrase was **eaten**, leaving a
  double space where `make defense` should be. That commit is the only one of my four with an internal
  double-space anomaly (`grep -c '[a-z]  [a-z]'` → 1, versus 0 for `55620fbf`, `e1ad7ece`, `3a4a1ca7`),
  which is the signature of the substitution and confirms it happened exactly once.

I had already measured `make -n defense` failing at the root and succeeding in `src/` — the correct
conclusion was sitting in my own output, and I read it as evidence about someone else because the
section I was writing was about concurrent tracks. **Recorded rather than quietly fixed:** the `make`
line is mine, `1da890cf`'s message is missing two words and cannot be rewritten without rewriting
history, and the lesson is that a `git commit -m` message must be single-quoted or backtick-free.
Neither error touches a style finding — the gate rc=0 is a real measurement of the suite — but both were
attributions I stated with more confidence than the evidence carried.

**The provenance, stated as what I can and cannot prove.** I did not write to that file, and there is no
write to it anywhere in this track's tool log; the one mention of the name in this report is line 95, a
parenthetical citing `GLOSSARY.md`'s own provenance note. What I can prove positively:

| Evidence | Measured |
|---|---|
| `PENDENCIAS.md` last **committed** | `a847ed8f`, **09:49:23** local — before this track's first tool call (≈10:16) |
| The worktree edit's mtime | **11:05:48** local |
| Concurrent tracks committing in that window | **four**, 13 commits between 10:45:55 and 11:23:02 (`40_readability_r9b` at 11:07:42, `39_mtl_r9b` at 11:09:22 and 11:16:21, `41_ai_tells_r9b` at 10:58:50, `42_excellence_r9b` at 11:23:02) |
| This track's writes | `38_style_r9b.md` only (11:12:17, 11:15:31), plus the two backups above |
| `dissertacao.pdf`, also dirty | mtime 11:20:09 at the time of measurement, identical to `build/main.pdf`'s — **not my build**, which completed at **10:18:19** local, about 62 minutes earlier. Re-checked later: both are now 11:38:32, a *third* build. Another track is building repeatedly into this tree. I staged neither |

The 11:05:48 edit falls inside another track's commit sequence and outside any write of mine. I cannot
prove a negative from mtimes alone, and I am not asking you to take my word for it — the two facts that
matter are that the tree is restored and green, and that **the work may still be live somewhere.**

**The risk, which is the actionable part.** If one of the four concurrent tracks is holding that
restructure in flight, my `git checkout --` just discarded **385 insertions / 279 deletions** of its
uncommitted work with no notice to it. That is why I copied the file first. Its shape, measured from the
preserved diff so you can identify the owner without opening it: **33 hunks, 367 added and 251 removed
content lines, exactly one heading removed (`### 2.8 CONSIDERATIONS.md — EXECUTADO nesta rodada; a fila
de decisao virou o §6`) and none added.** So it is a content rewrite across the file, in the shape of a
compression pass, and *not* a re-sectioning — which makes the loss of 2.8 look like the incidental
casualty of a rewrite rather than its purpose. Recovering it is `cp` from the preserved path; deciding
whether it should land is yours.

**No proposal attached.** I have no view on whether that restructure has value, because I never read it
as content — I measured its shape to write this paragraph and nothing more. It is not my file and not my
remit.

## 6 · Findings index

| id | severity | file | PDF page | one line |
|---|---|---|---|---|
| F-01 | REQUIRED | 2_fundamentals.tex | 22, 23 | forward pointer to "the next paragraph" (also misdirects by two paragraphs) and "deserves one statement" |
| F-02 | REQUIRED | 2_fundamentals.tex | 23 | `gradient conflict` defined but never named; 0 occurrences in the whole chapter |
| F-03 | RECOMMENDED | 2_fundamentals.tex | 23 | 66-word, two-semicolon sentence |
| F-04 | RECOMMENDED | 2_fundamentals.tex | 22 | "obliges one to say", the document's only impersonal *one* |
| F-05 | RECOMMENDED | apx_f_cosine.tex | 97 | Georgia "enters because the diagnostic ran on it cheaply" (pre-existing) |
| F-06 | RECOMMENDED | apx_f_cosine.tex, cosine.tex | 100, 7 | two clauses explaining why the table has a column; the first is false as history |
| F-07 | OPTIONAL | cosine.tex | 7, 100 | 194-word caption reproduced whole in the List of Tables |
