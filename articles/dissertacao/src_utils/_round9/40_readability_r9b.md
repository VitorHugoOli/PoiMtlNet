# 40 · Readability editor — round 9b, the changed prose only

> **Filename note.** First committed as `_round9/35_readability.md` in `195bf79e`, renamed here at the
> author's instruction because the 33–37 slots were already allocated. **No existing file was
> overwritten:** `git cat-file -e 195bf79e^:…/_round9/35_readability.md` returns *"exists on disk, but
> not in 195bf79e^"*, `git log --all` shows that path touched by exactly one commit (mine), and slot 35
> at `195bf79e^` was held by `35_wave_a_render_check.py`, a different filename. The rename is recorded
> by git as `R100`, a pure rename at 100 percent similarity. The earlier readability report is at
> `_round9/reviews/15_readability.md` (build `901a0408`, 08:41–09:14 today) and is untouched.

**Persona:** `reviewers/15_readability_editor.md` (senior academic editor; ten lenses; craft and reader
experience, not law enforcement — personas 02/03 own sentence mechanics and the ban lists).
**Reviewed:** `git diff c94d1f19..HEAD` over four files, read in the **rendered PDF**.
**Build:** `cd src && make defense` → `build/main.pdf`, **102 pages, `tex_errors=0`, exit 0**, ~7 min wall.
Every quote below is from that PDF's own text layer, with its page.
**Read first, in order:** `reviewers/15_readability_editor.md`, `WRITING_LAW.md` (all 193 lines,
including the §1 process-narration ban added today), `AGENT_GUARDRAILS.md` §4b (V1–V15c),
`GLOSSARY.md` (all 192 lines), `CLAUDE.md`, `reviewers/README.md` Common protocol.

**Severity:** REQUIRED / RECOMMENDED / OPTIONAL, as the task specifies. My persona's own scale
(Critical / Major / Minor / Strength) is given in brackets so the mapping stays visible.

**Relation to the prior readability pass.** `_round9/reviews/15_readability.md` ran against build
`901a0408` with scope "frame chapters only" and no build of its own. It found the §2.3 pointer defect
(its SF-4) and declared in its UNFINISHED §6 that it had not diffed the new §2.3 block against the
prior commit. **R-4 below is that finding, confirmed against the diff it could not run**; I claim no
originality for it. Appendix F, `tables/frame/cosine.tex`, the front matter and the rendered-vs-source
question were outside that pass entirely, and findings R-1, R-2, R-3, R-5, R-6, R-7, R-8 are new.
Finding-by-finding agreement is in the next section.

---

## Where I agree with `reviews/15_readability.md`, and where I differ

I read all 16 of its findings (12 should-fix + 4 nits, and its own tally reconciles with its rows — I
checked, because a header that miscounts its rows is this project's V13 fourth instance). Its scope was
Ch. 1 / 2 / 6 as prose; mine is the round's diff as rendered. **Four of its sixteen fall inside my
scope**; the other twelve are outside it and I neither confirm nor contest them.

| its finding | in my scope? | my position |
|---|---|---|
| **SF-4** pointer "next paragraph" | yes, §2.3 new block | **AGREE, and I close its open question.** It could not tell whether the defect was introduced this round. It was: the sentence is inside `beebd33b`. My **R-4** adds the measurement it lacked (the guarantees paragraph is the document's *first* mention of CAGrad, Aligned-MTL, GradNorm and FAMO), which suggests a second repair it did not consider. |
| **SF-7** "Two \<plural\> \<verb\>" ×6 | one of its six sites is in the new block | **AGREE on the pattern, DIFFER on one site.** See below. |
| **SF-8** the Wilcoxon sentence | Ch. 2, but **not** in this round's diff | **AGREE it is the chapter's hardest sentence** (66 words, measured; joint third longest of 185). Not a round-9 finding: `git diff c94d1f19..HEAD` does not touch it. I flag the overlap because its fix and my **R-1**'s fix land in the same paragraph pair. |
| **SF-10** §2.5 says the gap three times | yes, and it constrains my R-3 | **AGREE, and it changes my own recommendation.** My R-3 asks §2.5 to pick up the orthogonality thread; SF-10 says §2.5 is already saying its argument three times. Both can be right, so R-3 proposes a clause that *replaces* rather than adds, and says so. |

**The one place I differ, and it is narrow.** SF-7 lists six sites of the "announce the count, then
enumerate" template and treats them as one habit. One of the six, `2_fundamentals.tex:445`, is inside
the block I review, and it is a **different construction**. Measured, all six classified, N of N:

| site | paragraph starts | sentence | construction |
|---|--:|--:|---|
| `2_fund:82` | L82 | 1 of 10 | announce-then-enumerate, paragraph opener |
| `2_fund:624` | L624 | 1 of 4 | announce-then-enumerate, paragraph opener |
| `6_concl:110` | L110 | 1 of 12 | announce-then-enumerate, paragraph opener |
| `2_fund:164` | L150 | 9 of 12 | announce-then-enumerate, mid-paragraph |
| `6_concl:48` | L43 | 4 of 8 | announce-then-enumerate, mid-paragraph |
| **`2_fund:445`** | **L433** | **6 of 8** | **partitive: "Two of these papers"** |

"Two of these papers state the residual limitation themselves" does not announce a count and then
enumerate; it selects two members of a set the reader already has (the five methods named two sentences
earlier) and it enumerates nothing. Three of the six are the template as described; two are the same
construction mid-paragraph; the sixth is not the construction. **Consequence for the author:** if SF-7's
fix is applied by pattern-matching "Two \<plural\>", this sentence will be rewritten for a tic it does
not have, and rewriting it would cost the citation scoping ("two of these papers", `liu2021cagrad` and
`senushkin2023aligned`) that makes the attribution precise. I recommend leaving `:445` as it stands and
applying SF-7 to the other five.

I also **agree with its empty blocker section** and reach the same result independently: nothing in the
changed prose defeats comprehension.

---

## Verdict

**Defensible with two required fixes, neither of which is in the new prose itself.** The §2.3 Pareto
block earns its length and is the most disciplined new writing in the chapter. The problems are at the
seams: two paragraphs the source separates are rendered as one because a comment block bridges them
(R-1), and the cosine table's caption has grown to 197 words, the longest in the document, which the
front matter then reprints in full so that Table 11 alone consumes 38 percent of the List of Tables
(R-2). Appendix F is still an appendix, not an article, and I say so with numbers below.

**Top three:** R-1 (a paragraph break the reader does not receive, twice), R-2 (the caption is sized
like a subsection), R-3 (§2.5 does not pick up the thread §2.3 just started).

---

## What I measured, and how

Instrument: `check_audit_claims.live_text` / its `COMMENT = (?<!\\)%` rule, imported live, extended to
a paragraph-preserving variant that reproduces TeX's own rule (a blank line breaks a paragraph; a
comment-only line does not). Validated in both directions before use: the deleted out-of-disk sentence
is present in the raw file and absent from `live_text` (68.3 percent of `apx_f_cosine.tex` is comment),
live prose survives, and every parse asserts `len(rows) > 0` — a zero here would be a broken
instrument, not a clean result. Word counts strip inline math, `\cite`, `\ref` and macros, so they
count what a reader reads.

### Sentence statistics (rendered prose, comments stripped)

| set | n sents | mean | median | sd | >35w | >45w | max |
|---|--:|--:|--:|--:|--:|--:|--:|
| **Ch. 2 baseline** (28 paragraphs, new block excluded) | 168 | 26.9 | 25 | 14.9 | 25.6% | 10.1% | 80 |
| **new §2.3 block** (3 paragraphs, pp. 22–23) | 17 | **29.6** | 23 | 17.6 | 29.4% | **17.6%** | 66 |
| **Appendix F** (20 paragraphs, pp. 97–102) | 76 | 23.0 | 23 | 12.7 | 14.5% | 5.3% | 62 |
| Ch. 1 (7 advisor items applied) | 48 | 27.2 | 26 | 14.7 | 29.2% | 10.4% | 65 |

All four rows are counted the same way, on the comment-stripped **source**, so they are comparable.
**A correction to my own first reading, recorded because it changed a verdict.** I first measured the
new block on the extracted PDF text and got mean 27.7 over 18 sentences, which is within one word of the
chapter baseline; that is not comparable to the other three rows. The display equation (2.4) sits inside
the block's second sentence, and the extractor splits that sentence at the equation while the baseline
rows were never split anywhere. Counted consistently, the block's mean is **29.6, which is 2.7 words
above the chapter's own**, and the share of sentences over 45 words is **17.6 percent against 10.1**.
Both readings are defensible about what a *reader* experiences, since the equation does give the eye a
break, but only the source-consistent one may be compared to the baseline, and it does not support
"within one word". The block's length is still earned (see the word budget below); its *sentences* run
longer than the chapter's, which is finding R-5.

Variance holds up in either reading: sd 17.6 against the baseline's 14.9, so this is not the flattened
prose `WRITING_LAW` §4.3 warns about. Appendix F is the **easiest** prose in the changed set on every
measure. Neither is where the prolixity is.

### Paragraph and section budget

Chapter 2 is 5,014 prose words (cross-checked: section sum + preamble = all-prose sum). §2.3 is 1,303
of them (26.0 percent); the new Pareto block is 503 words, 38.6 percent of §2.3 and 10.0 percent of the
chapter. Its three paragraphs are 152 w / 6 sents, 222 w / 8, 129 w / 3, against a chapter median of
147.5 w. **None of the three is an outlier.** The chapter's two longest paragraphs are older prose at
319 words each (pp. 25–26 and p. 19).

### Appendix F against its peers

| appendix | body words | rendered pages |
|---|--:|--:|
| A Contributions | 369 | 3 |
| C AI-use disclosure | 156 | 1 |
| E Data ethics | 833 | 3 |
| **F Cosine** | **1,753** | **6** |
| (B errata, D benchmark — supplementary volume) | 2,159 / 893 | — |

Against the paper chapters: Ch. 3 = 4,440 words, Ch. 4 = 4,330, Ch. 5 = 5,435. Appendix F at 1,753 is
**32 percent of the shortest paper chapter** and grew only 40 words this round (1,713 → 1,753) while
adding three datasets. It is the longest appendix in the defense build and the second longest overall,
but it is not article-sized, and it did not become one this round.

### Captions, against all 25 in the document

Document-wide caption median 37 words, mean 48.7.

| caption | words | sents | rank |
|---|--:|--:|--:|
| **Table 11, cosine** | **197** | 6 | **1 / 25** |
| Table (Appendix D benchmark) | 129 | 6 | 2 |
| **Figure 8, cosine** | **97** | 4 | **3 / 25** |
| Table 10, MobiWac main results | 82 | 3 | 4 |

Both grew this round: Table 11's caption 129 → 197 words, Figure 8's 54 → 97.

### Commands

```bash
cd articles/dissertacao/src && make defense                 # 102 pp, tex_errors=0, exit 0
git diff --stat c94d1f19..HEAD -- articles/dissertacao/src/
git show c94d1f19:articles/dissertacao/src/chapters/apx_f_cosine.tex > /tmp/apxf_old.tex   # and 2_fundamentals, tables/frame/cosine
python:  sys.path.insert(0, 'src_utils'); from check_audit_claims import live_text, strip_text, COMMENT
         # paragraph-preserving stripper on the same COMMENT rule; asserts in both directions
         # pypdfium2 → per-page get_text_range() over build/main.pdf (102 pp, 257,713 chars)
         # captions(): brace-balanced \caption{...} bodies over chapters/**, tables/**, figures/**
         # merged_blocks(): prose → comment-only run → prose with no blank line = one rendered paragraph
```

---

## REQUIRED

### R-1 [Critical] — two paragraph breaks the source declares and the reader never receives

**Where:** `chapters/apx_f_cosine.tex:310–316`, rendered **p. 101**; and
`chapters/2_fundamentals.tex:755–767`, rendered **pp. 25–26**.

A five-line comment block sits between two prose paragraphs with no blank line on either side of it.
TeX drops comment-only lines without breaking the paragraph, so the two paragraphs are typeset as one.
This is the appendix's own documented hazard class from the other direction: the file's comment at
:170 warns that a comment block running into prose swallows the prose, and the blank lines there are
called load-bearing. Here the blank line is missing between the comment and the *following* paragraph,
so nothing is swallowed and nothing warns; the break is simply gone.

Rendered on p. 101, verbatim, as one paragraph:

> "The first axis is the tuning. Every one of Florida's twelve configurations is equivalent to zero on
> its own five folds, and their twelve means span [−0.00261, +0.00457] over the observations as
> recorded. Orthogonality survives every hyperparameter and procedure that was varied, so it is not an
> artifact of one choice among them. **The second axis is the data.** It now covers every dataset
> Chapter 5 reports on: the check-in counts run from 113,846 at Alabama to 4,089,892 at Texas and the
> region counts from 520 at Istanbul to 8,501 at California, so this axis spans a factor of thirty-six
> in volume and one of sixteen in the size of the region label set, with Georgia added under the same
> protocol. Equivalence holds at both ends. The result is not a quirk of Florida, and it is not an
> artifact of small data either: the two largest states behave like the two smallest."

**Why it harms the reader** (lens 8, paragraph quality; lens 2, coherence). The section opens by
promising "two axes, each answering a different objection", and the two-axis structure is the only
navigational device in it. Delivered as one 153-word block, the announced parallelism is invisible:
"The second axis is the data" arrives mid-paragraph, where a reader scanning for the second axis will
not find it. The source intends 53 + 100 words. Same mechanism in Ch. 2 at :755/:767, where a
122-word protocol paragraph and a 195-word statistics paragraph merge into the 317-word block that is
the chapter's joint-longest, and the merged result straddles the p. 25/26 break: 18 rendered lines on
p. 25 plus 4 on p. 26, so 22 of a 37-line page, about 0.6 of a page in one unbroken block.

I checked all four changed files for this pattern. `apx_f_cosine.tex` has two comment-bridged
junctions, one benign (inside the `figure` environment at :227/:249, where no paragraph is at stake)
and the one above. `2_fundamentals.tex` has two: the one above, and :888/:891, which is benign because
it falls mid-sentence. `1_introduction.tex` has nine, all inside `itemize` items or mid-sentence, none
losing a break. `tables/frame/cosine.tex` has two, both in the table preamble. **So: 15 junctions
examined, 2 defects.**

**Fix:** insert one blank line after the comment block at `apx_f_cosine.tex:315` and at
`2_fundamentals.tex:766`. No prose changes. This is also a candidate for `check_trapped_prose.py`'s
inverse: a comment block that bridges two prose runs is mechanically detectable, and the two instances
found here would be its first regression cases.

### R-2 [Critical] — the cosine table's caption is sized like a subsection, and the front matter reprints it whole

**Where:** `tables/frame/cosine.tex` caption, rendered **p. 100** (and **p. 7**, List of Tables).

197 words in six sentences, the longest caption in the document, against a document median of 37. On
p. 100 it occupies **15 of the page's 42 text lines — 36 percent of the page, more than the table it
captions (13 lines)**. Sentences four and five, verbatim from p. 100:

> "Alabama and Georgia show all five fold means positive with a 𝑡-test that rejects zero, and
> California's 𝑡-test also falls below the conventional threshold, but at five folds the exact sign
> test cannot return less than 0.0625, marked † where it sits at that floor. Each is therefore a
> consistent tendency rather than an established effect."

The body prose **on the same page** says it again, at greater length:

> "Neither is called significant here, and the reason is the design rather than the data. At five folds
> the exact sign test cannot return less than 0.0625, so no distribution-free test could reach
> significance however consistent the pattern. […] Both are therefore consistent tendencies rather than
> established effects. California makes the same point from the other side and is the reason the table
> carries a sign-test column at all: its 𝑡-test returns 0.048, below the conventional threshold, while
> its exact sign test returns 0.375 on four of five positive folds […]"

Measured overlap on p. 100: "0.0625" three times (caption, table footnote, body), "sign test" four
times, "consistent tenden…" twice, "established effect" twice, "conventional threshold" twice. The
reader meets the same qualification three times inside one page.

**And the front matter reprints it.** No caption in this document uses the optional short form
`\caption[short]{full}` — measured, 0 of 25 — so the List of Tables carries every caption in full.
Table 11's entry is 192 words over 15 lines and **is the only entry on p. 7**; the other ten tables
together take 320 words on p. 6 (median 25.5 words each, longest 80). One table is 38 percent of the
whole List of Tables by word count. Figure 8 does
the same to the List of Figures: 96 words, 8 lines, alone on p. 5, against a median of 54 for
Figures 1–7.

**Why it harms the reader** (lens 4, redundancy; lens 5, conciseness; lens 8, size). A caption is read
before the table and again when the reader returns to it; at 197 words it is neither. And a reader
paging the front matter to orient sees a two-page List of Tables whose second page is one diagnostic
appendix table. `WRITING_LAW` §5 asks tables for "a lead takeaway sentence" and figures for "2–4
self-contained interpretive sentences" — Figure 8's four are within that letter; Table 11's six are not
a lead sentence.

**Fix, three parts, none touching a number.** (a) Add short forms: `\caption[The cosine between the
next-region and next-category gradients on the shared trunk, by dataset.]{…}` on Table 11 and a
one-line equivalent on Figure 8. This alone recovers pp. 5 and 7 and is the cheapest of the three.
(b) Cut caption sentences four and five, whose content the body states more fully twelve lines below;
keep the † gloss, which the table footnote already carries. That brings the caption to about 110 words
and three sentences. (c) Keep sentence six (the Texas sign explanation): it explains a table cell that
looks like a typo, which is exactly a caption's job.

---

## RECOMMENDED

### R-3 [Major] — §2.3 opens a thread that §2.5 does not close

**Where:** new block pp. 22–23 against §2.5, **pp. 26–27**.

The new block ends by handing the reader a live question:

> "Orthogonality is not a conflict resolved but a conflict absent, which puts a limit on what any of
> these methods can contribute." (p. 23)

Measured in §2.5 (532 words, the chapter's designated "draws them together" section): occurrences of
"Pareto" **0**, "orthogonal" **0**, "cosine" **0**, "conflict" **0**, "Appendix" **0**. §2.5's
multi-task paragraph reads:

> "Multi-task learning is the mechanism that would let the two tasks share what they have in common,
> but it is not a free gain. Naive hard sharing can leave a task worse than its single-task model, and
> the elaborate gradient balancers proposed to prevent this frequently do not outperform a well-tuned
> fixed-weight baseline. Whether joint training helps therefore cannot be assumed […]" (p. 26)

That is the pre-Pareto argument, unchanged: negative transfer, then balancers underperform. The new
block's contribution — that in *this* task pair there is measurably nothing to balance, which is *why*
the balancers had nothing to give — is not in the summing-up.

**Why it harms the reader** (lens 2, coherence; lens 10, reader experience). §2.5 is the last thing
before Chapter 3 and is where a banca member forms the chapter's takeaway. A reader who has just been
told orthogonality bounds what any balancer can contribute, and then reads a summary saying only that
balancers empirically underperform, will read the new block as a digression. It is not one: it is the
chapter's strongest link to the arc.

**Fix:** one clause in §2.5's multi-task paragraph, after "fixed-weight baseline" — for this task pair
the reason is measurable, the two tasks' gradients being close to orthogonal on the shared parameters,
so there was little for a balancer to correct. No new claim; it restates p. 23 in the section built to
restate. Any cut to compensate is persona 14's gate, and note the prior pass's SF-10 argues §2.5 is
already saying its gap statement three times, so the author may prefer this clause to replace rather
than add.

### R-4 [Major] — "each method named in the next paragraph" still points past the paragraph that answers it (CONFIRMS prior SF-4)

**Where:** `2_fundamentals.tex:424–426`, rendered **p. 22**.

> "over 𝐾 tasks at shared parameters 𝜽, and each method named in **the next paragraph** is a different
> answer to how the weights 𝑤𝑘 , or the update direction they imply, should be set."

Measured in the render: the next paragraph is "Reaching that front is not what the balancing methods
promise" (p. 23), about guarantee strength. The paragraph that answers *how the weights should be set*
is "A family of methods tries to manage the conflict at the level of the gradients or the losses",
**three paragraphs later**, at character offset 2,285 on p. 23 against the pointer's target at 101.

What makes this a genuine ambiguity rather than a slip: I traced all occurrences of each balancer name
in Ch. 2. Nash-MTL, CAGrad, Aligned-MTL and PCGrad each appear exactly twice, **once in the guarantees
paragraph and once in the family paragraph** — and the guarantees paragraph is the first mention of
CAGrad, Aligned-MTL, GradNorm and FAMO anywhere in the document (all first render on p. 23). So the
pointer does land on a paragraph full of method names; it just is not the paragraph that says how their
weights are set.

**Why it harms the reader** (lens 3, clarity: no reference interpretable two ways). This is the prior
pass's SF-4, which could not establish whether the defect was introduced by this round. **It was:** the
pointer sentence is inside the block added by `beebd33b`, verified in `git diff c94d1f19..HEAD`.

**Fix:** the prior pass's proposal stands — "each of the balancing methods below is a different answer
to…". I add one alternative it did not consider: since the guarantees paragraph is where four of these
methods first appear, the cleanest repair may be to name them at the equation as a set ("uncertainty
weighting, GradNorm, PCGrad, CAGrad, Nash-MTL, Aligned-MTL and FAMO, discussed below, each answer this
differently"), which fixes the pointer and the first-mention order together.

### R-5 [Major] — the 66-word guarantees sentence is the one place in the new block where a reader must re-read

**Where:** `2_fundamentals.tex:433–440`, rendered **p. 23**.

> "Guarantees in this family are stated at that weaker level: Nash-MTL proves that its updates have a
> subsequence converging to a Pareto-stationary point, and reaches Pareto optimality only under an
> added convexity assumption on the losses that a deep network does not satisfy [47]; the fixed points
> of CAGrad are Pareto-stationary [48]; and Aligned-MTL converges to such a point for task weights
> fixed in advance [49]."

66 words, one colon and two semicolons, three methods with three different guarantee forms, plus a
negation ("does not satisfy") nested in the first. It is the longest sentence in the new block and
the fourth longest in the chapter. The Nash-MTL clause carries two distinct claims (what is proved, and
what is not) while the other two carry one each, so the parallelism the semicolons promise is false.

**Why it harms the reader** (lens 1, no re-reading; lens 9, rhythm). The paragraph around it is well
built: 22 / 24 / **66** / 7 / 44 / 30 / 13 / 16 words, and the seven-word "Not every method claims even
that much" is the best sentence in the new block. The 66-word sentence is the one bump.

**Fix:** split at the first semicolon. Sentence one keeps Nash-MTL with its two-part claim; sentence two
takes CAGrad and Aligned-MTL, which are genuinely parallel. No content moves.

### R-6 [Major] — Appendix F is reachable from exactly one sentence in the body

**Where:** measured across every `.tex` in `chapters/**` and `tables/**`.

Two `\ref` to Appendix F's labels exist: one in `2_fundamentals.tex` (the new p. 23 sentence) and one
inside Appendix F itself. In the render, "Appendix F" appears on **p. 23 and in the table of contents,
nowhere else**. Meanwhile a cosine result reaches the reader at three separate places: p. 63 (Ch. 5,
"+0.001 … four seeds on four Gowalla states"), p. 80 (Ch. 6, "averaged +0.001 … directional conflict
only"), and pp. 97–102 (the appendix, 4,650 epoch-level cosines on seven datasets). Neither p. 63 nor
p. 80 points to the appendix.

**Why it harms the reader** (lens 2, coherence; lens 10). A reader who meets the +0.001 figure at p. 63
or p. 80, notices it rests on four seeds and four states, and wants the fuller measurement has no
signpost — and a reader who reaches p. 97 having already read two versions of the finding may take the
appendix for a third restatement. The appendix's own p. 97 sentence handles the numerical relationship
correctly ("the two sets of numbers are not interchangeable and this appendix supersedes nothing
there"), but it is doing that work at the wrong end: it defends the appendix to a reader who has
arrived, rather than bringing the reader.

**Out of scope for me, and I note it rather than push it:** Ch. 5 is under review and its body is
deliberately untouched (the file's header comment states this), so the natural fix is closed. Ch. 6 is
frame prose and is not in my remit this round. **Recommendation for the author:** one cross-reference
in Ch. 6's gradient paragraph (p. 80) would connect the two, at the cost of one clause.

---

## OPTIONAL

### R-7 [Minor] — three phrases in Appendix F make the reader hold a definition they were given pages earlier

**Where:** pp. 97, 98, 101.

- p. 97: "and two of Florida's **carry a partial re-run on top of theirs**." The disclosure is right and
  belongs in the text; the phrasing leaves "on top of theirs" to be resolved against "five series of
  fifty values" earlier in the same sentence. Suggested direction: state the shape ("two of Florida's
  twelve configurations repeat their first fifteen epochs, so those series hold sixty-five values").
- p. 98: "**Florida's sixty fold series**, its twelve configurations over the same five folds." The
  appositive does the defining work, but "sixty fold series" then recurs on pp. 7 and 100 without it.
  Suggested direction: since the phrase is used three times in three places, give it its one gloss in
  the table's second column heading and let the prose use it bare.
- p. 101: "their twelve means span [−0.00261, +0.00457] **over the observations as recorded**." The
  qualifier is honest (it points at the partial re-run) but at this distance from p. 97 it reads as
  hedging of unknown target. Suggested direction: "over the series as recorded, including the two
  partial re-runs".

### R-8 [Minor] — "this appendix" five times in 1,753 words

Appendix F says "this appendix" five times, one per 350 words. Peers: Appendix E one per 277, Appendix
D one per 297, Appendix B once in 2,159. So the density is **normal for this document** and I flag it
only because two of the five are in consecutive paragraphs on p. 98 and one of those is doing rhetorical
rather than referential work: "this appendix will not accept for one claim a basis it rejects for
another" (p. 100). That sentence is good — it is the appendix's most persuasive line — and the fix, if
any, is to drop one of the two neighbours on p. 98, not this one.

---

## What is good, and should not be touched

1. **The Pareto block earns its length, measured.** 503 words for four registered terms, five methods'
   guarantee levels, and the disclaimer that the dissertation claims none of them. It is 10.0 percent of
   the chapter across three paragraphs of 152, 222 and 129 words, none of them an outlier against the
   chapter's 147.5-word median, and its variance is *higher* than the baseline (sd 17.6 vs 14.9) — the
   burstiness `WRITING_LAW` §4.3 requires, in prose that could easily have flattened. Nothing in it is
   padding. Its *sentences* do run longer than the chapter's (R-5), which is a different defect from
   length and has a one-line fix.

2. **The disclaimer is the best-placed sentence in the changed set.** p. 23: "This dissertation therefore
   claims no Pareto property of any kind for its models. Its verdicts are per-task scores measured against
   dedicated single-task models under the tests of Section 2.4." Two short sentences after a dense
   technical run, doing exactly what a reader needs at that moment, and forward-linking to §2.4. Protect
   the pair, including the full stop between them.

3. **"Not every method claims even that much."** p. 23, seven words. The rhythm break that makes the
   surrounding 44-word sentence readable, and the sentence a reader will remember the paragraph by.

4. **Appendix F's opening two paragraphs are the clearest exposition in the appendices.** p. 97: the
   failure mode, then the measurement, then the finding, then the consequence, then a one-sentence
   roadmap. The second paragraph's "a gradient balancer had nothing to balance" is the whole appendix in
   six words. This is the pattern the other appendices should be measured against.

5. **Appendix F got *easier* to read while covering three more datasets.** Sentence mean 23.0 against
   the chapter's 26.9, only 14.5 percent of sentences over 35 words against 25.6 percent, max 62 against
   80. Widening from four datasets to seven added 40 body words. That is a real editorial achievement and
   the opposite of what usually happens when coverage grows.

6. **The seven-dataset scope discipline reads well.** p. 97's "Six of the seven are the six Chapter 5
   reports on. The seventh, Georgia, is a further Gowalla state the dissertation does not otherwise use;
   it enters because the diagnostic ran on it cheaply and stays because dropping a measured dataset would
   be a choice about the evidence." Short, honest, and it answers the question a reader forms at the word
   "seven" in the sentence before. The ‡ marking in the figure caption is consistent with it.

7. **The deleted out-of-disk paragraph is gone from the render, and its removal left no seam.** I read
   pp. 101–102 for a gap where it stood: the coverage paragraph now runs into the architecture-boundary
   paragraph without a stumble, and the coverage fact survives in three other places including the figure
   caption. The cut improved the passage, not just its compliance.

8. **Ch. 1's seven advisor items read as if they had always been there.** The FAB-16 desemicolonization
   ("is a third and different problem, not addressed in this dissertation. Chapter 2 keeps the three
   tasks formally distinct.", p. 13) is a clear improvement, and the §1.2 heading now matches its
   content. Ch. 1's sentence statistics are unchanged from the chapter baseline (mean 27.2, sd 14.7).

9. **Appendix F is an appendix, not an article.** Answering the question as asked: 1,753 body words and
   six pages, against 4,330 for the shortest paper chapter; four sections, no abstract, no related work,
   no numbered contributions, and every section title is a question about the measurement. It reads as a
   diagnostic note, which is what an appendix should be.

10. **The caption on Figure 8 is defensible as a caption; the one on Table 11 is not.** At 97 words and
    four sentences, Figure 8 is at the top of `WRITING_LAW` §5's stated range but inside it, and it names
    every panel, the band, and the ‡ convention. Its only cost is the front-matter reprint (R-2a), which
    a short form fixes without touching the caption.

---

## Scores (1–10), changed prose only

| lens | score | one-line justification |
|---|--:|---|
| Readability | **8** | one 66-word sentence (R-5) and three deferred-definition phrases (R-7) in 2,256 changed words; everything else parses first time. |
| Coherence / flow | **6** | R-1's lost paragraph breaks and R-3's unclosed thread are both structural, and both are cheap to fix. |
| Clarity | **7** | one pointer that resolves two ways (R-4); no ambiguous pronoun or referent found elsewhere in the changed prose. |
| Conciseness | **6** | the prose is lean (the new block is at the chapter mean); the caption is not, and the caption is on the page three times over (R-2). |
| Consistency of voice | **9** | the new §2.3 block and Appendix F read as one author, and as the same author as the untouched chapter; no seam detectable at either boundary. |
| **Overall writing quality** | **7.5** | strong new prose let down at the seams: a break, a caption, and a summary that has not caught up. |

## Chapter-seam verdict

**The new frame prose and the re-typeset chapters read as one voice at the §2.3 boundary, and the seam
that does show is a lineage seam, not a style one.** The new block's Pareto vocabulary reappears in
Ch. 3's re-typeset prose on p. 31 ("MGDA finds Pareto-optimal descent directions") and on pp. 37 and 49
in Nash-MTL's own exposition — published prose the frame cannot restyle. Because §2.3 now defines
Pareto-stationary before the reader reaches either, those later mentions land as recall rather than as a
second definition, which is an improvement this round made. Ch. 4's unhyphenated "Pareto stationary"
inside `tables/courb/errata.tex` remains the one visible inconsistency, and the glossary already records
the decision to leave it (persona 04's scope, not mine).

## Out-of-scope handoffs

- The `[NEEDS SIGN-OFF]` on the new §2.3 paragraph's last two clauses and on Appendix F as a whole are
  claim questions (persona 07 / AGENT_GUARDRAILS C2), not readability ones. I read both flags; neither
  changes a finding above.
- Table 11's caption asserts "California's 𝑡-test also falls below the conventional threshold" while the
  body calls the same result "not even a leaning" — a tone difference I judged as redundancy (R-2), not
  as a numbers or claims question. Persona 06/07 own whether the two framings are consistent.
- R-1's mechanism is a source-engineering defect with a prose consequence. Persona 19 owns the gate
  proposal; I report the reader-facing half.
- **A law pointer in `apx_f_cosine.tex`'s deletion comment sends the next agent to the wrong section.**
  It reads `% _round9/30_cosine_six.md. WRITING_LAW §4 now carries the general rule (process
  narration).` The ban added today is **§1** ("No process narration, and this is a hard ban", under
  *Register: dissertation ≠ paper*); I searched §4 in full (2,956 characters, the AI-tell law) and it
  contains no occurrence of "process narration". A second comment three lines above is correct as
  written, since the *restating-a-section* half of that paragraph's defect genuinely is a §4 matter.
  This is a comment, so no reader receives it and it is outside my remit by the rule that separates what
  the reader gets from what an agent wrote. I pass it on because it will misdirect the next agent who
  follows it, and because a comment that cites the wrong clause of the law is the kind of thing that
  survives for rounds. Verified independently, not taken on report.

## What I did not do

1. **I did not read the other build modes.** Only `build/main.pdf` (defense, 102 pp). R-2's front-matter
   measurements are for that build; `make final` (99 pp) and `make ppgc` (103 pp) were not built, so I
   cannot say whether pp. 5 and 7 fall the same way there.
2. **I read Chapter 2 in full and the other chapters only where a thread from the changed prose led.**
   Ch. 3–5 were read at pp. 31, 37, 49, 62, 63 and 85 (Pareto and balancer mentions) and Ch. 6 at
   pp. 79–81, in service of R-3, R-4 and R-6. The paper chapters were not read as prose; this round's
   diff does not touch them.
3. **The figure image itself was not inspected.** R-2 judges Figure 8's *caption*; whether panels (a)–(c)
   are legible is persona 18's remit and I did not open the PNG.
4. **No aloud pass.** `WRITING_LAW` §4.2 prescribes reading a page aloud for uniform sentence weight. I
   measured the distribution instead (the tables above), which catches flatness but not every awkward
   cadence. The new block's sd of 17.6 is evidence against flatness, not proof of good rhythm.
5. **One of my own measurements was wrong on first pass, and it is recorded above rather than quietly
   fixed.** The new block's sentence mean came from the PDF text layer while its baseline came from the
   source, and the display equation splits a sentence in the former but not the latter — so "within one
   word of the chapter's mean" was an artifact of mixing two instruments, and the honest figure is 2.7
   words above. The full account is in the statistics section. The lesson for the next pass is
   mechanical: count every row of a comparison table with one instrument, and state which. Four of my
   fifty numeric self-checks failed on first run (this one, a paragraph sentence count, a median printed
   as an integer when it is 147.5, and another median off by half a word); all four are corrected in the
   text above.
