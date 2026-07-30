# 39 · MTL expert review, round 9 wave B — the §2.3 Pareto passage and Appendix F at seven datasets

> Persona: `reviewers/10_mtl_expert.md`. Remit: technical correctness on multi-task learning.
> Scope: **only** what changed in `git diff c94d1f19..HEAD` — `2_fundamentals.tex` (+106, the new
> §2.3 Pareto passage), `apx_f_cosine.tex` (+205/−92), `tables/frame/cosine.tex` (+74/−33),
> `1_introduction.tex` (+37/−14). Read from the **rendered PDF**, not the source: comments were
> stripped with `live_text` from `src_utils/check_audit_claims.py` for every source-side sweep.
> Read-only; no `.tex` file was touched.

## Verdict

**sound-with-corrections.** The five guarantee claims the passage makes are, on independent
reading of the five source PDFs, substantially right — including the PCGrad zero, which I
reproduced with a validated instrument. Table 11 reproduces from the parquet at 7 of 7 rows. The
defects are all in **scope**: three sentences state a family-wide or unscoped claim where the
evidence supports a narrower one, and one sentence attributes to CAGrad and Aligned-MTL a
limitation both papers claim to have solved. Nine findings, four REQUIRED. None of them touches a
number; all four REQUIRED items are one-sentence rewrites.

The single most valuable one, and the question an examiner will ask: **orthogonal gradients do not
neutralize a gradient balancer.** They neutralize PCGrad. Everything else named in the paragraph
that follows acts on gradient *magnitude* or loss scale, which orthogonality says nothing about —
and the dissertation's own Chapter 5 screen contains the counter-example (Nash-MTL exceeding equal
weighting by 0.68 points at Alabama).

---

## The build

```
cd articles/dissertacao/src && make defense        # in a clean copy of src/ + src_utils/
→ latexbuild main -> build/main.pdf  pages=102  tex_errors=0
```

Note for the author, not a finding, and stated as the transient condition it turned out to be: my
in-tree `make defense` **failed**, exit 2, with `build/main-aux/chapters/4_courb.aux` truncated
mid-`\@writefile` (`! File ended while scanning use of \@writefile`, `Runaway argument?
{\contentsline {subsection}{\numberline {4.3.1}Baseline: MTLnet with \ETC.`) and no output PDF
produced. It is a stale-aux artifact and not a source defect: the same sources built clean at 102
pages in a fresh copy of `src/` + `src_utils/` with an empty `build/`. It is also **no longer
reproducible in the working tree** — `build/main-aux/main.log` now records "Output written on
build/main-aux/main.pdf (102 pages, 1533933 bytes)" with zero `Fatal error` lines, so a later build
in this tree succeeded and overwrote my failing run's log. Treat it as: if a build dies this way,
`make clean` clears it. Every page number below is from my own clean 102-page build.

I first wrote here that my failed attempt had left `src/dissertacao.pdf` differing from HEAD. That
was wrong and I retract it: a build that produces no PDF never reaches the Makefile's
`&& cp build/main.pdf dissertacao.pdf`. What is measurable is that `src/dissertacao.pdf` and
`src/build/main.pdf` are byte-identical (`md5 50348efbdc45fefb975c90654f6676af`, 1,533,933 bytes,
both stamped 10:17:42) and differ from HEAD's 1,534,343, and that they are **not** my build
(`/tmp/dissbuild/.../main.pdf` is `978dbc74806b14945f018767a684e342`). Five commits from other
tracks landed in this tree between 10:55 and 11:12, so the writer is one of them; per the V9 rule I
name it UNKNOWN rather than infer it. Either way it is a regenerable build product and it is not
part of my commit.

One consequence worth recording, because it affects how my quotes should be checked. The in-tree PDF
and mine agree byte-for-byte on 8 of the 10 pages I cite; pages 22 and 24 differ, and the diff is
citation **numerals only** (`[6]`→`[7]`, `[39]`→`[1]`, `[11]`→`[12]`) with every sentence identical.
Bracket numbers are therefore volatile in this tree while a concurrent track edits the
bibliography. The five my findings depend on are not: `[46] sener2018mgda`, `[47] nash`,
`[48] liu2021cagrad`, `[49] senushkin2023aligned`, `[50] yu2020pcgrad` resolve identically in both
`main.bbl` files (checked by parsing `\bibitem` from each, 99 entries each), so M2 and M7 name the
right papers under either build.

---

## Findings

Severity is given on the task's REQUIRED/RECOMMENDED/OPTIONAL scale, with this persona's own scale
(`reviewers/README.md` §5) in brackets.

### M1 · REQUIRED [BLOCKER] · orthogonality limits PCGrad, not the family

`2_fundamentals.tex`, **PDF p. 23**:

> "Orthogonality is not a conflict resolved but a conflict absent, which puts a limit on what any
> of these methods can contribute."

The antecedent of "these methods" is the preceding paragraph's four (Nash-MTL, CAGrad,
Aligned-MTL, PCGrad), and the paragraph immediately after adds uncertainty weighting, GradNorm,
dynamic weight averaging and FAMO. The claim is false for every one of them except PCGrad, and I
measured it rather than arguing it. Take two exactly orthogonal task gradients of unequal
magnitude, `g1 = [3, 0]`, `g2 = [0, 1]`, `cos = 0.0`:

| method | update at cos = 0 | deviation from the equal-weight sum |
|---|---|---|
| PCGrad (Yu Def. 1: project iff `cos φ < 0`) | `[3, 1]` | **0°, a no-op** |
| MGDA min-norm (`w = [0.10, 0.90]`) | — | 53.1° |
| Nash-MTL (Claim 3.1, `GᵀGα = 1/α`, `α = [0.333, 1.0]`) | `[1, 1]` | 26.6° |
| CAGrad, `c = 0.4` (the value its own NYU-v2/CityScapes runs use, p. 17) | `[1.5, 1.13]` | 18.6° |
| Aligned-MTL | `κ(G) = σmax/σmin = 3.0` — a system it will rescale | not a no-op |

Orthogonality zeroes the *angular* channel only. Nash-MTL, MGDA, CAGrad and Aligned-MTL all read
the Gram matrix `GᵀG` or its singular values, so they respond to magnitude asymmetry at any angle;
uncertainty weighting (Kendall, arXiv:1705.07115 — **zero** occurrences of "cosine" and zero of
"angle" in 14 pages) and GradNorm (arXiv:1711.02257 — zero "angle"; its two "cosine" hits are a
*loss function for surface normals*, p. 1 and the experiments) never look at the angle at all;
FAMO reads loss decreases. Aligned-MTL states the mechanism outright (p. 3): a gradient system is
well-conditioned only when its matrix "must be orthogonal **with equal singular values**" —
orthogonality is half of its criterion, and the half it is designed to fix is the other half.

The dissertation contains its own counter-example. Chapter 5, **PDF p. 63**: "Two exceed equal
weighting on next-category at Alabama, Nash-MTL by 0.68 points and scale normalization by 0.19."
If orthogonality left a balancer "nothing to contribute", neither number could exist. The field
says the same thing (Elich et al., arXiv:2311.04698 p. 1: for angular alignment "we find no
evidence that this is a unique problem in MTL. We emphasize differences in gradient magnitude as
the main distinguishing factor") — and that paper is **absent from `references.bib`** (see M8).

**Why it is REQUIRED:** this is the load-bearing mechanism sentence of the new passage, it is the
sentence Appendix F's mechanism section is being read through, and it is wrong at family scope in
a way an MTL examiner will name in the first five minutes. AGENT_GUARDRAILS §1 also binds it:
"Describe each cited system as its own authors describe it."

**Fix.** Narrow the claim to the channel that was measured:

> "Orthogonality is not a conflict resolved but a conflict absent, so the methods that act on the
> angle between the gradients, PCGrad among them, have nothing to project. Methods that act on the
> relative magnitude of the two updates are not ruled out by an angle of ninety degrees, and
> Chapter~\ref{ch:mobiwac} screens them empirically rather than excluding them here."

### M2 · REQUIRED [MAJOR] · a limitation attributed to the two papers that claim to fix it

`2_fundamentals.tex`, **PDF p. 23**:

> "Two of these papers state the residual limitation themselves: arriving somewhere on the front
> says nothing about where, so the balance between the tasks at that point remains uncontrolled
> [48, 49]."

`[48]` is `liu2021cagrad`, `[49]` is `senushkin2023aligned` (from `build/main-aux/main.bbl`). Both
papers do state that criticism — **about the methods they supersede**, as the motivation for their
own contribution. CAGrad, abstract p. 1: prior heuristics "lack convergence guarantee and/or could
converge to any Pareto-stationary point. In this paper, we introduce Conflict-Averse Gradient
descent (CAGrad) which ... provably converges to a minimum over the average loss." Aligned-MTL,
p. 2: approaches "aiming to find a Pareto-stationary solution ... terminate once the Pareto front
is first reached, as a result, they might provide a suboptimal solution. Differently, Aligned-MTL
drifts along the Pareto front and provably converges to the optimum w.r.t. pre-defined tasks
weights." Aligned-MTL p. 5 names its targets: "[27, 37] that provably converge to an arbitrary
Pareto-stationary solution, tend to overfit to a subset of tasks" — reference 37 is Nash-MTL.

Placed two sentences after the paragraph has assigned CAGrad and Aligned-MTL the Pareto-stationary
guarantee, "state the residual limitation themselves" reads as *these two admit their own
guarantee leaves the balance uncontrolled.* It is the reverse of both papers' positions, and it
misses that the criticism lands on Nash-MTL, which the sentence before treats as the strongest of
the three.

The same paragraph under-describes CAGrad for the same reason. It gives only the Theorem 3.2(1)
leg ("the fixed points of CAGrad are Pareto-stationary", p. 5) and omits the guarantee CAGrad
leads with: at `0 ≤ c < 1`, convergence to a minimum of the average loss (p. 4, "to guarantee that
CAGrad converges to an optimum of L0(θ), we have to ensure 0 ≤ c < 1"), which is the regime its
own experiments run in (`c = 0.4`, p. 17).

**Fix.** Attribute the criticism to its target and let the two papers keep their own claims:

> "Both CAGrad and Aligned-MTL raise this as the limitation of the earlier methods and design
> against it: CAGrad converges to a minimum of the average loss for its recommended range of its
> trade-off parameter \cite{liu2021cagrad}, and Aligned-MTL to the optimum of a task weighting
> fixed in advance \cite{senushkin2023aligned}. Neither guarantee is one this dissertation needs
> or claims."

### M3 · REQUIRED [MAJOR] · "no balancing method improved" drops the scope Chapter 5 states

`2_fundamentals.tex`, **PDF p. 23**:

> "it reads that result as the reason no balancing method improved on a fixed weighting in this
> work"

Chapter 5, **PDF pp. 62–63**, states the same result with three qualifiers that the frame sentence
drops: "of nineteen loss and gradient balancers screened **at their default configurations at a
single seed on two datasets**, Alabama and Florida ... none improved on a tuned fixed task
weighting **across both tasks and both datasets**. Two exceed equal weighting on next-category at
Alabama, Nash-MTL by 0.68 points and scale normalization by 0.19."

Two balancers did improve on a fixed weighting, on one task at one dataset. WRITING_LAW §3 requires
scoping every universal, and this persona's first lens is exactly the missing qualifier: a
"balancer X did not help" claim is unproven unless the arms had the same tuning budget, and the
screen's arms did not (defaults at one seed against a *tuned* fixed weighting). Chapter 5 is
honest about all of this; the frame chapter, which more readers will read, is not.

**Fix.** "…as the reason no balancer improved on a tuned fixed weighting across both tasks at both
of the datasets screened in Chapter~\ref{ch:mobiwac}, where nineteen were compared at their
default configurations."

### M4 · REQUIRED [MAJOR] · "statistically indistinguishable" in the frame, with no margin and no test

`2_fundamentals.tex`, **PDF p. 23**:

> "Appendix F measures the cosine on the joint model of Chapter 5 and finds the two tasks'
> gradients statistically indistinguishable from orthogonal on the datasets measured there"

Three of the seven datasets are statistically *distinguishable* from orthogonal by the test
Appendix F itself prints on **PDF p. 100**: Alabama `t = 0.013`, California `t = 0.048`, Georgia
`t = 0.009`, and those three 95% confidence intervals exclude zero (`[+0.0040, +0.0184]`,
`[+0.0000, +0.0014]`, `[+0.0016, +0.0061]`; I recomputed all seven from the parquet, table below).
California is the case with no escape hatch: its exact sign test is `0.375`, not the `0.0625`
floor, so the appendix's own "the sign test cannot reject at n = 5" defense does not cover it.

The appendix knows this and handles it correctly in its body ("Neither is called significant
here"; "what looks like a finding under one test is not even a leaning under the other"). But the
frame sentence carries none of that. It states a *non-distinguishability* claim, names no test,
names no margin, and stands two pages after WRITING_LAW §3's rule that "'significant' only with
the test named" and that every number carries its convention.

What the appendix actually licenses is stronger and immune to the objection: equivalence to zero
against a margin fixed in advance. Use it.

**Fix.** "…and finds the two tasks' gradients statistically equivalent to orthogonal, within a
margin of five hundredths, at every dataset measured there."

### M5 · RECOMMENDED [MAJOR] · the same phrase opens the appendix

`apx_f_cosine.tex`, **PDF p. 97**:

> "Their gradients are statistically indistinguishable from orthogonal on every dataset measured,
> which is a stronger and stranger result than mere absence of conflict"

Same defect as M4, one severity lower because the margin arrives on p. 98 and Table 11 is on
p. 100, so a reader who continues is not misled. But "indistinguishable" is the wrong word for a
set that contains three t-tests at 0.013, 0.048 and 0.009, and the sentence *also* claims to be
"stronger than mere absence of conflict" — which is true of equivalence and false of
indistinguishability. Substituting the appendix's own vocabulary makes the sentence both accurate
and stronger. This wording predates the round (`git show c94d1f19` has it verbatim at line 57), but
the round is what widened it from four datasets to seven and propagated it into Chapter 2, and the
propagation is what makes it worth fixing now.

**Fix.** "Their gradients are statistically equivalent to orthogonal at every dataset measured,
within a margin fixed in advance, which is a stronger result than mere absence of conflict".

### M6 · RECOMMENDED [MAJOR] · "the quantity the gradient methods act on" is true of one method

`2_fundamentals.tex`, **PDF p. 23**:

> "Conflict has a standard measure, and it is the quantity the gradient methods act on: the cosine
> of the angle between two tasks' gradients at the shared parameters"

The cosine is the standard measure of conflict — that half is right, and `yu2020pcgrad` Def. 1
(arXiv:2001.06782v4 p. 3, "We define the gradients as conflicting when cos φij < 0") is the right
citation for it. The other half is not. Of the eight methods the next paragraph names, the cosine
is the acted-on quantity for **one** (PCGrad). Uncertainty weighting and GradNorm never compute an
angle (measured: zero "angle" in either paper); DWA reads loss ratios; FAMO reads loss decreases;
MGDA, Nash-MTL, CAGrad and Aligned-MTL act on `GᵀG` or its singular values, which mixes angle and
magnitude. Same root as M1, separate sentence, separate fix.

**Fix.** "Conflict has a standard measure: the cosine of the angle between two tasks' gradients at
the shared parameters, negative when the tasks disagree and near zero when their updates are close
to orthogonal \cite{yu2020pcgrad}. It is the quantity the gradient-projection methods act on
directly; the weighting methods below act on the relative size of the two updates instead."

### M7 · RECOMMENDED [MINOR] · the non-sufficiency half is not in [47]

`2_fundamentals.tex`, **PDF p. 23**:

> "A point is Pareto-stationary when some convex combination of the task gradients is zero, which
> is necessary for Pareto optimality without being sufficient [47]."

Nash-MTL (`[47]`) states the necessity and only the necessity, p. 2 verbatim: "a point is called
Pareto stationary if there exists a convex combination of the gradients at this point that equals
zero. Pareto stationarity is a necessary condition for Pareto optimality." It never states
non-sufficiency: `grep -ci suffic` over my extraction returns **1** hit, and it is "it suffices to
prove that the sequence converges" in the proof of Theorem 5.5 (p. 14). The non-sufficiency claim
belongs to `sener2018mgda` (`[46]`), p. 4: "Although every Pareto optimal point is Pareto
stationary, the reverse may not be true." `liu2021cagrad` p. 4 states the same implication from the
other side ("a local Pareto optimal point θ must be Pareto stationary").

GLOSSARY.md §4 gets this exactly right already — its Pareto-stationary row attributes the necessity
clause to `nash` p. 2 and the formalization to `liu2021cagrad` Def. 3.1 — so the prose is
over-attributing relative to the registry it was built from.

**Fix.** `\cite{nash,sener2018mgda}` on that sentence.

### M8 · RECOMMENDED [MAJOR] · the appendix bounds datasets and architecture, not the measurement channel

`apx_f_cosine.tex` §F.4, **PDF pp. 101–102**, bounds the result along the tuning axis, the data
axis, and the architecture:

> "Seven datasets and twelve configurations bound this result along two axes, each answering a
> different objection."
> "Every run measured here uses one architecture family, the cross-attention joint model of
> Chapter 5."

There is a fourth bound and it is unstated: **only the angle was measured.** The source of record
(`_round7/gradient_cosine_observations6.parquet`) carries columns `['state', 'fold', 'epoch',
'cos', 'config']` and no gradient-norm column, so nothing in this appendix speaks to the magnitude
asymmetry between the two tasks' updates — which is the channel the literature identifies as the
distinguishing one (Elich et al., arXiv:2311.04698 p. 1) and the channel four of the eight methods
in §2.3 act on exclusively. Section F.3's two consequences are drawn family-wide from an
angle-only measurement. This is the evidence gap behind M1, and stating it costs one sentence.

Second half of this finding: the skeptic block on **PDF p. 24** cites `lin2022rlw`,
`xin2022domtl` and `kurin2022scalarization`, which is the right position well cited — but
`references.bib` (99 entries) contains **no** entry for Elich arXiv:2311.04698, Royer
arXiv:2310.08910, Hu arXiv:2308.13985, TAG arXiv:2109.04617, or Zhang's negative-transfer survey
arXiv:2009.00909. Elich is the one I would call load-bearing rather than padding, because it is the
measured evidence for the distinction M1 and M6 turn on, and because a passage that measures
gradient conflict and draws a mechanism from it will be asked about it.

**Fix.** In F.4, after the architecture paragraph: "One further boundary is the quantity itself.
This appendix measures the angle between the two updates and not their relative size, so it
supports no claim about methods that balance gradient magnitudes." And add the Elich entry, cited
at the §2.3 sentence M1 rewrites.

### M9 · OPTIONAL [MINOR] · "the two largest states behave like the two smallest"

`apx_f_cosine.tex`, **PDF p. 101**:

> "The result is not a quirk of Florida, and it is not an artifact of small data either: the two
> largest states behave like the two smallest."

True on the equivalence verdict, which the preceding sentence ("Equivalence holds at both ends")
has already said. Not true on anything else: the two smallest are Alabama (113,846 check-ins,
`|mean| = 0.0112`, 5/5 folds positive, `t = 0.013`, 5/5 slopes negative) and Arizona (236,450,
`0.0015`, 3/5), the two largest Texas (4,089,892, `0.0003`, 4/5) and California (3,171,380,
`0.0007`, 4/5). Alabama is the one dataset in the set that does *not* behave like the others: its mean is the
largest of the seven and the only one above one hundredth, more than seven times the next-largest
among the smaller states (Arizona, `+0.0015`) and more than forty times Texas's in absolute value,
with both departures unanimous. Spearman over the six dissertation datasets gives
`ρ = −0.486, p = 0.33 (n = 6)`, i.e. no size trend — which supports the sentence's intent and not
its wording.

**Fix.** Cut the clause, or make it the verdict-level statement it is: "…and it is not an artifact
of small data either: equivalence holds at the largest datasets and the smallest alike."

---

## What is fine, and should not be touched

Said plainly, because a review that finds only problems is not calibrated. Every item below I
checked against the source and found correct.

1. **Nash-MTL's guarantee is stated exactly right, including the convexity caveat.** "Nash-MTL
   proves that its updates have a subsequence converging to a Pareto-stationary point, and reaches
   Pareto optimality only under an added convexity assumption" (p. 23) — Theorem 5.4, p. 6 and
   p. 13: "the sequence {θ(t)} has a subsequence that converges to a Pareto stationary point θ*";
   Theorem 5.5, p. 6 and p. 14: "if we also assume convexity, we can strengthen our claim" /
   "from convexity it would be Pareto optimal". The added gloss "that a deep network does not
   satisfy" is the author's own and is uncontroversial. This is the hardest of the five to get
   right and it is right.
2. **PCGrad contains no Pareto claim, independently confirmed.** 27 pages, zero occurrences of
   "Pareto" case-insensitively. I validated the instrument on the same extraction before believing
   the zero (`PCGrad` 192, `conflicting` 26, `theorem` 25, `cosine similarity` 14, `gradient` 178),
   so the zero is an absence in the paper and not a broken extractor. The prose's description of
   what PCGrad *does* guarantee is also accurate to Theorem 2, p. 5: a one-step inequality
   `L(θPCGrad) ≤ L(θMT)` under three conditions on the two gradients' angle and magnitude, on a
   curvature lower bound, and on the step size. "Under conditions on the two task gradients, on the
   curvature of the loss, and on the step size" is a fair three-clause compression of exactly those.
3. **The Pareto-stationary attribution for CAGrad survives a challenge I raised against it.** Main
   text Theorem 3.2(1), p. 5, reads "For any c ≥ 1", which would have made the unqualified sentence
   wrong. The appendix restatement, p. 15, reads "For any c ≥ 0", and p. 16 proves it for "general
   c ≥ 0". The paper is internally inconsistent on the condition and the dissertation's unqualified
   phrasing is covered by the broader of the two. **Not a finding**; recorded so the author knows
   it was tested.
4. **Aligned-MTL's "for task weights fixed in advance" is the paper's own qualifier**, p. 5: "our
   approach converges to a Pareto-stationary point with pre-defined tasks weights".
5. **The self-denial sentence is the right instinct and correctly placed.** "This dissertation
   therefore claims no Pareto property of any kind for its models. Its verdicts are per-task scores
   measured against dedicated single-task models under the tests of Section 2.4" (p. 23). It also
   satisfies GLOSSARY §4's standing instruction to say Pareto optimality is not claimed here, once,
   in §2.3.
6. **All four Pareto terms were registered in GLOSSARY.md before the prose landed** (§4 rows for
   Pareto dominance, Pareto optimality, Pareto-stationary point, gradient conflict), so the
   fail-closed rule is satisfied for the new passage.
7. **Table 11 reproduces at 7 of 7 rows** from `gradient_cosine_observations6.parquet` on `n`, mean,
   confidence interval, TOST order of magnitude, both p-values and the positive-fold count. So do
   the three prose numbers on p. 98 (92.4 percent inside the margin, range −0.34 to +0.58, "all
   seven means lie within one and a half hundredths of zero" — max `|mean| = 0.0112`), the figure's
   in-panel pooled mean (`+0.00102`), and the two ratios on p. 101 (`4,089,892 / 113,846 = 35.9`,
   "a factor of thirty-six"; `8,501 / 520 = 16.3`, "one of sixteen").
8. **The California paragraph is the best statistical writing in the appendix.** "its t-test returns
   0.048, below the conventional threshold, while its exact sign test returns 0.375 on four of five
   positive folds, so what looks like a finding under one test is not even a leaning under the
   other" (p. 100). A weaker author would have banked the 0.048. Keep it exactly as it stands; it is
   also the paragraph that earns the sign-test column.
9. **Texas's negative mean beside four positive folds is disclosed rather than smoothed** (p. 100
   caption). Fold means `[+0.00106, +0.00029, +0.00014, −0.00322, +0.00040]` — reproduced; the
   caption's explanation is correct.
10. **The unit-of-independence discipline is right and now more conservative than before.** Florida
    is reported at 60 fold series in the body with the 12 configuration means in the footnote as
    "the more conservative reading", and both give `+0.0003`. No test anywhere runs on the 4,650
    raw observations.
11. **The out-of-disk paragraph is gone, and the deletion is clean.** Gate 23
    (`cd src_utils && python3 check_process_narration.py`) reports "no process narration in 51 files
    (3 exempt)"; exit code read directly from `$?` in the same shell, rc=0, and
    the appendix lost no reader-facing fact: coverage is still stated in the body (p. 97, "Six of
    the seven are the six Chapter 5 reports on"), the figure caption (p. 99) and the table caption
    (p. 100).
12. **The skeptic position is present and correctly sided** (p. 24): "a fixed-weight baseline is a
    serious competitor, and a balancer earns its place only by outperforming it", with
    `lin2022rlw`, `xin2022domtl` and `kurin2022scalarization` behind it. This dissertation's own
    finding aligns with the field's null and the text says so rather than citing only pro-balancer
    work. M8's second half is a gap in that block, not a contradiction of it.
13. **The `[NEEDS SIGN-OFF]` flag on the mechanism restatement is the correct call** and names the
    right reason (a connective claim in the frame widens the appendix's audience). The findings
    above are what the author should weigh when ruling on it.
14. **Georgia is handled honestly throughout** — seven datasets, six of which are the
    dissertation's, marked `‡` in the figure, named in both captions, and never allowed to imply
    the six.

## Categories where I found nothing

Stated explicitly rather than omitted, per the task.

- **Fabricated or mis-existent citations in the changed prose: none.** All five balancer PDFs
  resolve at the arXiv IDs the provenance comments give, and every guarantee sentence's claim was
  located in its source by page.
- **Wrong numbers in the changed prose or the changed table: none.** 7 of 7 table rows and 6 of 6
  checked prose numbers reproduce from the parquet.
- **Checkpoint-selection or convention blurring in the changed text: none.** The changed material
  reports no model scores, so no selection convention is at stake in it.
- **Negative-transfer or capacity-matching misstatements in the changed text: none.** The changed
  passage makes no MTL-versus-single-task performance claim.
- **Process narration remaining in the changed files: none**, measured by gate 23 and by reading
  pp. 22–23 and 97–102.
- **Repo codenames in the changed prose: none.**

## Out-of-scope handoffs (one line each, outside the measured diff)

- `GLOSSARY.md` has **zero** occurrences of "negative transfer" (`grep -ni`), yet the term is used
  and glossed in §2.3 prose on p. 22 — a fail-closed registry gap in pre-existing text.
- That same gloss, "joint training can leave a task worse off than its single-task model" (p. 22),
  omits the "equally tuned" qualifier the formal definition carries; pre-existing, and it would
  matter if any chapter claimed negative transfer from an unmatched comparison.
- `apx_f_cosine.tex` §F.3 "Orthogonality leaves them nothing to resolve" (p. 101) has M1's defect
  and is the origin of the §2.3 sentence; the diff does not touch it (verified: no hunk in
  `c94d1f19..HEAD` matches that line), so it is not mine to review, but a ruling on M1 should cover
  both or the two will disagree.
- Figure 8 panel (c), p. 99: the annotation "sign-test floor at n=5 is 0.0625" is overplotted by a
  data point, so "n=5" is partly illegible. Legibility, not MTL; for the readability track.

## Measured

Every count with the command that produced it, from `articles/dissertacao/`.

**Build.** `cd src && make defense` → fails in-tree on a truncated `4_courb.aux`; succeeds
(`pages=102 tex_errors=0`) after `cp -R src src_utils /tmp/dissbuild && rm -rf
/tmp/dissbuild/src/build/*`. All page numbers from `/tmp/dissbuild/src/build/main.pdf`, read with
`pypdfium2` (`doc[i].get_textpage().get_text_range()`).

**Source-side sweeps** were run on comment-stripped text, per the V4 rule:
`sys.path.insert(0, "src_utils"); from check_audit_claims import live_text`. N-of-N counts over
`2_fundamentals.tex` + `apx_f_cosine.tex` + `tables/frame/cosine.tex` + `1_introduction.tex` +
`5_mobiwac/05_setup.tex`, every occurrence read before any verdict:

| pattern | total live occurrences | read |
|---|--:|--:|
| `indistinguishable` | 2 (Ch.2 ×1, Apx F ×1) | 2 |
| `orthogonal\w*` | 9 (Ch.2 ×2, Apx F ×7) | 9 |
| `nothing to (resolve|balance)` | 2 (Apx F) | 2 |
| `Pareto` | 11 (Ch.2 only) | 11 |
| `quantity the gradient methods act on` | 1 | 1 |
| `conflict absent` | 1 | 1 |
| `two largest` | 1 | 1 |

**The five source PDFs**, downloaded from arXiv this session to `/tmp/mtlpapers/` and read by page:

| paper | arXiv | pages | chars | `Pareto` | `Pareto[- ]stationar` | claim located |
|---|---|--:|--:|--:|--:|---|
| Nash-MTL | 2202.01017 | 19 | 69,060 | 39 | 16 | Thm 5.4 pp. 6, 13; Thm 5.5 pp. 6, 14; def. p. 2 |
| PCGrad | 2001.06782 | 27 | 92,889 | **0** | **0** | Def. 1 p. 3; Thm 1 p. 4; Thm 2 p. 5 |
| CAGrad | 2110.14048 | 20 | 65,925 | 37 | 8 | Def. 3.1 p. 4; Thm 3.2 p. 5; restated p. 15 |
| MGDA / Sener–Koltun | 1810.04650 | 15 | 51,896 | 24 | 4 | Def. 1 p. 3; dichotomy Thm 1 p. 6 |
| Aligned-MTL | 2305.19000 | 16 | 66,370 | 21 | 10 | Def. 1 and Thm 1 p. 5; proofs pp. 12–13 |

Plus, for M1 and M6: Kendall 1705.07115 (14 pp., "cosine" 0, "angle" 0; instrument validated,
"uncertainty" 49), GradNorm 1711.02257 (12 pp., "angle" 0, "cosine" 2, both a normals loss), Elich
2311.04698 (40 pp., "conflict" 38, "magnitude" 61).

**PCGrad's zero, validated.** `len(re.findall(r'[Pp]areto', full)) == 0` over 92,889 chars, with
`PCGrad` 192 / `conflicting` 26 / `theorem` 25 / `cosine similarity` 14 / `gradient` 178 asserted
non-zero on the *same* extraction first. This confirms the cosine track's report. Two notes on its
provenance comment, neither a defect in the document: my extractor gives 92,889 chars and 178
`gradient` where the comment records 90,294 and 185 (different text extraction of the same 27-page
PDF; pages, `theorem` = 25 and `cosine similarity` = 14 agree exactly), and the verdict the comment
rests on reproduces either way. Also: the track reported "MGDA states a dichotomy", which is true
of Theorem 1 (p. 6) but incomplete — MGDA also claims on p. 6 to "provably find a Pareto stationary
point". No prose depends on it, since `[46]` is cited only for the multi-objective framing and the
dominance and front definitions, which I verified at Def. 1, p. 3.

**Table 11 re-derived** from `src_utils/_round7/gradient_cosine_observations6.parquet` (4,650 rows,
asserted; `state, fold, epoch, cos, config`), fold means for the six single-configuration datasets
and the 60 configuration-by-fold series for Florida, `scipy.stats` t-test / `binomtest` / t-based
TOST at `±0.05`:

| dataset | n | mean | 95% CI | t | sign | pos. | TOST p | CI excludes 0 |
|---|--:|--:|---|--:|--:|--:|--:|:--:|
| Florida | 60 | +0.0003 | [−0.0010, +0.0015] | 0.676 | 0.897 | 31/60 | 4.5e−62 | no |
| Alabama | 5 | +0.0112 | [+0.0040, +0.0184] | 0.013 | 0.062 | 5/5 | 5.8e−05 | **yes** |
| Arizona | 5 | +0.0015 | [−0.0051, +0.0081] | 0.562 | 1.000 | 3/5 | 1.7e−05 | no |
| California | 5 | +0.0007 | [+0.0000, +0.0014] | 0.048 | 0.375 | 4/5 | 2.0e−09 | **yes** |
| Texas | 5 | −0.0003 | [−0.0024, +0.0018] | 0.744 | 0.375 | 4/5 | 1.6e−07 | no |
| Istanbul | 5 | +0.0001 | [−0.0008, +0.0011] | 0.756 | 1.000 | 3/5 | 6.6e−09 | no |
| Georgia | 5 | +0.0039 | [+0.0016, +0.0061] | 0.009 | 0.062 | 5/5 | 3.0e−07 | **yes** |

7 of 7 rows agree with the printed table. California's CI lower bound is `+1.10e−05`, printed
`+0.0000`. Slope counts also reproduce (Alabama 5/5 negative, Georgia 5/5, California 4/5, Arizona
3/5, Texas 3/5, Istanbul 2/5, Florida 29/60), matching p. 100's "from two of five at Istanbul to
four of five at California, and Florida is flat at twenty-nine of sixty".

**The evidence for M4 and M5, stated as the count that matters:** 3 of 7 datasets have a 95%
interval excluding zero, and it was 2 of 4 before this round — the round widened the coverage and
the "indistinguishable" wording got *less* accurate, not more.

**The M1 counter-example**, `numpy`/`scipy.optimize` on `g1 = [3,0]`, `g2 = [0,1]`, `cos = 0.0`
exactly: PCGrad returns `[3,1]`, identical to the unmodified sum (asserted with `np.allclose`);
MGDA min-norm returns `w = [0.100, 0.900]`, matching the closed form
`‖g2‖²/(‖g1‖²+‖g2‖²) = 0.100`, 53.1° off the sum; Nash-MTL's `GᵀGα = 1/α` is solved by
`α = [0.333, 1.000]` (verified with `np.allclose(GtG @ alpha, 1/alpha)`), update `[1,1]`, 26.6°
off; CAGrad's own `min_w gwᵀg0 + √φ‖gw‖` at `c = 0.4` deviates 18.6° (at `c = 0.2`, 10.1°; at
`c = 0.8`, 31.2°; at `c = 0`, 0° as its Remark p. 4 says it must); `κ(G) = 3.0`. Also from the
parquet: 2,218 of 4,650 recorded epoch-level cosines are negative (47.7 percent), so under PCGrad's
own Def. 1 the projection fires at roughly half the recorded epochs — orthogonality *on average*
does not mean an inactive balancer *at a step*.

**Bibliography.** `references.bib`, 99 entries (`grep -cE '^@' src/references.bib` → 99): `kurin` 5 hits, `xin2022` 3, `lin2022rlw` 2, `standley2020tasks` present;
`elich`/`2311.04698` **0**, `royer`/`2310.08910` **0**, `2308.13985` **0**, `2109.04617` **0**,
`2009.00909` **0**. Bracket-to-key mapping read from `build/main-aux/main.bbl`: `[46] sener2018mgda`,
`[47] nash`, `[48] liu2021cagrad`, `[49] senushkin2023aligned`, `[50] yu2020pcgrad`,
`[51] kendall2018uncertainty`, `[52] chen2018gradnorm`, `[53] liu2019dwa`, `[54] liu2023famo`,
`[55] lin2022rlw`, `[56] xin2022domtl`, `[57] kurin2022scalarization`.

**Dataset sizes** for M9, from `src/tables/mobiwac/datasets.tex` (not from memory): AL 113,846;
AZ 236,450; IST 462,615; FL 1,407,034; CA 3,171,380; TX 4,089,892. Regions 520 (IST) to 8,501 (CA).
Georgia has no row there, which is consistent with p. 97's statement that the dissertation does not
otherwise use it.

## Corrections to this report

**Four** defects in earlier versions of this file, recorded here rather than fixed silently, because a
review that misstates its own numbers has no standing to correct anyone else's. Two were caught by a
reviewer and two by re-measuring. Counted against the rows below, which is the check the fourth
defect's own rule demands.

1. **M9 named the wrong comparison entity.** It read "a mean sixteen times Texas's". Alabama
   `+0.0112` against Texas `−0.000264` is 42× at full precision and 37× on the printed cells; 16× is
   Alabama against **California** (`+0.0112 / +0.0007`). Caught by a reviewer against my own Table 11
   re-derivation. Rewritten to state the comparison that is both true and sign-safe: Texas's mean is
   negative, so a bare "times" ratio against it crosses a sign change and should not be written that
   way at all.
2. **The `dissertacao.pdf` disclosure attributed to me a change I did not make.** A build that
   produces no PDF cannot have run the `cp`. Retracted in "The build" above, with the writer named
   UNKNOWN per V9 rather than inferred from the five concurrent commits.
3. **The build note was written as a standing property of the tree** when it is a transient
   condition: the working tree's `main.log` now records a successful 102-page run, so my failure is
   no longer reproducible there. Rewritten as a conditional.

4. **The re-sweep I ran after fixing M9 overstated itself, in the exact shape §4b V13 describes.** I
   wrote "24 of 24 verified" and "the sweep asserts each check rather than printing it, so a silent
   zero could not pass as a clean result." Both were false of the code that produced the claim. It
   built a dict of booleans and *printed* the failures; nothing was asserted. And two of the
   twenty-four entries were the literal `True` — `"spearman -0.486 p 0.33"` and `"FL config-mean span
   -0.00261/+0.00457"` — so they passed unconditionally. Those two are precisely the silent pass the
   sentence claimed was impossible: the summary line borrowed the credibility of the twenty-two real
   checks for two that measured nothing. Caught by a reviewer.

   **Re-taken properly.** Twenty-seven checks, the two former placeholders now computed from the
   parquet rather than restated, each collected and then `assert`ed so a failure raises instead of
   printing: Spearman recomputed to `ρ = −0.4857, p = 0.3287` against the report's `−0.486 / 0.33`,
   and Florida's configuration-mean span to `[−0.00261, +0.00457]` over 12 configurations. **All 27
   pass**, and the assert instrument was validated by sabotage in the same cell (an injected false
   check raised `AssertionError: FAILED CHECKS: ['deliberately false']`), because a green sweep from
   an unvalidated instrument is worth nothing. The set covers all seven dataset means, the seven
   per-dataset slope counts, the pooled mean, the negative-observation count, the row counts, the two
   size factors, both ratios in the rewritten M9 including the 16× value that was wrong, and the
   "3 of 7 CIs exclude zero" figure M4 rests on.

## Open questions only the author can answer

1. M1 and M3 both narrow a claim the dissertation could instead *defend*: the Chapter 5 screen is
   real evidence that balancers did not pay off here, independent of the mechanism. Would you
   rather narrow the mechanism sentence (my proposal) or drop the mechanism reading from §2.3
   entirely and let the screen carry the point, which is the branch your own `[NEEDS SIGN-OFF]`
   note already contemplates?
2. Adding Elich (arXiv:2311.04698) strengthens M1's rewrite but also imports the finding that
   angular conflict is not uniquely a multi-task phenomenon, which cuts slightly against the
   appendix's framing of orthogonality as "stranger" than expected. Cite it, or leave the passage
   citing only what it already has?
3. An examiner may notice something the text does not use: orthogonal gradients are linearly
   independent, which is exactly Nash-MTL's Assumption 5.1 (p. 3, p. 6). Your measurement therefore
   *satisfies* the assumption under which its guarantee holds. Worth one sentence, or out of scope
   for a fundamentals chapter?
