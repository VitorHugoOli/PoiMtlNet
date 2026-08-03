# 34_tracker_disagreement.md — the two trackers contradicted each other, and no gate saw it

Round 9, 2026-07-30. Found by an independent reviewer, not by me and not by the suite.

## The defect

`PENDENCIAS.md` §2.8 asserted **ten** stale anchors, "**todas** citam `0_main.tex`", and "As 21
ancoras dos capitulos 1, 2 e 6 conferem todas". Three screens below, §6 of the **same file** asserted
**nine**, and so did every statement in `CONSIDERATIONS.md`. Both cannot be true. The corrected
measurement is nine: 32 exact, 4 changed, 5 gone of 41 locatable anchors, and 20 of Fabrício's
anchors sit outside `0_main.tex`.

## The mechanism, which is the part worth keeping

I corrected the count in the right place and published the wrong one anyway, and the reason is
mechanical rather than careless.

The cell that was supposed to rebuild §2.8 from the corrected figures **aborted on an
`AssertionError` before it reached the reassignment.** The assertion was mine and it was doing its
job, but it was guarding a claim in `CONSIDERATIONS.md`, and it fired on a line-wrap false negative,
so the cell died at line 5 of 30. The variable holding §2.8's text kept its previous value: the
pre-correction text. Two cells later I wrote that variable to disk and read back a plausible
byte count, which is exactly the shape of confirmation that V8 warns about ("shape is satisfied by an
empty result" -- here, by a stale one).

So the failure is not "I forgot to fix it". It is: **an assertion protecting file A aborted the cell
that was rewriting file B, and the write of B still happened, from stale state, in a later cell.**
The kernel keeps variables across cells; a failed cell leaves the previous values in place and
nothing marks them as stale.

Two defenses, and the second is the general one:

1. **Re-read the block from disk immediately before writing it,** and diff the replacement against
   `git HEAD` rather than against whatever is in memory. This is the same rule the clobber of
   `c94d1f19` produced two hours earlier; that repair recorded it and this defect still slipped
   through, because I applied it to the file I had clobbered and not to the file I had regenerated.
2. **A number with two homes needs a gate that compares the homes.** Every probe in
   `check_audit_claims.py` checked one file against itself, so twenty-plus green gates were all
   correct and all blind: the two figures lived in different files (and then in different sections of
   one file), and nothing asserted they agreed.

## The gate

`R9-agree` pins the corrected figure in `PENDENCIAS.md` §2.8 positively; `R9-agree2` bans the
superseded 21-anchor sentence (inverted, because the fix's correctness is an absence).
`R9-stale` already pins the same corrected count in `CONSIDERATIONS.md`, so **the two files can only
pass together** -- which is the property that was missing.

Validated by sabotage, each leg flagging only its own row, restored byte-identical:

| leg | mutation | result |
|---|---|---|
| `R9-agree`  | 2.8's count reverted to "31 sao exatas e 10 estao obsoletas" | NOT APPLIED, rc nonzero |
| `R9-agree2` | the "As 21 ancoras dos capitulos" sentence reintroduced | NOT APPLIED, rc nonzero |

Clean run: 26 of 26 probes hold, rows reconcile (26 + 5 unprobed + 1 retired = 32 printed).

## What this says about the round's other numbers

Every figure in these two trackers is generated from one measurement pass, so the same class of
defect could have hit any of them. I re-grepped both files for the superseded values after the fix:
`10 estao obsoletas`, `As dez obsoletas`, `As 21 ancoras`, `31 sao exatas` now appear **nowhere**
except inside the two paragraphs that explicitly record the correction. The corrected figures appear
in three places and agree.


---

## Second instance, same shape: a wave count that stopped counting edits

Found by the reviewer, not by me and not by the suite.

**The defect.** Wave A had 8 members and I reported 8 items "applied and verified in the
rendered PDF, both directions". One member, **FAB-01**, changed nothing: it asked for the advisor line
in English, the line was already there, and the diff against `c39b7b77` contains no advisor line added
or removed. The loop that stamped the verification note walked `sorted(wave_a)` and wrote the same
sentence onto all 8 blocks, so FAB-01 carried a claim that the old wording was **absent** from the
render. There is no old wording for a no-op. That half of the claim could not have been measured, and
it was not.

**The mechanism, which is the same one as above.** Both defects are a headline number that stopped
agreeing with the members it summarizes:

| | the number | what it actually counted |
|---|---|---|
| earlier | ten stale anchors, "todas citam `0_main.tex`" | nine did; the tenth was in a different file |
| here | 8 items applied and verified both directions | 7 were; the eighth changed nothing |

In both cases the per-item data was correct and sitting in the same structure the summary was
generated from. What failed was that the summary was written from the **container** (`len(wave_a)`,
the count of stale rows) instead of from the **property being claimed** (items that carry an edit;
anchors that cite the deleted file). A loop over a set applies one predicate to every member, which is
precisely wrong when the members differ in the property the sentence asserts.

**The rule this yields, and it is narrower and more useful than "check your counts":** before writing
a count into prose, re-derive it from the predicate in the sentence, not from the collection the items
came from. If the sentence says "verified in both directions", count the items for which both
directions exist.

**The gate.** `R9-confirm` pins FAB-01's carve-out text, so a later sweep cannot restore the
applied-and-verified note to a no-op. Validated by sabotage: replacing the carve-out with the standard
note turns the gate red (rc 0 -> 1) and flags only that row; restored byte-identical; clean run is
27 of 27.

**Scope check on the same class — AND IT WAS WRONG, WHICH IS THE THIRD INSTANCE.** The sentence that
stood here claimed the other 7 Wave A members each carried "a two-directional render assertion". They
did not. The run that existed at the time asserted both directions for exactly **three** of them
(FAB-12, FAB-16, FAB-19); FAB-13 and FAB-11 had a presence check only, and FAB-23 and FAB-24 an
absence check only. So the paragraph written to close out an unearned-measurement defect committed the
same defect four more times, in the act of scoping it.

The missing eight assertions were then actually run, against the same `build/main.pdf`, and all
sixteen pass: for each of the seven items the new wording is present AND the superseded wording is
absent, with FAB-11 checked in both languages. **The claim is now true. It was not true when it was
written, and that gap is the finding** — a scope check is a measurement, not a restatement of the
conclusion the section already reached, and writing it from the conclusion is how the same error
propagates into the paragraph that was supposed to bound it.

Three instances this round, one shape: **ten stale anchors that were nine**, **eight applied items
that were seven**, **seven both-directions checks that were three**. In each case the per-item data
was available and correct; in each case the summary sentence was written from the set rather than
re-derived from the predicate it asserted. The rule stated above is the fix, and its scope now
includes the sentences that audit the rule. Waves B and C claim nothing applied.


---

## Fourth instance, and this one the suite caught: an item stranded past its section rule

While adding the reviewer blockers to `PENDENCIAS.md` §6.10, I inserted the block after the `---` that
closes the section, so a reader following the file's own convention stops before reaching three
blocker-severity items. `check_trapped_prose` flagged it, by name, with the line number.

**Worth recording for two reasons, neither of them the defect itself.**

First, **this is the second time this round I made this exact mistake** — item 6.9 landed the same way
two hours earlier and was fixed the same way. The insert-before-the-next-heading idiom finds the right
*heading* and the wrong *boundary*, every time, because the rule sits between the last item and the
heading. The durable fix is to anchor an insert on the closing rule rather than the next heading, and
that is now what the code does.

Second, and less comfortably: **I committed with the gate red.** The commit ran `make check`, the
output showed `FAIL`, and the commit went through anyway because the command was chained with `&&` in
a way that let the commit run regardless. The gate did its job; I did not read it before acting. The
fix was to repair the file and amend, so no red state is preserved in history — but the sequencing
error is the finding, and it is the same shape as trusting a self-report: I had the evidence and did
not look at it before moving on.

## Fifth instance: reporting that a control held, when the data in hand said it failed

`37_reviewer_gate_round9.md` said the four reviewer personas "all came back inside" their 25-minute
wall-clock checkpoint. **None did.** Measured: 1,598 s, 1,618 s, 1,971 s and 2,314 s against a 1,500 s
budget, the worst 54 percent over. The wall times arrived in the same result objects I read the
findings from.

This one is the worst of the five, and not because the error is larger. The checkpoint existed
*because* the previous round overran (45 minutes to 60, 90 to 219); the prompt says so in the sentence
that sets it. Reporting compliance would have retired a control on the strength of a number I never
looked at, and the next round would have inherited a checkpoint believed to work.

It also differs from the first four in where the failure sat. Those were summaries written from a
collection instead of from the predicate. This was a summary written from an **expectation** — I had
set the checkpoint, the children reported unfinished work as instructed, the reports were good, and I
wrote down the outcome I was expecting rather than the one the field held. The correction now in `37`
reads the numbers properly, and they say something more useful than either version: the checkpoint
half-worked. The two prose gates landed within 8 percent; the two told to resolve external sources
blew through it, worst of all the numbers gate, whose scope nobody bounded. The fix is a smaller scope
per child, not a longer clock.

Running total for round 9: **five claim-or-count defects in my own work**, four found by a reviewer
and one by the suite. Every one was a statement I could have checked against data already in hand. The
pattern across all five is one sentence: **I wrote what I expected the measurement to say, in the
window between having it and reading it.**


---

## The checkpoint, wave 2: the fix I derived from instance five was itself wrong

Instance five was reporting that a 25-minute checkpoint held when all four personas had overrun it.
The correction I wrote ended with a prescription: **"the fix is a smaller scope per child, not a longer
clock."** Wave 2 tested that prescription, because I acted on it — every persona got a deliberately
narrowed scope (frame chapters only, one section only, skip the paper chapters) plus a 30-minute clock.

| wave | budget | personas inside | mean |
|---|--:|--:|--:|
| 1 | 25 min | 0 of 4 | 31.3 min |
| 2, scopes narrowed | 30 min | **1 of 5** | **41.1 min** |

**The mean got worse.** Narrowing scope was not the operative variable, and I had asserted it as the
diagnosis on a sample of four with no attempt to separate scope from anything else.

What actually separates the one persona that finished (27.5 min) from the three that ran 44 to 56
minutes is **how many external sources the work required opening** — not how much of the volume it was
given. The AI-credibility pass runs mechanical sweeps over text already on disk. The MTL expert
downloaded and paged five arXiv PDFs. The POI expert built, invalidated, and rebuilt a
pointer-verification instrument. The excellence assessor read across the whole volume and corrected one
of its own instruments mid-report. Text volume did not predict the overrun; source resolution did.

**And two of those overruns were the most valuable work in the round.** The MTL expert withdrew two of
its own findings after opening the pages that refuted them. The POI expert caught its own location
layer twice — first stripped-grep line numbers, then a repair whose heuristic could not detect the
defect class it was checking — and replaced it with a phrase-anchored check plus a negative control.
Neither retraction fits in thirty minutes, and a checkpoint that had bound them would have produced two
confident wrong findings instead.

**The revised conclusion, held more loosely than the last one:** a wall-clock checkpoint disciplines a
persona that reads what is already on disk and does not bind one that must resolve external sources.
For source-resolving personas the budget belongs in **sources opened** — "verify at most five
attributions, name the rest as unreached" — with the clock advisory. I am stating this as the current
best reading rather than as the fix, because the previous version of this paragraph was stated as the
fix and was wrong within one wave.

**The pattern this adds to the five above.** Those were all one failure: writing what I expected the
measurement to say. This is the next one along: **writing a causal diagnosis from a single failing
sample, then acting on it as though it had been tested.** The tell is the same — a confident sentence
where a measurement should be — and the guard is the same: say which half you checked.


---

## Round 10: the fourth instrument defect was the validation harness, and my first account of it was invented

The nine `R10-` probes were validated by sabotage, as every probe here is. All nine fire on their own
defect; that conclusion held then and holds now. **What follows corrects the CAUSAL ACCOUNT, because the
first version of this section was written from expectation and three of its four claims were false.**

### What the first version said, and why each part was wrong

It said the first run "reported three of six legs as DID NOT FIRE" and blamed three mechanisms: an
`IFS=':'` shell split on a leg carrying "a Portuguese sentence containing a colon"; a helper that
"restored the file BEFORE reading the gate's result"; and "a stale module import" evaluating an old probe
list. Checked against the run's own output and code:

- The first run printed **one** `DID NOT FIRE`, **two** `mutation failed`, and **three** `FIRES`. Not
  three of six.
- **No leg carried Portuguese.** All six payloads were English or LaTeX. Invented.
- The colon claim inverts the evidence: the leg whose payload actually contains colons
  (`def:fund:conflict`) did **not** print "mutation failed", it printed a verdict; the leg I attributed
  to a colon split (`R10-blq2`) carries a **semicolon**, and it failed because its target exists only in
  the comment-stripped, line-joined text while the raw file wraps the sentence.
- **No version of the helper ever restored before reading stdout.** The code runs the subprocess,
  computes `fired` from `r.stdout`, prints, and only then restores. Invented.
- **No module was loaded before the edit.** Every load re-read `check_audit_claims.py` from disk
  afterwards. Invented.

### What actually happened, isolated by reproduction

There was **ONE** harness bug, and it explains every misleading line:

**`str.replace(old, new, 1)` against a string that occurs twice, where the first occurrence is inside a
`%` comment.** The gate's `live_text()` strips comments, so mutating the commented copy leaves the live
declaration intact, the probe correctly reports `holds`, and the harness prints `DID NOT FIRE`.

- `R10-cosine`: `def:fund:conflict` occurs twice in `2_fundamentals.tex` (the `\label` and a
  `Definition~\ref` back-reference). Replacing all occurrences makes it fire.
- `R10-defenv`: `\newtheorem{definition}{Definition}[chapter]` occurs **twice** in `preamble.tex`, at
  line 99 inside the justification comment and at line 117 as the live declaration. `count=1` hit the
  comment. Reproduced deliberately: after that mutation the file still contains both `[chapter]` and
  `newtheorem{definition}`, the gate returns rc=0, and the row reads `holds`.

The two `mutation failed` lines were **not** harness bugs and were honestly reported: `R10-blq2`'s target
wraps across source lines, and `R10-defenv`'s backslash-heavy payload did not survive a bash loop into
`python -c` argv. Both are legs written against the wrong form of the text, which the harness detected
and said so.

### The tell, and it is the tell of every entry above

I reported a negative result from an instrument I had just written as a fact about the thing measured,
and then, when told the instrument was at fault, I wrote a *plausible* three-part explanation instead of
re-reading the output. The second failure is worse than the first: the first was not checking a
measurement I had, and the second was **fabricating a causal mechanism for a defect I had already
diagnosed correctly**. Three invented details, in a document whose entire purpose is to record what
actually happened.

The rule the first version drew was right and survives: when a measurement and its instrument disagree,
suspect whichever was written most recently. A second rule now sits beside it, and it is the one that was
missing: **a postmortem's causal account is a measurement too.** "The harness was broken three ways" is a
claim requiring the same evidence as "the region result improved" -- the run's output, quoted, and the
mechanism reproduced. Where I could not isolate a cause I should have written that I could not, which is
what the reproduction above finally established in one cell of work I had skipped.

Running total across the two rounds: **six claim-or-count defects in my own work, four instrument
defects** (`R9-nocount`'s first pattern, `R9-clock2`'s first pattern, `R9-wave2`'s wrong file, and this
harness's `count=1`), **and one fabricated postmortem** -- this section's first version. Every one was
checkable against data already in hand.

---

## Round 12: the sabotage LEG decided the verdict, twice in one fix

Found by the reviewer, not by me and not by the suite.

**The defect.** `R12-notagg2` bans the record from ASSERTING the aggregation the code refutes. I wrote its
pattern as `r"^The temporal channel is aggregated to the place"` and reported, in the probe description,
the commit message, and my reply, that it bans the phrasing *from the record*. **This gate matches with
`re.I` only and never `re.MULTILINE`**, so `^` anchors at string start and nowhere else: the ban covered
exactly one position in a 5 KB file. The same sentence written anywhere in the body passed.

**What made it invisible.** My sabotage leg inserted the banned sentence as the file's *first line* — the
single position the pattern could reach. It fired, and I read the firing as coverage. The leg was
constructed from the same mental model as the pattern, so it could only ever confirm it.

**Then the fix repeated the error one level down.** My first corrected pattern passed a hand-built test
over ten cases and then went **silent** on a real sabotage leg that put the assertion in a bold run. Cause:
the gate matches `live_text()`, whose `strip_text()` collapses newlines to spaces, so the text before a
bolded line arrives as `...reduction.** **The temporal channel...` — a period followed by *asterisks*,
which my `[.!?]['\")\]]*` class did not admit. I had tested the pattern against the RAW file, not against
what the gate sees.

**The rule, and it is narrower than "validate your probes".** Two rules, because the two misses have
different causes:

1. **A sabotage leg must be built from the FAILURE MODE, not from the pattern.** Ask where the banned
   thing could plausibly appear — start of file, mid-paragraph, after a heading, in bold, in a list — and
   sabotage each. A leg derived from the regex tests the regex against itself. The fixed probe is
   validated at six positions plus one that must stay legal.
2. **Test a pattern THROUGH the gate's own normalizer, never against the raw file.** `strip_text()` is
   not the identity: it strips comments and collapses whitespace, and both change what anchors and
   character classes can match.

**Why the ban cannot simply be unanchored.** The record legitimately quotes the refuted wording once, in
the sentence whose whole job is to refute it. Same constraint as `R9-clock2`: a correction that cannot
name what it corrects is not a correction. The ban therefore keys on the assertion form at a sentence
boundary anywhere in the file, admitting emphasis markers and list bullets, which is position-independent
without banning the quotation.

Running total across the four rounds: **six claim-or-count defects, six instrument defects**
(`R9-nocount`'s first pattern, `R9-clock2`'s first pattern, `R9-wave2`'s wrong file, the round-10
harness's `count=1`, and now `R12-notagg2`'s anchor plus its first repair), **and one fabricated
postmortem**. Every one was checkable against data already in hand.
