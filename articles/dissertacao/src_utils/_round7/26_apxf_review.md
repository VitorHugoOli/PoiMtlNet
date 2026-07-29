# 26 · Appendix F reviewed with the personas, and fixed — the gradient-cosine appendix

**Track:** review `src/chapters/apx_f_cosine.tex` with the reviewer personas and fix what the pass
finds. The appendix had never been reviewed: it was written late in round 7 and shipped on the
author's build without a persona pass.

**Read first, in the required order:** `src_utils/_round7/BRIEF.md` (the round brief; the task named
`_round7/AGENT_BRIEF.md`, which does not exist — `_round6/AGENT_BRIEF.md` and `_round7/BRIEF.md` do,
and BRIEF.md is this round's), `WRITING_LAW.md`, `GLOSSARY.md`, `AGENT_GUARDRAILS.md` §1–§4b.
**Personas run:** 06 (number auditor), 15 (readability editor, the roster's persona for
style/readability), 09 (stats & leakage skeptic) and 10 (MTL expert) for the ML claims, 18 (visual &
presentation, on the rendered figure), 19 (LaTeX source reviewer), 12 (banca simulator).

> **A roster correction, stated because the task named personas by number.** The task asked for
> "07 (style/readability)" and "13 (banca excellence)". The roster defines **07 as the claim &
> honesty auditor**, **13 as the UFV compliance checker**, **15** as the readability editor and
> **12** as the banca simulator, with **17** as the excellence assessor. I ran the personas the
> roster actually defines and did not invent one: 15 for style/readability, 12 for the banca lens,
> and 07 in its real role (claim honesty) because this appendix is claim-dense. 13 (UFV compliance)
> was not run: nothing in this pass touches margins, numbering, or front matter.

---

## STEP 1 · Where this appendix sits in the document's arc (read the whole PDF first)

I read `build/main.pdf` end to end before touching the prose, 101 pages as it stood at the start
(measured: `len(pypdfium2.PdfDocument("build/main.pdf"))` = 101). The structure, page spans measured
from the extracted text: front matter 1–12, Ch.1 13–16, Ch.2 17–26, Ch.3 27–42, Ch.4 43–57, Ch.5
58–76, Ch.6 77–80, References 81–88, Appendices A/C/E 89–95, Appendix F 96–101.

**The view: this appendix is the mechanism chapter the arc has been promising since Chapter 3, and
it is currently the only place in the document where that mechanism is actually measured.**

The dissertation's spine is a conditional answer: MTL helps *if* the representation and the sharing
topology are right. Chapter 3 closes with three candidate explanations for its null result, and the
third is "Architectural Restrictiveness" — hard sharing may be too rigid. Chapter 6 §6.2 then makes
the strong claim that closes the arc: "Under the check-in-level representation the two sequential
tasks coexist with essentially orthogonal gradients: sharing stopped hurting." That is a mechanistic
claim, and until this appendix existed the document's only support for it was a single sentence in
Ch.5 §5.2.4 reporting a development-time cosine of +0.001 over four seeds, with its largest
per-dataset mean +0.0032. **Appendix F is what turns that sentence into evidence** — a per-epoch
measurement on the shipped data preparation, four datasets, with an equivalence test rather than a
failure to reject.

Three consequences for how the appendix should read, all of which shaped my edits:

1. **It answers Chapter 3's third hypothesis, and it should be legible as that.** Ch.3 hypothesized
   restrictive sharing; this appendix shows there was little interference for a better sharing scheme
   to fix. §F.3's second paragraph already does this work and is the most valuable paragraph in the
   appendix. I compressed it but did not touch its logic.
2. **It must not read as superseding Chapter 5.** Ch.5 is under review and its numbers are its own;
   the two measurements differ (pooled +0.00118 here versus +0.001 there, and my largest per-dataset
   mean is Alabama's +0.0112, not +0.0032). The appendix already had the sentence that prevents a
   reader from reading a contradiction. It survives, tightened.
3. **Its central discipline is the one thing worth more than its result.** The appendix holds the
   positive mean and the downward drift to the *same* n = 5 standard, which is what commit 4445e7ab
   fixed. That is the strongest methodological writing in the document, and Ch.5 §5.6.2 and Ch.6 §6.2
   both practice the same restraint (naming what the freeze control rules out and what it leaves
   open). **The appendix belongs in the arc because it is consistent with that voice, not merely
   because the data existed.** I weakened none of it — see "What I refused to change".

**One arc-level gap I did not close** (author's call, flagged, not acted on): nothing in the main
text points at this appendix. It is reachable from the table of contents only. The appendix's own
header comment records the reason — a pointer from Ch.5 would be an edit to an under-review chapter —
but Chapter 6 §6.2 is frame prose the author owns, it makes the orthogonality claim in plain terms,
and a `Section~\ref{apx:cosine}` there would cost one clause and connect the claim to its evidence.
`grep` over `src/**.tex` for `apx:cosine` outside the appendix returns nothing.
**[NEEDS SIGN-OFF: add a pointer to Appendix F in Ch.6 §6.2's gradient paragraph.]**

---

## STEP 2+3 · Findings, by persona, with verdict and evidence

Severity per `reviewers/README.md` §5. Every number below came from output I read in the cell that
produced it; the commands are given so the author can re-run them.

### Persona 06 · Number auditor — **GATE FAIL at entry, PASS after the fixes**

**F1 · BLOCKER · The figure caption made three false statements about the figure.** This is the
author's complaint (a) turning out to be a number-integrity finding as well as a length one. All
three were measured from the PNG's pixels, not judged by eye:

| Caption said | The render shows | How measured |
|---|---|---|
| "one outline per dataset, with the number of observations in the key" | panel (a) is **one pooled histogram**; its in-panel annotation reads "n = 3,900 epoch-fold observations, 4 datasets" | exactly one blue RGB triple `(72,120,168)` occurs in the plot area; no per-dataset key exists |
| "the dashed line marks zero" | the zero line is **solid** | the column at cosine 0 carries grey in 539 of 562 plot rows, every row-gap = 1 px |
| "the horizontal axis is clipped at $\pm 0.20$ ... so the tails reaching $-0.34$ and $+0.58$ are outside the frame" | the axis runs **$-0.40$ to $+0.40$**, so $-0.34$ is **inside** and visible; exactly **one** observation of 3,900 ($+0.5802$) lies beyond it | tick positions measured at 224–225 px per 0.10 unit, calibrated on the $\pm0.05$ band; `(df.cos.abs()>0.40).sum()` = 1 |

Root cause is `AGENT_GUARDRAILS` §4b V6, the same class as the defect 4445e7ab fixed: that commit
rebuilt the figure from the four-dataset parquet and corrected the caption's panel-(c) sentence, and
nobody re-read the panel-(a) sentence against the new image. **Action: caption rewritten** (see
STEP 3 measurements).

**F2 · MAJOR · "They are the largest of the six" is false for Istanbul.** The closing paragraph
justified the three missing datasets by size. Measured from the dissertation's own table
(`grep -v '^[[:space:]]*%' tables/mobiwac/datasets.tex | grep -E '^(AL|AZ|FL|TX|CA|Istanbul)'`):

    check-ins ascending: AL 113,846 | AZ 236,450 | ISTANBUL 462,615 | FL 1,407,034 | CA 3,171,380 | TX 4,089,892
    regions   ascending: Istanbul 520 | AL 1,109 | AZ 1,547 | FL 4,703 | TX 6,553 | CA 8,501

Istanbul is the **smallest** of the six by region count and the third smallest by check-ins — smaller
than Florida, which *is* measured here. The sentence held for two of three named datasets and
inverted the truth for the third, and it made the resource explanation appear to cover a case it does
not. **Action: size claim scoped to California and Texas; Istanbul given its real reason** (it never
started — job 805120f1 was killed inside california's fold 1 and texas and istanbul had not begun,
per the appendix's own `[VERIFY]` block).

**F3 · MAJOR · The named source of record could not run.** The appendix's header declares
`src_utils/_round7/cosine_stats.py` its derivation, "whose printed output is quoted". It crashed on
the parquet it ships with:

    AssertionError: expected ['alabama','arizona','florida'], got ['alabama','arizona','florida','georgia']

`EXPECTED` and two assertions still declared 3 states / 3,650 rows after Georgia landed (ff69ba07).
A file cannot be the source of record for numbers it refuses to compute. **Action: fixed to 4 states
/ 3,900 rows; it now exits RC=0**, and I re-checked every value in the appendix and the table against
its output *and* against an independent recomputation from the parquet.

**F4 · ALL-CLEAR · Every other number in the appendix and the table reproduces exactly.** Two
independent derivations (my own, and the repaired script) against
`gradient_cosine_observations.parquet` and `gradient_cosine_figure_facts.json`:

| Quantity | Appendix / table | Recomputed | Status |
|---|---|---|---|
| observations, per-state split | 3,900 = 3,150 + 250 + 250 + 250 | identical | ✓ |
| pooled mean | +0.00118 | +0.001182 | ✓ |
| within margin | 91.3 % | 91.28 % | ✓ |
| range | $-0.34$, $+0.58$ | $-0.3407$, $+0.5802$ | ✓ |
| Florida | n=12 configs, mean +0.0003, CI $[-0.0012,+0.0017]$, t 0.70 / sign 0.77 | +0.00026, $[-0.00118,+0.00170]$, 0.6965 / 0.7744 | ✓ |
| Alabama | n=5 folds, +0.0112, $[+0.0040,+0.0184]$, t 0.013 / sign 0.063 | +0.01119, $[+0.00399,+0.01840]$, 0.0125 / 0.0625 | ✓ |
| Arizona | n=5, +0.0015, $[-0.0051,+0.0081]$, 0.56 / 1.00 | +0.00150, $[-0.00511,+0.00812]$, 0.5621 / 1.0 | ✓ |
| Georgia | n=5, +0.0039, $[+0.0016,+0.0061]$, 0.009 / 0.063 | +0.00385, $[+0.00158,+0.00612]$, 0.0093 / 0.0625 | ✓ |
| per-fold slopes, negative counts | AL 5/5, GE 5/5, AZ 3/5, FL 29/60 | identical | ✓ |
| slope t-tests | AL 0.006-class, GE 0.047-class, FL flat 0.71 | 0.0058, 0.0470, 0.7108 | ✓ |
| Florida config-mean span | $[-0.00261, +0.00457]$ | identical (min `T6_4_two_pass`, max `shipping_florida_mtl_ep50_seed42`) | ✓ |
| 12 of 12 configs equivalent at fold unit | asserted | 12 of 12, TOST p < 0.05 each | ✓ |
| TOST equivalence at every unit | asserted | holds at observation, fold-series and configuration levels for all four | ✓ |
| Alabama first-five-epoch mean | about $+0.058$ | +0.05824 (the 25 values of that block) | ✓ |
| check-in and region ranges on the data axis | 113,846–1,407,034; 1,109–4,703 | matches `tables/mobiwac/datasets.tex` | ✓ |

### Persona 09 · Stats & leakage skeptic — **survives a hostile examiner, with one disclosure added**

**F5 · MAJOR · An undisclosed duplication in the observations.** The appendix said "one
configuration on one dataset is five series of fifty values". Measured:

    df.groupby(['state','config','fold']).size().value_counts()   ->  50: 65 series, 65: 10 series

Two Florida configurations (`T6_4_two_pass`, `T6_4_infonce_tau0_5`) hold **65 rows per fold**: epochs
1–15 each appear **twice with different cosine values**, a partial re-run harvested on top of the
original series. 10 of 75 series, 150 of 3,900 rows. `AGENT_GUARDRAILS` §4b V2 is explicit that a
filter, skip or duplicate must be named with its count, and the sentence as written covered 65 of 75
series while reading as if it covered all of them.

I checked whether it changes anything before deciding how to handle it. It does not change the
appendix's claim: both configurations remain equivalent to zero at the fold unit (TOST p = 1.54e-05
and 2.90e-08), and deduplicating on `(state, config, fold, epoch)` moves the pooled mean
+0.001182 → +0.001526 and the within-margin share 91.28 % → 91.36 %. It *does* move those two
configuration means (`two_pass` −0.00261 → −0.00055 keeping the first row of each duplicated epoch),
and `two_pass` is the **minimum of the reported span**. **Action: the prose now names the partial
re-run, the span sentence says it is computed "over the observations as recorded", and the full
measurement with both readings is recorded in the file's comments and in `cosine_stats.py`'s
docstring.** I did not recompute the span on a deduplicated basis: which row of a duplicated epoch is
canonical is not determinable from the parquet, and silently picking one would be a worse defect than
disclosing the duplication. **[VERIFY: whether the two 65-row Florida configurations should be
deduplicated, and if so on which rule. The harvest that produced them is not in the parquet.]**

**F6 · ALL-CLEAR · The independence argument is correct and correctly applied.** The unit is the fold
(the configuration for Florida), every reported p-value is computed on that unit, the observation
count is explicitly disclaimed as not a sample size, and the sign-test floor at n = 5 is named in the
prose, in the table, in a dagger footnote and in the figure. The appendix reports equivalence at every
level of aggregation as a robustness statement rather than resting on one. This is stronger than the
statistical writing in most of the document.

**F7 · MINOR, not acted on · The margin is justified but not pre-registered.** §F.1 argues $\pm 0.05$
on relevance grounds ("under five percent of its length ... smaller than the epoch-to-epoch
variation"), which is the right kind of argument, but unlike Ch.5's two-point region margin there is
no record that it was fixed before the data were seen. §F.4's replication recipe does tell a
replicator to choose "a margin chosen before looking", which implicitly concedes the point. Adding a
pre-registration claim would require evidence I do not have; weakening the margin paragraph would
lose a good argument. **Left as is, flagged.** An examiner may ask; the honest answer is that the
margin was argued from the quantity's scale, not registered.

### Persona 10 · MTL expert — **sound**

**F8 · ALL-CLEAR · The mechanism claim is measured, scoped, and aligned with the field's null.** The
persona's prior (a tuned fixed-weight scalarization matches specialized optimizers) is exactly what
§F.3 argues from measurement rather than assertion, and the appendix does not overreach: it says
orthogonal gradients leave a balancer nothing to resolve *in this architecture*, ties that to Ch.5's
reported finding that no balancer improved on fixed weighting, and states the architecture limit as
the boundary. Ch.2 §2.3 already carries the skeptic block (Kurin, Xin, and the unitary-scalarization
defense), so the citation footing exists in the document.

**F9 · MINOR, not acted on · One clause invites a question the appendix cannot answer.** §F.4 says the
two tasks' projections "appear to be close to statistically independent given the shared
representation". The hedge is correct and the sentence says "would explain", but statistical
independence of the targets is not what was measured — gradient orthogonality is. The persona's lens 2
(Elich et al.: angular conflict is not uniquely MTL, magnitude differences dominate) would press here.
The existing hedging is adequate for an appendix and rewriting it risks losing the insight, which is
the appendix's most interesting speculation. **Left as is, recorded.**

### Persona 15 · Readability editor — this is the author's complaints (b) and (c)

**F10 · MAJOR · The prose was prolix, and it was prolix in a specific way.** Not padding, but
throat-clearing: "The usual reason is that the tasks disagree about how the shared parameters should
change, and the disagreement is visible in the gradients" says one thing twice; "the cosine is the
natural quantity here because it is scale-free: it reports the alignment of the two requested
updates and ignores their magnitudes, which differ between the tasks and change during training" is a
40-word chain. **Action: compressed throughout** (measurements below).

**F11 · MAJOR · It read as a lab report because it never took the reader's hand at the transitions.**
The author's words were "take the hand of the reader and go with him". Concretely: the opening stated
a fact about joint models and then a fact about gradients with no bridge to *why we are about to
measure something*; §F.4's two axes were introduced as "Four datasets and twelve configurations bound
this result along two different axes" and then not labeled, so the reader had to hold the mapping
themselves. **Action:** the opening now earns the measurement ("That has a direct measurement, and
this appendix runs it..."), the two axes are named as they arrive ("The first axis is the tuning...
The second axis is the data"), and the replication recipe reads as an invitation ("Anyone who wants
to know can run the same diagnostic and change nothing else"). Sentence-length variance was preserved
deliberately (`WRITING_LAW` §4.3): measured sd 12.4 → 11.0 words with min 5 and max 57 retained.

**F12 · MINOR · Two stub paragraphs.** A 13-word paragraph carried the TOST citation alone, and a
20-word signpost paragraph carried "Two departures ... are worth reporting". Both were merged into
their neighbors, which reads better and recovered the vertical space that took the appendix from six
pages to five. The merge of the citation stub was done by moving prose *above* the comment block
rather than by deleting the blank line beneath it — that blank line is the guard against the
trapped-prose defect this very file was bitten by, and it stays.

### Persona 18 · Visual & presentation — **needs a visual pass (one item, author's call)**

**F13 · MAJOR, NOT FIXED · Panel (a) is a single pooled histogram where three panels imply
per-dataset detail, and the figure's plotting code is not in the repository.** The caption now
describes the render truthfully, so nothing false ships. But the honest fix is the other direction:
panel (a) *should* show one outline per dataset, as the original caption claimed, because that is
what would let a reader see the four distributions the rest of the appendix discusses. I could not
make that change: `git log -S 'fig_gradient_cosine'` and a search across all tracked `.py` files find
no plotting script — only `cosine_stats.py`, which computes statistics and draws nothing. Commit
4045eb8d's message describes the figure's three panels but does not carry the code. Regenerating the
figure would mean writing a new plotting script from scratch and re-deriving the panel geometry,
which is a larger change than a caption fix and needs the author's decision.
**[NEEDS SIGN-OFF: whether to rebuild panel (a) as four per-dataset outlines, and where the plotting
code lives. If it is not recoverable, the appendix's figure has no committed derivation, which is a
reproducibility gap `TEMPLATE.md` §3 would not accept for a table.]**

**F14 · ALL-CLEAR · Float placement, table craft, legibility.** Table 11 is booktabs with the caption
above (ABNT), the figure caption below; both floats sit within a page of their first reference (table
on p.97 referenced p.97, figure on p.98 referenced p.98); the figure is `width=\textwidth`; no
overfull boxes in any build mode; the $\dagger$ footnote is inside the tabular as a `\multicolumn`
row, which renders correctly.

### Persona 19 · LaTeX source reviewer — **source-clean, after F3**

**F15 · ALL-CLEAR.** `% !TeX root` present; labels follow their captions; every reference is tied
(`Table~\ref`, `Figure~\ref`, `Section~\ref`, `Chapter~\ref`); label prefixes consistent
(`apx:`/`fig:`/`tab:`); the numeric table is a separate `\input{tables/frame/cosine}` rather than
inline cells; `\includegraphics` uses a relative width; the `lakens2017tost` key resolves (0 undefined
citations in all four build logs). The one engineering defect was F3, a script under `src_utils/`.

### Persona 12 · Banca simulator — **aprovado com correções menores**

**F16 · The question this appendix invites, and what the text now answers.** *"O senhor afirma que os
gradientes são ortogonais. Em quantos datasets, com quantas repetições independentes, e por que faltam
três dos seis?"* Before this pass the text answered the first two well and the third with a sentence
that was false for Istanbul (F2). It now answers all three: four datasets, five folds each (twelve
configurations at Florida), and the three missing ones blocked on a full disk rather than queued.
A second likely probe — *"o senhor chama isso de tendência e não de efeito; por quê?"* — the appendix
answers better than any other passage in the document, and that is its defense value.

---

## STEP 3 · The author's three complaints: measured before and after

All measurements are from the source and the built PDF, taken in the cell that printed them.

| | Before | After | Change |
|---|---:|---:|---|
| **(a)** figure caption, words | **199** | **102** | −49 % |
| | table caption, words | 152 | 128 | −16 % |
| **(b)** prose words (body, excluding floats, table and headings) | 1,965 | 1,860 | −5.3 % |
| | live non-comment non-blank source lines | 170 | 158 | −7 % |
| | **Appendix F pages in `build/main.pdf`** | **6** (96–101) | **5** (96–100) | **−1 page** |
| | defense build total pages | 101 | 100 | −1 |
| | sentence length: mean / sd / min / max | 24.1 / 12.4 / 4 / 56 | 22.4 / 11.0 / 5 / 57 | tightened, variance preserved |
| **(c)** em-dashes / contractions | 0 / 0 | 0 / 0 | law held |
| | -ly adverb density (`WRITING_LAW` §4: ≤0.8 %) | 0.58 % | 0.58 % | in band |
| | banned AI-tell words, banned templates, repo codenames | 0 | 0 | law held |

**Why the prose fell only 5 % while the caption fell 49 %.** Two required disclosures were *added*
during this pass: the partial re-run (F5) and Istanbul's real reason for absence (F2). The
compression itself removed about 160 words; the disclosures put about 55 back. The page count is the
honest measure of complaint (b), and it moved: the appendix now ends on a full page rather than
spilling 32 words onto a sixth. Cutting further would have meant cutting content, which the brief
forbids.

**On (c), what "enjoyable" meant in practice.** Compression alone would have made it terser, not
warmer. The changes that do the work are the transitions: the opening now motivates the measurement
before performing it; §F.4's axes are named as the reader meets them; the replication recipe addresses
a person rather than a procedure. What I did *not* do is add personality to the statistical
paragraphs — F6's discipline is the appendix's best feature and reads well precisely because it is
plain.

---

## What I refused to change (the brief's non-negotiables, verified intact)

Every item below was checked in the final source and the final render:

- **The independent unit is the fold** (the configuration for Florida), never the 250 or 3,900
  epoch-level values. Intact in prose, table, table caption and figure caption. The observation-count
  disclaimer survives.
- **Alabama and Georgia are 5/5 with sign_p = 0.0625, the exact test's floor at n = 5 — a consistent
  tendency, not a significant effect.** Intact for the positive mean *and* for the drift. No persona
  argued for an upgrade and I would have refused: this is the fix from 4445e7ab, and the two claims
  still rest on the same footing, stated one paragraph apart. Persona 15 wanted the drift paragraph
  shorter; I trimmed its wording and left every clause of its reasoning.
- **Four datasets only.** No sentence implies six. "Three of the dissertation's six datasets plus
  Georgia" survives as the coverage statement.
- **California, Texas and Istanbul are BLOCKED by a full disk (PENDENCIAS §2.9), not pending.** The
  prose now says this outright rather than leaving it to a comment: "blocked on that machine's free
  space rather than waiting in its queue". Previously the body said "ran out of disk" and then "until
  it is run", which reads as a queue.
- **Every number matches the facts JSON and the parquet.** Verified twice (F4). No number was changed
  by this pass; two false *statements about the figure* were (F1) and one false size claim was (F2).

---

## Edits applied

**`src/chapters/apx_f_cosine.tex`**
1. Figure caption rewritten: 199 → 102 words, and its three false statements about the render removed
   (F1). The detail it shed lives in the body and the table caption; nothing was lost.
2. Closing paragraph: size claim scoped to California and Texas, Istanbul given its real reason, and
   "blocked, not queued" stated in the prose (F2).
3. Partial re-run in two Florida configurations disclosed in §F.1, and the span sentence in §F.4
   qualified as computed "over the observations as recorded" (F5).
4. Compression and transition work across all four sections (F10, F11), two stub paragraphs merged
   (F12).
5. Header provenance comment updated: records that `cosine_stats.py` could not run, what was fixed,
   and which single number (Alabama's +0.058 first-five-epoch mean) comes from the script's epoch-block
   section rather than from the parquet directly.

**`src/tables/frame/cosine.tex`**
6. Caption 152 → 128 words. Cut the clause re-explaining why fifty per-epoch values are one
   trajectory (§F.1 makes that case in full). Kept all three honesty devices: the unit justification,
   the observation-count disclaimer, and the dagger's meaning.

**`src_utils/_round7/cosine_stats.py`**
7. `EXPECTED` and the three structural assertions corrected from 3 states / 3,650 rows to 4 states /
   3,900 rows; the script now runs (RC=0) and its printed output is quoted above (F3). Docstring
   records the 65-row series and how to measure them.

---

## Build state

Measured after the final edit, each value read from the cell that produced it:

    make defense    RC=0   build/main.pdf            100 pp
    make academico  RC=0   build/main_academico.pdf   97 pp
    make ppgc       RC=0   build/main_ppgc.pdf       101 pp
    make extra      RC=0   build/main_extra.pdf       19 pp

    all four logs: tex_errors 0 | overfull hbox 0, vbox 0 | undefined references 0, citations 0
    check_trapped_prose.py: trapped-prose suspects: 0
    python3 src_utils/sync_page_counts.py --write: 7 claim(s) updated (defense 101->100,
      academico 98->97, ppgc 102->101 across CLAUDE.md, PLAN.md, src_utils/codex_reviewer.md)

**A note on `make check`, because the brief demands the exit code be read honestly.** At the *start*
of this session `make check` returned RC=2, before I had changed anything. The failing gate was
"recorded page counts vs the measured build", reporting `src/build/main_academico.log has no page
count -- the build did not finish`: the academico PDF was present but its log was stale. Building all
four modes and re-running gave RC=0, so the entry failure was a stale build artifact and not a defect
in the appendix. I record it because a reader of this report would otherwise see RC=2 in the history
and assume I introduced it.

---

## [VERIFY] flags and could-not-confirm

1. **[VERIFY]** The two 65-row Florida configurations (`T6_4_two_pass`, `T6_4_infonce_tau0_5`):
   whether their duplicated epochs 1–15 should be deduplicated, and on which rule. Disclosed in the
   prose; not resolved. The harvest that produced the duplicates is not in the parquet, so no rule is
   derivable from the committed data.
2. **[VERIFY / NEEDS SIGN-OFF]** The figure's plotting code is not in the repository.
   `git log -S 'fig_gradient_cosine'` and a search of all tracked `.py` files find only
   `cosine_stats.py`, which draws nothing. Every number in the figure was verified against the parquet,
   but the image itself has no committed derivation. Consequence: F13 (rebuilding panel (a) as four
   per-dataset outlines) cannot be done without writing a new plotting script.
3. **[NEEDS SIGN-OFF]** A pointer to Appendix F from Ch.6 §6.2's gradient paragraph (STEP 1). Frame
   prose, author's own, one clause — but it is a cross-reference addition to a chapter outside this
   track's scope.
4. **Could not confirm:** that the $\pm 0.05$ margin was fixed before the data were seen (F7). The
   appendix argues it from the quantity's scale, which is a good argument but not a registration, and
   no artifact in the repo records a pre-commitment.
5. **Could not confirm:** the pre-existing `[VERIFY]` block's job-level claims about California's
   folds 2 and 3 (byte-identical, unattributable). I did not go to the GPU host; the appendix's
   existing caution stands unchanged and nothing in my edits depends on it.
6. **Not run:** persona 13 (UFV compliance) and persona 17 (excellence assessor). 13 measures margins,
   numbering and front matter, none of which this pass touches; 17 requires the complete document as
   its unit and would duplicate the STEP 1 arc reading at a much larger scope.
7. **Resolved after the commit:** `make check` ran to completion at commit 2731ff15 and returned
   **RC=0**, read from the exit status of a direct `make check` with no pipe. Its page-count gate
   printed "measured from the build logs: defense 100 pp, academico 97 pp, ppgc 101 pp / all recorded
   page counts agree with the build", and its verification-commands gate reported "16 documented
   command(s) executed; 7 carried a machine-checkable expectation; 0 failed." The self-configuring
   Makefile needed no `texenv.sh` sourcing and the python preflight raised nothing.

## Post-commit state

    commit 2731ff15   review(apx-f): the caption described the pre-Georgia figure, and three of its
                      statements were false
    make defense RC=0 (100 pp) · make extra RC=0 (19 pp) · make check RC=0

The commit contains only this track's seven files. The working tree carries the author's concurrent
edits to other chapters, `preamble.tex`, `content.tex` and the build files; none of those were staged
or touched.

### F17 · MINOR · `make check` has a race against the build it measures (found by accident, reported)

Running `make defense; make extra; make check` back to back returned **RC=2**, with the page-count
gate reporting `src/build/main.log has no page count -- the build did not finish`. The build had
finished: its own last line read `Output written on build/main.pdf (100 pages, 1539694 bytes)` and
`latexbuild main -> build/main.pdf pages=100 tex_errors=0`. Re-running `make check` alone, seconds
later and with nothing else changed, returned **RC=0** and printed "all recorded page counts agree
with the build".

The gate greps `Output written on \S+ \((\d+) pages` out of `build/main.log`
(`src_utils/sync_page_counts.py:80`). `latexbuild.sh` copies the log from `build/main-aux/` into
`build/`, so a `make check` launched immediately after a build can read the file mid-copy, find no
match, and conclude the build never finished. The failure mode is indistinguishable in its output
from a genuinely truncated build, which is what makes it worth recording: **its message names the
wrong cause, and the suggested remedy (`sync_page_counts.py --write`) would write nothing useful.**

Not fixed, because a fix belongs to the build/gate machinery rather than to this track, and because
the correct fix is not obvious from one observation (a `sync` in `latexbuild.sh`, an atomic
`os.replace` on the copy, or a retry in the gate). Verified with a clean sequential run separated by
`sleep 3`: **`make defense` RC=0 (100 pp), `make extra` RC=0 (19 pp), `make check` RC=0**, and
`main.log` reporting tex_errors 0, overfull 0/0, undefined 0/0. That is the state at HEAD.
**[VERIFY: the `make check` page-count race. Reproduce by running `make defense && make check` with
no pause; expect RC=2 and the "did not finish" message on a build that did finish.]**
