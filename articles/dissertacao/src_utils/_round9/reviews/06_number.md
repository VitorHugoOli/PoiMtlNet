# Round 9 — Persona 06, number auditor (Gate G2, rules N1–N5)

**Persona:** `reviewers/06_number_auditor.md` — the numeral-extraction gate, fresh eyes
(AGENT_GUARDRAILS §4b L6: I wrote none of the text under audit).
**Build commit:** `03b53d16`
**Volumes read:** `src/build/main.pdf`, 102 pp (defense) — the scoped volume. I did **not** open
`main_extra.pdf` (20 pp), `main_academico.pdf` (99 pp) or `main_ppgc.pdf` (103 pp); see UNFINISHED.
**Date:** 2026-07-30
**Verdict:** **GATE FAIL** — one blocker (an orphan derived ratio, N2/N3).

## Commands run (working directory `articles/dissertacao` unless stated)

No build, no `make check`, no `make selftest`, no write outside this file.

```bash
git log --oneline -1                      # -> 03b53d16
git show --stat beebd33b ; git show --stat c4d84604
git show c4d84604 -- src/chapters/apx_f_cosine.tex        # the scope-paragraph diff
git show beebd33b -- src/chapters/2_fundamentals.tex | grep '^+' | grep -v '^[[:space:]]*%' \
  | grep -oE '[0-9][0-9.,]*' | sort | uniq -c | sort -rn   # numerals added by the Pareto commit
# page-indexed text of the defense build (pypdfium2, 102 pages) written to /tmp/na/main.pdf.txt,
# then every numeral token extracted and each repeated fact grepped back across all 102 pages
cat src_utils/_round7/gradient_cosine_tests6.csv
cat src_utils/_round7/gradient_cosine_figure_facts6.json
cat src_utils/_round7/gradient_cosine_tests6_README.md
cat src/tables/mobiwac/datasets.tex
sed -n '544,552p' ../../research/embeddings/hgi/README.md
sed -n '1,120p'  ../../docs/studies/closing_data/RESULTS_BOARD.md
sed -n '1,60p'   ../../docs/studies/closing_data/joint_best/JOINT_BEST_RESULTS.md
grep -n -E "0\.72|0\.66|69\.88|101\.9|56\.16" \
  ../../docs/results/closing_data/capacity_matched_stl_cat/README.md
grep -n -i -m20 -E "food|third" \
  ../../docs/studies/closing_data/archive/provenance/CATEGORY_DISTRIBUTION.md
# ratio-ledger sweep, whole tree, for the two derived factors of Appendix F:
cd /Users/vitor/Desktop/mestrado/ingred && \
  grep -rn -E "thirty-six|factor of thirty|of sixteen in" --include=*.md --include=*.tex \
  --include=*.py --include=*.json .        # -> ONE hit, the prose itself
```

Instrument note (V3): the page-indexed extraction was validated against a known-present string
before use — `4,089,892` returns p.66 (inside Table 8) and p.101 (Appendix F prose), which is the
pair the audit turns on, so the extractor sees both table cells and body text. Source greps that
touch `.tex` are read with the comment lines visible and quoted separately, because this tree's
provenance comments quote the very values being searched (V4); every WHAT below states whether the
quoted line is prose or a comment.

---

## Findings

### BLOCKER — the two size-span factors in Appendix F are derived numbers with no source

**WHERE:** `src/chapters/apx_f_cosine.tex:317-319`; rendered on **PDF p. 101** (defense build).

**WHAT (prose, verbatim from the PDF):**

> "the check-in counts run from 113,846 at Alabama to 4,089,892 at Texas and the region counts from
> 520 at Istanbul to 8,501 at California, so this axis spans **a factor of thirty-six in volume and
> one of sixteen** in the size of the region label set"

**WHY:** AGENT_GUARDRAILS §2 **N2** ("Agents quote; they do not compute. No mental arithmetic,
rounding, aggregation, percentage conversion, or delta-taking in prose. Derived quantities come from
a script committed to the repo … then are quoted") and **N3** ("Any number an agent writes must be
traceable to its source file in the handoff note"). The persona protocol makes the consequence
explicit: "Orphan numbers = BLOCKER."

The four *inputs* are clean: 113,846 / 4,089,892 / 520 / 8,501 all appear in
`src/tables/mobiwac/datasets.tex` and are quoted into this file's own comment block at
`apx_f_cosine.tex:362-363`. The two *ratios* are not. A tree-wide sweep for `thirty-six`,
`factor of thirty` and `of sixteen in` (command above, run from `ingred/`) returns exactly one hit:
line 318 itself. There is no ledger line, no script, and no `_round9/30_cosine_six.md` entry
carrying either factor — and `30_cosine_six.md` is otherwise a complete ledger for this commit
(it carries the seven-row table, the md5 gate, the per-dataset job stamps).

This is also new text: `git show c4d84604` shows the sentence replaced an earlier one that made a
qualitative claim only ("spans an order of magnitude in volume"), so the two factors entered at
`c4d84604` without a derivation.

**FIX:** author's call between two clean options. (a) Restore a qualitative span — "spans more than
an order of magnitude in volume and more than an order of magnitude in the size of the region label
set" — which needs no arithmetic and is readable off the four quoted endpoints. (b) Keep the factors
and add the ledger: a one-line committed derivation (or a line in `30_cosine_six.md`: value → the
two `datasets.tex` cells → the quotient) so the numbers are quoted rather than computed in prose.
Note that under (b) the same rule applies to the *rounding direction*: both factors as written round
in the direction that flatters the span, and N4 requires a declared rounding.

---

### SHOULD-FIX — Chapter 5 and Appendix F name different datasets as the volume maximum

**WHERE:** `src/chapters/5_mobiwac/05_setup.tex:22`, rendered **PDF p. 65**, against
`src/chapters/apx_f_cosine.tex:317`, rendered **PDF p. 101**. Table 8 itself is on **PDF p. 66**.

**WHAT (both prose, verbatim from the PDF):**

p. 65: "The states range from about 114,000 check-ins and 1,109 regions in Alabama to about **3.2
million check-ins** and 8,501 regions in **California**."

p. 101: "the check-in counts run from 113,846 at Alabama to **4,089,892 at Texas**".

Table 8 (p. 66, from `src/tables/mobiwac/datasets.tex`): `TX … 4,089,892` and `CA … 3,171,380`.

**WHY:** AGENT_GUARDRAILS §2 **N4** cross-check class "the same fact quoted twice must match to the
digit"; persona 06 procedure step 4, "derived quantities (deltas, ranges, 'at least X') recomputed
FROM THE TABLE (min/max over the right cells)". The p. 65 sentence is a range statement whose upper
endpoint is not the maximum of its own table: Texas carries 4,089,892 check-ins against California's
3,171,380. I take the intent to be a range ordered on *region* count (with the check-in figure
co-quoted at each endpoint), which is defensible in isolation — but Appendix F now states the
check-in maximum explicitly and attributes it to Texas, so a reader who reads both pages is handed
two different volume ceilings for the same six datasets, twenty-five pages apart.

**FIX:** scope the p. 65 clause to the axis it actually orders on, e.g. "ordered by region count,
the datasets run from 1,109 regions and about 114,000 check-ins at Alabama to 8,501 regions and
about 3.2 million check-ins at California; Texas carries the largest corpus at about 4.1 million
check-ins (Table 8)". Wording is the author's call; the requirement is that one of the two pages
names Texas as the volume maximum.

---

### SHOULD-FIX — the two scale floors on p. 72 point at a section that contains neither

**WHERE:** `src/chapters/5_mobiwac/06_results.tex:76-78`, rendered **PDF p. 72**. The pointer
resolves to `sec:mobiwac:setup-metrics` = §5.5.3, `src/chapters/5_mobiwac/05_setup.tex:127-132`.

**WHAT (prose, verbatim from the PDF p. 72):**

> "For scale: always predicting the most common category reaches only **about 7 percent macro-F1**
> (the majority-class floor), and a random region top-ten guess is right **at most about two
> percent** of the time (Section 5.5.3)."

§5.5.3, verbatim from source (`05_setup.tex:129-132`), in full: it defines "its reference point is
the majority-class floor, the macro-F1 of always predicting the most common category" and, for
region, "its reference point is the dedicated model." **No value appears there, and no random-guess
baseline appears there at all.**

**WHY:** **N3** (every numeral traceable) and **N4** (captions/pointers vs content); WRITING_LAW §3
requires every number to carry its reference point *and its convention*. Two defects compound here.
First, the pointer sends the reader to a section that carries neither figure — an L4 cross-reference
defect in the number-bearing direction. Second, the values themselves are unscoped aggregates over
six datasets that disagree: the majority-class macro-F1 per dataset in
`docs/studies/closing_data/archive/provenance/CATEGORY_DISTRIBUTION.md:113-118` is quoted there as
**AL 7.28%, AZ 7.25%, FL 5.66%, TX 6.76%, CA 7.04%, Istanbul 7.15%** — so "about 7 percent" is
right for five datasets and 1.3 points off at Florida, which is the dataset whose joint-vs-dedicated
category gain is the *smallest* (+5.33, p. 73). I found no source at all for the "two percent"
region figure; the only quantity it can plausibly come from is a ten-of-520 ratio at Istanbul, which
would be prose arithmetic and therefore an N2 violation on top of the missing ledger line.

**FIX:** two changes, both mechanical. (1) Repoint to a section that carries the values, or add the
values to §5.5.3 and keep the pointer. (2) Give each floor its scope, quoted not computed:
"reaches 5.7 to 7.3 percent macro-F1 across the six datasets (majority-class floor;
`CATEGORY_DISTRIBUTION.md`)". For the region figure, either add a ledger line naming the source and
the dataset it is computed at, or cut the clause — the sentence survives without it, since the
dedicated model is the operative reference point for region (§5.5.3's own words).

---

### SHOULD-FIX — an open `[VERIFY]` on a metric convention ships inside the built volume

**WHERE:** `src/chapters/2_fundamentals.tex:173-174` (prose) with the flag at
`2_fundamentals.tex:188-190` (comment); rendered **PDF p. 19**.

**WHAT (prose, verbatim from the PDF p. 19):**

> "the category F1 rose monotonically across them, from 0.7388±0.0205 at the published setting to
> 0.8186±0.0123 at the adopted one, on a zero-to-one scale, with the spread taken across the five
> folds."

The flag, verbatim from the source comment: `% [VERIFY: averaging convention of the swept "Cat F1"]
Every source records "Cat F1" without naming macro or weighted averaging, so the sentence says
"category F1" and not "macro-F1"; author to confirm the convention, or drop the two values and keep
the clause qualitative.`

**WHY:** **N5** ("Every reported cell states its convention (metric, selection rule, n, seeds ×
folds)"). The values themselves trace cleanly — `research/embeddings/hgi/README.md:544` gives the
header "5 folds × 50 epochs" and `:548`/`:551` give `0.7388 ± 0.0205` and `0.8186 ± 0.0123` verbatim,
which I read this session — so this is not a source mismatch. What is missing is the averaging
convention, and the effect on the reader is concrete: this is the only category-quality figure in the
document on a zero-to-one scale, twenty-five pages before macro-F1 out of 100 is established as the
category metric, so a reader cannot tell whether 0.8186 is comparable to the 56.82 / 64.51 of
Table 10.

**FIX:** the author's call, and his own flag already frames it correctly — confirm the convention and
name it, or drop the two values and keep the clause qualitative ("the category F1 rose monotonically
across the four settings"). Flagging it here because an open `[VERIFY]` inside a *built* volume at a
gate commit is a gate item, not a drafting note.

---

## All-clear list (what I verified, grouped)

**Chapter 2 §2.3, the ~106 lines added at `beebd33b`.** Numeral extraction over the added prose
(command above, comments stripped) returns **no data numerals at all** — only citation keys' years
and the `K` / `w_k` of Equation 2.4. The Pareto material is definitional throughout: dominance,
Pareto optimality, the front, Pareto-stationarity, and the per-method guarantee levels (Nash-MTL
subsequence + added convexity; CAGrad fixed points; Aligned-MTL with pre-set weights; PCGrad making
no Pareto claim) carry no measured quantity. **Nothing to trace and nothing orphaned.** The
`% NUMBERS: none quoted` provenance line in the section's own comment block is accurate.

**Appendix F, the cosine table and prose, against `gradient_cosine_tests6.csv` +
`gradient_cosine_figure_facts6.json` (read this session).** Every cell of Table 11 (PDF p. 100)
matches its CSV row exactly or as a disclosed 4-dp rounding: Florida `n=60`, 3,150 obs,
`+0.0003` / CI `[−0.0010, +0.0015]` / TOST `10⁻⁶²` / `0.68 / 0.90` / 31-of-60; Alabama `+0.0112`,
CI `[+0.0040, +0.0184]`, `0.013 / 0.063†`, 5/5; Arizona `+0.0015`, 3/5; California `+0.0007`,
`0.048 / 0.38`, 4/5; Texas `−0.0003`, 4/5; Istanbul `+0.0001`, 3/5; Georgia `+0.0039`,
`0.009 / 0.063†`, 5/5. Prose figures likewise: 4,650 observations, seven datasets, Florida 3,150
across twelve configurations, 250 each elsewhere, 92.4 percent inside the margin (`92.37` in the
JSON, declared rounding), range `−0.34 to +0.58` (`[-0.3407, 0.5802]`), configuration-mean span
`[−0.00261, +0.00457]`, the `n=12` reading at `+0.0003` with TOST `1.3 × 10⁻¹⁶` (`1.28e-16` in
`cosine_stats6.py:33`), the `0.0625` sign-test floor at `n=5`, Alabama `+0.0112` / Georgia `+0.0039`
fold means, and the Texas caption's "one fold at −0.0032" (`−0.00322` in the README). The
observation counts reconcile without arithmetic in prose: the appendix states the 12 × 5 × 50
structure and names the partial re-run that carries Florida to 3,150, which
`cosine_stats6.py:60-61` records as 65 rows in 10 of its 60 series.

**The seven-vs-six dataset discipline.** The single highest-risk item at `c4d84604`, and it holds
everywhere I checked: Table 11's caption, Figure 8's caption, and §F.1 all say "the six datasets of
Chapter 5 and Georgia" or equivalent, Georgia carries its `‡` in the figure caption, and the closing
sentence reads "all six of the dissertation's datasets plus Georgia". The stale strings the commit
message claims to have removed are genuinely absent from the appendix's pages: `3,900`,
`"four datasets"`, `91.3`, `"all four cases"`, `"three of the dissertation's six"` return zero hits
on pp. 97-102. `"four datasets"` does survive on **p. 76**, in Chapter 5, about the geographic
shortlist on Alabama/Arizona/Florida/Istanbul — a different measurement, correctly scoped there, and
not a stale cosine string.

**Never-cite sweep (reviewers/README §Sources; C3, absolute).** Zero hits in the 102-page volume for
`75.87` (the superseded reg-VOID bf16 TX category cell), `5.22` (the fp16 California collapse),
`2.37` (TX bf16). `2.41` appears once, on **p. 57**, inside Chapter 4's CoUrb results table — a
CoUrb F1 cell, not the voided TX region delta, and outside the never-cite list's scope.

**Joint-best vs diagnostic-best (N5, the distinction that must never blur).** Clean. Table 10
(p. 72) carries the joint-best lane throughout — Istanbul 63.32 / 75.35, AL 64.51 / 69.70, AZ
65.79 / 59.46, FL 79.84 / 77.41, TX 77.24 / 67.06, CA 77.05 / 65.69 — every cell matching the
**joint-best (deploy)** column of `joint_best/JOINT_BEST_RESULTS.md`, which I read this session. The
diag-best comparands (64.54, 65.84, 79.85, 75.44, 69.80, 59.56, 77.42) return **zero hits** in the
volume; the one apparent exception, `69.80` on p. 56, is a CoUrb F1 cell in Chapter 4. The
convention is named where the numbers are used: the table caption's selection rule, the p. 72
sentence "the joint model at the epoch selected by its joint validation score (the geometric mean of
the two task metrics)" with the ≤0.06 / ≤0.11 sensitivity to the alternative rule, and Appendix A's
"the joint-best convention: both tasks are read at the same checkpoint rather than each at its own
best epoch". Dedicated ceilings 54.74 / 56.82 / 56.43 / 74.51 / 69.79 / 70.60 match the
`CEILINGS_N20_FINAL`-derived column of the same file, i.e. the n=20 best-vs-best set, not the frozen
v16 ceilings the board warns against mixing.

**Abstract ↔ body ↔ Chapter 6 parity on the headline.** The three loci agree to the digit and each
carries its convention. Abstract (p. 3): "twenty fitted models per configuration, four random
initializations over five fixed folds, and paired tests on the four initialization means … at all
six, by 5.3 to 9.4 macro-F1 points **under a joint-best selection**"; Chapter 6 (p. 79): same range,
same n structure, "four of six, Istanbul, Florida, California, and Texas … Alabama and Arizona";
Chapter 2 (p. 27) and Chapter 1 (p. 14) both give the four-of-six / other-two split with the
two-point TOST margin. The `5.3 to 9.4` rounding is declared in `content.tex:146-148` against the
per-dataset deltas (FL +5.34 low, AZ +9.40 high), and the un-rounded `+5.33 to +9.35` in
Chapter 5 (p. 73, `06_results.tex:135`) is the joint-best restatement of the same pair — the
`fig4_deltas_diss.py:42-44` values (Istanbul 8.58, AZ 9.35) match Figure 7's rendered labels.
The four-of-six count matches the four `↑` rows against the two `≈` rows of Table 10.

**Chapter 6's capacity-matched control (pp. 79-80), against
`docs/results/closing_data/capacity_matched_stl_cat/README.md` read this session.** `hidden_dim=752`
and `101.9%` (README:29), best arm mean `56.16` sd `1.89` at Alabama (README:39), `69.88` sd `0.26`
at California (README:47), the narrow-width comparands `56.82` and `70.60 ±0.07` (README:54-55), and
the shortfalls `−0.66` AL / `−0.72` CA (README:54-55, 65) all match, with n=20-per-arm and
three-arms/sixty-total stated in the prose. The `64.51` used as the joint comparand is the
joint-best value, consistent with Table 10 — and the source comment at `6_conclusion.tex:144-149`
records that as a deliberate N5 fix rather than a drift.

**The development-time cosine, scoped in both places it appears.** p. 63
(`5_mobiwac/02_related.tex:157-159`) and p. 80 (`6_conclusion.tex:202-205`) both give `+0.001` over
four seeds on four Gowalla states and both flag that Georgia is not one of the dissertation's six;
p. 98 states that the appendix "supersedes nothing there" and names why the two sets are not
interchangeable. Same quantity, same value, same hedge, three loci.

**Reference points at the frame level.** The 93-percent predictability bound appears twice (p. 17,
p. 25) and both instances carry the same scope limit ("specific to next-location prediction at
coarse spatial resolution" / "not … a ceiling on seven-class category macro-F1 or on region
ranking"), with §2.4 naming the majority-class floor, the first-order Markov floor, and the
dedicated single-task model as the operative ceiling.

## Could-not-verify (fail-closed)

1. **Whether the "roughly a third of the check-ins" Food share (p. 24, `2_fundamentals.tex:634-635`)
   is the intended reading of its source.** `CATEGORY_DISTRIBUTION.md:39-94` gives the Food share
   per dataset as 34.19 / 34.01 / 24.69 / 30.98 / 32.72 / 33.41 percent, and that file's own header
   (`:29`) says it is the "check-in/task-sample weighted `next_category` distribution" — i.e. a share
   of *next-visit target labels*, which is also what `datasets.tex`'s Majority column reports. The
   prose says "a third of the check-ins", not of the next-visit labels. "A representative state"
   is unnamed. I am not calling this a finding: the numbers are consistent with the sentence at four
   of six datasets and the hedge is honest, but I could not establish which quantity the sentence
   means, and no ledger line names one. **Missing:** a ledger line giving value → file → field, and
   a named state (or "in four of the six datasets").
2. **The region-side "two percent" of p. 72.** No source located anywhere; reported above as a
   should-fix rather than a blocker only because the sentence is explicitly framed as scale
   ("For scale:") and no verdict rests on it. **Missing:** the file and field it comes from.
3. **Chapters 3 and 4 tables against the published CBIC/CoUrb papers.** Not attempted; see
   UNFINISHED. Anything I report about pp. 40-43 and pp. 56-58 would be from the built PDF alone,
   which under N1 is not a source for those chapters.

## COUNTS

**blockers: 1 / should-fix: 3 / nits: 0**

Nits: none found. I looked for the usual class — rounding drift between a caption and its table,
digit disagreement between two statements of one fact, a statistic named as the wrong kind — and in
the material I covered the captions match their tables and the repeated facts match to the digit.
An empty section is a result, so it is not padded.

## UNFINISHED

The 25-minute checkpoint arrived with the following inside my scope and not reached. None of it is
"probably fine"; it is unaudited.

1. **Chapters 3 and 4 (PDF pp. 29-58), the CBIC and CoUrb result tables.** Roughly 40 percent of the
   volume's numeral tokens sit here (Tables 2-7). N1 routes them to the published papers' own tables
   plus the documented errata and `CoUrb_2026/slides/judge_feedback.md`, none of which I opened. The
   whole G2 gate for these two chapters is outstanding, including the cross-convention check the
   persona protocol calls for (Ch.3/Ch.4 conventions vs Ch.5's, each named in its own chapter).
2. **The Resumo (Portuguese) side of the abstract parity pair.** I verified the English abstract on
   p. 3 against the body. The Resumo is on p. 2, which I did not extract; `content.tex:141-145`
   asserts the two blocks correspond sentence for sentence, and per §4b V2 that assertion is not
   evidence. Claim-parity includes numbers, so this is a real gap.
3. **`main_extra.pdf` (20 pp, the supplementary volume).** Not opened. Appendix D of that volume
   holds the label-history benchmark values (the GLOSSARY records FL 0.3617, AL 0.2800, AZ 0.3232,
   CA 0.3242, IST 0.3016) and Appendix A of the defense build points readers at it, so it carries
   number-bearing content that should agree with the defense volume.
4. **Exhaustive numeral extraction.** 2,757 numeral tokens across 102 pages were extracted; I traced
   the frame chapters (1, 2, 6), Chapter 5's headline and reference-point numbers, and Appendix F
   exhaustively, plus targeted sweeps (never-cite, joint-best/diag-best, repeated-fact agreement)
   across all 102 pages. The remaining untraced mass is Chapters 3-4 (item 1) and the appendices
   B/D/E number content. The persona's own rule applies: "No sampling on gate day: the extraction is
   exhaustive or the gate did not run." For Chapters 3-4 it did not run.
5. **`main_academico.pdf` (99 pp) and `main_ppgc.pdf` (103 pp).** Not opened. The page-count deltas
   against the 102-page defense build imply different front/back matter, so a number that renders
   differently in one target would not have been caught here.
