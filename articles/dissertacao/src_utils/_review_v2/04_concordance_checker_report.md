# 04 · Concordance checker — cross-chapter consistency

**Build audited:** `src/dissertacao.pdf` (94 pp, written 2026-07-25 23:43:53) + `src/chapters/*.tex`
as of the same timestamp. **Date:** 2026-07-26. **Persona:** `reviewers/04_concordance_checker.md`.
Read-only. Fresh eyes: I drafted none of this text.

## Verdict

**COHERENT, with two seams that need work.**

The correction round did the thing a correction round usually breaks, and mostly did it right: the
inferential-unit rewording (`n = 4` per-seed means over 20 fitted models) propagated to all three
sites that carry it, in the same words, with the same hedge. The claim-scope qualification of the
Conclusão Geral now matches Chapter 5 clause for clause. What it did **not** do is propagate the
fixed-partition caveat into the chapter that owns the protocol, and it left one wording
substitution in Appendix B that no longer quotes the chapter it describes.

No BLOCKER at concordance level. Two MAJOR, three MODERATE, four MINOR.

## Top 3 findings

1. **F-01 (MAJOR)** — the fixed-partition consequence is stated in Chapter 1 but absent from
   Chapter 5, the chapter that runs the protocol.
2. **F-02 (MAJOR)** — Appendix B Table 12 quotes a "Chapter wording" replacement that does not
   appear in Chapter 3 in that form.
3. **F-03 (MODERATE)** — the freeze-control comparand basis is reconciled in Chapter 5 but the
   Conclusão Geral repeats the control without carrying its single-seed footing.

---

## Ranked findings

### F-01 · MAJOR · Fixed-partition caveat lives in the frame, not in the protocol chapter

Chapter 1 carries the caveat, added this round:

> "All / four seeds reuse the same fold partition, so the reported intervals do not / cover
> uncertainty over resampled user splits."
> — `src/chapters/1_introduction.tex:245-247`

Chapter 5's own statistics subsection states the pairing and the n, but never the consequence:

> "On every dataset, both models use four seeds ($4\times5=20$ measurements) and the tests pair the
> per-seed means ($n{=}4$), with a Holm correction~\cite{holm1979} across the six next-category
> comparisons and, separately, across the four next-region ones."
> — `src/chapters/5_mobiwac.tex:394`

I searched Chapter 5 for every phrasing of the consequence (`do not capture`, `split-to-split`,
`another partition`, `different partition`, `reuse the same fold`): zero hits in prose. A reader who
reads Chapter 5 on its own — the normal way an examiner reads a results chapter — gets the intervals
without the scope statement that Chapter 1 promises. The frame is currently more careful than the
chapter it summarizes, which is backwards.

*Direction:* the caveat belongs where the intervals are reported (Section 5.5.3 or at the interval
list in 5.6.2), with Chapter 1 continuing to echo it. This is a concordance finding, not a
statistics ruling — persona 09 owns whether the caveat is correctly worded.

### F-02 · MAJOR · Appendix B Table 12 quotes a replacement that is not the chapter's text

Appendix B Table 12 records this substitution:

> "``both our MTL and Single models significantly outperform HMRM across all POI / categories'' &
> ``outperform HMRM in every POI category'' (the comparison is stated by scope instead of / by
> significance)."
> — `src/chapters/apx_b_errata.tex:213-216`

Chapter 3 actually reads:

> "As shown in Table~\ref{tab:cbic:category}, both our MTL and Single models outperform HMRM
> \cite{chen2020modeling} in every POI category in terms of F1-score, precision, and recall."
> — `src/chapters/3_cbic.tex:302`

The literal string `outperform HMRM in every POI category` does not occur in `3_cbic.tex` (grepped;
zero hits). The substitution is *substantively* right — `significantly` is gone, the scope
qualification is there — but the errata table presents its right-hand column as the chapter's
wording, and it is a paraphrase of it. Appendix B is the document's own audit trail; an examiner who
spot-checks one row and finds the quote does not match will discount the whole table.

*Direction:* either quote the chapter verbatim in the right-hand cell, or mark the column as a
summary rather than a quotation. Author's call which.

### F-03 · MODERATE · The freeze control travels to Chapter 6 without its footing

Chapter 5 states the control and its measurement basis in the same breath, correctly:

> "The control predates the results of / Table~\ref{tab:mobiwac:results} and was measured at one
> random initialization / over five folds, so its second comparison is to the joint scores of the /
> development configuration current at the time ($63.56$, $63.39$, $79.82$), which / it matched to
> within $0.3$, and not to the joint cells reported here."
> — `src/chapters/5_mobiwac.tex:664-668`

Chapter 6 repeats the control with the dataset scope but drops the single-seed footing:

> "First, the freeze control / reported in Chapter~\ref{ch:mobiwac}: with the region pathway frozen,
> the category gain survives, at / the three datasets where the control was run (Alabama, Arizona,
> Florida), so the gain / does not come from the region task teaching the category task"
> — `src/chapters/6_conclusion.tex:92-95`

The scope ("the three datasets where the control was run") is present and is an improvement. The
footing (one random initialization over five folds, against the n = 20 of everything else in the
chapter) is not. Chapter 6 is where a reader in a hurry reads the mechanism claim.

*Direction:* one clause carrying the footing, matching Chapter 5's wording.

### F-04 · MODERATE · Table 9 and Table 10 report two different "single-task" columns and only Chapter 5 knows it

Table 9 (`5_mobiwac.tex:443-454`) prints a "Check-in level" column: Istanbul 54.65, AL 55.87,
AZ 57.13, FL 75.15, TX 69.95, CA 70.26. Table 10 (`5_mobiwac.tex:568-580`) prints a "Dedicated"
column: 54.74, 56.82, 56.43, 74.51, 69.79, 70.60. These differ by −0.70 to +0.95 and are *supposed*
to: Table 9's is one fixed configuration, Table 10's is the per-dataset-tuned n = 20 ceiling. The
distinction is stated:

> "The check-in-level column keeps one fixed configuration, not the per-dataset-tuned dedicated
> model of Table~\ref{tab:mobiwac:results}."
> — `src/chapters/5_mobiwac.tex:411`

and the freeze-control deltas correctly point at Table 9 by name (`5_mobiwac.tex:662`), which I
verified reconciles exactly: 63.50 − 55.87 = +7.63, 63.67 − 57.13 = +6.54, 79.79 − 75.15 = +4.64.
The seam is that no frame chapter mentions the two conventions exist. Chapter 2's evaluation section
(`2_fundamentals.tex:479-481`) names "the dedicated single-task model" as *the* operative ceiling,
singular.

*Direction:* Chapter 2's ceiling sentence could acknowledge that Chapter 5 reports two comparison
arms under different tuning budgets, or Chapter 5's Table 9 lead-in could be pointed at from the
frame. Low urgency; the chapter itself is internally sound.

### F-05 · MODERATE · Table 9's own column heading does not name its convention

Table 9's caption reads "same single-task model, training configuration, and folds (mean and fold
sd, seed 0)" (`5_mobiwac.tex:438-440`) and the column is headed `\textbf{Check-in level}`
(`:445`). The convention is in the caption; the disambiguating sentence is 27 lines earlier at
`:411`. Given F-04, a column heading that carried the convention would remove the whole class of
confusion.

*Direction:* heading or caption gains "one fixed configuration". Presentation detail; flagged here
because it is the mechanism by which the F-04 seam becomes a misreading.

### F-06 · MINOR · "everywhere" appears twice, both times legally scoped

`2_fundamentals.tex:618-620` and `6_conclusion.tex:88`. Both are immediately followed by the region
scope ("and on the next region at four of six datasets", "outperforms or matches them on the region
task"), so neither is a bare universal. WRITING_LAW §3 bans bare "everywhere"; these are not bare.
Recorded so a later editor does not "fix" a compliant sentence.

### F-07 · MINOR · Duplication sweep: three shared passages, all sanctioned

12-gram sweep across all chapters plus the abstract. Results:

| Pair | shared 12-grams | verdict |
|---|---|---|
| `1_introduction.tex` ↔ abstract (`0_main.tex`) | 30 | **sanctioned** — the abstract is a compression of the intro funnel by design |
| `1_introduction.tex` ↔ `6_conclusion.tex` | 17 | **sanctioned** — the research question and the protocol boilerplate |
| `4_courb.tex` ↔ `apx_b_errata.tex` | 5 | **sanctioned** — Appendix B quotes the declared addition it documents |
| `5_mobiwac.tex` ↔ `6_conclusion.tex` | 1 | **sanctioned** — one gradient-cosine clause |

The intro/conclusion runs are: "monolithic 64 dimensional place embedding with decomposed spatial
temporal and categorical encoders"; "whether multi task learning helps point of interest prediction
for the next category and next region tasks and"; "under user disjoint cross validation with twenty
fitted models per configuration four seeds over one fixed set of five folds". The third is exactly
the protocol sentence this round rewrote, and it is *identical* in both places, which is the
correct outcome. No padding found; the frame does not repeat the papers.

### F-08 · MINOR · Cross-reference lint: clean

98 labels defined, 64 referenced, **0 dangling** `\ref` targets. I resolved every section-level
pointer to its target and checked the target is the right one (not merely that it compiles — the
Viegas defect class). All 40 section/chapter pointers land correctly; spot-verified in the rendered
PDF that in-text "Section 5.6.2", "Section 5.5.3", "Section 5.5.4", "Table 8/9/10" resolve to the
right numbered objects. 34 labels are defined but never referenced (mostly equation and appendix
labels) — harmless.

**One build-level exception, which belongs to persona 05 and 18 and is repeated there:** the
citation key `russwurm2024geographiclocationencodingspherical` does not resolve, rendering as
`(??)` on pages 21, 45, 49, 50 of the defense build. That is a cross-reference failure in the
`\cite` half of L4.

### F-09 · MINOR · Time-capsule prefaces present and bidirectional

All three article chapters carry a preface. Checked the forward/back pointer pairs:

- Ch.3 preface (`3_cbic.tex:14-31`) points forward: "Chapters~\ref{ch:courb} / and~\ref{ch:mobiwac}
  revise that verdict"; and, corrected this round, "The chapter's preference for the Nash-MTL
  optimizer is likewise / a conclusion of the time ... and Chapter~\ref{ch:mobiwac} does not rely on
  it" (`:25-27`). I verified the correction is warranted: `4_courb.tex:115` does train with Nash-MTL
  ("Multi-task training uses the Nash-MTL regularizer"), so the previous "the following chapters do
  not rely on it" was false and the narrowed pointer to Ch.5 is correct.
- Ch.4 preface (`4_courb.tex:12-15`) and Ch.5 preface (`5_mobiwac.tex:16-`) present.
- Appendix B points back at all three (`apx_b_errata.tex`, sections per article).

Time-capsule integrity holds. Ch.3's terminology bridge ("the term ``Next-POI Prediction'' as used
in the reproduced article denotes the frame's \emph{next category} task", `3_cbic.tex:27-31`) is the
mapping sentence GLOSSARY §1 mandates, and it is where the law places it.

---

## Cross-reference lint table

| Check | Result |
|---|---|
| `\ref` targets defined | 64/64 resolve |
| `\ref` targets *correct* (not just compiling) | 40/40 section+chapter pointers verified against the rendered TOC |
| `\cite` keys in bib | 98/98 present in `references.bib` |
| `\cite` keys in the rendered reference list | **97/98** — `russwurm2024...spherical` missing (see F-08) |
| Figure/table numbers in prose vs floats | all match |
| Orphan bib entries | 1 (`liu2014geographical`, uncited) |

## Numbers that appear in more than one chapter

Handed to persona 06 for value-correctness; my finding is agreement or disagreement only.

| Fact | Sites | Agree? |
|---|---|---|
| Category gain range | `0_main.tex:290` "5.3 to 9.4"; `6_conclusion.tex:75` "5.3 to 9.4"; `5_mobiwac.tex:612` "+5.33 to +9.35" | **yes** — frame rounds, chapter is exact, both name macro-F1 |
| AL dedicated ceiling 56.82 | `5_mobiwac.tex:574`, `6_conclusion.tex:101` | **yes** |
| AL joint 64.51 | `5_mobiwac.tex:574`, `6_conclusion.tex:102` | **yes** — and the round's 64.54→64.51 fix is correct: Table 10 reports joint-best, and Ch.6 now matches it |
| CA narrow ceiling 70.60 | `5_mobiwac.tex:578`, `6_conclusion.tex:112` | **yes** |
| Inferential unit | `0_main.tex:285-287`, `1_introduction.tex:243-244`, `6_conclusion.tex:72-73` | **yes** — all three say twenty fitted models, four seeds, one fixed set of five folds, tests on the four per-seed means |
| Region scope | `0_main.tex:292-294`, `1_introduction.tex:131-133`, `2_fundamentals.tex:619-621`, `5_mobiwac.tex:620-622`, `6_conclusion.tex:21-22, 74-77` | **yes** — four of six, TOST two-point margin at the other two, at all five sites; Arizona never upgraded |

## Promises vs delivery

- Abstract ↔ Resumo structural parity: **holds.** Same eleven moves in the same order, same numbers,
  same hedges, same joint-best convention named in both (`0_main.tex:284-296` / `:260-296` PT). Claim
  and value halves handed to personas 07 and 06.
- Objectives ↔ chapters: 1:1, verified against `1_introduction.tex` §1.3 and the chapter map.
- Conclusão Geral claims nothing the body did not establish, and *does* claim the
  dissertation-level synthesis ("The representation, together with the / sharing topology built on
  it, is what the answer depends on", `6_conclusion.tex:89-90`) which no single chapter claims. This
  is exactly the coletânea requirement and it is met.
- Chapter 2's hinge paragraph (`2_fundamentals.tex:610-622`) pre-motivates Chapters 3/4/5 in order,
  one clause each. Intact.

## Tightest seams (do not touch)

1. **The Ch.3 → Ch.4 → Ch.5 arc statement** in `6_conclusion.tex:14-22`. Three chapters, one
   sentence, each with its own verdict and scope, no drift. This is the paragraph that makes the
   coletânea read as one investigation.
2. **The task-name bridge** in the Ch.3 preface (`3_cbic.tex:27-31`). It disarms the single most
   likely examiner confusion (the published paper's "Next-POI Prediction" is the frame's *next
   category*) before the reader can trip on it.
3. **The Ch.4 split-axis and seed declaration** (`4_courb.tex:238`). It states a protocol weakness
   in the reproduced work, names the chapter that fixes it, and does so without editorializing.
   Concordance-wise this is the best seam in the document.
4. **Table 9's coincidence footnote** (`5_mobiwac.tex:457-458`): "The matching place-level value at
   Alabama and Istanbul ($26.56$) is a coincidence of / two independent runs; their per-fold values
   differ." Pre-empts a reader's "did you copy a cell?" — exactly the register this document should
   keep.

## Open questions for the author

1. F-02: should Appendix B Table 12's right-hand column be a verbatim quote or an acknowledged
   summary? It is currently presented as the former and reads as the latter.
2. F-01: is the fixed-partition caveat's absence from Chapter 5 deliberate (because Chapter 5 is a
   reproduced submitted paper and the caveat is a frame-level addition)? If so, Appendix B's
   MobiWac section should record it as a declared departure, since it is currently not in Table B.4.

## Out-of-scope handoffs

- Persona 05: the unresolved `russwurm...` cite key and the `(??)` renders.
- Persona 06: whether "5.3 to 9.4" is the correct rounding of the exact +5.33/+9.35 range.
- Persona 09: whether the fixed-partition caveat, as worded in Chapter 1, is statistically adequate.
- Persona 18: page 71 renders as floats only.
