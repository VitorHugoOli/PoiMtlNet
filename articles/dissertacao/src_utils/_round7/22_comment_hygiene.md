# 22_comment_hygiene.md -- the comments: deduplicated, trimmed, extended, and the sign-off queue dated

**Track:** the author's words -- *"trim, delete, merge and summarize the comments; keep the important
ones and ADD new ones so we can transfer knowledge to future agents and devs; there are a lot of
comments that are not useful; and take care with the [NEEDS SIGN-OFF] ones -- keep or update the ones
that need attention so I can take care of them later."*
**Date:** 2026-07-29. **Base:** commit `0bfc9e5e`. **Commits:** `d9584fca`, `ce45c051`, `13b5e7b0`.

## 0 - The one-paragraph version

Round 6 measured the CHAPTERS' comments and recommended against compressing them, because 95 percent
of those lines carry a traceable fact. **That verdict stands and this round did not touch chapter
provenance.** The build files are a different problem, and it is not fact-free commentary: it is
DUPLICATION, plus three files describing a two-line file as "four lines". Five stories were told
across the tree, three copies of one count were wrong, and one of the duplicated build-shape copies
documented a `make final` command that silently produces the DEFENSE document. Each story now has one
canonical home and the rest hold pointers; the wrong counts are fixed; six new comments document traps
this round's refactors CREATE; all 46 sign-off markers now carry the round and date they were raised;
and a gate catches both mechanical classes, validated against the tree that carried them.

Nothing rendered changed. For all 23 `.tex` files touched, non-comment content is byte-identical to
`0bfc9e5e` -- with one exception that is not mine (`main.tex`'s `\FINALBUILD` -> `\ACADEMICOBUILD`,
from the concurrent rename track). Builds 108/105/109 pp, `tex_errors=0`, `make check` RC=0.

## 1 - Measurements, before and after

| what | before | after | how |
|---|---|---|---|
| `main_ppgc.tex` content lines | 2 | 2 | `grep -v '^[[:space:]]*%' src/main_ppgc.tex \| grep -cv '^[[:space:]]*$'` |
| files claiming that file has FOUR lines | 3 | 0 | `grep -rn 'four-line file\|four lines of content' src/ CLAUDE.md src_utils/README_SRC.md` |
| "three builds, one source" told in full | 5 files | 1 (canonical `src/main.tex`) | `python3 src_utils/check_comment_hygiene.py --verbose` |
| nested-`\if` hazard told in full | 4 files | 1 (`src/main.tex`, at the switch definitions) | same |
| halt-on-error vs nonstopmode told in full | 5 files | 1 (`src_utils/README_SRC.md`) | same |
| decorative pure-rule comment lines in `0_main.tex` | 10 | 2 | `grep -c '^[[:space:]]*%[[:space:]]*[-=]\{4,\}[[:space:]]*$' src/0_main.tex` -- run against `0bfc9e5e` and the current tree. **An earlier draft of this row said 12 -> 4, from a broader pattern that also matched `%%`-prefixed rules; the command in this cell returns 10 and 2, and the row now reports what the command returns.** 8 removed, and 8 is also the line-count delta below. |
| `0_main.tex` total lines | 468 (at `0bfc9e5e`) | 530 | `git show 0bfc9e5e:articles/dissertacao/src/0_main.tex \| wc -l` vs `wc -l < src/0_main.tex`. **NET GROWTH of 62 lines, not a reduction: 8 decorative lines were removed and 70 lines of knowledge-transfer comment added. Measured on comment lines alone: 200 -> 262, delta +62 (`git show 0bfc9e5e:articles/dissertacao/src/0_main.tex | grep -c '^[[:space:]]*%'` vs the same grep on the file).** An earlier draft of this row said 538 -> 530, which measured an intermediate working state against itself and read as a net trim. This track is not a net line reduction and was never going to be -- half the brief was to ADD comments. |
| sign-off markers in `src/` | 46 | 46 | `grep -rn 'NEEDS SIGN-OFF' src/ \| wc -l` |
| markers whose own line states a round | 5 | 46 | `grep -rn 'NEEDS SIGN-OFF' src/ \| grep -ciE 'round *-?[0-9]\|v1 assembly'` |
| markers whose own line lacks round AND date | 41 | 0 | the same sweep, inverted |
| markers deleted | -- | **0** | the count above is unchanged |
| defense / academico / ppgc pages | 108 / 105 / 109 | 108 / 105 / 109 | `grep -o '([0-9]* pages' src/build/main*.log` |
| `tex_errors` in all three logs | 0 | 0 | `grep -c '^! ' src/build/main*.log` |
| `make check` | RC=0, 18 gates | RC=0, 20 gates | `(cd src && make check); echo $?` |
| gates in the suite | 18 | 20 | one is mine, one is the concurrent timing track's |
| `main.tex` total lines | 74 | 143 | `wc -l`; the canonical build header, the nested-`\if` mechanism and the TEXMFVAR trap all live here now |
| `main_ppgc.tex` total lines | 19 | 25 | `wc -l`; the corrected count plus its measuring command plus the pointer |

**A skip named, per the brief.** The duplication counts above are measured over 8 LIVE files
(`src/main.tex`, `src/main_ppgc.tex`, `src/main_academico.tex`, `src/0_main.tex`, `src/Makefile`,
`src_utils/README_SRC.md`, `CLAUDE.md`, `PLAN.md`). **Frozen audit trails are deliberately NOT in
scope and were not edited:** `_round6/`, `_round7/`, `_review_v1/2/3/`, `_specialists_v1/2/`,
`_archive/`, `_gates/`, `PENDENCIAS.md`, `CODEX_AUDIT.md`, `CODEX_VS_PERSONAS.md`,
`codex_reviewer.md`, `storyline/archive/`. Those record what was true when written; editing them to
remove a duplicate would falsify the record. This is the same exclusion rule every rename in
`LATEX_UPGRADE.md` uses. `LATEX_UPGRADE.md` itself is left alone for the same reason: its
"four lines"/"two lines" passage is the FINDING that named the defect (F-5), not a copy of it.

## 2 - Step 1: the duplicated stories

`LATEX_UPGRADE.md` candidate 5, which the author endorsed in principle, with its one load-bearing
correction applied: **the canonical home is the `src/main.tex` header, not `README_SRC.md`.**
`README_SRC.md` is a sibling of `src/` and does not travel with an Overleaf paste, so a reader who
pasted the source would hold only pointers to a file they do not have. `main.tex`'s header is the one
copy guaranteed to be in front of whoever opens the source.

| story | before | after | canonical home |
|---|---|---|---|
| three builds, one source | `Makefile`:1, `main.tex`:5, `README_SRC.md`:63, `CLAUDE.md` §1, `LATEX_UPGRADE.md`:140 | `main.tex` header tells it and SAYS it is canonical; `Makefile`, `README_SRC.md`, `main_ppgc.tex`, `main_academico.tex` point at it | `src/main.tex` |
| the nested-`\if` scanning hazard | `main.tex`:16, `main.tex`:40, `README_SRC.md`:64, `LATEX_UPGRADE.md`:174 | described once, at the switch definitions, with the mechanism; header and README point there | `src/main.tex` |
| the usermode TeX tree / TEXMFVAR | `Makefile`:6, `README_SRC.md`:94/103/104, `build.sh`:24-34, `texenv.sh`:5-25 | `README_SRC.md` keeps the operational table; `main.tex` carries the SYMPTOM (a font map reporting as a missing font) because that is what a pasted-source reader hits | `src_utils/README_SRC.md` (+ the symptom in `main.tex`) |
| halt-on-error vs nonstopmode | `Makefile`:9, `README_SRC.md`:114/118/139, `LATEX_UPGRADE.md`:185/241, `build.sh` x11, `AGENT_HANDOFF.md` x8, `PENDENCIAS.md` x3 | `README_SRC.md` keeps the lesson; `Makefile`/`build.sh` carry the FLAGS without re-explaining them (a recipe containing a flag is not a telling) | `src_utils/README_SRC.md` |
| "main_ppgc is N lines of content" | 7 places, **3 of them wrong** ("four lines"): `Makefile`:35, `main.tex`:12, `main_ppgc.tex`:8 | all say TWO; `main_ppgc.tex` and `main_academico.tex` carry the measuring command next to the claim | the files themselves, gated |

**The count was measured, not inherited from any of the seven.** `src/main_ppgc.tex` has 19 lines, 16
of them comments, 1 blank, and **2 content lines**. The `Makefile`'s copy was corrected by the
concurrent build track in the same session; `main.tex` and `main_ppgc.tex` are mine.

### 2.1 - The trap inside the duplicate (F-1), fixed

`main.tex`'s header quoted this as the way to produce the deposit build, and claimed `make final` did
exactly it:

    pdflatex "\newif\ifdefensebuild\defensebuildfalse\input{main.tex}"

It never did, and the quoted command is worse than stale: the command-line `\newif\ifdefensebuild` is
undone when `main.tex` re-executes `\newif\ifdefensebuild`, the `\ifdefined` test then finds no switch
macro, and the run takes the DEFENSE branch. An operator following that comment uploads the
full-front-matter document to AcademicoPG. The paragraph now points at the Makefile recipe -- the one
place the command is written down -- and records the trap.

The same block inverted the nested-`\if` attribution: it said the scanning hazard was something
`\ifdefined` HAS, when `\ifdefined` is what AVOIDS it. That is an invitation to "simplify" the guard
back into the bug. The mechanism is now stated once, correctly, at the switch definitions.

## 3 - Step 2: the trim, and what was NOT removed

**Removed: 8 lines, all from `0_main.tex`, every one a pure rule of dashes.** Each bracketed a one- or
two-line label and carried nothing: the `% PRE-TEXTUAL`, `% TEXTUAL`, `% POST-TEXTUAL` banners (two
rule lines each) and the two around the `chapterpreface` label. The labels themselves stay -- they are
the navigation of a 530-line preamble, which is exactly what round 6 said not to remove.

**Kept: 29 other pure-rule lines across `src/**.tex`** (31 in the tree now, of which 2 are `0_main.tex`'s surviving pair; `find src -name '*.tex'` + `grep -c '^[[:space:]]*%[[:space:]]*[-=]\{4,\}[[:space:]]*$'`). Each delimits a provenance or ledger block longer
than two lines, where the rule is doing structural work: the `2_fundamentals.tex` citation/number
ledgers, the paper-chapter section maps, the cover metadata block. A classifier ran before every
deletion and no line carrying a path, `:line`, number, date, commit hash or finding id was touched.
`AGENT_GUARDRAILS` N3 makes provenance mandatory, so deleting a provenance comment would be a law
violation rather than a cleanup.

**Reduced rather than deleted: `README_SRC.md`'s build-shape section.** Its table and switch
explanation are replaced by a pointer plus the reason the pointer runs that direction. No fact left
the tree: everything the table said is in the canonical header, and what genuinely belongs outside the
Overleaf paste (TeX tree, `texenv.sh`, verification protocol, gate suite) stays here.

**Not attempted: moving provenance blocks out of the `.tex` files.** Round 6 offered this and declined
to recommend it without the author's explicit call, on the grounds that a provenance comment next to
its value is read by whoever edits that value and a provenance file one directory away is not. That
reasoning is sound and this round did not overrule it.

## 4 - Step 3: the knowledge-transfer comments added

Each names what breaks and how the breakage presents to whoever hits it, written for someone who has
never seen this tree. Six were requested; all six landed, plus one the refactor made necessary.

1. **`\checkandfixthelayout[fixed]` is load-bearing** -- `0_main.tex`, at the four layout lines.
   MEASURED on this preamble's own package stack, with only the option changed:
   `[fixed]` gives `\lowermargin` 56.9055pt = 2.0000 cm; the default `classic` gives 45.9046pt =
   1.6134 cm. The mechanism, from `memoir.cls:1039`: `classic` REPLACES the height with
   `floor(702.78308/17.99446) = 39` lines plus 12pt `\topskip` = 713.78394pt, which is 11.00pt TALLER
   than the margins asked for, and a taller block under a fixed top margin can only grow downward. It
   presents as nothing at all: no warning, no overfull box, every gate green, and a bottom margin
   about 4 mm short of the UFV specification on every page.
2. **Three isolated aux trees** -- `0_main.tex`, at the `\include` list, because that is where the
   per-chapter aux files come from. Observed this session on the SHARED tree: `make final` launched
   while another session was building died with `! Extra }, or forgotten \endgroup` while reading
   `build/chapters/4_courb.aux`. That reads as a defect in chapter 4 and is a truncated aux. The
   comment says to read either that message or the `Runaway argument? ... \@writefile` form as
   evidence of a concurrent build BEFORE looking for a LaTeX defect, and notes that the `.aux` files
   are deliberately not copied back into `build/` (an aux on a path BibTeX searches is how four
   undefined citations once shipped).
3. **The precompiled preamble format dump** -- `0_main.tex` header, where a preamble editor will see
   it. A dump loaded after the preamble moved means the build succeeds and the PDF is stale, with
   nothing in the log naming the cause. The comment states that the staleness key makes that
   unreachable, that `fastbuild.sh` refuses rather than building from a stale key, that the plain
   targets never load it, and that if a `fast` build ever disagrees with a plain build, the plain
   build is the document.
4. **The `final` -> `academico` rename** -- `main.tex`. My note listed the five tools that hardcode
   the stem and derive nothing. The rename landed mid-session; the concurrent track rewrote the note
   into its completed form and kept the tool list, which is the correct outcome.
5. **The TEXMFVAR trap** -- `main.tex`, deliberately inside `src/` because `README_SRC.md` does not
   travel with an Overleaf paste. A missing font MAP reports as `Font ntx-Regular-tlf-ot1r ... not
   found`, i.e. as a missing font. The comment says not to go looking for the font and not to probe
   the variable (`kpsewhich -var-value TEXMFVAR` reports an unreadable path here).
6. **The longtable `\small` asymmetry** -- `tables/frame/bib_errata.tex`, the trap that stopped the
   build for six commits. In the two longtable files `\small` is set OUTSIDE the environment by an
   explicit brace pair; in the three float-container siblings it is set INSIDE. That looks like an
   inconsistency and is not one: `\begin{table}` is a float and opens a group of its own, so a font
   change inside it ends with the table, while `longtable` opens no group and a `\small` placed inside
   would leak into the rest of the document. Because the group is a bare brace pair spanning the whole
   table rather than a named environment, dropping one half is invisible to a reader and fatal.
7. **The counterpart warning**, added to `tables/courb/errata.tex` -- because a reader tidying up
   starts at whichever file they open first, and the warning only in the longtable file protects only
   one direction. It says both forms are correct, that they are not interchangeable, and points at the
   full reasoning.

## 5 - Step 4: the sign-off inventory

**46 markers, 19 files, 0 deleted.** Before this round 5 stated a round on the marker line and 41 did
not, so the queue could not be triaged: nothing on the line said whether a marker was raised during
v1 assembly or the previous day. Each now carries the round and date it was RAISED.

**How each date was established, and one defect in my own method.** `git blame` against HEAD, then a
correction: `0_main.tex` is modified in the working tree, so blaming a WORKING-tree line number
reported four wrong dates. Positions were re-derived by matching marker text in `HEAD` (two identical
`author caveats` lines disambiguated by order), which moved four dates. Then 11 markers blamed to
`4e84cf7a` -- the mechanical per-section chapter split -- carry that commit's date rather than their
own; each was traced past it with `git log -S` on its marker text. Final: **35 dated by blame, 11
traced past the split, 0 left at the split date.** Round windows come from this repository's own
commit subjects (07-23/24 assembly phases and rounds 2-3, 07-26 round 4, 07-27 round 5, 07-28 round 6,
07-29 round 7).

**Liveness.** All 46 still point at a live question. Two had a subject that later work moved, and per
the brief they are FLAGGED, not deleted:

- **`apx_a_contributions.tex`** said *"AGENT_GUARDRAILS C4 still mandates the containment device and
  needs the matching update (not my file)"*. **That is no longer true:** C4 was amended 2026-07-27 by
  author decision and the edit landed -- the rule now prohibits any BRACIS material and explicitly
  voids the disclosure half. The stale sentence is replaced by a note saying so. The sign-off itself
  stays open on its own terms (it is about the appendix's prose).
- **`apx_c_ai_disclosure.tex`** asked three things, one of which round 6 already answered: no model
  version string is derivable from the provenance trail, which closes the `COD-013` row for that file.
  Narrowed to the two questions still needing him, with the third recorded as answered and the note
  that he can still overrule it.

Anchor by the PHRASE, not the line number -- a third of the previous round's coordinates went stale
within one commit, and the numbers below are true only as of `13b5e7b0`.

| # | file : line | anchor phrase (grep for this, not the line number) | raised | what he must decide | live? |
|--:|---|---|---|---|---|
| 1 | `0_main.tex`:274 | `[NEEDS SIGN-OFF: Resumo CUT, round 6, 2026-07-28` | round 6, 2026-07-28 | The pair was cut on the author's \| ruling that both blocks were too long. MEASURED on the rendered pages of the 105-page defense build, on one stated convention (catalog header and keyword block stripped; h... | yes |
| 2 | `0_main.tex`:331 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| author caveats 2.1/3-4-5,` | round 6, 2026-07-28 | author caveats 2.1/3-4-5, 2026-07-24, carried through the cut | yes |
| 3 | `0_main.tex`:374 | `[NEEDS SIGN-OFF: Abstract CUT, round 6, 2026-07-28` | round 6, 2026-07-28 | Cut as the claim-parity pair of the \| Resumo above (WRITING_LAW §6): same claims, same numbers, same hedges, sentence for sentence, and cut in the same pass so the two cannot drift. MEASURED on the rendered... | yes |
| 4 | `0_main.tex`:409 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| author caveats 2.1/3-4-5,` | round 6, 2026-07-28 | author caveats 2.1/3-4-5, 2026-07-24, carried through the cut | yes |
| 5 | `chapters/1_introduction.tex`:79 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| gate L3 fix A-1. Orig` | v1 assembly, 2026-07-23 | gate L3 fix A-1. Original sentence was near-verbatim with Ch.5 ("one artifact to train, version, and deploy, and one forward pass whose single set of inputs produces both answers at once"); reworded here so ... | yes |
| 6 | `chapters/1_introduction.tex`:249 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-014, 2026-07-26` | round 4, 2026-07-26 | "twenty repetitions per configuration \| (four seeds, five folds)" -> the 20 fits kept, the inferential unit (n = 4) named, plus the fixed-partition clause. Sources: GLOSSARY rows "paired superiority test" +... | yes |
| 7 | `chapters/2_fundamentals.tex`:41 | `[NEEDS SIGN-OFF: raised round 3, 2026-07-24 \| gate fix B-5, 2026-07-24` | round 3, 2026-07-24 | The §2.1 "93% is the ceiling any model | yes |
| 8 | `chapters/2_fundamentals.tex`:177 | `[NEEDS SIGN-OFF: NUM-4, round6, 2026-07-28` | round 6, 2026-07-28 | Number RE-ANCHORED and its convention added. \| The clause previously read "the category F1 on Alabama, over five folds, rose monotonically from 0.74 to 0.82 across the swept values": two means with no sprea... | yes |
| 9 | `chapters/2_fundamentals.tex`:267 | `[NEEDS SIGN-OFF: COD-015, round6, 2026-07-28` | round 6, 2026-07-28 | New frame prose: the Check2HGI loss, which the \| document did not carry anywhere. The author approved adding it. WHY HERE AND NOT IN Ch.5, decided on the science: this paragraph is the point where the disse... | yes |
| 10 | `chapters/2_fundamentals.tex`:341 | `[NEEDS SIGN-OFF: COD-013, round6, 2026-07-28` | round 6, 2026-07-28 | New frame prose. The author approved naming the \| joint model's descent from MTLnet; it was carried only by the lineage table's row order. Placed HERE and not in Ch.5 on the science: the fact is about the R... | yes |
| 11 | `chapters/2_fundamentals.tex`:719 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| gate L3 fix A-2. The ` | v1 assembly, 2026-07-23 | gate L3 fix A-2. The "weekday lunch / Saturday night out" image now appears only in Ch.1 (the signed-off mechanism beat); this hinge sentence was reworded to state the same static-vector limit without duplic... | yes |
| 12 | `chapters/3_cbic/method.tex`:171 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR -- this is a claim` | round 6, 2026-07-28 | AUTHOR -- this is a claim change in published co-authored prose, and it REMOVES a stated advantage of the architecture this chapter adopts, so it runs against the chapter's own interest, the same property Ap... | yes |
| 13 | `chapters/3_cbic/results.tex`:111 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR -- four sentences ` | round 6, 2026-07-28 | AUTHOR -- four sentences of new protocol detail added to a published chapter. Every fact is recovered from the released code and named above; nothing about a tuning budget is asserted, per your instruction. ... | yes |
| 14 | `chapters/4_courb.tex`:27 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| one added sentence in a P` | round 6, 2026-07-28 | one added sentence in a PUBLISHED chapter's preface. The preface is FRAME prose written for the dissertation, not translated article text, so it carries no errata cost -- but it is a public pointer to a stat... | yes |
| 15 | `chapters/4_courb/methodology.tex`:81 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR -- a claim narrowi` | round 6, 2026-07-28 | AUTHOR -- a claim narrowing in published co-authored prose. The alternative device this collection already uses, reproducing the sentence and externalizing the correction in a footnote (3_cbic.tex, the Nash ... | yes |
| 16 | `chapters/4_courb/results.tex`:42 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR -- one new sentenc` | round 6, 2026-07-28 | AUTHOR -- one new sentence of protocol detail in a published chapter, recovered from the released code. No number and no claim about a result is affected. | yes |
| 17 | `chapters/5_mobiwac.tex`:20 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| new-to-chapter frame ` | v1 assembly, 2026-07-23 | new-to-chapter frame text (the time-capsule preface mandated by NORTH_STAR section 3/section 4 Ch.5); claims drawn from the approved spine and the paper's claim whitelist (PAPER_PLAN section 3). | yes |
| 18 | `chapters/5_mobiwac/02_related.tex`:14 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| new-to-chapter recap ` | v1 assembly, 2026-07-23 | new-to-chapter recap subsection (the bridging device mandated by NORTH_STAR section 3: Ch.5 recaps BOTH the Ch.3 artifact and the Ch.4 finding). Content from the approved spine (NORTH_STAR sections 2 and 6) ... | yes |
| 19 | `chapters/5_mobiwac/02_related.tex`:203 | `[NEEDS SIGN-OFF: raised round 5, 2026-07-27 \| AUTHOR -- "nineteen", "+0` | round 5, 2026-07-27 | AUTHOR -- "nineteen", "+0.68" and "+0.19" are new numbers in the chapter. | yes |
| 20 | `chapters/5_mobiwac/06_results.tex`:46 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| restored element. Thi` | v1 assembly, 2026-07-23 | restored element. This figure was cut from the 8-page MobiWac build (author decision 2026-07-09); its four numbers are fully stated in the prose above. Restored here because the dissertation is not page-limi... | yes |
| 21 | `chapters/5_mobiwac/06_results.tex`:219 | `[NEEDS SIGN-OFF: raised round 5, 2026-07-27 \| AUTHOR` | round 5, 2026-07-27 | ground. Value unchanged; only the inference is narrowed. \| AUTHOR \| [round5, persona 03 S3-02] "cell" for a table result is verdict *never* in articles/[mobiwac]/GLOSSARY.md §3 ("this audience reads 'cell'... | yes |
| 22 | `chapters/5_mobiwac/06_results.tex`:242 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| AUTHOR. Two things only y` | round 4, 2026-07-26 | AUTHOR. Two things only you can settle. (1) Ch.5 is PUBLISHED-adjacent text under the errata regime; this rewrites an interpretation sentence, so it needs an Appendix B row, which I have NOT written pending ... | yes |
| 23 | `chapters/5_mobiwac/07_discussion.tex`:58 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR. You approved soft` | round 6, 2026-07-28 | AUTHOR. You approved softening the attribution and adding the scope clauses. Two things to confirm. First, the sentence is now three sentences rather than one, which is a bigger change to the section opener ... | yes |
| 24 | `chapters/5_mobiwac/07_discussion.tex`:95 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR -- a limitations s` | round 6, 2026-07-28 | AUTHOR -- a limitations sentence in the chapter reproduced from the submitted paper. The change makes the paragraph slightly less favorable to the study, which is its point. | yes |
| 25 | `chapters/6_conclusion.tex`:23 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-015, 2026-07-26` | round 4, 2026-07-26 | The unqualified "outperforms both dedicated | yes |
| 26 | `chapters/6_conclusion.tex`:73 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| two added sentences in th` | round 6, 2026-07-28 | two added sentences in the author's own frame prose. No number changed. | yes |
| 27 | `chapters/6_conclusion.tex`:74 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-009 + idiom law, 2026` | round 4, 2026-07-26 | Two changes in one sentence pair. | yes |
| 28 | `chapters/6_conclusion.tex`:128 | `[NEEDS SIGN-OFF: raised round 5, 2026-07-27 \| AUTHOR` | round 5, 2026-07-27 | rests on the freeze control and the capacity-matched control. \| AUTHOR | yes |
| 29 | `chapters/6_conclusion.tex`:144 | `[NEEDS SIGN-OFF: raised round 3, 2026-07-24 \| gate fix B-2, 2026-07-24` | round 3, 2026-07-24 | 64.54 -> 64.51 to match Ch.5 Table 3, the | yes |
| 30 | `chapters/6_conclusion.tex`:171 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-013, 2026-07-26` | round 4, 2026-07-26 | The interim sentence ("A partial California run, | yes |
| 31 | `chapters/6_conclusion.tex`:252 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-013, 2026-07-26` | round 4, 2026-07-26 | "closes the parameter-count explanation" | yes |
| 32 | `chapters/apx_a_contributions.tex`:15 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| whole appendix -- new` | v1 assembly, 2026-07-23 | is still open on its own terms -- it is about this appendix's prose, not about C4. Counts trace to ../../src_utils/etl_tooling_contribution_evidence.md + ../../_archive/handoffs/handoff_tooling.json. ETL not... | yes -- narrowed |
| 33 | `chapters/apx_a_contributions.tex`:20 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-023, 2026-07-26` | round 4, 2026-07-26 | "the three published studies" -> "the three | yes |
| 34 | `chapters/apx_a_contributions.tex`:35 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-006 consistency, 2026` | round 4, 2026-07-26 | "supported every experiment in the | yes |
| 35 | `chapters/apx_a_contributions.tex`:55 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| REV-006, 2026-07-26` | round 4, 2026-07-26 | The former sentence ("The same codebase | yes |
| 36 | `chapters/apx_a_contributions.tex`:99 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| whole section -- new fram` | round 6, 2026-07-28 | whole section -- new frame prose. The author approved adding reproducibility content, following the pattern of Appendix D, where each number names the script that produced it and the output file it lives in.... | yes |
| 37 | `chapters/apx_a_contributions.tex`:137 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| "supplied on request" is ` | round 6, 2026-07-28 | "supplied on request" is a weaker commitment than the previous sentence implied, and a banca may ask why. If the files are published before the defense, revert this paragraph to the stronger claim and delete... | yes |
| 38 | `chapters/apx_b_errata.tex`:92 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23 \| whole appendix -- new` | v1 assembly, 2026-07-23 | whole appendix -- new frame prose around quoted ledger content. | yes |
| 39 | `chapters/apx_b_errata.tex`:147 | `[NEEDS SIGN-OFF: raised round 4, 2026-07-26 \| AUTHOR. You previously ru` | round 4, 2026-07-26 | AUTHOR. You previously ruled this REPORTED, NOT CORRECTED. I have NOT changed that ruling -- the published sentence stands. What changed is that the appendix now names the preservation instead of staying sil... | yes |
| 40 | `chapters/apx_b_errata.tex`:182 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| AUTHOR -- new appendix pr` | round 6, 2026-07-28 | AUTHOR -- new appendix prose describing additions to a published chapter. | yes |
| 41 | `chapters/apx_b_errata.tex`:266 | `[NEEDS SIGN-OFF: raised round 5, 2026-07-27 \| AUTHOR -- the errata poli` | round 5, 2026-07-27 | AUTHOR -- the errata policy (NORTH_STAR section 5.7) covers content departures; this is the first purely typographical one to get its own paragraph. | yes |
| 42 | `chapters/apx_b_errata.tex`:420 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| whole section -- new fram` | round 6, 2026-07-28 | whole section -- new frame prose making a public statement about a published co-authored result. The author has the co-author's agreement; the ADVISOR conversation is still pending, which is why it is one co... | yes |
| 43 | `chapters/apx_b_static_scope.tex`:59 | `[NEEDS SIGN-OFF: raised round 6, 2026-07-28 \| this paragraph now says s` | round 6, 2026-07-28 | this paragraph now says something WEAKER about Chapter 3 than the author's ruling assumed. He ruled the static-task problem is CoUrb's alone; the measurement says Ch.3's channel is indirect rather than absen... | yes |
| 44 | `chapters/apx_c_ai_disclosure.tex`:11 | `[NEEDS SIGN-OFF: raised v1 assembly, 2026-07-23, narrowed round 7 2026-0` | v1 assembly, 2026-07-23 | whole appendix -- the author must confirm scope and tool naming. STILL LIVE and still his call. The THIRD question this marker used to ask, "whether to name specific model versions he can source", is ANSWERE... | yes -- narrowed |
| 45 | `chapters/apx_d_ceiling.tex`:10 | `[NEEDS SIGN-OFF: raised round 5, 2026-07-27 \| whole appendix -- new fra` | round 5, 2026-07-27 | whole appendix -- new frame prose. Alternative offered to the author: fold it into Ch.5 as one paragraph and drop the chapter, which would need the Ch.5 cross-reference (5_mobiwac.tex:376) removed. | yes |
| 46 | `chapters/apx_e_ethics.tex`:6 | `[NEEDS SIGN-OFF: raised round 5, 2026-07-27 \| AUTHOR -- whole appendix;` | round 5, 2026-07-27 | Licenses: src_utils/DATASET_LICENSING_FINDINGS.md (Figshare, Hugging Face and GitHub records re-opened 2026-07-27). Pipeline facts: src/etl/gowalla/main.py:22-25, src/etl/massive_steps/stage_1.py:84-90, src/... | yes |
| | **46 markers, 19 files** | | **v1 assembly 8 · round 3 2 · round 4 10 · round 5 6 · round 6 20** | | **46 live · 0 deleted** |

## 6 - Step 5: the gate

`src_utils/check_comment_hygiene.py`, wired into `check.sh` between the doubled-macro and
trapped-prose gates. Two classes, both mechanical, neither visible to any existing gate -- every copy
of a duplicated comment is legal LaTeX, and a comment that is merely WRONG raises nothing.

- **CLASS A**: a canonical story told in full in more than one file. Each story declares one canonical
  home and a `tell` pattern matching the SUBSTANTIVE explanation, so a pointer does not read as a
  duplicate; otherwise the gate would fight its own fix.
- **CLASS B**: a self-describing line count that disagrees with the file it describes.

**Validated in both directions against the real historical tree,** reconstructed from `0bfc9e5e` with
`git show`:

    CLASS B pre-fix: 3 findings  (main_ppgc.tex:8, main.tex:12, Makefile:35)   fixed tree: 0
    CLASS A pre-fix: 1 finding   (README_SRC.md as a second full telling)      fixed tree: 0

Five self-tests build synthetic defective and fixed trees and run BEFORE any verdict, so a broken
checker reports itself broken rather than reporting a clean tree. On the current tree: 3 stories each
told once, 9 count claims examined across 8 files, 0 skipped, RC=0, 0.09 s.

### 6.1 - Three defects in my own instrument, and how each was caught

Recorded here and at the code, because §4b V3 says to interrogate the instrument rather than trust its
silence, and because each is a live trap for whoever extends this gate.

1. **The docstring asserted a validation that had not run.** I wrote "2 violations -- `main.tex` and
   `main_ppgc.tex`" from what I expected the check to do. The first real run found `main_ppgc.tex:8`
   and `Makefile:35` and **missed `main.tex:12`**: at `0bfc9e5e` that copy names its subject on
   line 10 and states the count on line 12, TWO lines below, while the original window reached one
   line FORWARD and none backward. The file the note credited as detected was
   the one copy the checker could not see -- root cause R1/R2 of `AGENT_GUARDRAILS` §4b, committed
   inside the gate written to prevent it. The window is now the surrounding comment paragraph and all
   three copies are found; the docstring records the error rather than hiding it.
2. **A clean result measured on nothing.** `git worktree` failed with "Operation not permitted" and
   left an EMPTY directory, against which the checker reported 0 findings. That is the shape of every
   false pass in this repository's history. The reconstruction now asserts the tree exists before
   reporting.
3. **A story that could never fire.** The halt-on-error story's first `tell` pattern matched NOTHING,
   including its own canonical home, so it passed vacuously -- the "gate that has never fired" of
   `AGENT_GUARDRAILS` §7. A story with zero tellers is now a FINDING. The pattern was re-anchored on
   the text `README_SRC.md` actually contains.

Two false positives on the real tree, both narrowed and both recorded at the code: `1.5 line spacing`
(the UFV manual's spacing) parsed as a count of five lines, and `0_main.tex`'s own prose was measured
against `main_ppgc.tex` by an over-broad self-description fallback.

## 7 - Coordination with the concurrent tracks

Two other tracks were editing `src/Makefile`, `src/main.tex`, `src/main_ppgc.tex` and `src/0_main.tex`
while this ran. Their work was kept and improved, never reverted:

- The **build track** (format dump + per-target aux trees) had already corrected the `Makefile`'s
  "four-line" claim and documented the aux hazard there. I did not touch the `Makefile`, and wrote the
  aux-tree comment where a CHAPTER author meets it -- the `\include` list in `0_main.tex` -- rather
  than duplicating theirs. `check.sh` is left uncommitted with their timing harness in it; my gate
  registration rides along in their working copy for them to land.
- The **`academico` rename track** landed mid-session. It adopted the canonical-header convention this
  track established, rewrote my "pending rename" note into its completed form, and kept the tool list.
  `main_academico.tex` arrived carrying the same "TWO lines of content" self-description, so it was
  added to the gate's counted files and its scope: 9 count claims are now checked instead of 5.
- `README_SRC.md` still said "TWO entry files" and `make final` after the rename; the entry-file count
  and the shim paragraph are updated. The `make final` / `main_final.pdf` strings in the build-command
  block are left alone deliberately -- `sync_page_counts.py`'s `CLAIMS` regexes still anchor on them,
  and that file is theirs.

## 7.1 - Two things my commit boundaries do NOT say cleanly

Disclosed because the commit messages imply a tidier split than happened, and a reader auditing by
commit would be misled.

- **`d9584fca` is titled as the sign-off commit but also carries part of the `0_main.tex` comment
  work** (9 of the added lines match the `[fixed]` / aux-tree / format-dump blocks). `0_main.tex` was
  staged whole, and the two pieces of work were in it at the same time. The content is right and both
  pieces are described in the round's commits; the attribution by commit is not clean.
  `git show d9584fca -- articles/dissertacao/src/0_main.tex` shows exactly what landed there.
- **`ce45c051` carries one NON-comment line that is not mine:** `main.tex`'s
  `\ifdefined\FINALBUILD` -> `\ifdefined\ACADEMICOBUILD`, from the concurrent rename track, which
  was in the working copy when I committed the file. Committing the file whole PRESERVES their work
  rather than reverting it, which is the required behaviour, but it means this commit is not
  comment-only despite its subject line. It is the single exception behind the "non-comment content
  byte-identical" claim throughout this report.

## 8 - [VERIFY] flags and what could not be confirmed

- **[VERIFY: the bottom-margin figure is derived from `\lowermargin`, not from the rendered page box.]**
  The 2.0000 cm / 1.6134 cm figures come from `\the\lowermargin` in a minimal document on this
  package stack, converted at 28.45275 pt/cm. That is the length TeX uses to lay out the page, and it
  reconciles exactly with `memoir.cls:1039`'s arithmetic, but it is not a measurement of ink on the
  rendered page. If the author wants the margin certified against the PDF page box, that is a separate
  measurement.
- **[VERIFY: round windows are inferred from commit subjects, not from a recorded schedule.]** The
  date-to-round mapping comes from this repository's own commit subjects. Where a marker was raised on
  a day that carried more than one round token, the later round is used. Any marker whose stamp
  matters to a decision should be checked against its commit hash, which is in the inventory's origin
  data.
- **Could not confirm: the two-line shim claim for `main_academico.tex` at the moment it was created.**
  The file arrived from the concurrent track during this session. Its claim is correct NOW (measured:
  2 content lines) and the gate holds it, but I did not observe it before their edits settled.
- **RESOLVED (was a [VERIFY] flag): `check.sh` is committed and carries my gate.** The registration
  was in the working copy of a file the concurrent timing track owns; they have since committed it
  (`5e6250d5`), and `git show HEAD:articles/dissertacao/src_utils/check.sh | grep -c
  check_comment_hygiene` returns 1. `make check` RC=0 across 20 gates on the committed tree. The
  checker itself is at `13b5e7b0` and also runs standalone.
- **Not measured: whether any of the 46 sign-off subjects is one the author has already settled
  verbally.** Liveness here means the marker's subject still exists in the tree and no later note
  supersedes it. Only he knows which he has already decided.

## 8.1 - Three of my own comments were wrong, found by review after the first commits

Landed as `01e1fbbc`. Each is a comment that compiles fine, breaks nothing, and misleads whoever
believes it -- the exact class this round's gate exists for, which makes getting them wrong inside
that gate's own provenance the sharper lesson.

1. **`README_SRC.md` documented a target that no longer produces the file it names.** The build
   block said `make final` -> `build/main_final.pdf`. After the rename that target forwards to
   `academico` and writes `build/main_academico.pdf`; the recipe sent the author looking for a file
   that is not produced. Fixed, with the forwarding behaviour stated, plus two further stale
   references in the same file (the verification recipe, and a `build/main_final.log` path in the
   aux-tree paragraph). **I had left these deliberately, on the stated grounds that
   `sync_page_counts.py` anchors on those strings. It does not** --
   `grep -n README_SRC src_utils/sync_page_counts.py` returns nothing. The text stayed wrong
   because the justification for leaving it was never checked. The `PENDENCIAS.md` and `CLAUDE.md`
   anchors that regexes DO use are untouched and remain theirs.
2. **The checker blamed its own false positive on the wrong guard.** The `0_main.tex` case came
   from the FORWARD half of the window -- the header's `1.5 line spacing` line is immediately
   followed by one that genuinely names `main_ppgc.tex` -- so the fix that mattered was the decimal
   exclusion in `COUNT_CLAIM`. The `rel in counted` restriction is an independent narrowing and did
   not clear this case.
3. **The post-mortem in §6.1 above, and in the checker's docstring, had the window backwards.**
   It said `main.tex:12` escaped because the window "reached only one line back" and that its
   subject sat "three lines" away. Measured at `0bfc9e5e`: the subject is on line 10 and the count
   on line 12, so TWO lines above, and the original window reached one line FORWARD and none
   backward. Both now stated from the file rather than from memory.

Revalidated after the fixes, both directions on the reconstructed pre-fix tree: CLASS B 3 -> 0,
CLASS A 1 -> 0, gate RC=0, `make final` still builds by forwarding.

## 9 - How to re-verify all of it

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao
source src_utils/texenv.sh

# the two gated classes, both directions, with the self-tests
python3 src_utils/check_comment_hygiene.py --verbose      # EXPECT: RC=0, 3 stories, 9 count claims

# the count that was wrong in three places
grep -v '^[[:space:]]*%' src/main_ppgc.tex | grep -cv '^[[:space:]]*$'    # EXPECT: 2
grep -rn 'four-line file\|four lines of content' src/ CLAUDE.md src_utils/README_SRC.md  # EXPECT: nothing

# the sign-off queue
grep -rn 'NEEDS SIGN-OFF' src/ | wc -l                                   # EXPECT: 46
grep -rn 'NEEDS SIGN-OFF' src/ | grep -ciE 'round *-?[0-9]|v1 assembly'   # EXPECT: 46

# nothing rendered changed (comments only)
for f in $(git diff --name-only 0bfc9e5e..HEAD -- 'articles/dissertacao/src/**/*.tex' | sed 's|articles/dissertacao/||'); do
  diff <(git show "0bfc9e5e:articles/dissertacao/$f" | grep -v '^[[:space:]]*%') \
       <(grep -v '^[[:space:]]*%' "$f") >/dev/null || echo "NON-COMMENT CHANGED: $f"
done   # EXPECT: only src/main.tex (the concurrent \FINALBUILD -> \ACADEMICOBUILD rename)

# the build and the suite
(cd src && make defense && make academico && make ppgc && make check)     # EXPECT: RC=0
grep -o '([0-9]* pages' src/build/main.log src/build/main_academico.log src/build/main_ppgc.log
                                                                          # EXPECT: 108 / 105 / 109
```
