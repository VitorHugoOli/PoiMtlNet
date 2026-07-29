# Round 7 · 27 — comment trim in `src/*.tex` and `src/chapters/*.tex`

**Scope.** The 19 LaTeX sources at `src/` top level and `src/chapters/` top level. NOT `src_utils/`
(round 7 item 20/22 already trimmed the build files), NOT the per-section subdirectories
(`src/chapters/3_cbic/`, `4_courb/`, `5_mobiwac/`), NOT `src/tables/`.

**Permission used.** Comments only. No live line was changed in any file; §"PDF identity" proves it
two ways. Three defects that need a live-line change are reported, not applied.

## 1 · Measurement, before and after

A comment line is a line whose first non-blank character is `%`; a block is a maximal run of them.
Measured by `src_utils/_round7/_trimwork/measure.py`. "After" is the trimmed tree with the author's
concurrent prose edits excluded (see §5), so every number below is attributable to this pass.

| file | lines | comment lines | pct | longest block | sign-off |
|---|---|---|---|---|---|
| `content.tex` | 339 -> 324 | 167 -> 152 ** | 49.3 -> 46.9 | 27 -> 27 | 4 |
| `main.tex` | 163 -> 138 | 152 -> 127 ** | 93.3 -> 92.0 | 89 -> 79 | 0 |
| `main_academico.tex` | 37 -> 34 | 34 -> 31 ** | 91.9 -> 91.2 | 34 -> 31 | 0 |
| `main_extra.tex` | 277 -> 277 | 117 -> 117 | 42.2 -> 42.2 | 41 -> 41 | 1 |
| `main_ppgc.tex` | 28 -> 27 | 25 -> 24 ** | 89.3 -> 88.9 | 25 -> 24 | 0 |
| `preamble.tex` | 278 -> 254 | 176 -> 152 ** | 63.3 -> 59.8 | 36 -> 29 | 0 |
| `chapters/1_introduction.tex` | 271 -> 281 | 54 -> 64 ** | 19.9 -> 22.8 | 25 -> 25 | 2 |
| `chapters/2_fundamentals.tex` | 766 -> 723 | 321 -> 278 ** | 41.9 -> 38.5 | 57 -> 50 | 5 |
| `chapters/3_cbic.tex` | 65 -> 62 | 36 -> 33 ** | 55.4 -> 53.2 | 19 -> 19 | 0 |
| `chapters/4_courb.tex` | 49 -> 45 | 37 -> 33 ** | 75.5 -> 73.3 | 17 -> 16 | 1 |
| `chapters/5_mobiwac.tex` | 56 -> 53 | 29 -> 26 ** | 51.8 -> 49.1 | 15 -> 15 | 1 |
| `chapters/6_conclusion.tex` | 287 -> 287 | 103 -> 103 | 35.9 -> 35.9 | 23 -> 23 | 7 |
| `chapters/apx_a_contributions.tex` | 251 -> 246 | 152 -> 147 ** | 60.6 -> 59.8 | 45 -> 45 | 6 |
| `chapters/apx_b_errata.tex` | 438 -> 440 | 221 -> 223 ** | 50.5 -> 50.7 | 92 -> 94 | 5 |
| `chapters/apx_b_static_scope.tex` | 72 -> 72 | 23 -> 23 | 31.9 -> 31.9 | 18 -> 18 | 1 |
| `chapters/apx_c_ai_disclosure.tex` | 92 -> 92 | 54 -> 54 | 58.7 -> 58.7 | 54 -> 54 | 1 |
| `chapters/apx_d_ceiling.tex` | 127 -> 127 | 27 -> 27 | 21.3 -> 21.3 | 12 -> 12 | 1 |
| `chapters/apx_e_ethics.tex` | 98 -> 98 | 11 -> 11 | 11.2 -> 11.2 | 6 -> 6 | 1 |
| `chapters/apx_f_cosine.tex` | 352 -> 352 | 154 -> 154 | 43.8 -> 43.8 | 60 -> 60 | 1 |
| **TOTAL (19 files)** | **4046 -> 3932** | **1893 -> 1779** | **46.8 -> 45.2** | | **37** |

`**` marks a file whose comment count changed. **114 comment lines removed; 114 total lines
removed; no prose line added or deleted.** Comment share of the 19 files: 46.8 percent -> 45.2.

Longest blocks before the pass: `apx_b_errata.tex` 92, `main.tex` 89, `apx_f_cosine.tex` 60,
`2_fundamentals.tex` 57, `apx_c_ai_disclosure.tex` 54.

## 2 · What was cut, and the rule for each class

The standard was applied file by file, not mechanically. Every cut falls in one of four classes.

**(a) The same story told in a second place — keep ONE at the site that would break, leave a
pointer.** `main.tex` carried the front-matter switch explanation twice, once as a bullet list and
once as prose; merged into the switch description. `content.tex`'s Resumo and Abstract cut blocks
each carried the full measurement convention and the claim-floor argument; the convention now
appears once at the Resumo block and the Abstract block points at it. `preamble.tex`'s split
rationale and inherited-skeleton provenance were two accounts of one decision; merged.
`main_academico.tex` and `main_ppgc.tex` each retold what belongs in a shared file; each now
states its own two-line count, its own build command, and points at `main.tex` for the build shape.
The three paper-chapter shims carried an identical six-line SECTIONS block; each is now three lines.
`2_fundamentals.tex` repeated "Draft 1 (2026-07-21). Obeys WRITING_LAW.md + GLOSSARY.md.
Citation/number ledger at the section end." once per section; stated once in the file header.

**(b) Historical process commentary that no longer changes behaviour.** `content.tex`'s
`\needspace` block spent nine lines on how an earlier diagnosis went wrong through an escaping bug;
the measurement that settles the choice is kept, the post-mortem of the bug is not. `main.tex`'s
page-counter block held two superseded corrections told in full; both corrected values and the
reason the counter goes wrong are kept in one statement of the standing hazard.

**(c) A twenty-line block that says what five lines would.** `2_fundamentals.tex`'s Check2HGI loss
provenance (52 lines) kept its source transcription, every code path, both scope caveats, its
[VERIFY] flag and its glossary resolution, at 33 lines. The MTLnet-descent block and the HGI sweep
block were tightened the same way: every path, number and spread retained.

**(d) Narration of what the next line plainly does.** Removed where found; it was rare in these
files, which is why the round 7 finding that "95 percent of comment lines carry a fact" was true.

**Kept in full, deliberately:** `apx_a_contributions.tex`'s source ledger (every line carries a
path or a quoted protocol statement), `apx_f_cosine.tex`'s header (measured facts, the
unit-of-independence warning, the claim deliberately not made), `apx_b_errata.tex`'s reconciliation
tables, `apx_c_ai_disclosure.tex`, `apx_d_ceiling.tex`, `apx_e_ethics.tex`, `6_conclusion.tex`.
Six of the 19 files were left untouched.

**Sign-off markers: 37 before, 37 after, none moved out of its file.**
```
cat src/*.tex src/chapters/*.tex | grep -c 'NEEDS SIGN-OFF'          -> 37   (was 37 at HEAD)
grep -ro 'NEEDS SIGN-OFF' --include='*.tex' src/ | grep -v '/build/' -> 49   (was 49 at HEAD)
```
All 37 in-scope markers carry a round or a date (`grep -ciE 'round *-?[0-9]|v1 assembly|20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]'` returns 37).

**On the brief's figure of 53.** No record in the repository claims 53. The bare grep
`grep -ro 'NEEDS SIGN-OFF' --include='*.tex' src/` returns exactly 53 — it includes
`src/build/fmt/_body.tex`, a generated untracked copy of the body region, which duplicates four
markers. `src_utils/_round7/22_comment_hygiene.md` §"sign-off markers in src/" documents that same
`src/build/` exclusion as load-bearing and recorded 46 at the time. The source counts are 49 across
`src/` and 37 within this pass's scope.

## 3 · PDF identity

**Instrument.** `_trimwork/fingerprint.py` extracts the text layer page by page and hashes each page
plus the whole document. `_trimwork/selftest_fingerprint.py` validates it in three directions
before it is trusted: a one-word change in a two-page document must report DIFFER localized to page
2, a byte-identical copy must report IDENTICAL, and a page-count change must report DIFFER. Run:
`SELFTEST PASS`.

**The comparison that proves the claim.** Two isolated trees were built from scratch in the
workspace, outside the repository, so nothing else running could touch their aux trees:

- **A** = the trimmed comments.
- **B** = the same 16 files at `HEAD` (`e771d331`), untouched.

A mechanical check confirmed the two trees are comment-only apart: stripping every `%` line from
all 54 sources under `src/` gives **0 differing live lines**. Both trees then built:

```
TREE A   main 101 pages, tex_errors=0     main_extra 19 pages, tex_errors=0
TREE B   main 101 pages, tex_errors=0     main_extra 19 pages, tex_errors=0

defense  A vs B  ->  IDENTICAL   101 pages, 255811 chars, doc sha c3a6920e1213b9f7, 0 differing pages
extra    A vs B  ->  IDENTICAL    19 pages,  43165 chars, doc sha 858cacb009499712, 0 differing pages
```

**Convergence, and why it matters here.** The first clean build of a tree is NOT the document.
Measured on both trees: after `rm -rf build`, `make defense` produces a PDF whose citation numbers
are wrong (`[1]` where the settled document has `[39]`) because the three-pass recipe runs bibtex
once, against an aux file that did not yet exist on pass 1. Build again and it settles; build a
third time and nothing moves:

```
tree B, 1st build after rm -rf build   101 pages, 255811 chars, doc c3a6920e1213b9f7
tree B, 2nd build                      101 pages, 255800 chars, doc a9d9d19f2faea763
tree B, 3rd build                      101 pages, 255800 chars, doc a9d9d19f2faea763   converged
tree A, 2nd build                      101 pages, 255800 chars, doc a9d9d19f2faea763
tree A, 3rd build                      101 pages, 255800 chars, doc a9d9d19f2faea763   converged
```

`a9d9d19f2faea763` is **the fingerprint taken before any edit of this pass**, at 11:41 from the
repository's own warm build tree. So the identity claim closes three ways: the trimmed tree equals
the untouched tree, both equal the pre-edit baseline, and consecutive builds of each are stable.

```
defense  A converged vs B converged  ->  IDENTICAL  101 pages, 255800 chars, 0 differing pages
defense  A converged vs 11:41 baseline -> IDENTICAL (same doc sha a9d9d19f2faea763)
extra    A vs B                      ->  IDENTICAL   19 pages,  43165 chars, 0 differing pages
```

**This also retracts an earlier reading of mine.** Partway through I compared the 11:41 baseline
against a single fresh build and recorded a 50-page difference in citation numbering as unexplained,
flagged `[VERIFY]`. That difference was the one-bibtex-pass artifact above, not a property of the
document, and the flag is withdrawn. The lesson is the one this round keeps relearning: a number
read from a build that has not converged is not a measurement of the document.

**The live working tree is at 100 pages, and that is the author's change, not this pass.**
Converged, the repository tree renders 100 pages / 253647 chars / doc `f1260ef1e1a0c7df`. It carries
the author's in-flight prose rewrite of `apx_f_cosine.tex` (+199/-149 lines) and
`tables/frame/cosine.tex` (+15/-10), which is net shorter and drops the last page. Tree A isolates
this pass by taking those two files from `HEAD`, and renders 101, matching the baseline exactly.
Nothing in `src/build/` was committed.

## 4 · Gates

```
python3 src_utils/check_comment_hygiene.py   rc=0   7 self-tests PASS; 10 files; 3 canonical
                                                   stories each told once; self-describing
                                                   counts agree with their files
python3 src_utils/check_trapped_prose.py     rc=0   0 suspects
python3 src_utils/check_torn_sentences.py    rc=0   0 suspects
python3 src_utils/check_tracker_refs.py      rc=0   every live PENDENCIAS citation resolves
```
Each rc was read from the command's own exit status, not through a pipe.

## 5 · Concurrency: what in the working tree is NOT mine

While this pass ran, the author committed `e771d331` (Makefile self-configuration, the title, the
supplementary-volume merge) and is **still editing two files**:

| file | live-line diff vs HEAD | mine? |
|---|---|---|
| `src/chapters/apx_f_cosine.tex` | 276 | **no** — prose rewrite in progress. My only change to this file is one comment repoint. |
| `src/tables/frame/cosine.tex` | 18 | **no** — never opened by this pass. |

Every other file I edited has **0** live-line diff:
```
for f in $(git diff --name-only -- articles/dissertacao/src/ | grep '\.tex$'); do
  git diff -U0 -- "$f" | grep -E '^[-+]' | grep -vE '^(\+\+\+|---)' \
    | grep -vE '^[-+][[:space:]]*%' | grep -vE '^[-+][[:space:]]*$' | wc -l
done
```
returns 0 for all 15 remaining files. The `_trimwork/A` tree used for the identity proof takes
`apx_f_cosine.tex` from HEAD plus my single comment line, and `tables/frame/cosine.tex` from HEAD
in both trees, so the author's in-flight work cannot influence the result either way.

## 6 · Findings that need a live-line change: reported, not applied

**F1 · Trapped prose in `1_introduction.tex`, and a gate that cannot see it.** The lead-in
"The collection is organized as follows:" sits inside a comment block, so rendered page 15 runs
from "rather than silently edited." straight into the first bullet with no lead-in. Verified in the
render. `check_trapped_prose.py` returns 0 suspects for this file because the swallowed text is on
a comment line rather than on a prose line following a comment. Not fixed: restoring it adds a line
to page 15 and this pass must leave the render identical. A flagged comment now sits at the site.

**F2 · `content.tex`'s header named a switch that does not exist.** It listed
`\ifdefensebuild, \ifapprovalsheet, \ifacademicobuild`; a live-line sweep for
`ifacademicobuild` across all of `src/` returns nothing. The real third name is the
`\ACADEMICOBUILD` selector. Fixed as a comment.

**F3 · 22 stale line coordinates left by the 2026-07-28 chapter split, plus 5 secondary ones.**
Comments across seven files cite `4_courb.tex:226`, `3_cbic.tex:261`, `5_mobiwac.tex:376` and
similar, into files that are now 45, 62 and 53 lines long. Each was re-anchored to its verified
location in the split tree (`4_courb/results.tex:14`, `3_cbic/results.tex:30`,
`5_mobiwac/05_setup.tex:34`, ...), located by searching for the quoted phrase rather than by
arithmetic. A sweep now reports 0 out-of-range coordinates and 0 landing on an `\input` line.

**F4 · 20 comment references to `0_main.tex` and 5 to `0_extra.tex`, both deleted.** Repointed to
`preamble.tex` / `content.tex` / `main_extra.tex` after verifying each destination holds the thing
being cited (`\extravolume` at `preamble.tex:193`, the appendix-lettering block in `content.tex`,
the chapter counter in `main_extra.tex`).

**F5 · `preamble.tex` declared a superseded title as SELECTED.** The comment named the 2026-07-24
working title as live while `\titulo` carried the author's 2026-07-30 replacement, which the folha
de rosto prints. Corrected from the author's own commit message; the former working title is
preserved as alternate 1.

**F6 · An orphaned comment fragment and a contradicted count in `apx_a_contributions.tex`.** The
line `% from it:` began nothing, and the PRESENT/ABSENT list below it enumerated nine absent paths
under a heading that had been corrected to five. Rewritten to state the five, to account for all
thirteen, and to explain that the thirteenth entry is a directory the hashing instrument cannot
classify.

**F7 · A stale count claim in `apx_b_errata.tex`.** Its round-4 note listed seven line coordinates
for Chapter 4's eight marked additions, taken before the split. Re-anchored to phrases.

**F8 · The em-dash gate stopped covering the preamble and the body when `0_main.tex` was split.**
`src_utils/check.sh:94` sweeps `$CH 0_main.tex`, where `$CH` is `chapters/*.tex chapters/*/*.tex`.
`0_main.tex` no longer exists, and its successors `content.tex` and `preamble.tex` are in neither
term, so nothing sweeps them. Verified in BOTH directions: an em-dash planted on a live line of
`content.tex` leaves the gate printing `OK` (the probe was removed immediately; `content.tex`'s
live-line diff against HEAD is 0), and the gate still fires correctly on a chapter file. Two other
sweeps in the same file take `$CH` alone and are unaffected. The fix is a live-line change in a
build file, outside this pass's permission: `$CH content.tex preamble.tex`. Reported, not applied.

## 7 · [VERIFY] flags

1. **The three-pass recipe does not converge from a cold `build/`.** `src_utils/latexbuild.sh` runs
   pdflatex, bibtex, pdflatex, pdflatex. From an empty aux tree that is one bibtex pass too few:
   the first PDF carries wrong citation numbers and the run still reports `tex_errors=0`, so nothing
   flags it. A second invocation settles it. This is a real trap for anyone who measures a page
   count or a fingerprint after a clean build, and it is upstream of this pass. Author's call
   whether to add a fourth pass or a convergence check; I did not change a build file.

## 8 · Could not confirm

- Whether the trapped sentence F1 was ever present in a rendered PDF. It is absent from the current
  render and from the 11:41 baseline; earlier PDFs were not examined.
- Whether the author intends the `apx_f_cosine.tex` prose rewrite now in flight to land in this
  round, and whether the resulting 100-page defense build is the intended length. The page-count
  registers in `src/` were not touched by this pass and still record 101.
