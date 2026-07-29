# 21 · Applying LATEX_UPGRADE.md, and persona 19 re-reviewing the result

> **Track:** apply `src_utils/LATEX_UPGRADE.md` (persona 19's source/build-engineering review:
> five candidates §1, five findings F-1..F-5 §2, a do-not-touch list §3, three author-decided
> additions A-1/A-2/A-3 §4), then re-run that persona over the changed tree.
> **Working directory for every command below:** `articles/dissertacao/`, after
> `source src_utils/texenv.sh`.
> **Concurrent track:** the build-speed track was editing `src/Makefile` and `src_utils/check.sh`
> throughout. Every anchor below was re-read immediately before the edit; §0 records what their
> work had already changed by the time I started, because three of the ten items I was sent to
> apply were already applied by it.

---

## 0 · The tree I actually found, which is not the tree LATEX_UPGRADE.md reviewed

`LATEX_UPGRADE.md` was written 2026-07-28 against a 51-line `Makefile`. When I started, the
build-speed track had already replaced it with a 174-line one (per-target aux trees, `all3`,
`fast`/`fast3`, twelve per-checker gate targets, `help`). Two consequences, both load-bearing:

1. **F-2 was already fixed by that rewrite.** All three recipes read
   `bibtex …; test $$? -le 1`, not `bibtex … || true`. My job on F-2 became *validating* a fix
   somebody else wrote, in both directions, which is what §3 below reports.
2. **The inherited build state did not reproduce in the repository tree.** `make final` and
   `make ppgc` both died at
   `! Extra }, or forgotten \endgroup. l.100 }` reading `build/chapters/1_introduction.aux`,
   while `build.sh` reported both builds clean, `tex_errors=0`. That is not a source defect: it
   is two sessions writing one shared `build/chapters/` aux tree, which is the exact failure the
   concurrent track's per-target aux trees were built to end. To keep my measurements free of
   their in-flight builds I did every experiment in an isolated copy of `src/`
   (`diff -r -x build -x dissertacao.pdf` clean against the repo before each run) and re-measured
   in the repository only after their Makefile landed.

    | tree | `make defense` | `make final` | `make ppgc` |
    |---|---|---|---|
    | repo `src/`, old shared aux, mid-flight neighbour | RC 0 | **RC 2** | **RC 2** |
    | isolated copy, same source, nobody else building | RC 0, 108 pp | RC 0, 105 pp | RC 0, 109 pp |
    | repo `src/`, after their per-target aux trees (`make all3`) | RC 0 | RC 0 | RC 0 |

    The middle row is the inherited claim reproduced (108/105/109, `tex_errors=0` in all three).
    `build.sh` reporting `tex_errors=0` for a build `make` could not complete is
    `science/AGENT_HANDOFF.md` §2.3b in the wild, in a form the round-6 fix does not cover: the
    error was in an `.aux` file, so the recovered PDF came out fine and only `-halt-on-error`
    saw it. `[VERIFY]` flag V-1 in §7.

---

## 1 · Step 1: every candidate and finding re-verified against the live tree

Anchored by phrase, measured, one verdict each. Nothing was applied that is not verified here.

### The five candidates (§1 of LATEX_UPGRADE.md)

| # | Candidate | Verdict | The measurement |
|---|---|---|---|
| C-1 | Move both `\newif`+`\ifdefined` blocks from `main.tex` into `0_main.tex` | **Still applies, still not recommended alone** — and now has a new dependent | `mkformat.py` extracts `main.tex`'s switch region by the phrases `\newif\ifdefensebuild` and `\newcommand{\finalbuildfirstpage}` (`SWITCH_FIRST`/`SWITCH_LAST`). Moving that block into `0_main.tex` breaks the format dump unless `mkformat.py` moves with it. Not applied: out of my track's scope, and the review itself calls it a waypoint that should not be committed. |
| C-2 | Merge `0_main.tex` back into `main.tex` | **Applies differently: the reference count has grown** | The review measured "55 live references to `0_main`". Now **88 lines across 24 live files** (`grep -rn '0_main' .` minus `/build/`, `_archive/`, `_review_v*/`, `_round6/`, `_gates/`, `_specialists_v*/`, `PENDENCIAS.md`, `CODEX_AUDIT.md`, `CODEX_VS_PERSONAS.md`; the count includes 19 lines in `LATEX_UPGRADE.md` itself and 21 in the new `mkformat.py`, and three `__pycache__` binary-match lines are reported by grep as one line each and are included in the 88). The new dependency is structural, not documentary: `mkformat.py` splits `0_main.tex` at `\begin{document}` into complementary byte ranges. Merging the files now would also have to re-derive that split. The review's "post-defense refactor" timing verdict stands and got stronger. Not applied. |
| C-3 | Delete `main_ppgc.tex`; inject `\APPROVALSHEET` from the Makefile | **Still applies, still not recommended** | `main_ppgc.tex` still holds exactly two content lines (`\def\APPROVALSHEET{}`, `\input{main.tex}`), verified with comments stripped: `grep -vn '^[[:space:]]*%' src/main_ppgc.tex \| grep -v '^[0-9]*:[[:space:]]*$'` returns those two. The Overleaf-contract argument is unchanged. Not applied; A-2 goes the opposite way and adds a third such file. |
| C-4 | DRY the three Makefile recipes into a parametrized pattern | **No longer applies as written** | The Makefile is no longer 51 lines with three six-line recipes; it is 174 lines and each recipe now carries a per-target `-output-directory` and a copy-back. `define`/`call` over that shape would hide the aux isolation the concurrent track just documented at length. Not applied. |
| C-5 | One canonical "three builds, one source" explanation; the others point to it | **Applies, and its own evidence changed** | The review cited a "four lines"/"two lines" contradiction across four documents. Two of the three wrong copies are gone: `Makefile` now says "a two-line file that sets `\APPROVALSHEET`", and `README_SRC.md` says "TWO lines" in both places. The two survivors are `main.tex:12` ("a four-line file") and `main_ppgc.tex:8` ("deliberately four lines of content") — that is F-5, and I fixed both. The larger consolidation (shrinking `README_SRC.md` and `CLAUDE.md` §1 to pointers) is documentation restructuring across files another track owns; not applied, flagged in §7. |

### The five findings (§2 of LATEX_UPGRADE.md)

| # | Finding | Verdict | The measurement |
|---|---|---|---|
| F-1 | `main.tex` header documents a `make final` command that silently produces the DEFENSE build | **Still applies, and the trap is real, not theoretical** | I ran the documented command verbatim, three passes, `-halt-on-error`, into a separate output directory: `pdflatex … -jobname=main_final "\newif\ifdefensebuild\defensebuildfalse\input{main.tex}"` produced **108 pages**, byte-different from both real builds, with `Abstract` on rendered page 3. `make final` produces 105 pages with `Table 12 …` on page 3, and `make defense` produces the same 108-page front matter as the documented command. So the command labelled "the AcademicoPG deposit body" emits the defense document under the deposit's jobname. Fixed. |
| F-2 | `bibtex … \|\| true` masks fatal BibTeX errors | **Already fixed by the concurrent track; I validated it in both directions** | The three recipes now read `bibtex …; test $$? -le 1`. Direction 1 (defect present): appending a malformed entry to `references.bib` makes `bibtex` exit **2**, `.blg` prints `(There was 1 error message)`, and `make final` exits **2** with `make: *** [final] Error 1`. Direction 2 (clean tree): the same `bibtex` invocation exits **0** and the build completes. Not re-applied — re-"fixing" it would have been the defect the brief warns about. |
| F-3 | The Makefile does not preflight the TeX environment it documents as mandatory | **Still applies** | `grep -n 'TEXMF\|texenv\|preflight' src/Makefile` before my change returned four lines: one prose mention of `TEXMFHOME` in the header comment and three `TEXMFOUTPUT=` assignments inside recipes. No target tested anything. Fixed. |
| F-4 | Optional: a two-line `main_final.tex` for build-tool parity | **Still applies; superseded in name by A-2** | No second thin entry file existed (`ls src/main*.tex` → `main.tex`, `main_ppgc.tex`). Applied as `main_academico.tex` per A-1/A-3, which is the author's decision on the name. |
| F-5 | `main_ppgc.tex` line-count self-description | **Still applies, in exactly two places** | `grep -n 'four-line\|four lines' src/main.tex src/main_ppgc.tex src/Makefile src_utils/README_SRC.md CLAUDE.md` → `src/main.tex:12`, `src/main_ppgc.tex:8`. The other documents already say two. Fixed. |

### The do-not-touch list (§3) and the author-decided additions (§4)

Everything in §3 was left alone: the `\ifdefined\MACRO` guard pattern, `main_ppgc.tex` as a
no-content shim, the C-1 `\finalbuildfirstpage` measurement block, the E-5 `hyperfootnotes` note,
the preamble hygiene ledger, the `% !TeX root` directives, `-halt-on-error` in the build flags,
`\include`-per-chapter. The one apparent exception is not one: §3 protects the `\ifdefined` guard
*pattern*, and A-3 renames the macro it tests, which the author decided and which leaves the
pattern intact.

A-1/A-2/A-3 all applied — §4 below.

---

## 2 · What changed, anchored by phrase

Every anchor is a phrase, not a line number. Where the concurrent build-speed track had already
edited the same block, I re-read the file immediately before the edit and rebased onto their text.

### `src/main.tex` — F-1, F-5, A-3

| Anchor phrase (before) | Anchor phrase (after) | Why |
|---|---|---|
| `main.tex+final -> the AcademicoPG deposit body` | `main_academico.tex -> the AcademicoPG deposit body` | A-1/A-2: the build has its own entry file now, so the header's three-build table names three entry files. |
| `it is a four-line file that sets one switch` | `Each shim holds TWO lines of content` | F-5. The file has two content lines; the count was checkable and stated wrongly inside the source. |
| `pdflatex "\newif\ifdefensebuild\defensebuildfalse\input{main.tex}"` … `make final does exactly this` | `THE MAKEFILE'S academico: RECIPE IS THE ONE PLACE THE COMMAND IS WRITTEN DOWN` + a `WHY THIS PARAGRAPH NO LONGER QUOTES A COMMAND` block carrying the measurement | F-1. The header no longer quotes any runnable command; it points at the single place the command lives. The quoted command is preserved only inside the explanation of why it was wrong, where it cannot be copied as instruction. |
| `This avoids a nested-\if scanning problem that \ifdefined has` | a block explaining the scanning mechanism, ending `This pattern is the fix, NOT the hazard -- a comment here claimed the reverse until 2026-07-29` | F-1's second half. The attribution was inverted; a reader acting on it would have "simplified" `\ifdefined` back into the raw `\newif` pattern that carries the hazard. |
| `\ifdefined\FINALBUILD \defensebuildfalse \else \defensebuildtrue \fi` | `\ifdefined\ACADEMICOBUILD \defensebuildfalse \else \defensebuildtrue \fi` | A-3. The guard *pattern* is untouched (§3 protects it); only the macro it tests is renamed. |
| `NAMING, PENDING: the author has approved renaming this build academico … It is NOT applied in this file yet` | `NAMING: this build was called final until 2026-07-29 … If you rename it again, sweep the tools that hardcode the stem` + the eight-tool list | The pending note became a done note, and the sweep list grew from five tools to eight (see §4). |

Not renamed, deliberately: **`\finalbuildfirstpage`**. `src_utils/mkformat.py` extracts `main.tex`'s
switch region by that exact phrase (`SWITCH_LAST = r"\newcommand{\finalbuildfirstpage}"`), the
author's decision covers the switch macro and the external names rather than this counter, and the
reader never sees it. The comment above it now says so, and the surrounding prose was updated from
"the final build" to "the academico deposit build" so the block does not contradict the rename.

### `src/main_academico.tex` — new file (A-2, closing F-4)

Two content lines, `\def\ACADEMICOBUILD{}` and `\input{main.tex}`, on the `main_ppgc.tex` pattern
with the same do-not-add-content header. **Verified equivalent to the injection it replaces**: I
built the old command line (with the renamed macro) and the new entry file into separate output
directories, three passes each, and compared with `verify_format.py --all`:

    pair: text IDENTICAL (digits masked, 270385 chars compared)
    pair: digit sequence IDENTICAL (4480 runs)
    pair: 105 pages, all media boxes equal
    pair: bookmark tree IDENTICAL (107 entries)

### `src/Makefile` — A-1, F-3

- `final:` recipe → `academico:`, with `-jobname` and the aux tree renamed to `main_academico`,
  and the `"\def\FINALBUILD{}\input{main.tex}"` injection replaced by `main_academico.tex` as the
  entry file. Three passes, unchanged shape.
- `final:` survives as a **forwarding alias** that prints what replaced it and builds `academico`.
  A renamed target that simply vanishes turns a half-remembered command into a silent no-op.
- `all3` now builds `defense academico ppgc`; `fast-final`/`fast3-final` → `fast-academico`/
  `fast3-academico`; the header target map, the `make help` text and `.PHONY` follow.
- **New `preflight` target** (F-3), a prerequisite of `defense`, `academico`, `ppgc`,
  `fast-defense`, `fast-academico`, `fast-ppgc`, `format` and `fast3`.

  The naive guard the review sketched — `test -n "$$TEXMFHOME"` — would have been a gate firing on
  the wrong condition, so I measured first. `kpsewhich -var-value TEXMFHOME` returns
  `$HOME/Library/texmf` whether or not the variable is exported, and `abntex2.cls` resolves either
  way; an unset `TEXMFHOME` is **not** what breaks this build. What breaks is `TEXMFVAR`, whose
  default is `$HOME/Library/texlive/2026basic/texmf-var`, a tree with no `pdftex.map` at all. So
  the preflight tests the condition that actually fails: `pdflatex` on `PATH`, and an `ntx-` entry
  in the map under whatever `TEXMFVAR` resolves to. With `texenv.sh` sourced that map holds 168
  `ntx-` lines.

### The tools — A-1

`build.sh` (`run_final` → `run_academico`, the jobname, both `bibtex` calls, the mode dispatch, the
`BUILT` list, the `label` map, both usage strings), `sync_page_counts.py` (the `measured()` stem
tuple and four `CLAIMS` keys), `sync_deliverables.py` (the source path), `fastbuild.sh` (the target
case, with the alias **normalized** rather than stem-mapped, because the run-time driver file is
named after the target key), `mkformat.py` (the `TARGETS` key), `verify_format.py` (`STEMS`), and
`_round6/VERIFY_LIST.md` (three executable command blocks). Each carries a one-line comment naming
the rename and its date.

---

## 3 · The sweep the brief warned about: FIVE tools, measured EIGHT

The brief named five files that hardcode the old stem. Grepping the live tree found **eight**, and
the eighth was found only by *running the gate suite* rather than by grepping:

| # | File | What holds the stem | In the brief's list? |
|---|---|---|---|
| 1 | `src/Makefile` | target, jobname, aux tree, echo, `all3`/`fast*` deps, `.PHONY` | yes |
| 2 | `src_utils/check.sh` | **nothing** — it reads `build/main.log` only | yes, but no edit needed |
| 3 | `src_utils/build.sh` | `run_final`, jobname, 2 bibtex calls, mode dispatch, `BUILT`, `label` map, 2 usage strings | yes |
| 4 | `src_utils/sync_page_counts.py` | `measured()` stem tuple + 4 `CLAIMS` keys | yes |
| 5 | `src_utils/sync_deliverables.py` | `DELIVERABLES` source path | yes |
| 6 | `src_utils/fastbuild.sh` | target `case` → `STEM` | **no** |
| 7 | `src_utils/mkformat.py` | `TARGETS` key → stem + switch list | **no** |
| 8 | `src_utils/verify_format.py` | `STEMS` | **no** |
| 9 | `src_utils/_round6/VERIFY_LIST.md` | three EXECUTABLE bash blocks | **no** |

Rows 6–8 are the concurrent track's new files, which did not exist when `LATEX_UPGRADE.md` or the
brief was written. **Row 9 is the interesting one.** `_round6/` is a frozen historical directory,
and the rename convention in `LATEX_UPGRADE.md` §4 explicitly excludes frozen reports. But
`src_utils/check_verify_list.py` *executes* the bash blocks in `VERIFY_LIST.md`, so its A4
page-numbering probe — `for stem in ("main", "main_final", "main_ppgc")` — is live tooling wearing
a frozen report's clothes. After the rename it printed

    FAIL     VERIFY_LIST.md: python3 - <<'PY'
             exit 1
             lines=1, expected 3

and `make check` exited 2. Fixed by renaming the stem in the three executable blocks (the build
reproduction command, the A4 probe, the A5 `Hfootnote` command) and the one prose sentence that
would otherwise contradict them; the historical mention at "then `main_final`" is left as history.
**`check.sh` needed no edit at all** — it reads `build/main.log`, the defense stem, which did not
change. The claim "five tools" would have been wrong in both directions.

### What was deliberately NOT renamed, and why each would have been a defect

- **Two `sync_page_counts.py` `CLAIMS` regexes** still read `build/main_final\.pdf` and
  ``make final` -> `main_final\.pdf``. They match what `CLAUDE.md` and `PENDENCIAS.md` *actually
  say*, and those documents are another track's to reword. Renaming a pattern to match a document
  that has not changed is precisely how this tool went silent on 2026-07-28: it printed nothing,
  exited 0, and four page-count claims went unchecked. The internal keys are renamed to
  `academico`; the patterns are not. A comment in the file says so.
- **The `dissertacao_v3_final.pdf` workspace name** in `sync_deliverables.py`. That is the name of
  an already-saved artifact; renaming it would fork the artifact's version history rather than
  continue it. Only the repo-side source path moved.
- **`\finalbuildfirstpage`** — see §2.
- **`main_final` in `CLAUDE.md`, `PLAN.md`, `PENDENCIAS.md`, `README_SRC.md`, `codex_reviewer.md`,
  `reviewers/19_latex_source_reviewer.md`.** `LATEX_UPGRADE.md` §4 A-1 lists these as in scope for
  the rename, and I did not touch them: they are documentation owned by other round-7 tracks that
  are editing the same files concurrently, and two of them are load-bearing for the page-count gate
  above. `[VERIFY]` flag V-2.

**Residual old-stem count, read from the grep output immediately before writing this:** 11
occurrences of `main_final` in live source and tooling
(`src/*.tex src/Makefile src_utils/*.sh src_utils/*.py src_utils/_round6/VERIFY_LIST.md`). Nine are
provenance comments naming the old name in an explanation; two are the `CLAIMS` regexes above. Zero
are executable references to a file that is no longer written. The command:

```bash
grep -rn 'main_final' src/*.tex src/Makefile src_utils/*.sh src_utils/*.py \
  src_utils/_round6/VERIFY_LIST.md | wc -l          # EXPECT: equals=11
```

---

## 4 · Measurements, before and after

All from `articles/dissertacao/` with `src_utils/texenv.sh` sourced. The "after" column was read
from the tool's own output at the time of writing, not from an earlier run.

| What | Before | After | How |
|---|---|---|---|
| `make defense` | 108 pp | 108 pp | `grep -a 'Output written' src/build/main.log \| tail -1` |
| deposit build | 105 pp (`main_final.pdf`) | 105 pp (`main_academico.pdf`) | same, on `main_academico.log` |
| `make ppgc` | 109 pp | 109 pp | same, on `main_ppgc.log` |
| `tex_errors`, all three | 0 / 0 / 0 | 0 / 0 / 0 | `grep -ac '^! ' src/build/<stem>.log` |
| overfull hbox / vbox, all three | 0 / 0 | 0 / 0 | `build.sh` + direct log regex |
| undefined cites / refs, all three | 0 / 0 | 0 / 0 | flattened-log regex (LaTeX wraps at 79 cols) |
| bibtex problems / oversized floats | 0 / 0 | 0 / 0 | `.blg` read separately from the `.log` |
| `Hfootnote` warnings, all three | 0 | 0 | log regex |
| `build.sh src both` | `DEFENSE … FINAL …` RC 0 | `DEFENSE … ACADEMICO …` RC 0 | `bash src_utils/build.sh src both` |
| `sync_page_counts.py` | `defense 108, final 105, ppgc 109`, 10 claims, RC 0 | `defense 108 pp, academico 105 pp, ppgc 109 pp`, 10 claims, RC 0 | `python3 src_utils/sync_page_counts.py` |
| `make check` | RC 0, 18 gates (brief); 19 gates after the concurrent track's timing table | RC 0, 19 gates, 1.597 s | `(cd src && make check)` |
| `make check-scripts` | not present at round start | 13 gates ran, 0 skipped, 1.109 s | `(cd src && make check-scripts)` |
| entry files producible as a compile root | 2 of 3 | 3 of 3 | `ls src/main*.tex` + the equivalence proof in §2 |
| tools hardcoding the deposit stem | 8 (5 named in the brief + 3 unnamed) + 1 executable frozen doc | 0 executable | §3 table |

**Rendered verification (not the `.tex`).** `pypdfium2`, on the three PDFs on disk:

    main            108 pp | first numbered: physical  11 prints  11  OK
    main_academico  105 pp | first numbered: physical   8 prints   8  OK
    main_ppgc       109 pp | first numbered: physical  12 prints  12  OK

    front matter, pages 1-6:   cover  Resumo  Abstract  List-of-Figures
      main                       Y       Y        Y           Y
      main_academico             n       n        n           Y
      main_ppgc                  Y       Y        Y           Y

That is the switch doing exactly what the header says: the deposit build drops the cover, the
Resumo and the Abstract, keeps the lists, and numbers from physical page 8.

---

## 5 · Validation in both directions (AGENT_GUARDRAILS §7, and the brief's rule 4)

Three fixes, each shown failing on a tree carrying the defect and passing on the fixed one.

**F-1, the wrong-PDF trap.** The proof *is* the failing direction: I ran the documented command
verbatim rather than reasoning about it.

```bash
# in an isolated copy of src/, three passes, -halt-on-error
pdflatex ... -jobname=main_final "\newif\ifdefensebuild\defensebuildfalse\input{main.tex}"
```

    documented command -> 108 pages, "Abstract SILVA, Vitor Hugo Oliveira..." on rendered p.3
    make final         -> 105 pages, "Table 12 - Wording substitutions..."   on rendered p.3
    make defense       -> 108 pages, same p.3 as the documented command

Three distinct sha256 sums; the documented command's output matches the defense build's front
matter, not the deposit build's. The comment invited an operator to upload the wrong document.

**F-2, the bibtex guard** (already fixed by the concurrent track; validated, not re-applied):

```bash
printf '\n@article{r7_deliberate_break,\n  author = {Broken, A.,\n  title = {no closing brace\n' >> references.bib
make academico            # -> RC 2, "(There was 1 error message)", make: *** [academico] Error 1
bibtex build/.../main_academico   # -> exit 2 on its own
# restore references.bib from the repo copy, confirm identical
make academico            # -> RC 0, 105 pages; bibtex exits 0 on the clean tree
```

Direction 1 fails, direction 2 passes, and the restore was confirmed with `diff -q` against the
repository's own copy rather than assumed.

**F-3, the preflight.** Three directions, because there are two failure modes and one success:

```bash
env -u TEXMFHOME -u TEXMFVAR -u TEXMFCONFIG PATH=/usr/bin:/bin make preflight
#   -> "BUILD ENVIRONMENT NOT SET UP: pdflatex is not on PATH." RC 2
env -u TEXMFHOME -u TEXMFVAR -u TEXMFCONFIG PATH=/Library/TeX/texbin:/usr/bin:/bin make preflight
#   -> "BUILD ENVIRONMENT NOT SET UP: the newtx font MAP is not on TEXMFVAR."
#      names the resolved path and the expected file. RC 2
env ... make academico   # -> refuses at preflight, WITHOUT running a pdflatex pass
(source src_utils/texenv.sh && cd src && make preflight)   # -> RC 0
```

Without the guard, the same environment runs three passes and then dies with
`!pdfTeX error: pdflatex (file t1xtt): Font t1xtt at 657 not found` — measured, and it names the
wrong thing, which is why F-3 exists.

**The page-count gate, which is the one the brief said would go silent.** I tried to make it go
silent and could not, but the attempt found something worth recording:

```bash
# a copy of sync_page_counts.py pointed at a stem that is NEVER written
python3 src_utils/_r7_stale_probe.py
#   -> "no src/build/main_NONEXISTENT.log -- build first, this script reads the real log", RC 1
# the same copy pointed at the OLD stem
python3 src_utils/_r7_stale_probe.py
#   -> "measured ... academico 105 pp ... all recorded page counts agree", RC 0   <-- WRONG
```

The second case passed **because a pre-rename `build/main_final.log` was still on disk holding
105**. A stale build artifact made a stale tool look correct. `make clean` then `make all3`
removed it; the probe was deleted after use. This is the silent-gate hazard in person, and the
lesson is narrow and worth keeping: **after renaming a build output, `make clean` before you trust
any tool that reads `build/`.**

---

## 6 · Persona 19 re-review, run over the tree AFTER the changes

`reviewers/19_latex_source_reviewer.md`, its procedure steps 1–8, over the changed tree. It wrote
`LATEX_UPGRADE.md`, so it is judging whether its own recommendations were applied faithfully and
what the changes introduced. Its hard limit is read-only on tracked source; I ran it as a review and
then applied the two findings that were cheap, which is a departure the persona permits only because
the author's brief asks for it ("fix what is cheap and flag the rest") — noted so the boundary is
visible.

### Verdict: **source-clean**

No defect that ships a wrong or broken PDF. Zero undefined citations, zero undefined references,
zero TeX errors, zero overfull boxes across all three builds; no duplicate bibliography key; no
trapped prose; and — the one that matters for this track — each of the three builds is now
demonstrably the document it claims to be, verified in the render rather than in the source.

### Scorecard (8 dimensions, one evidence line each)

| # | Dimension | Score | Evidence |
|---|---|---|---|
| 1 | Preamble hygiene | **GOOD** | 20 live `\usepackage` loads, **zero duplicates**, zero l2tabu obsolete commands (`\bf`/`\it`/`\centerline`/`\over`) in `0_main.tex`. Font stack is `newtxtext,newtxmath` per TEMPLATE §8. `hyperref` is loaded by `abntex2` itself, with the E-5 `\PassOptionsToPackage{hyperfootnotes=false}{hyperref}` placed before the class for the documented reason. One l2tabu-list name appears: `\usepackage[utf8]{inputenc}` — see F19-3. |
| 2 | Build health | **GOOD** | All three: `tex_errors=0`, `Fatal=no`, overfull hbox 0, overfull vbox 0, undefined cites 0, undefined refs 0, `Float too large` 0, `Hfootnote` 0, `Label(s) may have changed` 0. Underfull hboxes 17/16/17 — cosmetic, `\OnehalfSpacing` in a justified one-column book, and not a margin risk (hand-off to 13). One package warning per build, identical: babel `Name 'brazil' is deprecated. Use 'brazilian' instead` (F19-4). |
| 3 | Bibliography integrity (mechanical) | **GOOD** | `src/references.bib` (the only file `\bibliography{references}` loads): **100 entries, zero duplicate keys**. All three `.blg` present, **0 errors**, 1 warning each. The documented collision set is **absent**: `grep -c '^@.*{Wang_2023,'` and the same for `Liu_2023`/`Lai_2024` each return **0** here. Two DGI-family keys exist and are two *different* papers — `velickovic2019deep` is titled "Deep Graph Infomax", `velivckovic2017graph` is "Graph Attention Networks" — so this is two works, not one fragmented key. Backend is `bibtex` + `abntex2cite[num]` + `abntex2-num.bst`, matching the settled decision. |
| 4 | Cross-reference plumbing | **NEEDS-WORK (minor)** | 50 `.tex` files swept, comments stripped and comment *tails* truncated. **259 live `\ref`-family calls: 249 tied with `~`, 9 preceded by a plain space, 1 other.** Zero labels before their caption; zero hardcoded "Figure 3.2"-style numbers; label prefixes consistent (`sec` 53, `tab` 16, `apx` 13, `eq` 9, `fig` 7, `ch` 6, `fn` 1). The 9 untied refs are F19-1. |
| 5 | Graphics & floats (source) | **GOOD** | 7 `\includegraphics`: **zero hardcoded pt/cm widths, zero absolute paths**; 5 carry `width=\textwidth`, 2 carry no option at all (F19-5). By extension, counted rather than recalled: **4 vector `.pdf`** (all MobiWac: `fig1_dataflow`, `fig2_model`, `fig3_embquality`, `fig4_deltas`) and **3 raster `.png`** (CBIC's `cbic_mtlnet_arch`, CoUrb's `arquitetura_modelo` and `distribuicao_estados` — the published figures), which sums to the 7. Float placements: 18 `[htbp]`, 3 `[htb]`, **zero `[H]`**. No `svg`, so no `--shell-escape` dependency. |
| 6 | Two-build correctness | **GOOD — this is what changed** | Three entry files, each producing its own document, verified in the render: `main` 108 pp with cover+Resumo+Abstract; `main_academico` 105 pp with none of the three; `main_ppgc` 109 pp with all three plus the approval sheet. First numbered page prints its own physical position in all three (11/8/12). The new thin entry file is byte-equivalent to the injection it replaced across text, digits, page boxes and bookmarks. The F-1 trap is gone and its explanation remains. `\finalbuildfirstpage` still carries its measured derivation. |
| 7 | Maintainability & reproducibility | **NEEDS-WORK (pre-existing)** | `% !TeX root` present in **all 50** files and every target resolves, including the two new roots. `src/` stays Overleaf-pasteable: **zero live references from `src/` into `src_utils/` or a parent**. `\sd{}` used 37 times across the table files. But **1 of 16 table files declares script provenance in its header** — `tables/frame/lineage.tex`, which points at `tables/README.md` for its extraction provenance (F19-2). 13 manual layout commands, 10 of them in `0_main.tex`'s front matter where they are structural (`\vspace*{\fill}` on the cover), 3 `\vspace{2pt|6pt}` inside MobiWac tables. |
| 8 | Portability | **NEEDS-WORK, and newly better** | Engine deps are declared and minimal; no shell-escape, no absolute paths. The deposit build was un-openable outside `make` until this pass and now is not — all three are a selectable compile root. Remaining risk is environmental rather than in the source: this machine's TeX is **BasicTeX** (`/usr/local/texlive/2026basic`) plus a usermode tree, which is why the preflight was worth adding. |

### Ranked findings

**F19-1 (LOW, mine to have caught) — 9 of 259 cross-references are not tied with `~`.**

> `This chapter is organized as follows: Section \ref{sec:courb:related}` — `chapters/4_courb/intro.tex`

Four in `4_courb/intro.tex` (one line, four `Section` refs), four in `4_courb/results.tex`
(`Table`, `Figure`, `Table`, `Figure`), one in `apx_c_ai_disclosure.tex`. **Measured in the render
before assigning severity:** zero cases where the label word is separated from its number by a line
break in the defense PDF, so nothing is visibly wrong today. This is latent — a reflow can break
"Section" from its number. *Direction:* `Section~\ref{...}`. **Not applied:** eight of the nine are
in `chapters/4_courb/`, a **PUBLISHED** chapter under the errata regime (NORTH_STAR §4), so even a
whitespace edit there is the author's call, not mine. Flagged, `[VERIFY]` V-3.

**F19-2 (LOW, pre-existing) — 15 of 16 table files carry no script-provenance header.**

TEMPLATE §3 and AGENT_GUARDRAILS N2 want numeric table cells generated by a committed script rather
than hand-typed; only **`tables/frame/lineage.tex`** says where its numbers came from, via
`% frame/lineage.tex -- one table. Extraction provenance and verification: tables/README.md`. The
other 15 carry no such line, `tables/frame/bib_errata.tex` among them. This is the *source side* of
number integrity, and the value check belongs to persona 06. *Direction:* add a one-line provenance
comment naming the producing script or the source table per file. Out of my track's scope. Handed
off.

*(Correction, same pass: an earlier draft of this finding and of dimension 7 named
`tables/frame/bib_errata.tex` as the file that carries provenance. The count of 1 was right and the
file was wrong — `bib_errata.tex` is in the list of 15 that lack it, which my own producing cell had
printed. A reader following the direction would have skipped the wrong file. Re-measured by printing
both lists rather than only the count.)*

**F19-3 (INFORMATIONAL) — `\usepackage[utf8]{inputenc}` is on l2tabu's obsolete list for modern
engines but is correct here.** This document compiles with `pdflatex`, not LuaLaTeX or XeLaTeX, and
`inputenc` remains the supported way to declare UTF-8 input for `pdflatex`. Flagged only so a future
"modernize the preamble" pass does not delete it on a checklist, and so this review is not read as
having missed it.

**F19-4 (INFORMATIONAL) — one package warning per build:** babel's
`Name 'brazil' is deprecated. Use 'brazilian' instead`. It comes from `abntex2` loading `babel` with
the legacy alias, not from this preamble. Changing it means overriding the class's own option.
Cosmetic; the warning count is 1 and it is identical across all three builds, which is why the build
claim above says "one package warning" rather than "zero".

**F19-5 (LOW) — two `\includegraphics` carry no width option:**
`figures/mobiwac/fig3_embquality.pdf` and `fig4_deltas.pdf` in `5_mobiwac/06_results.tex`. Natural
size happens to fit the text block today. A vector figure regenerated at a different canvas size
would overflow with no warning that names the cause. *Direction:* `width=\linewidth`. **Not
applied:** `5_mobiwac/` is the chapter under review, where every edit must land in both the
dissertation and `articles/[mobiwac]/src/` plus an `ERRATA.md` entry (NORTH_STAR §4). Flagged,
`[VERIFY]` V-4.

**F19-6 (FIXED IN THIS PASS, and it was mine) — both two-line shims cited stale line coordinates.**

> `% This file is deliberately TWO lines of content (:22-23 below; ...` — `main_academico.tex`, my
> own new file: the lines are at :31-32.
> `% ... (:18-19 below; ...` — `main_ppgc.tex`: the lines are at :24-25.

The same F-5 class the rename pass had just removed from `main.tex`, reintroduced by me and by the
concurrent comment pass, and invisible to `check_comment_hygiene.py`, which guards the *count* but
not the *coordinates*. Both now anchor to the two commands by name. Committed as `688856cc`.

**F19-7 (INFORMATIONAL, blocked) — the linter appendix cannot be produced on this machine.**
The persona's step 5 calls for `chktex` and `lacheck` findings triaged into load-bearing and noise.
**Neither binary exists here:** `command -v chktex` and `command -v lacheck` both return nothing, and
neither is in `/Library/TeX/texbin`, because this is BasicTeX rather than a full TeX Live. I did not
install anything into the user's TeX tree to obtain them. What I ran instead, mechanically over the
50 live-filtered `.tex` files, covers the load-bearing subset the persona names: missing `~` before
refs (F19-1), labels before captions (0), hardcoded reference numbers (0), obsolete commands (0),
duplicate packages (0). Quote direction and `$…$`-vs-`\(…\)` were **not** checked and are the gap.
`[VERIFY]` V-5.

### What is already engineered well (do not undo)

Everything on `LATEX_UPGRADE.md` §3's list still holds and I add three items the changes created:

- **The `preflight` target tests the condition that actually breaks the build**, not the variable a
  reader would guess. If someone later "simplifies" it to `test -n "$$TEXMFHOME"`, the gate stops
  discriminating: measured, that variable's default is already correct on this machine.
- **The forwarding `final:` alias and `build.sh`'s accepted `final` mode.** A renamed target that
  vanishes turns a half-remembered command into a silent no-op. Both say what replaced them.
- **The three per-target aux trees** (the concurrent track's work). They are what makes the three
  builds concurrently safe, and they are the reason the shared-aux corruption in §0 cannot recur.

### Out-of-scope handoffs (one line each)

- **13 (UFV compliance):** the deposit build's printed numbering now reads 8 on physical page 8 in
  `main_academico.pdf` — re-measure margins/font/numbering against the manual on the renamed file.
- **18 (rendered pages):** two MobiWac figures have no `width` option (F19-5); judge whether their
  natural size is what you want on the page.
- **03 (prose):** none. No prose changed in this pass; every edit was a comment, a Makefile recipe,
  or a tool.
- **05/06 (citation truth / number values):** F19-2's hand-typed table cells are a number-integrity
  surface, not a markup one; the values are yours.

---

## 7 · `[VERIFY]` flags and what I could not confirm

- **V-1 — `build.sh` reported `tex_errors=0` for two builds `make` could not complete.** §0's first
  table. The error was inside `build/chapters/1_introduction.aux`, so nonstopmode recovered and
  wrote a full PDF, and the round-6 `tex_errors` fix did not see it because the `.log` it read was
  from a *different* jobname's successful pass. The concurrent track's per-target aux trees make
  the collision unreachable for the three `make` targets, so this is closed by their fix rather than
  by mine, but the general shape — a shared-aux corruption that only `-halt-on-error` sees — is not
  gated anywhere. Worth a gate; not mine to add this round.
- **V-2 — six documentation files still say `make final` / `main_final.pdf`.** `CLAUDE.md`,
  `PLAN.md`, `src_utils/PENDENCIAS.md`, `src_utils/README_SRC.md`, `src_utils/codex_reviewer.md`,
  `reviewers/19_latex_source_reviewer.md`. `LATEX_UPGRADE.md` §4 A-1 puts them in scope; I left them
  because other round-7 tracks are editing them concurrently and because two of them are matched by
  `sync_page_counts.py` patterns, so the doc reword and the pattern change must land together or the
  gate goes silent. **Whoever rewords them must change the matching `CLAIMS` regex in the same
  commit** and confirm `sync_page_counts.py` still reports that row rather than printing `UNMATCHED`.
- **V-3 — 9 untied `\ref` calls, 8 of them in the PUBLISHED CoUrb chapter** (F19-1). Not applied:
  the errata regime makes even a whitespace edit there the author's decision. Nothing renders wrong
  today (measured: 0 label-word/number line breaks in the defense PDF).
- **V-4 — 2 `\includegraphics` with no width option in the chapter under review** (F19-5). Not
  applied: an edit there must land in `articles/[mobiwac]/src/` too, plus an `ERRATA.md` entry.
- **V-5 — the linter appendix is missing.** `chktex` and `lacheck` are absent from this machine
  (BasicTeX, not full TeX Live) and I did not install into the user's TeX tree. Quote direction and
  `$…$`-vs-`\(…\)` were not checked. Everything else the persona names as load-bearing was checked
  by other means, listed in F19-7.
- **V-6 — candidate 5's documentation consolidation is only half done.** `main.tex`'s header is now
  the canonical telling and the concurrent track's comment pass reduced the other copies to
  pointers, which is the substance of C-5. What remains is the *prose* in `README_SRC.md` and
  `CLAUDE.md` §1 — another track's files. Related to V-2.

### Could not confirm

- **That the `academico` name is what the author wants in the delivered documentation.** The rename
  is his decision (`LATEX_UPGRADE.md` §4 A-1), and I applied it to the build machinery and the tools.
  Whether the author-facing docs should read `academico` or keep `final` as the familiar name is a
  wording call I did not make. V-2.
- **That `\finalbuildfirstpage` should keep its name.** A-3 renames the switch macro and says
  nothing about the counter. I left it, wrote the reason into the source, and flag it here rather
  than deciding: it is `mkformat.py`'s extraction anchor, so renaming it is a two-file change.
- **The underfull-hbox count's significance.** 17/16/17 across the three builds. They are cosmetic
  in a justified one-column book under `\OnehalfSpacing` and none produce an overfull box, but I did
  not inspect the 17 loci in the render. Persona 13's surface, not mine.
- **Whether the two round-6 known false positives are truly gone.** `science/AGENT_HANDOFF.md` §3.5
  says `check.sh` "currently exits 1 on two known false positives... Read the output rather than the
  exit code". Measured today: `make check` exits **0**, and the "Pareto" and "this article" sweeps
  are now exemptions carrying their reasoning. So §3.5's instruction is stale. Not my file to
  correct, and it points the next agent at the wrong habit.

---

## 8 · Commits

| Commit | What it carries |
|---|---|
| `ce45c051` (another track's) | `main.tex`'s F-1/F-5/A-3 edits, swept in while the file was staged. Their commit message discloses the one non-comment line. |
| `6b6dc8a8` (another track's) | **The whole rename and preflight**: `main_academico.tex`, `Makefile`, `build.sh`, `sync_page_counts.py`, `sync_deliverables.py`, `fastbuild.sh`, `mkformat.py`, `verify_format.py`, `VERIFY_LIST.md`. Swept in the same way. A **`git notes` correction is attached to it** carrying the full defect/fix/measurement record, per the repo's rule that a wrong commit message gets a note rather than a rewrite: `git log --notes -1 6b6dc8a8`. |
| `688856cc` (mine) | F19-6: both two-line shims cited stale line coordinates. |

**The coordination cost, stated plainly.** Two sessions were editing `src/Makefile`, `src/main.tex`
and `src_utils/check.sh` at once. I re-read every file immediately before editing and rebased when
an anchor had moved, which worked — no work was reverted in either direction, and their bibtex fix
and aux-tree isolation are intact. What did not work is commit boundaries: their `git add` of a
shared file swept my staged work into their commits twice. Nothing was lost, and the `git notes`
entry makes the record recoverable, but the history now attributes this track's work to two commits
whose subjects describe something else. Next time, in a concurrent-edit round: commit each file the
moment it is verified, rather than batching a nine-file change.
