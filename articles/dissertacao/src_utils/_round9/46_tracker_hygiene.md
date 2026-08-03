# 46 — PENDENCIAS.md hygiene: what was classified, what was corrected, and what is UNFINISHED

Round 9, 2026-07-30. Track owner of `src_utils/PENDENCIAS.md`. Author's instruction: *"let's hygnazie
the PENDENCIAS.md file removing the itens that is done and is unecessary"*, kept *"direct and easy to
iterate, matain and add my points"*.

**Headline, stated first because it is the part that did not get done.** The archiving step —
step 2 of this track — **did not happen**. Not one item moved to `_archive/PENDENCIAS_RESOLVIDOS.md`.
The classification found exactly **one** item that is both closed and not in-flight (2.22), and
repointing its **28 live citations** could not be completed in the time left. What did land is four
author-facing facts that were **stale or false**, each re-measured. Details in §4.

---

## 1 · Classification: every item read by its own decision block

28 numbered items live in the file: **16 in §2, 1 in §5, 11 in §6**. Each was read for its
`(C) O que eu preciso de voce` section and its `> **AUTHOR:**` ruling, not pattern-matched.

| class | n | items |
|---|--:|---|
| IN-FLIGHT — a parallel track is applying the ruling; not mine to close | 12 | 2.9, 2.11, 2.12, 2.14, 2.15, 2.16, 2.18, 2.19, 2.20, 2.21, 2.23, 5.6b |
| STILL ASKING HIM — no ruling yet, or a blank `DECISAO SUA` | 12 | 2.1, 2.5, 2.24, 6.1–6.8, 6.11 |
| RECORD, not a pendency — closed by its own text, kept as a pointer | 3 | 2.8, 6.9, 6.10 |
| CLOSED and archivable | 1 | 2.22 |

**2.22 (Appendix F) is genuinely closed, verified three ways, and is the one archive candidate.**
Its fifteen points were probed one at a time on comment-stripped, wrap-tolerant prose (`live_text`
from `check_audit_claims.py`), after first asserting the instrument returns real prose:
`live_prose_chars=11020`, twelve encodable points, **`FAILED_POINTS=[]`**. Point 14's added sentence
was found by searching the concept's nouns rather than an imagined phrasing: *"Orthogonal gradients
also do not mean that the tasks share no knowledge: the two streams still exchange information
through the cross-attention trunk"*. Point 0 (the relettering) renders: the defense PDF prints
`APPENDIX D. Why the Two Tasks Do Not Compete on the Shared Trunk` in the body header on pp. 99–101
and the appendices run A, B, C, D in the table of contents on p. 11. The sign-off marker is gone
from the file and replaced by two `[SIGNED OFF 2026-07-30]` records at lines 67 and 552.

**`2.8` was NOT archived even though it is closed**, deliberately: two probes in
`check_audit_claims.py` pin strings inside its block (`R9-agree` requires
`**32 sao exatas e 9 estao obsoletas**` present, `R9-agree2` requires `As 21 ancoras dos capitulos`
absent), and those two exist *because the two trackers once contradicted each other for twenty
minutes*. Moving the block silently breaks the positive probe. It is a record, it is three lines of
"nothing to decide here", and it costs the author nothing where it sits.

## 2 · What is UNFINISHED, and why

**Step 2, archiving 2.22.** Blocked on citation repointing, not on the classification.
`check_tracker_refs.py` fails when a live citation names a heading that no longer exists, and 2.22
is the most-cited coordinate in the repository: **28 citations across 10 files** — 15 inside
`apx_f_cosine.tex` alone, plus `AGENT_GUARDRAILS.md`, `WRITING_LAW.md` (×2), `content.tex`,
`main_extra.tex`, `check.sh`, `check_register.py` (×2), and **four in fixture trees**
(`_fixtures/check_register/{clean,dirty}/src_utils/check_register.py` ×2 each, plus the fixture
README). The fixture copies are byte-identical to the live checker — `diff -q` returns 0 for both —
and `selftest_all.py` copies the live script into each fixture tree before running it, so repointing
the live file without the copies makes `make selftest` compare a file against itself and disagree.
The convention to use is established (`PENDENCIAS_RESOLVIDOS 5.3 (arquivado 2026-07-30)`, already in
ten source comments). **28 comment rewrites across 10 files, with a gate run after each, is the
remaining work.** Nothing is half-applied: the item is untouched and the gate is green.

**Not attempted, and why:** §6 is out of scope by the author's explicit instruction. §4 was already
retired by another track (`3ea775e6`); by the time this track reached the file, a parallel track had
also **removed the §4 heading entirely** — so step 3 of this track was already done twice over, and
the only remaining trace is the header's section-order sentence, which still names §4 (below).

## 3 · The concurrency hazard was real, and it hit twice

`PENDENCIAS.md` was rewritten under this track while it worked. The first application of three edits
was **clobbered wholesale**: the file went from 100,560 bytes with all three edits present to 98,952
bytes at 19:24 with `grep -c` returning **0 for all three**. All six edits were then re-applied in a
single atomic write-and-rename and committed at once (`5b4d0b4c`), which is the only reason they are
in history. Two other observations from the same cause:

- The sign-off marker count **moved from 56 to 54 during this session** (measured against `HEAD`
  and the working tree per file: `6_conclusion.tex` 9→8, `apx_b_errata.tex` 6→5). That is a parallel
  track removing markers under the author's rulings, not a measurement error. The item now says the
  number moves.
- `check_tracker_refs.py` was **red for a period through no edit of mine**: a live `### 5.6b`
  heading was invisible to its `^#{2,4}\s+(\d+)\.(\d+)\b` pattern (`\b` does not sit between `6` and
  `b`) while `PENDENCIAS_RESOLVIDOS 5.6b (arquivado 2026-08-02)` in `6_conclusion.tex:286` truncated to `5.6`. The same track that
  introduced the citation fixed the gate's pattern; both patterns now carry `([a-z]?)(?![\w.])`.

## 4 · What DID land: four facts, each re-measured (commit `5b4d0b4c`)

**4.1 — 2.1 said 53 markers, and its command printed zero lines.** The documented command was
`grep -rn "NEEDS SIGN-OFF" src/ --include="*.tex" | grep -v ":\s*%"`. Every marker lives inside a `%`
comment, so `-v` deleted precisely what it was counting: run directly, `rc=1`, **zero lines**.
Replaced with a per-file count excluding the generated build copy. Measured at `5c074a2a` plus the
working tree: **54 markers in 21 files**, of which **52 carry a body** (`[NEEDS SIGN-OFF: ...]`) and
**2 are bare back-references**; **58** if `src/build/` is included, because `build/fmt/_body.tex` is
a generated copy. Two independent instruments (Python `re` and `grep -rc`) agree per file, delta 0.
A wrap-tolerant pattern finds **57** across the whole tree — the extra one is a marker quoted inside
a `[SIGNED OFF]` record in `apx_f_cosine.tex`, and the naive pattern provably misses it (asserted
both directions). Also removed the item's false claim that `check_verify_list` counts these markers:
**no gate does** — searched every `.py` and `.sh` in `src_utils/`.

**4.2 — 2.5 said only 1 of the 2 `.drawio` is in the repository. It is false, and it had already
been diagnosed.** Both are present. `find . -name '*.drawio'` returns four in the repo;
`figures/mtlnet_poi_new.drawio` (13,640 B, `fontSize=14`) and
`figures/courb/arquitetura_modelo.drawio` (14,588 B, `fontSize=13`) are the two that matter. Commit
`b89a9876` names this exact defect in its own message — the instrument was `ls src/figures/*.drawio`,
a non-recursive glob blind to `figures/courb/` — but **the correction never reached the file**; the
pre-compression revision `ffce2375` said *"Ha `.drawio` para as duas"*. Restored the label
percentages the compression pass dropped, quoted from `LEFT_OUT.md` LO-6: **45.3** and **44.4**
percent of an 11.96 pt body. Also confirmed `figures/cbic_mtlnet_arch.png` is byte-identical to
`articles/CBIC___MTL/imgs/mtlnet_poi.drawio.png` (sha256 `0dc7e9dc…`), which is why raising its type
is the author's call and not a repair.

**4.3 — the §5 sweep command was comment-blind and overstated its own finding.** It printed five
full lines, which reads as five unfixed items. Comment-stripped, four of the five phrases are absent
from live prose entirely (`leakage-guarded`, `equivalence is well powered`, `revise that verdict…`,
`mean reciprocal rank` — the last also absent as `MRR` and as `reciprocal`, patterns proved against
the tree), surviving only in provenance comments that quote the old wording. The fifth resolves to
`src/tables/cbic/errata_wording.tex`, the errata table, where the string **is** the evidence. The
replacement strips `%` lines before matching, agrees with `live_text()`, still satisfies
`# EXPECT: lines=5`, and prints four empty lines.

**4.4 — §3 carried two wrong rows.** The banca row named **`0_main.tex`, deleted at `2b9b853d`**,
and claimed *"os colchetes aparecem no PDF"*. They do not: `\membrobancaA`, `\membrobancaB` and
`\databanca` are set at `preamble.tex:217-219`, but the signature block that would typeset them is
**commented out** at `abntex2-UFV.sty:166-170`, and a pypdfium2 search of all three builds returns
**zero** occurrences of `Banca member`, `defense date` or `pending advisor`. Corrected to say so,
and to note the useful consequence: nothing invented reaches the PDF, and neither will the real
names when they arrive. Separately the `\finalbuildfirstpage` row said **8**; `main.tex:95` says
**9**, and `main_academico.pdf`'s first body page is physical 9 printing 9 (its front matter is 8
pages, not 7). The approval-sheet row was checked and is **correct as written**: the placeholder
renders on p. 2 of `main_ppgc.pdf` and is absent from `main.pdf`.

## 5 · Compression accounting

**No block was compressed.** Measured on the committed blobs, which is the only stable measurement
available — the working tree is being rewritten concurrently and gave three different byte counts in
four minutes (102,255 / 101,634 / 102,255):

```bash
cd /Users/vitor/Desktop/mestrado/ingred
git show 5b4d0b4c^:articles/dissertacao/src_utils/PENDENCIAS.md | wc -c   # 97426, 1395 lines
git show 5b4d0b4c:articles/dissertacao/src_utils/PENDENCIAS.md  | wc -c   # 101390, 1427 lines
```

**Before 97,426 bytes / 1,395 lines → after 101,390 bytes / 1,427 lines.** It GREW, because four
corrections each carry the command that produced them, per `AGENT_GUARDRAILS` §4b V1. So the "no
author-facing fact was lost" obligation is satisfied trivially: nothing was removed except **two
false sentences and one broken command**, each replaced by a measured statement. The header's
per-item shape and its *"Para ADICIONAR um ponto seu"* instruction were re-read after the edits and
remain accurate.

**One header inaccuracy left for the author, deliberately.** The *"Ordem das secoes"* sentence still
reads `… -> §3 (terceiros) -> §4 (o que auditar primeiro)`, but **§4 no longer exists** — another
track removed the heading. Fixing it is a one-clause edit; it is left because the same file was being
rewritten concurrently and a seventh edit racing the other track risked losing the six that matter.

## 6 · A build failure this track hit and cleared (not caused by it)

`make defense` failed twice with `! Extra }, or forgotten \endgroup` at `l.100` and then
`! File ended while scanning use of \@writefile`. The fault was **not in any source file**: the
commit under this track touches one markdown file and no `.tex`. The error was inside
`build/main-aux/`, the generated LaTeX cache, which two racing builds had left internally
inconsistent (`1_introduction.aux` ended with three spurious closing braces; `4_courb.aux` was
truncated mid-`\@writefile`). Proof the sources were sound throughout: `make ppgc` (102 pp) and
`make extra` (22 pp) built with `tex_errors=0` from the same tree using their own aux directories.
Cleared that one aux tree (gitignored output; the project's own `make clean` is `rm -rf build/*`) and
`make defense` returned `rc=0`, 101 pp. Restoring `build/fmt/` via `make format` was then needed
because a `VERIFY_LIST` block asserts `build_copy_is_generated True` against
`src/build/fmt/_body.tex`.

## 7 · Verification, every exit code read directly

| what | result |
|---|---|
| `bash src_utils/check.sh` | **rc=0**, all 25 gates; 24 documented commands executed, 15 with a machine-checkable expectation, **0 failed** |
| `make defense` / `academico` / `ppgc` / `extra` | rc=0 / 0 / 0 / 0 — **101 / 98 / 102 / 22 pp**, `tex_errors=0` in all four |
| `sync_page_counts.py` (read-only) | rc=0, *"all recorded page counts agree with the build"* — no `--write` needed |
| `check_tracker_refs.py` | rc=0, 28 numbered sections, self-test both directions |
| `check_verify_list.py`, `check_audit_claims.py`, `check_meta_claims.py`, `check_wordcount_claims.py` | rc=0 each, run after the last edit |

## 8 · For the author

1. **2.22 is ready to archive and I did not do it.** 28 citations in 10 files must be repointed to
   `PENDENCIAS_RESOLVIDOS 2.22 (arquivado …)` first, four of them in fixture copies that must stay
   byte-identical to `check_register.py`. Say the word and it is one focused pass.
2. **The header still advertises a §4 that no longer exists.** One clause.
3. **2.1's title now carries a number that will be wrong tomorrow** (54, as tracks remove markers).
   It says so, and gives the command. If you would rather the title carried no number at all, say so.
