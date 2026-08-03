# Round 11, item 31 — Definition 2.7 split into two definitions

**Baseline commit:** `28df0cc0`
**File edited:** `src/chapters/2_fundamentals.tex` (the only file this item was permitted to touch,
and the only file I edited).
**Report written:** `src_utils/_round11/31_def27_split.md`

---

## 1 · The item and the author's ruling, verbatim

The item: split Definition 2.7, which carried two concepts under one head.

The author's authorization, verbatim:

> "podemos seguir com a definicao separada do 2.7"

And, in the same message, lifting the page budget:

> "pode melhorar o texto da fundamentacao sem preocupacao de paginas, vamos remover algumas coisas
> de appendix em sequencia"

The reason the split is a defect repair rather than a preference: round 10 merged the two concepts to
hold the page count, and its own report says so. `src_utils/_round10/29_ch2_definitions.md` §4:

> "Definitions 2.7 and the former 2.8 merged into one two-sentence block, since the second was the
> first's complement"

listed under the heading "The definitions cost one page on the first build (defense 106 → 107). I did
not accept that: five compaction passes brought it back". And §7 item 2 of the same report, left open
for the author:

> "**Definition 2.7 merges two concepts under one head** ("Place embedding and check-in-level
> representation"). I merged them to hold the page count at 106. If the author prefers them separate,
> splitting them costs one definition number and about four lines"

## 2 · What the LIVE text said before my edit

Verified in the live file immediately before editing (the brief's line numbers were correct at
`:235-249`, but I re-measured rather than trusting them; `read_file` on lines 228-251):

```latex
\begin{definition}[Place embedding and check-in-level representation]\label{def:fund:placelevel}
A place embedding assigns one vector $\mathbf{e}_p$ to each POI $p$, so every check-in
at that POI enters the model with the same representation. A check-in-level
representation instead assigns one vector to each check-in $x_i$, so two visits to the
same POI may enter the model with different representations.
\end{definition}

Definition~\ref{def:fund:placelevel} carries one limitation for this research: a
weekday morning and a Saturday night at the same place have identical inputs at the
representation level. CTLE shows an alternative by learning context-aware and
time-aware location embeddings whose vectors change with the
visit~\cite{lin2021ctle}. Check2HGI reaches the check-in level of
Definition~\ref{def:fund:placelevel} by adding the check-in below the place in the
hierarchy.
```

Both cross-references pointed at the same label while meaning opposite halves of the block: `:242`
meant the place embedding, `:247` meant the check-in level. That is the defect.

## 3 · What I changed, and where

One edit to the definition block and its following paragraph, plus one source-level reflow.

**(a) The split.** At what was `:235-248` (now `:371-397` after the parallel item's insertions
above it; see §7):

- `\begin{definition}[Place embedding]\label{def:fund:placelevel}` — sentence one only.
- The limitation paragraph, ending at the CTLE citation, now sits BETWEEN the two definitions.
- `\begin{definition}[Check-in-level representation]\label{def:fund:checkinlevel}` — sentence two.
- The Check2HGI sentence follows the second definition as its own paragraph.

**This is a split, not a rewrite.** The wording of each half is round 10's, character for character,
with ONE deletion I am naming because it is a wording change however small: the word **"instead"** was
dropped from "A check-in-level representation *instead* assigns one vector...". It contrasted with a
sentence that is no longer in the same block, so it had lost its antecedent. Nothing else in either
sentence changed.

**The two cross-references are repointed to the half each one means:**

| Site | Now reads | Means |
|---|---|---|
| `:385` | `Definition~\ref{def:fund:placelevel} carries one limitation...` | the PLACE half. Correct: the limitation is the place embedding's. |
| `:396` | `Check2HGI reaches the check-in level of Definition~\ref{def:fund:checkinlevel}` | the CHECK-IN half. Correct: it is the level being reached. |

**(b) Readability, item 4, §2.2 only.** Round 10's report names exactly two compactions inside this
subsection: the merge itself (undone above) and "the duplicated place-embedding limitation sentence"
(that sentence is the limitation paragraph, which is present and now correctly placed). The other
three compactions it lists, the inlined `H_i`, `g_cat` and `f_cat`/`f_reg` display equations, are in
OTHER subsections and belong to the parallel item, so I left them alone per the "do not roam"
instruction. That left one thing worth doing in this subsection: a ragged source line under
"The check-in level" was reflowed (`It adds a fourth level below the` / `place and learns...` →
`It adds a fourth level below the place and learns one vector per visit`). Source formatting only;
the rendered text is identical.

**(c) A provenance comment** above the split records why the merge existed and why it was undone,
so the next agent does not re-merge it to save a page.

## 4 · Renumbering verified, not assumed

- **Labels vs references.** `grep -n 'label{def:fund\|ref{def:fund'` over the live file: twelve
  labels, and every `\ref` target is in the label set. Both new labels resolve. `make check`'s
  unresolved-`\ref`/`\cite` gate (which reads the compiled `.log`) passed.
- **No literal "Definition 2.N" in prose.** `grep -vn '^[[:space:]]*%' chapters/2_fundamentals.tex |
  grep -n 'Definition [0-9]'` returns nothing, rc=1. Comments stripped first, per GUARDRAILS V4.
  Every number comes from `\ref`.
- **R10-cosine probe.** It pins `def:fund:conflict`, which the split does not touch. Confirmed from
  the gate's own output rather than by assumption: `/tmp/r11_check2.log:172` reads
  `holds       R10-cosine gradient conflict is DEFINED with the cosine in Chapter 2, not only named`.
- **The rendered PDF.** Text layer extracted from `build/main.pdf` (pypdfium2, regex over
  `Definition N.N (Head)`), not eyeballed. Twelve definition heads, in order:

  | # | Head | Page |
  |---|---|---|
  | 2.1 | Check-in | 18 |
  | 2.2 | Check-in history | 19 |
  | 2.3 | Category classification | 19 |
  | 2.4 | Next-category prediction | 19 |
  | 2.5 | Next-region prediction | 19 |
  | 2.6 | Next-place prediction | 19 |
  | 2.7 | Place embedding | 22 |
  | 2.8 | Check-in-level representation | 22 |
  | 2.9 | Hard parameter sharing | 24 |
  | 2.10 | Soft parameter sharing | 25 |
  | 2.11 | Negative transfer | 25 |
  | 2.12 | Gradient conflict | 26 |

  Sequence `2.1 … 2.12`, consecutive, no gap and no duplicate. The former 2.8-2.11 became 2.9-2.12
  exactly as the brief predicted.

## 5 · The four commands (rule 6), run separately from `src/`, exit codes read directly

| Command | rc |
|---|---|
| `make defense` | **0** (`pages=108 tex_errors=0`) |
| `make ppgc` | **0** (`pages=109 tex_errors=0`) |
| `make check` | **0** (25 gates, all under threshold) |
| `make selftest` | **0** (`PROVEN 5 | FAILED 0 | UNPROVEN or HALF 12 of 17`) |

`make academico` is not one of rule 6's four commands, but I ran it too (**rc=0**, `pages=105
tex_errors=0`) because the page gate reads its log and that log was stale; see the correction in §7.
`make check` and `make selftest` were then re-run against the fresh log, both rc=0 again.

None was piped into `grep`; each exit code is the make invocation's own.

**A measurement caveat I am naming rather than smoothing over.** My first `make defense` after the
split reported 107 pages; the next two reported 108 with no source change between them, so 107 was an
unconverged first pass and **108 is the settled count**. I report the stable value, from two
consecutive agreeing runs.

## 6 · Page counts, and why I did NOT run the sync script

| Build | Baseline (28df0cc0 record) | Now |
|---|---|---|
| defense | 106 | **108** |
| academico | 103 | **105** |
| ppgc | 107 | **109** |

The document grew by two pages, of which **this item accounts for one** (round 10 measured the merge
as worth exactly one page, and my first split build moved 106 → 107). The second page is the parallel
item's; see §7.

**`sync_page_counts.py --write` was NOT run by me, and did not need to be.** By the time I finished,
`CLAUDE.md`, `PLAN.md` and `src_utils/codex_reviewer.md` already carried 108/105/109: the agent running
the parallel item had synced them. I verified rather than assumed, running the script read-only from
`src/`:

```
$ python3 ../src_utils/sync_page_counts.py
measured from the build logs: defense 108 pp, academico 105 pp, ppgc 109 pp
all recorded page counts agree with the build      rc=0
```

Writing would have been a no-op. Note that this is the authorized case the brief describes (the author
lifted the budget, so growth is genuine), and had the records been stale I would have written them.

## 7 · A concurrent writer in the same file, disclosed

`src/chapters/2_fundamentals.tex` was modified at 03:01:37 by another agent while I was building, after
my edit. `git status` shows that agent also touched `CLAUDE.md`, `GLOSSARY.md`, `PLAN.md`,
`src/chapters/1_introduction.tex`, `src/dissertacao.pdf` and `src_utils/codex_reviewer.md`. Consequences
I checked rather than assumed:

- My split survived intact: both labels, both repointed references and the provenance comment are
  present in the live file (`grep`, after their edit).
- Their insertions are ABOVE mine, which is why my block moved from `:235` to `:371`. The line numbers
  in §3 are the live ones as of this report and will move again.
- All three targets were rebuilt AFTER their 03:01:37 edit, so every exit code and page count in §5
  and §6 describes the combined tree, not my edit alone. The 108/109 counts and the twelve-definition
  PDF extraction are from those post-edit builds.

  **A correction to an earlier version of this sentence, which was wrong and was caught in review.**
  It read "I rebuilt all three targets AFTER their 03:01:37 edit" at a point when I had rebuilt only
  `defense` and `ppgc`. I never ran `make academico`, so the 105 pp in §6 was, at that moment, read
  by `make check` and `sync_page_counts.py` out of `build/main_academico.log` written at 03:00:31,
  i.e. BEFORE the 03:01:37 source edit: a stale log certifying a page count for a document state that
  no longer existed. This is exactly the GUARDRAILS §4b V2 failure (writing from what I intended the
  check to do rather than from what ran), and it is the second time in this file's history that a
  page count was certified from a log rather than a build. `make academico` has since been run:
  rc=0, `pages=105 tex_errors=0`, log written 03:21:23. The VALUE 105 was correct; the STATEMENT
  about how it was obtained was not, and the value is only now actually verified. `make check` and
  `make selftest` were re-run afterward against the fresh log, both rc=0, and the page gate again
  reports `defense 108 pp, academico 105 pp, ppgc 109 pp` with `all recorded page counts agree with
  the build`.

  Timestamps after the correction, all three logs postdating the 03:01:37 edit:

  | File | Modified |
  |---|---|
  | `chapters/2_fundamentals.tex` (last source edit) | 03:01:37 |
  | `build/main.log` | 03:05:52 |
  | `build/main_ppgc.log` | 03:08:08 |
  | `build/main_academico.log` | 03:21:23 |
- `git diff --stat` on the chapter reports 169 insertions / 20 deletions, which is BOTH items' work.
  Mine is the definition block, its two references, the reflow and the comment; the rest is theirs.

## 8 · Sources opened this session

No external source was opened. Budget was 6; **0 used**. This item is a structural split of existing
in-repo wording with no new claim, no new citation and no new number, so no external record was needed.
Everything quoted above comes from files in this repository: `src/chapters/2_fundamentals.tex`,
`src_utils/_round10/29_ch2_definitions.md`, `AGENT_GUARDRAILS.md`, `WRITING_LAW.md`, `GLOSSARY.md`,
`src_utils/sync_page_counts.py`, `src_utils/check_audit_claims.py`, and the built `build/main.pdf`.
Unreached, and disclosed as such: the CTLE record (`lin2021ctle`) was not re-opened, because the
sentence citing it is unchanged round-10 prose that I moved rather than wrote.

## 9 · Glossary position (no proposal needed)

Both terms are already registered, so the fail-closed rule is satisfied without a new row:

- `GLOSSARY.md:135` — `place embedding | representação (embedding) em nível de POI`
- `GLOSSARY.md:134` — `check-in-level representation | representação em nível de check-in`

The two new definition HEADS use exactly those registered names. I propose no new term.

## UNFINISHED

Nothing from this item is outstanding. Three things I did not do, each deliberate:

1. **The other three round-10 compactions** (the inlined `H_i`, `g_cat` and `f_cat`/`f_reg` display
   equations) are NOT restored. They live outside §2.2 and the brief assigns them to the parallel item.
2. **No second-pass critic** re-read the split. Round 10's report already carries this same gap for the
   eleven definitions; the split does not widen it, but it is not closed either.
3. **Nothing is committed.** The working tree carries both items' edits interleaved, and I did not
   commit or stage anything.
