# 44 — Decompaction: undoing what the page budget forced in round 10

Baseline commit: **8f17f294**. One file edited: `src/chapters/2_fundamentals.tex`. No other file in
the tree was touched.

---

## 1 · The author's instruction, verbatim

> "Sim corrija essa compactacao"

> "Corrija a compactacao que a suspensao do orcamento de paginas deixou pendente (as tres equacoes
> em display que foram inlinadas e os quatro lead-ins encurtados na rodada 10)."

Approved without qualification.

---

## 2 · What round 10 claims it compacted, and what the live chapter actually held

Round 10's report is `src_utils/_round10/29_ch2_definitions.md`. Its §4 ("Thinness") is the only
place the compaction is itemized, and it reads:

> "- three display equations inlined (`H_i`, `g_cat`, `f_cat`/`f_reg`) — the definition environment
>   already sets them off;
> - Definitions 2.7 and the former 2.8 merged into one two-sentence block, since the second was the
>   first's complement;
> - four lead-in sentences shortened or cut (\"Three definitions separate the targets\", \"One further
>   target must be named because…\", \"and it is a measured quantity rather than a figure of speech\",
>   the duplicated place-embedding limitation sentence)."

The same report's §7 item 2 records the merge as held "to hold the page count at 106", and §4 opens
"The definitions cost one page on the first build (defense 106 → 107). I did not accept that: five
compaction passes brought it back."

**Every claim was verified against the live file rather than taken from the report.** The mechanism
was `git diff aaf4e7eb d7e8c598 -- src/chapters/2_fundamentals.tex`, the round-10 commit against
its parent, read in full, then each surviving string located in the live file.

| Round 10's claim | Found in the round-10 diff? | State in the LIVE file before my edit | Action |
|---|---|---|---|
| `H_i` display inlined | Yes: `-\begin{equation}` / `-    H_i=(x_{i-\ell},\ldots,x_{i-1}),` | inline, `:67` | **restored** |
| `g_cat` display inlined | Yes: `-    g_{\mathrm{cat}}(\mathbf{e}_p)\longrightarrow c_p.` | inline, `:103` | **restored** |
| `f_cat` / `f_reg` displays inlined | Yes, two removed `equation` blocks | inline, `:110` and `:115` | **restored** |
| lead-in "Three definitions separate the targets" | Yes, replaced by "The first target is static and reads a place" | clipped form live at `:98` | **restored** |
| lead-in "One further target must be named because…" | Yes, replaced by "The models this section reviews were built for a fourth target." | clipped form live at `:135` | **restored** |
| lead-in "…and it is a measured quantity rather than a figure of speech" | Yes: the live sentence ended at "a measured quantity." | clipped form live at `:835` | **restored** |
| lead-in "the duplicated place-embedding limitation sentence" | Yes, the sentence "Every place embedding has the same limitation for this research: it gives a place one fixed vector for all visits." was removed | **GONE, and legitimately so** | **not restored, see §3** |
| Definitions 2.7 + 2.8 merged | Yes | **already undone** by round 11 item 31 | nothing to do |

**Anything the report claims that I could not find: none.** All seven compaction items are traceable
to the round-10 diff.

**Anything I found that the report does not claim.** Round 10's diff also rewrapped two paragraphs
without changing their words (the cross-attention paragraph at old `:486-489`, and the
"Shared parameters can produce…" sentence), and it split the balancer paragraph into two. Neither is
a compaction: the first is line-wrapping, the second is the GER-09 lineage rewrite the report does
describe in §3. I left both alone. Round 10's §4 also says "shortened **or cut**", and the fourth
lead-in was the one that was cut rather than shortened; the distinction turns out to matter (§3).

---

## 3 · Why the fourth lead-in is NOT restored

Round 10's fourth item is not a clipped lead-in. It is a deduplication, and the passage it belonged
to has since been rebuilt by another round.

The removed sentence was:

> "Every place embedding has the same limitation for this research: it gives a place one
> fixed vector for all visits."

Its content became the body of Definition 2.7 itself ("A place embedding assigns one vector
$\mathbf{e}_p$ to each POI $p$, so every check-in at that POI enters the model with the same
representation"), and round 11 (item 31) then split that merged definition head into Definition 2.7
(Place embedding) and Definition 2.8 (Check-in-level representation), moving the limitation paragraph
to sit between them. The live text at `:390-401` carries a round-11 provenance comment recording
exactly that. Restoring the sentence now would state the same limitation three times in eleven lines,
which is a readability regression, not a decompaction. The brief's own rule applies: *"If you find
your target GONE because another agent removed the surrounding prose, report that instead of
restoring it in a new place."* Reported here rather than restored.

That leaves **three equations and three lead-ins** actually restored, against the brief's expectation
of three and four. The brief's belief about the three equations ("a hidden-state combination, a
category gradient, and a pair of task-head functions") was **half right and is corrected here**: the
first is not a hidden-state combination but the check-in **history** $H_i$, and the second is not a
gradient but the static **category classifier** $g_{\mathrm{cat}}$. The pair of task-head functions
is correct. The names in round 10's own report (`H_i`, `g_cat`, `f_cat`/`f_reg`) are what the diff
shows.

---

## 4 · The LIVE text before each edit, and what it says now

Each target was re-read in the live file immediately before the edit (`read_file` on the specific
span, never a line number carried in from the brief), and each edit carried enough surrounding
context to be unambiguous. Line numbers below are **post-edit**.

### 4.1 · `H_i`, the check-in history — `:81-85`

Before (live, `:65-69`):

```
\begin{definition}[Check-in history]\label{def:fund:history}
The check-in history of length $\ell$ preceding check-in $x_i$ is the ordered sequence
$H_i=(x_{i-\ell},\ldots,x_{i-1})$. A target label is withheld from it when one of the
sequential tasks is trained.
\end{definition}
```

After: the sequence is a display `equation` (unlabelled, punctuated with a period since the sentence
ends there), and the withheld-label sentence starts a new line. Renders as **(2.2)**.

### 4.2 · `g_cat`, category classification — `:117-121`

Before (live, `:102-105`):

```
Category classification predicts the category of a POI from its representation:
$g_{\mathrm{cat}}(\mathbf{e}_p)\longrightarrow c_p$. At evaluation time, POI $p$ is
held out from the classifier-training fold, so ``unknown POI'' means unknown to the
classifier, not necessarily absent from the graph used to learn $\mathbf{e}_p$.
```

After: display equation, then the held-out-POI sentence. Renders as **(2.3)**.

### 4.3 · `f_cat` and `f_reg`, the two sequential tasks — `:127-131` and `:134-138`

Before (live, `:108-116`): both were inline after the colon, each as the definition's last clause.
After: each is a display equation inside its own definition. Renders as **(2.4)** and **(2.5)**.

### 4.4 · Lead-in, §2.1.1.2 — `:113`

Before (live, `:98-99`): "The first target is static and reads a place; the other two are sequential
and read a history."
After: "Three definitions separate the targets. The first is static and reads a place; the other two
are sequential and read a history."

### 4.5 · Lead-in, §2.1.2 — `:150-151`

Before (live, `:135`): "The models this section reviews were built for a fourth target."
After: "One further target must be named, because it is the task that the models this section reviews
were built to solve."

Round 10 quotes the original as "One further target must be named because…" and does not give the
rest, so the subordinate clause is written fresh rather than reconstructed. **One draft was discarded
before building:** "…because the models this section reviews were built for it rather than for the
three tasks above" would have added a `rather than` to a document whose negative-parallelism density
is under a standing ceiling (`src_utils/check_negative_parallelism.py`, 3.60 per 1,000 prose words).
The clause above avoids the construction. Gate green after the edit.

### 4.6 · Lead-in, the gradient-conflict subsection — `:850-851`

Before (live, `:835-836`): "Gradient conflict describes one source of negative transfer, and it is a
measured quantity."
After: "…and it is a measured quantity rather than a figure of speech." This is round 10's own
wording, restored verbatim from its report.

### 4.7 · Provenance comment — `:51-64`

A comment block above `\subsubsection{Check-ins and histories}` recording what was restored, that
round 10 did it for the page count and said so, why the fourth item is not restored, and why no
restored equation carries a label.

**On labels.** None of the three restored equations is labelled. The chapter's convention is that a
label exists because something points at it: the nine labelled equations are `eq:fund:check2hgi`,
`-disc`, `-term`, `eq:fund:mtl-total`, `eq:fund:cosine`, `eq:fund:macro-f1`, `eq:fund:acc10`,
`eq:fund:joint-selection`, and the chapter's one prose reference is `Equation~\ref{eq:fund:mtl-total}`
at `:936`. The check-in tuple at `:81` (2.1) was already an unlabelled display equation before this
round and stayed one. Adding labels nothing references would create unused labels, and the three
concepts are already referenceable as Definitions 2.2 through 2.5, which is how the prose points at
them (`Definition~\ref{def:fund:catclf}` and the rest at `:143-145`).

---

## 5 · Sources opened this session

**None external, and none needed.** This item restores wording that already exists in this
repository: round 10's report for what it compacted and in what words, the round-10 commit diff
(`aaf4e7eb..d7e8c598`) for the exact LaTeX that was removed, and the live chapter for where it went.
No citation was added, moved, or reworded; no number was written; `references.bib` was not touched.
The one clause written fresh (§4.5) is a connective sentence about this document's own structure and
carries no claim about any cited work.

Internal records read: `AGENT_GUARDRAILS.md` §0-§4, `WRITING_LAW.md` (in full),
`src_utils/_round10/29_ch2_definitions.md`, `src_utils/check_audit_claims.py` (the R10 and R11 probe
blocks), `src_utils/check_negative_parallelism.py` (its ceiling and its rule about raising it),
`src_utils/check.sh`, `src/Makefile`, and `src_utils/_round12/42_hgi_trim_and_07.md` §on the failing
gate.

---

## 6 · The six exit codes, run separately from `src/`, each read directly

One command per invocation, `echo "RC=$?"` immediately after, never piped into another program.

```
make defense     RC=0    build/main.pdf              104 pages, tex_errors=0
make academico   RC=0    build/main_academico.pdf    101 pages, tex_errors=0
make ppgc        RC=0    build/main_ppgc.pdf         105 pages, tex_errors=0
make extra       RC=0    build/main_extra.pdf         26 pages, tex_errors=0
make check       RC=2    91 of 91 probes hold; 0 claim(s) not applied; ONE gate red, NOT MINE (§7)
make selftest    RC=0    every required checker fires on its defect and is silent on the clean fixture
```

### Page counts

| Build | Baseline 8f17f294 (per the brief) | After this round's tree |
|---|---|---|
| defense | 108 | 104 |
| academico | 105 | 101 |
| ppgc | 109 | 105 |
| extra | 22 | 26 |

**These deltas are not mine to claim.** Two other agents are editing this same file and a third
edited `apx_a_contributions.tex` (76 lines deleted) during this round; the defense document is four
pages shorter than baseline while my own edits add six display equations' worth of vertical space and
three sentences. My contribution to the count is a small increase, and the net decrease comes from
the concurrent trims. Measure per-agent attribution from the individual diffs, not from this table.

### `sync_page_counts.py --write`: NOT run, and it was not needed

The page-count gate is **green**: `make check` prints "measured from the build logs: defense 104 pp,
academico 101 pp, ppgc 105 pp" followed by "all recorded page counts agree with the build". Another
agent in this round had already synced the records to the shorter document. Rule 6 authorizes
`--write` when that gate goes red; it did not, so I did not run it.

---

## 7 · The one red gate, and why it is not mine

`make check` returns 2 on a single gate, "the author-facing verification commands actually return
what they claim":

```
FAIL     VERIFY_LIST.md: python3 -c "
         output does not contain 'repair_in_prose: True'
```

Attribution, measured rather than assumed:

```
git show 8f17f294:./src/chapters/apx_a_contributions.tex \
  | grep -c "stratified its folds by sample rather than by user"   -> 1   (present at baseline)
grep -c "stratified its folds by sample rather than by user" \
  src/chapters/apx_a_contributions.tex                             -> 0   (absent now)
git diff --stat 8f17f294 -- src/chapters/apx_a_contributions.tex   -> 12 insertions, 76 deletions
```

The block in `src_utils/_round6/VERIFY_LIST.md:466` asserts a sentence in
`chapters/apx_a_contributions.tex`, which a concurrent agent deleted this round. That file is not one
my item names, so I did not touch it. The same failure is independently reported by the agent who
wrote `src_utils/_round12/42_hgi_trim_and_07.md` §on `make check`, from a run that predates my edit,
which confirms the gate was already red before I arrived. **Flagged for whoever owns
`apx_a_contributions.tex`.** I did not weaken, reword, or repoint the probe, and I did not edit
`VERIFY_LIST.md`.

Every probe that reads Chapter 2 holds, including the two that pin this chapter's definition
machinery (`R10-defenv`, `R10-cosine`) and the one that pins the round-11 split I decided not to
re-merge (`R11-def27`, watching `\label{def:fund:checkinlevel}`).

---

## 8 · Verified in the RENDERED PDF, not in the source

`build/main.pdf` text layer via `pypdfium2`, both directions asserted wherever both apply.

**New wording present.** Page 19 renders, in this order (extracted text, `\r\n` shown as line
breaks):

```
Definition 2.2 (Check-in history). The check-in history of length ℓ preceding check-in 𝑥𝑖 is the
ordered sequence
𝐻𝑖 = (𝑥𝑖−ℓ, . . . , 𝑥𝑖−1).                                                                  (2.2)
A target label is withheld from it when one of the sequential tasks is trained.
2.1.1.2 The three experimental tasks
Three definitions separate the targets. The first is static and reads a place; the other two
are sequential and read a history.
Definition 2.3 (Category classification). Category classification predicts the category of a POI
from its representation:
𝑔cat(e𝑝) −→ 𝑐𝑝.                                                                             (2.3)
At evaluation time, POI 𝑝 is held out from the classifier-training fold, so “unknown POI” means
unknown to the classifier, not necessarily absent from the graph used to learn e𝑝.
Definition 2.4 (Next-category prediction). Next-category prediction maps a check-in history to
the category of the next visit:
𝑓cat(𝐻𝑖) −→ 𝑐𝑖 .                                                                            (2.4)
Definition 2.5 (Next-region prediction). Next-region prediction maps a check-in history to the
region of the next visit:
𝑓reg (𝐻𝑖) −→ 𝑟𝑖 .                                                                           (2.5)
```

All three restored equations are on their own line, centered, and numbered (2.2, 2.3, 2.4, 2.5 with
2.1 the pre-existing check-in tuple). The two restored lead-ins on that page open on full sentences.
"rather than a figure of speech" renders on page 26.

**Old wording absent**, all three searched against the full text layer and all three returning no
page:

| Assertion | Result |
|---|---|
| "The first target is static and reads a place" | absent |
| "The models this section reviews were built for a fourth target" | absent |
| "negative transfer, and it is a measured quantity." | absent |

**No unresolved reference.** `full.count("??")` over the whole 104-page text layer returns **0**, and
Chapter 2's equation numbers run (2.1) through (2.13) with no gap. The one prose equation reference
resolves: page 26 renders "sets the weights 𝑤𝑘 of Equation 2.9". The typographic apostrophe was
accounted for: the assertions above avoid ASCII apostrophes, and the extracted text confirms the
document uses ’ and “ ”.

---

## UNFINISHED

Nothing from my item is outstanding. Three things are open, none of them work I was permitted to do:

1. **The red `make check` gate belongs to `apx_a_contributions.tex`** (§7). Either the deleted
   sentence returns to that appendix, or `src_utils/_round6/VERIFY_LIST.md` block 6 is retired by
   whoever owns that deletion. Not my file; not repointed by me.
2. **Round 10's fourth compaction item is deliberately not restored** (§3), which is a judgment the
   author may overrule. If he wants the place-embedding limitation sentence back as a third
   statement of the same limitation, it is one line at `:395`, and the round-11 provenance comment
   there should be amended to say so.
3. **Page-count attribution across the three concurrent agents in this file is not established**
   (§6). The table records the tree's counts, not mine; a per-agent number would require serializing
   the round, which I could not do.
