# AGENT_GUARDRAILS.md — process law for AI-assisted writing (v1, 2026-07-18)

> **Scope.** How humans and agents WORK on this dissertation: what an agent may and may not do,
> the verification gates every chapter passes, and the disclosure obligations. The word-level
> law is [`WRITING_LAW.md`](WRITING_LAW.md); this file is about process. It is evidence-based:
> §8 lists the 2025–2026 measurements each rule answers to. The rules are deliberately
> fail-closed — when in doubt, an agent STOPS and flags, it does not improvise.

---

## 0 · The three prime directives

1. **Agents draft; the author owns.** No chapter, claim, or number is "done" until the author
   (Vitor) has read and approved it. AI cannot be an author (CNPq/ICMJE/COPE consensus); the
   author is accountable for every word, so every word must be verifiable by him.
2. **Nothing enters the document from model memory.** Every citation, number, name, date, and
   quoted claim traces to a file in this repo or a source opened during the session. If the
   provenance is "the model recalled it", it does not go in.
3. **Fail closed.** A claim that cannot be verified is removed or flagged `[VERIFY: …]` — never
   smoothed over. An agent that is uncertain says so in its handoff note; self-reported success
   is not trusted (the author audits independently — standing project feedback).

## 1 · Citation integrity protocol

**Why (measured):** GPT-class models fabricate ~20% of citations in literature-review settings
(6–29% by topic sparsity; ~7–8% even with web search); fabricated citations in published papers
rose about tenfold from 2023 to early 2026 (1/2,828 papers → 1/277), still accelerating; 100
hallucinated citations were found in accepted NeurIPS 2025
papers that 3–5 reviewers each missed; arXiv now bans authors over it. Existence-checking alone
misses the two dominant subtler classes: partial attribute corruption (~27%) and
claim-not-supported ("semantic") errors.

**The rules:**

- **R1. No bib entry from memory, ever.** A reference enters `references.bib` only with (a) a
  resolvable identifier (DOI / arXiv ID / ACM-DL / SBC-SOL / publisher landing page) checked
  against the source of record (Crossref/OpenAlex/publisher), AND (b) the PDF or landing page
  actually opened, AND (c) the cited claim located in the source (page/section noted in a bib
  comment). The MobiWac bibliography (every entry web-verified, with quotes in bib comments) is
  the template — and the preferred donor: reuse its entries verbatim where the dissertation
  cites the same works.
- **R2. Attribute fidelity.** Author list, venue, year, pages copied from the source of record,
  not retyped from another paper's bibliography. Venue names per the bibliography style chosen
  in [`TEMPLATE.md`](TEMPLATE.md); one accuracy check per citation ("describe cited systems as
  their authors describe them").
- **R3. Claim-support audit.** Before any advisor handoff, an adversarial pass samples ≥20% of
  citations (100% for new-this-pass entries) and verifies the SENTENCE citing them is actually
  supported — not merely that the reference exists.
- **R4. Inherited errata stay fixed.** The known CBIC/CoUrb citation errata (owner list:
  NORTH_STAR §4) are corrected in the dissertation bibliography regardless of the errata policy
  chosen for chapter text (NORTH_STAR §5.7).
- **R5. AI output is not a source.** Never cite a model or its answer as evidence; never launder
  a model claim through a real-looking citation.

## 2 · Number integrity protocol

**Why (measured):** models corrupt numbers even when given correct input (numeric spans are a top
faithfulness-error class); in whole-paper generation studies, the most polished agent drafts
carried the MOST fabrications (>10/paper), and 57% of one system's papers contained wrong or
hallucinated numerical results. Polish anti-correlates with groundedness — which is exactly the
failure mode a good-looking dissertation chapter invites.

**The rules:**

- **N1. Single source of truth per chapter.**
  - Ch.5 (MobiWac): `docs/studies/closing_data/RESULTS_BOARD.md` §1 (via its §3 file map to the
    JSONs) + the claim whitelist in `articles/[mobiwac]/PAPER_PLAN.md §3`. Never from prose, never
    from memory.
  - Ch.3 (CBIC) and Ch.4 (CoUrb): the published papers' own tables (with the documented errata,
    NORTH_STAR §4, as the only sanctioned corrections — CoUrb's audited win-count/means come from
    `articles/CoUrb_2026/slides/judge_feedback.md`).
  - Frame chapters: may only repeat numbers already sourced in a chapter, with the same hedges.
- **N2. Agents quote; they do not compute.** No mental arithmetic, rounding, aggregation,
  percentage conversion, or delta-taking in prose. Derived quantities come from a script
  committed to the repo (or the existing analysis scripts), then are quoted.
- **N3. Every numeral is traceable.** Any number an agent writes must be traceable to its source
  file in the handoff note (a "numbers ledger": value → file → field). Numbers without a ledger
  line fail the gate.
- **N4. Numeral-extraction audit (the gate).** Before advisor handoff, extract every numeral+unit
  from the chapter (script or manual sweep) and match each against its ledger source, exact or
  declared-rounding. Orphan numbers block the handoff. Also cross-check: abstract vs body,
  captions vs table content, prose interpretation vs the statistic named (the sciwrite-lint
  check classes).
- **N5. Convention named.** Every reported cell states its convention (metric, selection rule,
  n, seeds×folds) per WRITING_LAW §3; the MobiWac joint-best vs diagnostic-best distinction must
  never blur.

## 3 · Claim registry

- **C1. The whitelist governs.** Scientific claims about MobiWac results come only from
  `PAPER_PLAN.md §3` (CAN-say / must-NOT-say) + the decisions ledger in
  `articles/[mobiwac]/CLAUDE.md §3`. Claims about CBIC/CoUrb come from their published texts,
  time-indexed per WRITING_LAW §3.
- **C2. New claims need sign-off.** Any claim not derivable from the registry (including
  "obvious" connective claims in the frame, e.g. about what the arc "shows") is proposed in the
  handoff note and enters the text only after the author approves. The Introdução Geral's
  arc-narrative sentences are claims — they get the same treatment.
- **C3. Never-cite lists are absolute** (inherited from the MobiWac board): STAN v4-collapse
  numbers, ReHDM v2 row, VOID fp16/bf16 collapsed cells, pre-bugfix findings flagged in
  `docs/PAPER_FINDINGS.md`.
- **C4. BRACIS containment.** **No BRACIS result, number, or claim appears anywhere in the
  dissertation**, and its C2-era region-cost claim is never reissued. The rejected manuscript is not
  disclosed to the reader at all.
  > **Amended 2026-07-27 by author decision** (NORTH_STAR §5 item 11; the matching edit was owed to
  > this file and is applied here). C4 previously required the *opposite* of its current form: it
  > mandated a containment device, citing BRACIS "only as an earlier unpublished iteration". Appendix
  > A §A.2, which carried that disclosure, was removed on the author's grounds that the trail of a
  > rejected-then-reworked manuscript adds reading complexity without serving the reader, and that
  > reworking after a reject is common practice with the conclusion unchanged. The *prohibition* half
  > of C4 survives and is now the whole rule; the *disclosure* half is void. Nothing in the document
  > asserts a correction relative to that manuscript (swept 2026-07-27: every "earlier" or
  > "corrected" passage names CBIC, CoUrb, the submitted MobiWac manuscript, or a development-time
  > data preparation as its own antecedent).

## 4 · Long-form failure-mode countermeasures

**Why (measured):** long outputs degrade (repetition in ~45% of long generations; quality
collapse past ~2k words in most models); models retrieve mid-context constraints unreliably
(context rot), so "keeping the whole thesis in context" silently drops notation defined two
chapters earlier; models reuse discourse skeletons across sections; register drifts across
model versions over a months-long project.

**The rules:**

- **L1. Bounded drafting.** No agent drafts a whole chapter in one pass. Work section-by-section
  against the approved outline (NORTH_STAR §3), ≤ ~1,500 words per drafting unit, each unit
  reviewed before the next.
- **L2. Consistency lives in files, not context.** Notation, canonical names, and claim scopes
  are enforced by checking against WRITING_LAW §2, the term registry
  [`GLOSSARY.md`](GLOSSARY.md) (a term not in the registry may not be used; agents propose,
  the author approves, the entry lands BEFORE the term does), and the claim registry — never by
  trusting that the agent "remembers" earlier chapters. Every session re-reads the law files
  first (CLAUDE.md §0 order).
- **L3. Cross-chapter duplication check.** Before advisor handoff: n-gram/near-duplicate sweep
  across chapters (paper-chapters legitimately share background; the frame must not repeat
  itself or them beyond the sanctioned recap subsections).
- **L4. Cross-reference lint.** Every `\ref`/`\cite`/section pointer resolves; no "as discussed
  in Section X" pointing at the wrong target (a known Viegas defect class).
- **L5. Translation fidelity gate** (if CoUrb is translated): a separate verification pass
  compares source PT and target EN sentence-by-sentence for claim-strength drift — quantifiers,
  hedges, and numbers must map 1:1.
- **L6. Fresh-eyes audits.** Style and consistency audits are run by an agent that did NOT write
  the text under audit (or by the author), never self-certified by the drafting agent.

## 4b · Meta-claim protocol: claims about the WORK, not about the science

**Why this section exists, measured.** Round 6 (2026-07-28) ran 13.3 hours and produced 61 commits.
**Seventeen of the 61 were rework** — repairing something the round itself had broken or misclaimed —
and reading all seventeen, **fourteen were genuine rework worth 2.4 hours**, of which **twelve
(86%) share one property**:

> The wrong statement was never about the dissertation. It was about **the work**: what a check
> covered, what a command returned, how many files a sweep touched, whether a gate passed. Every one
> was written from what the agent *intended the check to do* rather than from re-reading what the
> check *actually printed*.

§1 and §2 protect the *science* (citations, numbers in the text). Nothing protected the *record of
the work*, and that is where the time went. The four root causes, with the count each cost:

| # | Root cause | Commits | The instance |
|---|---|--:|---|
| **R1** | The record described something other than what ran | 5 | A count command annotated `# 13` returned 15; a "sweep of every command" skipped four blocks and counted the skips as passes; a switch test printed one question and compiled a different case. |
| **R2** | The instrument was blind to, or narrower than, the claim built on it | 4 | `FPDFText_GetFontSize` reports the size declared *inside* an embedded object and ignores `\includegraphics` scaling, so it read 6.97 pt after a figure was rescaled; a line-based `grep` missed a `\path{}` sharing a source line, giving 8-of-12 for a true 9-of-13. |
| **R4** | An own count was wrong, or the filter did not match the question asked | 3 | A row count of 27 reported over 30 instances; a subsection-heading count measured "does it print" when the paragraph's subject was "was it published". |
| **R3/R5** | A stale inline revision was read as current; a new guard was itself defective | 2 | A flag raised against a correct claim by landing inside a section headed *(superseded)*; a new gate recursed into its own caller and reported zero skips while doing it. |

### The rules

- **V1. A number about the work carries the command that produced it.** Any count, coverage claim,
  or pass/fail statement written into a durable record (`PENDENCIAS.md`, a report, a commit message,
  a provenance comment) must be accompanied by the exact command that yields it, runnable from a
  stated working directory. *A number without its command is an opinion with a digit in it.*
- **V2. Re-read the output; do not paraphrase the intent.** Before writing "the sweep found N" or
  "all X pass", read the tool's own last lines again and copy from them. If the code that produced
  the claim contained a `continue`, a `skip`, an `except: pass`, or a filter, **the claim must name
  what was excluded and how many.** An unreported skip counted as a pass is the single most common
  defect in this repository's history.
- **V3. Distrust a clean result from an unvalidated instrument.** Before believing a measurement,
  ask what the instrument is blind to, and prove it can see the defect by running it against a case
  where the defect is present. This extends §7's gate rule from *gates* to **every ad-hoc `grep`,
  `wc`, API call, and one-off script** — most of R2 was ad-hoc, not a gate.
- **V4. Greps over this source strip comments first.** Non-negotiable, and its own rule because it
  caused three separate defects in one day. This tree carries dense provenance comments that *quote
  the very strings being searched for*, so an unfiltered sweep always over-reports. Use
  `grep -vn '^[[:space:]]*%' "$f" | grep '<pattern>'` — filter the **file**, not the `grep -n`
  output, because `:[0-9]*: *%` misses an indented comment.
- **V5. Anchor on the revision header, not the first matching line.** Records in `docs/studies/`
  keep their own superseded revisions **inline** under headings that say so. A search landing inside
  one finds a real sentence that stopped being true. Check the revision header before quoting.
- **V6. Correcting a number at its source is not correcting the claim.** After fixing any count,
  grep the *superseded value* across all durable docs. When 8-of-12 became 9-of-13, the old figure
  survived in four other places, including the author's own push list — which listed eight files
  when nine needed pushing.
- **V10. A per-item rate comes from that item's own start and end, never from the batch total.**
  On 2026-07-29 one job trained six datasets in sequence. To cost the dataset still running I divided
  the JOB's elapsed time (which already covered three completed datasets) by that dataset's epoch
  count, and published 1.49 min/epoch, 74 min/fold, 6.2 h. Its own stamps give 0.446 min/epoch and
  22.3 min/fold, a 3.3x error, and the difference decided the outcome: at the wrong rate a per-fold
  resubmission looked impossible, so a recoverable dataset was reported to the author and to a
  sub-agent as unrecoverable. In the same note another dataset's cost had been derived correctly from
  its own stamps; the convention simply was not applied to the one still in flight. Also: a progress
  bar in a harvested log can be tens of minutes stale (identical bar bytes reappeared 40 minutes
  after the job had been killed), so a bar showing progress is not evidence a process is alive — check
  the status file and the log's mtime against the clock.

**V11. A verification line belongs in a commit message only if the run happened AFTER the last edit
  in that commit, and only if its exit code was read directly.** Three instances now, and the third
  is the one that shows the rule is needed rather than obvious. Commit 324b1269 recorded page counts
  from a cell that had printed byte sizes. Commit e771d331 closed with "make check RC=0" for a tree
  in which the gate had not been run since five separate edits landed, and the only post-change run
  available had printed rc=2 -- a result I read, called "a real failure I need to diagnose", and
  reproduced, in the same span in which I wrote the sentence claiming it passed. That commit's own
  message elsewhere corrects an earlier claim for exactly this class, which is the part worth
  keeping: knowing the rule and citing it in the same paragraph did not stop me applying it to
  someone else's claim and not my own.
  The failure is not carelessness about the code, it is writing the ritual closing line of a commit
  message from the shape a good commit is supposed to have. Two mechanical defenses, both cheap:
  run the gate as the LAST action before `git commit`, in the same cell if possible; and if the
  most recent run you can point at predates any edit in the commit, write what you actually know
  ("gate not re-run after the merge") rather than the sentence you expect to be true.
  A build measurement and a gate measurement are separate claims. Both were in that one line, one
  was real and one was invented, and the true half is what made the false half read as credible.
  **FOURTH INSTANCE, 2026-07-30, inside the round convened to stop this, by the track auditing
  PENDENCIAS §2.** Commit `a07e547b` closes with *"bash src_utils/check.sh -> rc=0 (22 gates; page
  counts agree)"*. The suite exited **1** in the same cell that made the commit; that cell printed
  `DEFENSE_RC=0`, `TRACKER_RC=0`, `CHECK_RC=1`, and the sentence claiming 0 had already been drafted.
  Two details make it worth a fourth entry rather than a tally mark. FIRST, the red gate was
  `check_trapped_prose`, and what tripped it was **the comment block that commit itself added** -- so
  the false green covered precisely the gate the commit broke, which is the least recoverable place to
  put one. SECOND, eleven other exit codes in that cell were 0, and the eye stopped at the eleven that
  confirmed the expected shape; this is V12's mechanism (the reader stops at the first thing answering
  the question it came with) operating on a batch of exit codes rather than on a timing table. The
  content of that commit was sound and verified in the render, which is exactly what made the closing
  line read as credible -- the same pattern as the third instance.
  Two consequences, and the second is new. (1) When a cell runs several checks, the gate line must be
  written **after** reading the last one, and per-check, never as one summary verdict -- V13's
  per-item rule applies to exit codes in a batch, not only to findings. (2) A false claim already
  committed is **corrected in the history, not silently in a later commit body**: the repair here is
  `git notes add` on `a07e547b`, which plain `git log` displays under the message it corrects, chosen
  because rebasing would have rewritten the hashes of parallel tracks' commits stacked on top.
  **Notes do not travel on `git push` by default** -- publishing one needs
  `git push origin refs/notes/commits`, and a correction that stays local is not a correction.

- **V9. When the locating step returns nothing, the location is UNKNOWN — do not infer it from a
  neighbour.** On 2026-07-29 two `Overfull \hbox` warnings appeared in one build. The cell that tried
  to attach a source file to each printed nothing at all. Rather than report "file not established",
  I noticed that the SECOND box's line range matched a paragraph in the new appendix, and asserted
  that BOTH boxes were in that file — then sent a sub-agent a directive naming
  `apx_f_cosine.tex` "paragraph at lines 33-45" and prescribing a `width=` fix for a graphic there.
  Lines 33-45 of that file are an unbroken block of `%` comments; no paragraph can be typeset there,
  so the box was never there. One match does not license the pair. A `\hbox` whose body prints as
  `[][]` carries no text and therefore no evidence of its own file: TeX's own file stack in the log
  is the only source, and if you cannot resolve it, say so.

- **V8. A file is not data; open it and count.** Before reporting that any run — local or remote —
  produced a measurement, open one output file and count the non-empty cells in the column you came
  for. Exit status, file count, row count, byte size and header shape are **all satisfied by an empty
  result**. On 2026-07-29 three remote runs reported `rc=0`, each harvesting five well-formed CSVs
  with the target column entirely `NaN`, because the diagnostic that fills it is opt-in and defaults
  off; the claim "per-state data is arriving" was made on the strength of the shape. One `awk` line
  would have caught it. This applies with equal force to a `.parquet` you just wrote, a figure you
  just rendered, and a table you just extracted.

- **V7. A gate may not invoke its own caller, and a skip is never silent.** A new check must
  self-test in both directions (§7), must not call the suite that calls it, and must report skipped
  items with the reason. Ordering matters: a broad "is this a note?" test placed before a narrow
  guard will swallow the cases the guard exists for.


**V12. A tool that prints a diagnostic is not a tool you have read. Look at the output, not the exit
  code.** On 2026-07-29 `check.sh` gained a per-gate timing table. From that commit to 2026-07-30 the
  suite ran on the order of fifty times and **33 commit messages claimed `make check` RC=0**. Every
  one of those runs printed, at the top of that table, `264.144s` against a suite total of `265.288s`
  and eighteen other gates under a quarter-second. The number was on screen dozens of times. Nobody
  read it, including the agent that had added the table the day before specifically to make slow
  gates visible.
  What made it invisible was the exit code. `make check` returned 0, 0 satisfies the commit ritual,
  and the eye stops at the first thing that answers the question it came with. A diagnostic that
  only a curious reader will notice is a diagnostic that will not be noticed.
  Two consequences, and the second is the general one. FIRST: when a tool emits a measurement, give
  it a THRESHOLD and let it complain -- the timing table now flags any gate over 5 s, because a
  number that has to be interpreted by a human every run will not be. SECOND, and this outlives the
  timing case: **the author noticing a problem before the agent does is itself a finding about the
  agent's instrumentation, not just about the bug.** Twice in this project he asked "why is this
  slow" and twice the answer was a large defect sitting in plain output. When that happens, fix the
  defect AND ask what should have surfaced it unprompted.

**V13. A per-item verification must be stated per item, and a claim that justifies new work must be
  checked HARDER than one that does not.** Two instances in one hour, 2026-07-30, and they share a
  root.
  FIRST, the batch claim. A commit fixing seven tools closed with "SIX MORE, each verified firing".
  Five were genuinely exercised. Two -- a scope floor and a missing-source branch -- had only been
  read in the diff, and one of them I had explicitly noted was exiting at argparse before its new
  logic ran, then counted as verified anyway. The summary line borrowed the credibility of the five
  for the two. If verification is per item, say it per item, or the strongest evidence in the batch
  launders the weakest.
  SECOND, the motivated claim. The commit introducing a new self-test runner asserted that "not one"
  of the twenty-one tools had ever been tested against its own defect. False, and contradicted by
  the gate script I had read that session: six checkers run internal self-tests before reporting,
  and one has a separate regression suite of real shipped defects that the gate executes. My own
  summary in the same turn said "five with no self-test at all", which presupposes the others have
  them -- the blanket claim contradicted my own prose a paragraph away.
  The mechanism is worth naming because it is not carelessness. A new tool is easier to justify
  against a total absence than a partial one, so the overstatement flattered the thing I had just
  built. **When a claim's function is to justify work you want to do, that is the claim to verify
  first, not last.** The real gap was narrower and still sufficient; stating it accurately cost
  nothing and would have been more persuasive.
  THIRD INSTANCE, in the artifact written to fix the second. Correcting the "not one had ever been
  tested" overstatement, I moved `sweep_guard` into the untested column on the reasoning that
  check.sh's "OK (4 self-tests)" must cover a substituter rather than the guard. I never opened
  sweep_guard.py. Those four tests exercise its own substitute() and assert_distinct(), one of them
  written after deliberately breaking the guard and watching the suite stay green; sabotaging its
  `n == 0` branch makes them exit 1. The evidence was in a grep I had run one cell earlier and read
  past: check.sh's own text says "sweep_guard self-tests do not pass -- the guard itself is broken".
  **So the rule is symmetric, and this is the part that generalizes: a coverage claim in EITHER
  direction is a measurement of an artifact, and it comes from opening the artifact.** Counting
  lines that match /self-?test/ in a tool's stdout cannot tell you whether tests exist or what they
  cover -- it is a proxy, and I used it because it was cheap and because understating coverage feels
  like the safe error. It is not safe: a tool wrongly listed as unproven gets re-tested at cost, and
  a queue that misdescribes its own contents is the thing this file exists to prevent.
  FOURTH INSTANCE, and it needs no domain knowledge to catch, which is why it is the most embarrassing
  of the four. The corrected table was headed "four of the FOURTEEN checkers" and listed TWELVE rows:
  check_tex_root and check_verify_list appeared in no column at all -- and they were exactly the two
  files I had still not opened, so the sweep_guard error was reproduced in the artifact fixing it.
  **A table whose headline counts N must have N rows. Count them.** A total that does not reconcile
  with its own rows is an arithmetic error visible to any reader, and it discredits the measurements
  that ARE right by sitting next to them.
  Opening the two files also produced the finding the proxy could never have: check_negative_parallelism
  HAS a self_test(), and disabling one of its four detectors leaves it exiting 0. A self-test that
  does not cover the detector is worse than none, because its presence reads as proof. That is why
  the classification is now by SABOTAGE -- break the logic, read the exit code -- and not by whether
  a `def self_test` exists.
  FIFTH AND SIXTH INSTANCES, both on 2026-07-30, and together they name the sub-rule that was missing.
  (a) "Five of seven candidates accounted" came from five probes, one of which had printed ZERO
  evidence for the item it was counted for and another of which was never run. (b) "The mechanical part
  is measured" was written when one of the two mechanical lists had parsed to ZERO terms -- it is a
  bullet list, not a table, so the table regex matched nothing -- and the other's parse had been
  rejected as wrong and never redone. The only real measurement was a term list I had typed by hand.
  **A PARSE THAT RETURNS ZERO ROWS IS NOT A CLEAN RESULT; IT IS A BROKEN INSTRUMENT, AND IN THE OUTPUT
  IT IS INDISTINGUISHABLE FROM "NO VIOLATIONS".** Before believing a zero, assert the parser found the
  rows it was meant to find: `assert len(rows) > 0` costs nothing, and its absence is how an unmeasured
  section becomes a measured one in a durable record. The same asymmetry runs through all six
  instances -- a count, a coverage claim, or a sweep verdict written from the subset that produced
  evidence, with the silent subset inheriting the verdict. The structural fix is not vigilance but
  SHAPE: report per item, with each item's own evidence beside it, because a summary line is precisely
  where the unmeasured members hide.
  SEVENTH INSTANCE, immediately after writing that fix, and it shows the fix was stated too weakly.
  I reported six flagged terms "each inside its condition" having printed 7 of their 18 occurrences --
  zero contexts for two of the six terms, and 3 of 11 for another. WORSE: the one occurrence I DID
  print contradicted the verdict I gave it. The glossary bans "head" as internal jargon and exempts it
  only "when describing OTHER systems"; the sentence on screen read "an earlier configuration whose
  region head was driven by a transition prior" -- OUR configuration -- and I wrote that the
  other-systems exemption applied. That is not a sampling error. **Reading the evidence is not the same
  as evaluating it: a context line printed and then classified against the rule from memory is
  unverified.** So the per-item shape rule needs its second half: report per item, AND for each item
  quote the specific evidence next to the specific clause of the rule it satisfies. If the quote and
  the clause cannot sit in one sentence without contradiction, the verdict is wrong -- and if a term
  has N occurrences, N of N get read before any verdict covers them, because the violating one is
  never the one you sampled. (The real violation was fixed: "region head" -> "region output".)
  EIGHTH AND NINTH, both in the tracker cleanup, and they are the two shapes of a miscounted set.
  (a) A per-item table headlined "15 of 20" over a set of 21, with the DROPPED member being item 10 --
  the one whose own block says its numbers do not reproduce, i.e. precisely the member the header
  existed to surface; the same header said "five open" above six rows. (b) "All nine archived items are
  gate-confirmed" when SEVEN have a probe: the gate printed nine rows, so I counted rows and called it
  items, while one row (NUM-4) belonged to a finding outside the moved set and two moved items had no
  probe at all. **A COUNT OF THE INSTRUMENT'S OUTPUT IS NOT A COUNT OF YOUR SET.** Before writing "all
  N are covered", join the two lists explicitly and name the unmatched members on both sides -- rows
  without an item, and items without a row. Both of the uncovered items were in fact fine (one verified
  in the render, one a register that needs no probe), which is the trap: the verdict was right and the
  warrant was not, so nothing downstream would ever have caught it.

**V14. A parallel track's self-report is not evidence that its edit landed, and an outcome table is a
  claim about the work.** This is the rule the whole 2026-07-30 recovery round exists to write.
  On 2026-07-28 an audit outcome table was annotated with sixteen rows reading **APPLIED**. Two days
  later the author read the tracker and found the fixes were not in the document. Measured: of the
  nine instructions he personally gave, EIGHT were never applied, five under rows asserting they were
  done. COD-006's row reads '"before any result was read" and "well powered" removed'; both strings
  were still in the source, in the dissertation and in the submitted paper.
  THE MECHANISM, from the commit graph rather than from memory. The commit that wrote the rows says
  in its own message "No source touched" -- it is pure bookkeeping. Of the four fix commits those
  rows cite, THREE never touched the file the row is about (COD-006 cites a commit that touches
  5_mobiwac/05_setup.tex zero times; same for COD-015a and COD-003). The fourth touched its file and
  fixed a DIFFERENT HALF of the finding, and the row credited the whole. Round 6 ran eight parallel
  tracks: each reported what it intended, the table recorded the reports, and no step re-read the
  source.
  WHY IT SURVIVED THIRTEEN HOURS, and this generalizes past audits. Twenty gates were green the
  entire time and every one of them was right — the single artifact with NO gate was the document
  that certifies all the others. A green suite reads as a clean document. A verdict column with a
  real commit hash reads as a citation. And nobody re-reads a closed row: the natural next action
  after marking a table APPLIED is to ARCHIVE the file, which is what happened, so the claim outlived
  the only moment anyone would have questioned it.
  THREE CONSEQUENCES. (1) Every APPLIED row needs a machine-checkable probe -- a string that must be
  absent because it was removed, or present because it was added; `check_audit_claims.py` is the
  gate, and rows whose subject is a process are listed as unprobed BY NAME rather than counted as
  passes. (2) When a finding has several parts, credit the PART, never the finding: "APPLIED" on a
  multi-part row is how COD-013 passed on work done to its other half. (3) Do NOT archive an audit
  on the strength of its own outcome table. Re-measure first; archiving is what turned these claims
  into history.
### Scope discipline for delegated work (the other 2.6 hours)

Round 6 lost **2.6 hours (19%)** waiting on the slowest sub-agent in each of five waves. The worst
was 5.4 hours; its cell log shows **84 inspection cells and 57 git-archaeology cells against 21
build cells** — it was not compute-bound, it was scope-bound.

- **S1. Every delegated task carries an explicit archaeology budget.** State how far back history
  may be walked and what to do on exhaustion (report the gap as a `[VERIFY]` flag, do not keep
  digging). Unbounded "recover the record" tasks are the ones that run five hours.
- **S2. A wave's slowest member sets the wave's cost.** Split any track whose scope is open-ended
  (a whole-document audit, a full-history recovery) into independently landable pieces, so a
  straggler delays one finding rather than the whole wave.
- **S3. Time-boxed checkpoint.** A child running past ~90 minutes without landing writes its partial
  findings to its report file and says what remains. Partial results in hand beat complete results
  three hours later.

## 5 · Review gates (the pipeline every chapter passes, in order)

```
G0 OUTLINE   author approves the section outline (scope + claims to be made)
G1 DRAFT     agent drafts per L1; handoff note lists: numbers ledger, new-claim proposals,
             [VERIFY] flags, sources opened
G2 FACT GATE (fail-closed) citation protocol §1 + number protocol §2 + claim registry §3
             + cross-ref lint L4. Any failure returns to draft.
G3 STYLE GATE (statistical, separate pass, fresh eyes) WRITING_LAW §7 checklist: AI-tell sweep,
             idiom sweep, variance/burstiness read-aloud, discourse-skeleton variety, register
G4 AUTHOR    Vitor reads and approves (edits welcome; approval recorded in git)
G5 ADVISOR   only after G2–G4 are green
```

- Gates G2 and G3 are **separate passes** (fact ≠ style; merging them measurably weakens both).
- Audit intensity scales with AI share: a chapter that is mostly re-typeset published text gets
  the standard pass; heavily AI-drafted frame prose gets the full adversarial treatment
  (contamination is bimodal — heavy-reliance documents carry most fabrications).
- Git discipline supports provenance: AI-drafted and author-drafted content land in
  distinguishable commits (`draft(ai): …` vs `edit(author): …`), so the disclosure statement
  (§6) is reconstructible from history rather than remembered.

## 6 · AI-use disclosure (required, not optional)

**The landscape (verified 2026-07-18):** no binding UFV/PPGCC rule yet, but (a) **CNPq Portaria
nº 2.664/2026** mandates declaring any generative-AI use (tool + purpose) for CNPq-linked
researchers and forbids submitting AI-generated content as human-authored; (b) UFV/DPE published
a recommended declaration format (03/2026); (c) CAPES directives are converging on the CNPq
policy; (d) every major publisher (ICMJE, Elsevier, Springer, IEEE, ACM) requires disclosure;
(e) PPGCC separately requires an **anti-plagiarism certificate** for the defense.

**The rules:**

- **D1.** The dissertation carries an AI-use disclosure note (placement: open decision,
  NORTH_STAR §5.9) naming: the tools and model versions, the scope of use per part (drafting,
  editing, formatting, code), and the human-verification steps applied (this file's gates).
  Honest, specific, one page maximum.
- **D2.** The disclosure is drafted from the git provenance trail (§5), not from recollection.
- **D3.** Raise with the advisor EARLY (it is also his risk); if he wants committee
  pre-authorization, obtain it before mass drafting, not after.
- **D4.** The anti-plagiarism certificate is a defense blocker — schedule it in PLAN.md, and
  remember AI-assisted text still must not lift verbatim prose from sources (paraphrase +
  citation discipline as usual; the coletânea's own papers are exempt self-material, stated in
  the organization section).

## 7 · Known agent biases this file counters (name them to catch them)

| Bias | Counter |
|---|---|
| **Sycophancy** (agreeing with the author's slip instead of checking — e.g. the CoUrb→CBIC order) | Evidence beats instruction on facts; discrepancies are flagged with sources, decisions stay the author's (this file exists because the check caught exactly that). |
| **Plausible confabulation** (citations, numbers, "recalled" details) | §1–§2 fail-closed protocols. |
| **Polish over grounding** (the best-looking draft carries the most errors) | G2 before G3; polish never substitutes for a ledger line. |
| **Overclaiming / verdict inflation** (upgrading "matches" to "outperforms", widening scopes) | Claim registry §3 + WRITING_LAW §3 verb-test binding. |
| **Padding** (length as a proxy for quality) | Outline-bound drafting (L1); every section must earn its pages; the Viegas example is ~100 pages total — that is the calibration, not a target to exceed. |
| **Fake cohesion** (template transitions, uniform section shapes) | G3 skeleton-variety check; WRITING_LAW §4.4. |
| **Variance compression in edit passes** (homogenizing the author's voice) | Edits preserve burstiness; a pass that only smooths is rejected (WRITING_LAW §4.3). |
| **Self-certification** (agent declares its own output verified) | L6 fresh-eyes rule; author audits independently. |
| **Trusting the tolerant tool** (two checks disagree; the one reporting success is believed) | The source did not compile for six commits while `build.sh` reported "104 pp, 0 overfull, 0 undefined": under `-interaction=nonstopmode` pdflatex recovers from an error and still writes a PDF, and the checker never looked for TeX errors. `make` (`-halt-on-error`) produced nothing the whole time. **Rule: `tex_errors=0` is part of every build claim; a PDF existing is not evidence the source is correct; when two tools disagree about one artifact, distrust the one reporting success.** (2026-07-28, §2.3b of `science/AGENT_HANDOFF.md`.) |
| **A gate that has never fired** (a check whose passing carries no information) | Validate every new gate in BOTH directions before trusting it: run it against a tree where the defect is present and confirm it fails, then against the fixed tree and confirm it passes. Four of this repository's checkers were wrong at least once by being tuned only on the case in front of them. |
| **Costing an item from the batch's total** (per-item rate divided out of an aggregate that includes other items) | §4b V10: derive each item's rate from its own start and end. A 3.3x error this way turned a recoverable dataset into a reported write-off. |
| **Generalising from the one that matched** (a locating step returns nothing; the location is inferred from a neighbouring hit that did match) | §4b V9: report the location as unresolved. Two overfull boxes, one line range matched a file, and both were attributed to it — the other was in a block of comment lines where nothing can be typeset. |
| **Taking shape for substance** (exit 0, right filename, right row count, empty column) | §4b V8: open one output file and count non-empty cells in the column you came for. Three remote runs in one night reported success with an all-`NaN` target column; the failure was invisible in every signal except the data itself. |
| **Silent correction** (fixing a published number/claim without a trail) | Errata policy (NORTH_STAR §5.7): every departure from a published source is listed and approved. |
| **Reporting the intent instead of the output** (writing what the check was *meant* to cover) | The largest single defect class in this repository: 12 of the 14 genuine rework commits in round 6 (R1+R2+R4 of §4b's table = 5+4+3). §4b V1–V2: a number about the work carries its command, and any `continue`/`skip`/filter in the producing code must be named in the claim with its count. |
| **Believing an instrument you have not interrogated** (a clean reading from a tool blind to the thing measured) | §4b V3. `FPDFText_GetFontSize` returned 6.97 pt for a figure that renders at 11.15 pt, because it reports the size declared inside the embedded object and ignores `\includegraphics` scaling. The reading was not wrong; the question was. |

## 8 · Evidence base (why these rules; verified 2026-07-18)

Citation fabrication rates and cases: Lancet/Columbia corpus study (fabricated citations 1/2,828
papers 2023 → 1/277 early 2026); JMIR Mental Health 2025 (GPT-4o 19.9% fabricated, 6–29% by
sparsity); GPTZero NeurIPS-2025 audit (100 fabricated refs in 53 accepted papers); Ansari 2026
taxonomy (66% total fabrication, 27% attribute corruption); Springer retraction of a fabricated-
citation ML book (2025); arXiv 1-year ban policy (2026). Number corruption: NAACL-Findings 2025
multi-doc faithfulness; AI-Scientist evaluation (57% papers with wrong numbers); PaperRecon/U-Tokyo
2026 (polish↔hallucination trade-off). Long-form: LongGenBench (repetition in ~45% of long
outputs); HelloBench (quality collapse past ~2k words); context-rot / Ref-Long (mid-context
constraint loss); QUDsim (discourse-skeleton reuse); syntactic-template detection (Shaib et al.);
"Voice Under Revision" (variance compression, Claude 78% of features). Tells: Kobak et al.
Science Advances 2025 (excess vocabulary, ≥13.5% of 2024 abstracts); Matsui 2025 (tell-avoidance
already measurable); Terčon & Dobrovoljc 2025 survey (POS-profile tells); refsmmat per-model word
rates (Claude "genuinely" ~10×). Policy: CNPq Portaria 2.664/2026; UFV/DPE guide 03/2026; CAPES
GT 2025 (+ NT 3/2025 via secondary sources — verify before citing verbatim); ICMJE 04/2025;
publisher policies (Elsevier/Springer/IEEE/ACM); U. Georgia / U. Toronto thesis policies; Unifesp
Res. 17/2025; Unicamp PRPG 2025. **Full findings with every URL:**
[`docs/research/ai_writing_evidence_2026-07-18.md`](docs/research/ai_writing_evidence_2026-07-18.md)
(kept verbatim; also the source pool for the dissertation's own disclosure appendix if needed).
