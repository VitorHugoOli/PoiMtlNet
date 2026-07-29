# 25 — Round 7 post-mortem: where round 6's 13 hours went, and what round 7 actually changed

**Written 2026-07-29 by the trackers track, for the author and for the next agent.**
**Working directory for every command below:** `articles/dissertacao/`.

This is not a status report. Round 7 fixed the largest cost centre and left the second-largest
partly open, and the class of defect that cost round 6 the most **was hit thirteen more times during
round 7 itself**, by me among others. §4 lists them with their commits. A post-mortem that reads as
a success report is useless, so the honest summary is one line:

> **The machine got faster and the record did not get more truthful.**

---

## 1 · What I could re-derive, and what I could not

The brief handed me four causes with hours attached. I re-derived each from the repository and the
session log rather than quoting them, and the derivations disagree with the brief in places. Where
they disagree I give my number, my method, and why the two differ — the brief's figures are not
wrong so much as differently scoped.

| Cause, as the brief states it | What I measure | Method |
|---|---|---|
| 13.3 h round | **12.53 h active**, 21.20 h wall | cell timestamps for 2026-07-28, summed with idle gaps over 20 min excluded. At a 30-min threshold it is **13.35 h**, which is where the brief's 13.3 comes from |
| 61 commits | **63** on 2026-07-28 | `git log --since='2026-07-28 00:00' --until='2026-07-29 00:00'`. 62 of them fall after 09:00; the 63rd is `ba90aa6d` at 02:44, the compile blocker, which belongs to the round that opened it |
| 3.4 h compilation, 127 invocations at ~97 s | **168 single-target compile invocations**, 3.9-5.7 h depending on per-pass cost | see §2 |
| 2.6 h waiting on the slowest child in 5 waves | **2.44 h** past the second-last finisher, **3.39 h** past the first | see §3 |
| 2.4 h rework, 17 of 61 commits, 14 genuine, 12 meta-claims | **not re-derivable**; I can reproduce neither the 17/14/12 split nor the 2.4 h | see §4 |

**What I could not confirm, stated plainly.**

- **The 2.4 h rework figure.** Time cannot be attributed to a commit from the log: commits are
  points, not intervals, and round 6 ran up to eight children concurrently, so wall-clock gaps
  between commits overlap work that was not rework. The 2.4 h in `AGENT_GUARDRAILS` §4b was measured
  by whoever wrote §4b with material I do not have. I take the *classification* (14 genuine, 12
  meta-claim) as that pass's finding and do not re-assert its hours as mine.
- **The 17/14/12 split.** My own subject-line classifier gives 39 of 63 round-6 commits as
  correction-shaped, which is far too broad: it catches every commit that fixes a defect in the
  *dissertation* as well as those fixing a claim about the *work*, and §4b's whole point is the
  distinction. A subject line does not carry it reliably. I report §4b's split as inherited and
  measure round 7's own rework instead, where I read every commit body (§4).
- **"82 build cells"** in the plan description and **"127 target invocations"** in the brief are
  two different quantities and neither matches mine. I cannot reconstruct either criterion.

---

## 2 · Compilation: the cause round 7 genuinely fixed

### What round 6 spent

I counted invocations, not cells, because one cell often runs three targets:

```bash
# Round 6 build invocations, counted in COMMAND POSITION in the session's own cell sources.
# Not a grep for the string: a target named inside an echo, a heredoc, a diff hunk or a comment
# is excluded, and the exclusions are counted and reported rather than dropped.
```

**Round 6 (2026-07-28): 168 single-target compile invocations** — `make defense` 75, `make final`
51, `make ppgc` 30, plus `make academico` 0 (the target did not exist yet) — with **59 `make check`**
and **43 `build.sh`**. Named exclusions, so the number is auditable: 7 references inside cells whose
language was `diff`, and 20 inside heredocs or Python string literals.

Per-pass cost is the variable, and it is noisy. The build-speed track measured the same command on
the same tree at **105, 111, 122.7 and 128 s** across four runs; I measured `make defense` today at
**83 s** with another track's build running concurrently. So:

| per-pass cost assumed | 168 invocations cost | share of the 12.53 active hours |
|---|---:|---:|
| 83 s (my measurement today) | **3.87 h** | 31% |
| 97 s (the brief's figure) | **4.53 h** | 36% |
| 122.7 s (the build-speed baseline) | **5.73 h** | 46% |

The brief's "3.4 h / 26%" is the low end of that band. **On every assumption, compilation was the
single largest cost of round 6** — between a third and a half of the active time — and that is the
claim worth keeping, not any one figure inside the band.

### What round 7 changed, measured today

Round 7 precompiled the preamble into a format dump (`f388d9d0`, `mkformat.py` + `fastbuild.sh`) and
added two multi-target paths. Timed in one sitting, each forced by touching a chapter first:

```bash
cd /Users/vitor/Desktop/mestrado/ingred/articles/dissertacao && source src_utils/texenv.sh && cd src
for t in fast3 all3 defense; do touch chapters/2_fundamentals.tex; \
  s=$(date +%s); make $t >/dev/null 2>&1; echo "$t rc=$? $(( $(date +%s) - s ))s"; done
make check >/dev/null 2>&1; echo "check rc=$?"
```

| path | what it builds | measured |
|---|---|---:|
| `make fast3` | all three targets, format-accelerated | **13 s** |
| `make all3` | all three targets, plain, `-j3` | **115 s** |
| `make defense` | one target, plain | **83 s** |
| `make check` | 20 gates | **2 s** |

**Three targets in 13 s against 115 s parallel or ~249 s serial at today's per-pass cost.** Per
target that is 4.3 s against 83 s, a 19x reduction. Applied to round 6's own mix, 168 invocations at
the format cost would have been **0.20 h instead of 3.87 h** — the round's largest cost centre
reduced to a rounding error.

**Round 7's own mix, for contrast:** 43 single-target compiles, 8 `all3`, 9 `fast3`, 54 `make check`.
That is 1.28 h of building in a 3.42 h round, **37%** — proportionally no better than round 6. The
tooling got 19x faster and the *share* did not move, because the round spent its cheaper builds on
more of them. Worth knowing before anyone expects the next round to be proportionally cheaper.

**Two things about the 87% claim.** The inherited "the preamble is 87% of a pass" was **not
re-derived this round**; the build-speed track recorded that it measured whole-build 122.7 -> 15.4 s
and never decomposed a single pass (`20_build_speed.md` §-table, "as measured here" row). The 87% is
consistent with the speedup but is not a round-7 measurement. And the format carries a checksum
staleness guard that **refuses** to build against a changed preamble, which is what makes it safe to
use freely — a stale format would be silently wrong, which is this repository's worst failure shape.

---

## 3 · The straggler cost: measured, and only partly addressed

### What it was

```bash
```

Round 6 launched its children in four waves (grouping launches within 2 minutes of each other).
Measured from the session log:

| wave | launched | n | last finish | idle past 2nd-last finisher | idle past FIRST finisher |
|---|---|--:|---|---:|---:|
| 1 | 09:20 | 3 | 11:16 | 0.39 h | 0.67 h |
| 2 | 11:26 | 3 | 16:49 | **1.31 h** | **1.82 h** |
| 3 | 17:20 | 3 | 18:45 | 0.27 h | 0.28 h |
| 4 | 17:32 | 5 | 19:12 | 0.47 h | 0.62 h |
| | | | **total** | **2.44 h** | **3.39 h** |

The brief's 2.6 h sits between my two framings, which is the right place for it: "waiting on the
slowest" is 2.44 h if you count from when the wave was all-but-done, 3.39 h if you count from when
the first child could have unblocked the next stage. **Wave 2 is 54% of it.**

### The worst child, profiled

The 5.4-hour child is wave 2's claim-scoping track — **session frame `743c418b`, which is a frame id
and not a commit** (do not try to resolve it with `git`; it is the delegated child's own execution
record). It is not the wave-2 straggler by accident: it *is* wave 2's cost. Its cell log:

| class | cells |
|---|--:|
| inspection (`grep`/`sed`/`head`/`cat`/`wc`/`pypdfium2`/`ls`) | 109 |
| git archaeology (`log`/`show`/`diff`/`blame`/`cat-file`/`hash-object`) | 49 |
| build | 21 |
| **total cells** | **193** |

Classes are non-exclusive — a cell that greps a file it just built counts in two — so these do not
sum to 193; 123 cells were bash, 44 were diffs, 26 were Python. The brief's "84 inspection and 57
git-archaeology against 21 build" is the same shape under a narrower classifier. The conclusion holds
under both: **the child was scope-bound, not compute-bound.** Twenty-one build cells cannot account
for five hours; 158 read-and-dig cells can.

### What round 7 did about it, and what it did not

`AGENT_GUARDRAILS` §4b now carries **S1** (every delegated task states an archaeology budget and what
to do on exhaustion), **S2** (split open-ended scopes into independently landable pieces), and **S3**
(a child past ~90 minutes writes partial findings and says what remains). Those are the right three
rules and they are **law, not mechanism**. Nothing enforces them: no gate can see a child's cell mix,
and a task that says "recover the record" will still run five hours if nobody bounds it in the brief.

**Round 7's own waves went better** — 3 children in 2.54 h, straggler cost 0.41 h — but that is weak
evidence for S1-S3, because round 7's tracks were narrow by construction (build speed, LaTeX upgrade,
comment hygiene) and were never at risk of a five-hour archaeology dig. **The rules are untested
against the case they were written for.** The next open-ended audit is the test.

---

## 4 · The rework class: not fixed, hit thirteen times this round

This is the section the brief asked for and the one I would want read first.

§4b's countermeasures are V1-V7 plus four new gates. They are good rules. They did not prevent the
class. Reading every round-7 commit body, **thirteen of the round's thirty-one commits correct a
statement about the work** — a count, a coverage claim, an exit code, a line coordinate, a
self-description — rather than anything about the dissertation:

| commit | what was wrong | §4b class |
|---|---|---|
| `adbb6952` | The guardrails' own gate hard-coded `9` in an f-string; R1+R2+R4 is 12 of 14 | R4 |
| `01e1fbbc` | Three of my own source comments were wrong **about their own subject** | R1 |
| `dfea92e3` | `fastbuild.sh` exited 1 after every clean build: `grep -c` returns 1 on a zero count | R2 |
| `b0db492d` | "18 gates" in a report; another track made it 19 within the hour | R4 |
| `445bb172` | Five numbers about my own tooling wrong, and the outline check **could not fail** | R1+R4 |
| `688856cc` | Two shims cited line coordinates already stale when written | R3 |
| `b1ea08f3` | A stale cell said the format renumbers citations; a controlled test says it does not | R3 |
| `d5db7855` | Two evidence lines named the wrong file and split a total wrongly | R4 |
| `cde5932d` | Four numbers in my own report had gone stale | R4 |
| `fb715824` | A count in `LATEX_UPGRADE.md` was unguarded, and already drifted | R4 |
| `9d3eae8b` | The missing-files count is **five**, not nine — third value for the same quantity | R2+V6 |
| `e1a1e4cf` | **`RC=0` written for a run that printed `RC=1`** — third instance that round | R1 |
| `3bd47d5d` | "Four tracker citations repointed" when three were patched | R1 |

Two more from the round's last hour belong to the same family even though their subject is science:
`4045eb8d` (four figure defects a geometry check could not see, found only by looking at the render —
pure R2) and `cc85b437` (250 serially-dependent cosines treated as independent, pushing a p-value).

**The three sharpest instances, because the mechanism matters more than the count:**

1. **`e1a1e4cf` — the exit code.** A resolved `[VERIFY]` flag claimed `RC=0 across 20 gates`; the run
   had printed `RC=1`. The agent read the output and wrote the success it expected. Its own body calls
   this worse than the earlier over-claim it corrects, because that one asserted a run never made,
   while this one **contradicted a run it had made and read**. V2 exists to prevent exactly this and
   did not.
2. **`445bb172` — a gate that could not fail.** A new outline check was incapable of returning a
   failure. It passed, its passing was reported, and its passing carried zero information. This is the
   bias `AGENT_GUARDRAILS` §7 names ("a gate that has never fired") landing inside the round that
   added the rule.
3. **`51a29f6b` — three remote jobs, zero data, reported as data.** A submitted job hit `rc=127` six
   times in one second (non-login shell, no conda on `PATH`), the loop swallowed each code, the
   tarball step still ran, and the completion notification said **SUCCESS with zero output files**.
   The second attempt trained cleanly and harvested well-formed CSVs whose diagnostic column was
   **entirely empty** (the diagnostic is opt-in and defaults off) — and that empty result was passed
   to a sub-agent as data. The shape is R2 exactly: the tool ran, produced output, and the output was
   about something other than what was meant to be measured. **A zero is the most dangerous reading,
   because zero findings is also what success looks like.**

**And two of mine, in this very track.**

The first was inherited and I shipped it before catching it. `PENDENCIAS.md` §2.1 told the author
that *"cada arquivo acima foi verificado individualmente como intacto no remoto"* for the fifteen
paths of the destructive worktree commit, above a command that checked **four**. The sentence
promised more coverage than the command delivered — V1 and V2 exactly — and it survived my own
rewrite of that section because I re-read the *prose* and not the *block under it*. It is now a loop
over all fifteen paths read from the commit itself, with no hand-typed list: **15 paths, 0 absent**
(14 deletions whose files remain in `origin/mobiwac`, plus the modified `README.md`). The corrected
text also states what the check does and does not establish — path existence in the remote, not
byte identity — because that distinction is what makes it the right question for a commit that was
never pushed.

The second: two of my page-coordinate probes disagreed within ten
minutes. The cause was not a bad probe: another round-7 track was editing `src/` while I measured, so
the build moved under me (16 modified files and four new ones under `src/` at the time of writing). I
caught it because the disagreement was visible; had only one probe run, I would have written down a
page number that was already wrong. **Every coordinate in my deliverables is therefore a phrase, not
a line or a page.** Concurrent tracks make line coordinates a liability, not just a fragility.

### What the four new gates do and do not cover

| gate | catches | blind to |
|---|---|---|
| `check_verify_list.py` | A documented command that does not return what its prose claims | Claims with no command attached — most of the thirteen above |
| `check_comment_hygiene.py` | A file whose self-count disagrees with itself; a story told twice | Any count about something other than the file it lives in |
| `check_tracker_refs.py` | A `PENDENCIAS N.M` citation that stops resolving | Whether the item it points at is still true |
| `check_tex_root.py`, `check_doubled_macro.py` | Two silent LaTeX classes | Nothing in the meta-claim family |

The pattern: **the gates catch claims that live in a file the gate can read and check against
itself.** Twelve of the thirteen round-7 instances were claims in commit messages, reports, and cells
— none of which any gate reads. That is not a gap to be closed by a fifteenth gate. `check_verify_list`
already exposes the honest version of this: it reports **11 executed-but-not-asserted** blocks
separately, so nobody can inflate the verified count. More surface would help less than the one habit
V1 already names, which is that **a number about the work carries the command that produced it.**

### The one measurable improvement in this class

Round 6 discovered its rework at the end, in a sweep. Round 7's thirteen instances were each caught
and committed **within minutes to hours of the claim**, several by the concurrent tracks reviewing each
other (`b0db492d` was corrected by a track that had changed the count within the hour). The class was
not prevented; its **latency** dropped. That is a real gain and it is not the gain the round was
aiming for.

---

## 5 · The state I am handing over

**Trackers, this track's own work** (commit `ba5dd5b3`):

- `PENDENCIAS.md` 342 -> 464 lines, but the author's queue now begins at line **49** instead of 107
  (`grep -n '^## §2' src_utils/PENDENCIAS.md`; my commit message `ba5dd5b3` says 41, which was a
  figure from a draft before the reading guide grew — the file is the truth, not the message). §1
  (66 lines of closed round-6 work) moved to `_archive/PENDENCIAS_RESOLVIDOS.md` **with all 19 commit
  hashes**, each verified to resolve.
- `CODEX_AUDIT.md` archived by `git mv` after a sweep of all 26 COD-/NUM- ids, all 16 `DECISAO`
  boxes, and the outcome table. **Nine of the author's own decisions are not in the document**, five
  of them under outcome-table rows that say "APPLIED". They are now `PENDENCIAS.md` §5, each with the
  command that measures it. Four live pointers repointed (`CLAUDE.md`, `codex_reviewer.md`,
  `science/AGENT_HANDOFF.md`, `_archive/README.md`).

**Two things are open and are not mine to close:**

1. **`make check` exits 2**, and did so **before** my edit. Proved by swapping `PENDENCIAS.md` for its
   HEAD bytes and re-running: identical single failure, then restored and checksum-verified. The
   failing assertion is in `_round6/VERIFY_LIST.md` and it is a real defect in live work —
   `main_academico.pdf` prints **8** on physical page **9**. This is the same class as round 6's C-1
   (the deposit build numbering from the wrong page), on the one build that is deposited.
2. **A destructive local commit still sits in the `mobiwac` worktree.** `6c4267ba`, subject "add the
   five missing reproducibility artifacts", is `15 files changed, 10 insertions(+), 2028
   deletions(-)`. Verified read-only today: it is still `HEAD` of that worktree, `origin/mobiwac` is
   still `3c57197c`, and nothing has moved (reflog unchanged). `PENDENCIAS.md` §2.1 holds the
   procedure.

   **A hazard I found while checking it, and did not fix.** `check_verify_list.py` executes every
   fenced `bash` block in `PENDENCIAS.md`, and §2.1's block contains `git reset --hard`, `git commit`
   and `git push origin mobiwac`. Today it is harmless only by accident: the sandbox cannot write
   `.git/` in that worktree, so every command fails with "not a git repository" and the gate reports
   the block as *ran, no EXPECT annotation*. **Outside the sandbox, on the author's own machine,
   `make check` would attempt the reset and the push.** I verified the dispatch path against a
   throwaway repository rather than the real one, and I have not changed the gate or the block —
   both belong to other tracks' remits. **This is the highest-consequence thing in this file.**

---

## 6 · For the next agent: the four habits that would have saved this round

Not new rules. These are V1-V7 restated as the actions that would have caught the thirteen:

1. **Read the exit code, then write it.** Not the output's tone, not the lines you recognize. `RC=` is
   a number the shell gives you; copy it. Three of the thirteen are this.
2. **Before believing a zero, prove the instrument could have returned non-zero.** A zero-findings
   sweep and a broken sweep are the same reading. `445bb172` and all three remote jobs in `51a29f6b`
   are this.
3. **After correcting a count, grep the old value everywhere.** The missing-files count was wrong
   three times (8-of-12, 9-of-13, five) and each superseded value outlived its correction. V6 exists;
   it needs doing.
4. **Anchor by phrase, and assume the tree is moving.** Concurrent tracks edit the same source. A page
   number measured ten minutes ago may already be false, and you will not be told.

And one that is not in §4b, learned here: **when two of your own probes disagree, that is the most
valuable signal you will get all round.** Do not reconcile them by picking the one that fits. Find out
which instrument was pointed at the wrong tree — mine was, and the disagreement is the only reason I
know.
