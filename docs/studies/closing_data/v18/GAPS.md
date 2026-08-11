# v18 — what is missing, for future audit

> **Status: the board is COMPLETE and the numbers are sound. Nothing below invalidates a result.**
> What is missing is **traceability** — the ability for a future reader to prove *which code, which
> machine, and which recipe* produced a given number. The charter (§7 "PROVENANCE.md — every rundir:
> state, seed, PID, path, recipe, commit SHA") asks for this explicitly and we are short of it on
> 30 of 72 cells.
>
> Written 2026-08-11 so the gaps can be assigned. Each item states **what to add, where, and how to
> get it** — no re-training is required for any of them.

## 0 · Board state (verified 2026-08-11 01:50)

| | |
|---|---|
| cells | **72 / 72** — 6 states × 4 seeds {0,1,7,100} × 3 families (cat, reg, joint) |
| n per cell | 20 (4 seeds × 5 folds) |
| missing results | **none** |
| seed spread, region | texas 0.023 · florida 0.035 · istanbul 0.036 · california 0.058 · arizona 0.166 · alabama 0.213 pp |

All spreads are far below the region effect at the large states (+1.93 texas / +1.96 california), so
the headline result is not in question.

---

## 1 · GAP A — 30 cells have no traceable `commit_sha` (HIGH)

`commit_sha` is absent or literally `"unknown"`. These cells cannot be tied to a state of the code.

| seed | family | states affected |
|---|---|---|
| 7 | cat | arizona, california, florida, istanbul, texas |
| 7 | joint | arizona, california, florida, istanbul, texas |
| 7 | reg | arizona, florida, istanbul |
| 100 | cat | alabama, arizona, california, florida, istanbul, texas |
| 100 | joint | alabama, arizona, california, florida, istanbul, texas |
| 100 | reg | alabama, arizona, florida, istanbul, texas |

**All 30 are seed 7 / seed 100 — i.e. the rented-lane cells.** The local seed 0/1 cells all carry a
SHA (`run_wave.sh` captures `git rev-parse HEAD` at wave start).

**What to add:** the SHA the lane image was built from, into each sidecar's `commit_sha`.
**Where to get it:** the Modal driver, now at `pipelines/modal/` (moved out of the retired
`v18_2/` folder on 2026-08-11) — whatever
commit was baked into the image, or the checkout the lane ran. If a single SHA covers all lane cells,
one value fills all 30. **If the SHA is genuinely unrecoverable, say so in PROVENANCE.md rather than
back-filling a guess** — an invented SHA is worse than an admitted gap.

**Update 2026-08-11 (florida s7/s100 joint, 2 of the 30):** unrecoverable, and the reason
generalises to the other rented cells. The `/repo` on the Modal Volume was uploaded at
2026-08-10T03:28Z as a **tar of the worktree**, not `git archive`, and carries no `.git`. The two
nearest commits (`a0df1fe3` 03:44Z, `d8a4cf04` 04:47Z) are both **later** than the upload, so HEAD
at upload time was an earlier commit that was never recorded. Even a recovered SHA would not certify
the code, because the payload was the worktree rather than a committed tree. Recorded as
`commit_sha_note` in each sidecar instead of a back-filled value.

## 2 · GAP B — 10 cells have no `recipe_version` (MEDIUM)

| seed | family | states |
|---|---|---|
| 0 | reg | all six |
| 1 | reg | alabama, arizona, florida, istanbul |

These are the **kept** region cells — banked before `recipe_version` was introduced on 2026-08-09.
Their recipe is known and unchanged (region was never retuned: `max_lr 3e-3`, `freeze_alpha=True`,
`alpha_init=0.0`, `logit_adjust_tau=0`).

**What to add:** `"recipe_version": "v18-approved-2026-08-09 (FINAL_SETTINGS.md)"` plus the `recipe`
string already used by the newer reg sidecars. Pure metadata back-fill, no re-run.

## 3 · GAP C — hardware is mixed and mostly undeclared (HIGH for the write-up)

| hardware | cells |
|---|---:|
| local A40 (declared or presumed) | 45 |
| `modal:NVIDIA H100 80GB HBM3` | 19 |
| `modal:NVIDIA A100-SXM4-40GB` | 8 |

**42 cells carry no `lane_host` field at all.** They are presumed local, but presumption is not
provenance.

Two things follow:

1. **Back-fill `lane_host` on the 42** (e.g. `"local:NVIDIA A40"`), so every cell declares its
   machine.
2. **The mixing is not uniform across a state's pool, and that must be disclosed.** The clearest
   case: **texas region has 3 seeds on the local A40 and 1 (seed 100) on a Modal A100** — and that
   Modal cell is the lowest of the four (64.9295 vs 64.9436 / 64.9514 / 64.9528).

   This matters more than it looks. `v18_2/EXECUTION_PLAN.md` measured cross-hardware deviation on
   one identical cell at **~0.086 pp** (A40 30.7654 / A100 30.6790 / H100 30.7049). That is
   **3.7× the entire seed spread of texas region (0.023 pp)**. So for this metric the *machine* is a
   larger source of variation than the *seed*.

   It does not threaten the +1.93/+1.96 region result. But describing these cells as "4 seeds"
   without stating that one ran on different silicon would be inaccurate. Either disclose it in
   `METHODOLOGY.md`, or close it: **re-running texas s100 reg locally costs ~57 min** and makes that
   pool single-machine.

**Update 2026-08-11 — the mixing is not confined to texas region.** Enumerating every
`(state, family)` pool by the silicon each seed ran on shows that **essentially every pool is
mixed**: seeds 0 and 1 are local, seeds 7 and 100 are rented, across all six states and all three
families. Texas region is not the exception, it is one instance of the rule. The disclosure in
`METHODOLOGY.md` therefore cannot be a footnote about one cell; it has to state the design
honestly — *seeds 0/1 local, seeds 7/100 rented* — and note that the measured cross-hardware
deviation (0.086 pp) exceeds the seed spread of several region pools.

Closing this by re-running is no longer a 57-minute job: it would mean re-running every seed
7/100 cell locally, which is the whole rented wave. The realistic close is disclosure, not
homogenisation.

**florida s7/s100 joint `lane_host` back-filled 2026-08-11**, with a caveat worth checking:
the requested tier was `A100-40GB`, but every heartbeat sample from both cells reports
`memory.total = 81920 MiB` (80 GiB). The device *name* had already rolled out of the captured
`.out` tail, so the sidecars record `modal:A100-class 80GB` with an explicit `[VERIFY]` rather than
a model name nobody measured. **Whoever audits this should confirm which tier Modal actually
served** — the other 8 cells are labelled `A100-SXM4-40GB`, and if those were also 80 GiB parts the
existing labels are wrong too.

## 4 · GAP D — the tie certificate exists on only 3 of 24 region cells (MEDIUM)

`ambiguous_rows` (how many validation rows have an undecidable hit@k) is recorded on **3 of 24**
region cells — the ones produced after 2026-08-10. Measured values where it exists: 1–2 rows out of
585 092 (california) and 1 of 766 083 (texas), i.e. **≤ 0.0003 pp** of Acc@10 — negligible, and ~1 %
of the seed spread.

The remaining 21 cells were scored on the legacy topk path where the question does not arise, so
there is nothing to measure retroactively. **What to add:** a one-line `"scored_on"` /
`"scoring_path"` field on those 21 stating they used the legacy full-logit + topk CPU path, so a
reader can tell the two populations apart without archaeology.

## 5 · GAP E — two display bugs in the generated report (LOW, but user-visible)

1. **`V18_RESULTS.md` §1 `n` column is wrong for texas and california.** It prints `20`, taken from
   `joint_cat_paired["n"]` (`make_results.py:156`), but it labels a row whose **region half** may
   have a different n. Now that the board is 72/72 both halves are 20, so **this is currently
   correct by accident** — the bug is that the column cannot represent a row whose two halves
   disagree, which was the case for ~26 h. Fix: print n per half, or assert they match.
2. **`status.json` says `phase: "done"` while cells were still running** (`updated_at`
   2026-08-11T01:44). `status_update.py:107-109` derives `current_n` from **joint cells only** ×5,
   so it structurally cannot see a lagging cat or reg family. Fix: count all three families.

## 5b · GAP E item 2 — CLOSED 2026-08-11

`status_update.py` derived `current_n` from **joint cells only**, so a state whose joint cells were
all banked reported its full n while its dedicated cells were still missing. Observed live on
florida 2026-08-10: joint s7/s100 landed and the state read n=20 while `florida_s7_cat`,
`florida_s7_reg`, `florida_s100_cat` and `florida_s100_reg` did not yet exist.

Fixed: a seed now counts toward `current_n` only when **all three families** are banked for it, and
the status file gained a `seeds_complete` map so a reader sees *which* seeds are complete rather
than inferring it from a total. Verified by execution, not inspection: hiding `florida_s7_cat` drops
florida from 20 to 15 with `seeds_complete: [0, 1, 100]`, and restoring it returns 20.

Item 1 of GAP E (the `n` column that cannot represent a row whose halves disagree) is **still
open** — it remains correct only because both halves happen to be 20.

## 6 · GAP F — charter deliverable `TASKS.md` was never created (LOW)

§7 lists `TASKS.md` ("the charter and task list"). Every other §7 deliverable exists: `README.md`,
`V18_RESULTS.md`, `PROVENANCE.md`, `AUDIT.md`, `METHODOLOGY.md`, `score_all.py`,
`data/v18_results.json`, `PROGRESS.md`, `status.json`, `log.md`.

## 7 · Open scientific decisions (NOT data gaps — author calls)

Recorded here so they are not mistaken for missing data:

1. **Practical-significance floor.** The generator awards "**beats**" to deltas as small as
   +0.04 pp because pairing on identical folds collapses the variance. Three independent arguments
   say those should not be reported as wins: fold dependence (folds within a seed share ~80 % of
   training data), the tuned-comparator bias (the dedicated arm got a per-state LR search the MTL
   arm did not), and effect size (~+0.04 pp is not a claim anyone can defend). **Not yet decided.**
2. **P1 capacity-matched region control** (`POSTPONED.md`) — the experiment that would decide whether
   the +1.93/+1.96 region advantage is multi-task sharing or simply the dual-tower's extra
   parameters. The 1-fold triage already showed it survives severing the trunk *and* deleting the
   category task, which points at capacity. Still held.

---

## Priority for whoever picks this up

| # | gap | effort | why it matters |
|---|---|---|---|
| 1 | **A** — 30 missing `commit_sha` | metadata only | charter §7 requirement; without it 30 cells cannot be tied to code |
| 2 | **C** — declare `lane_host` on 42 cells + disclose the texas mix | metadata (+57 min if closing texas) | the machine is a larger variance source than the seed here |
| 3 | **B** — 10 missing `recipe_version` | metadata only | trivial, and completes the set |
| 4 | **D** — scoring-path field on 21 reg cells | metadata only | lets a reader separate the two scoring populations |
| 5 | **E** — display bugs | small code fix | **item 2 CLOSED 2026-08-11**; item 1 still open (correct only by accident) |
| 6 | **F** — `TASKS.md` | writing | last unmet §7 deliverable |

**None of these require re-training.** Items 1–4 are back-fills into existing sidecars; 5 is a fix in
`make_results.py` / `status_update.py`; 6 is a document. The only optional compute is the ~57 min
texas s100 re-run under item C, and that is a homogeneity choice, not a correctness one.

---

# Addendum — added post-hoc 2026-08-11 by the session that ran the four closing reg cells

> **Everything below this line was written after the audit above, by a different session than the
> one that produced it.** It is kept separate on purpose: the audit is a snapshot of what was found,
> and this is what one later session changed and measured. Where a gap is narrowed rather than
> closed, that is said. Nothing above was edited except one stale path (`v18_2/modal/` →
> `pipelines/modal/`), which this session broke by moving the folder.

## A1 · What this session closed

| gap | before | after | basis |
|---|---|---|---|
| **C** — `lane_host` | 43 missing | **42 missing** (this session's 3; the parallel session filled 2 more, and re-generation reset some keys to an explicit `None`) | declared `local:NVIDIA A40` on texas s7, california s7, california s100 — the three cells this session launched and watched on nespedgpu. **Not** inferred; the other 40 are left undeclared deliberately. |
| **D** — tie certificate | 21 of 24 reg cells missing | **20 of 24** | texas s100 back-filled from its own p1 result JSON (`[{2,2},{1,1},{1,2},{2,1},{1,0}]`), copied verbatim, not recomputed. |

Both back-filled fields carry a `post_hoc_fields` block in the sidecar naming who added them and on
what basis, so a reader can tell a driver-written field from a hand-written one.

## A2 · A defect behind gap D, now fixed at the source

The three reg cells that *had* the certificate all ran on the **A40**; the one that lacked it ran on
**Modal**. That was not a coincidence: `run_lane.sh`'s `sidecar_write` never copied
`ambiguous_rows`, while the A40 driver did. Identical recipe, identical code, and the disclosure
depended on which machine you happened to use.

`run_lane.sh` now reads the certificate from the p1 result JSON the cell just wrote and emits it in
the sidecar. **Future rented reg cells will not need this back-fill.**

## A3 · Cross-hardware: the accumulated evidence says the effect is at or below the reporting quantum

Gap C's concern rests on a cross-hardware deviation of **~0.086 pp**, and the section above
(written by the session that enumerated the pools) establishes something the original text missed:
**the mixing is universal**, not a texas-region quirk. Seeds 0/1 are local and 7/100 are rented
across all six states and all three families. That removes "re-run the one odd cell" as an option
and makes disclosure the only realistic close — which is the right conclusion.

What this session adds is a tighter measurement of *how much it actually matters*. Same fold, same
seed, same data, same code, A40 against H100 (california s100, fold 0):

| | Acc@1 | Acc@5 | Acc@10 | MRR | s/fold |
|---|---|---|---|---|---|
| local A40 | 0.3344 | 0.5455 | **0.6283** | 0.4346 | 555.7 |
| Modal H100 | 0.3344 | 0.5456 | **0.6283** | 0.4346 | 268.8 |

**Acc@10 — the banked metric — is identical to 4 dp.** Only Acc@5 and F1 move, by 1e-4.

This is consistent with the other cross-machine agreements the board already contains rather than
standing alone: texas cat has three seeds inside **0.008 pp** while spanning an A40 and an H100
(s0/s1 local 36.3225/36.3144, s7 rented 36.3190), and california reg's two rented-vs-local seeds
sit **0.0055 pp** apart. Several independent pairs, all agreeing at or below the quantum.

**How to read it.** The 0.086 pp figure came from an uncontrolled comparison (different family,
whole cells, three machines at once); the controlled pairs put the effect at ~0. Three honest
limits remain: the controlled pair is *one fold*; it compares A40↔H100 while several banked cells
ran on an **A100**, which no controlled pair covers; and none of this is a proof of bit-identity —
it is repeated agreement at the precision the board reports.

**Practical consequence:** disclose the design (seeds 0/1 local, 7/100 rented) because it is true
and cheap to state, but do not treat the mixture as a threat to the results. The evidence available
does not support re-running anything on those grounds.

## A4 · Gap A is worse than "missing" — the 30 cells carry a literal `"unknown"`

The field exists; its value is the string `unknown`. That matters for tooling: a naive
`if not d.get("commit_sha")` audit reports **0 missing** and passes. Any future check must test the
value, not the key.

Two lane cells now carry real SHAs (`85bcc588`, `25942582`), so the mechanism works — the 30 are a
capture gap in the driver at the time, not a limitation. This session did **not** back-fill them:
the runs predate it and assigning a SHA to someone else's execution would be exactly the invented
provenance the gap text warns against.

## A5 · Untouched

**B** (10 cells without `recipe_version`), **E** (both display bugs — `status.json` still derives
`current_n` from joint cells only, so it structurally cannot see a lagging cat or reg family), and
**F** (`TASKS.md`) are unchanged. The remaining 40 `lane_host` and 30 `commit_sha` back-fills are
left open for the same reason: they concern cells this session did not run.
