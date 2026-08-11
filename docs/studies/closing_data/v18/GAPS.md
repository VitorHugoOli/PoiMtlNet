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
**Where to get it:** the Modal driver under `docs/studies/closing_data/v18_2/modal/` — whatever
commit was baked into the image, or the checkout the lane ran. If a single SHA covers all lane cells,
one value fills all 30. **If the SHA is genuinely unrecoverable, say so in PROVENANCE.md rather than
back-filling a guess** — an invented SHA is worse than an admitted gap.

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
| 5 | **E** — two display bugs | small code fix | one is currently correct only by accident |
| 6 | **F** — `TASKS.md` | writing | last unmet §7 deliverable |

**None of these require re-training.** Items 1–4 are back-fills into existing sidecars; 5 is a fix in
`make_results.py` / `status_update.py`; 6 is a document. The only optional compute is the ~57 min
texas s100 re-run under item C, and that is a homogeneity choice, not a correctness one.
