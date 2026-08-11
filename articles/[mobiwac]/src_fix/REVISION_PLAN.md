# REVISION_PLAN.md, what changes in the MobiWac text, and why

> **Status: PROPOSAL. Nothing in `sections/`, `tables/` or `main.tex` has been modified.**
> The working copy `src_fix/` is a byte-identical branch of `src/` (commit `da97ecf7`), minus build
> residue. `src/` is the submitted version of record and stays untouched.
>
> **This document is the approval gate.** Read §1 (what the measurements say), then §2 (the one
> decision only the author can make), then §3–§6 (the file-by-file edit list). No core file is
> touched until the author approves.

---

## 0 · What was run to produce this plan

| step | what | where the numbers live |
|---|---|---|
| joint-best recovery | the single-served-checkpoint convention recovered for **all 24 joint cells** (6 datasets x 4 seeds), which the board carries as `null` | **banked to `docs/results/closing_data/v18/joint_best_perfold.json`**: 15 cells from the rundir artifacts on the training host, 9 from the lane archive under `docs/results/closing_data/v18_2/`, plus one fold recomputed with the same scoring function that writes the artifact |
| integrity check | every banked cell's fold-mean diagnostic-best reproduces its per-cell sidecar in `docs/results/closing_data/v18/` to 4 decimals; the banked per-fold arrays agree with the board's own arrays to 5e-05 | 24/24 cells, gate in the banked file's `integrity` field |
| statistical battery | superiority and non-inferiority for both tasks at all six datasets, under **both** epoch-selection conventions, at two footings (n=20 pooled folds; n=4 per-seed means), Holm-corrected within each task family | §1.1, §1.2 |
| supporting contrasts | region against the protocol-matched Markov-1 floor; category against the majority-class and Markov-1 floors; shared-trunk arms paired fold-by-fold against **loss-weight-matched** joint arms | §1.3, §1.4 |
| representation contrast | the place-level (HGI) dedicated category arm re-measured **under the current tuned recipe**, same folds, same head, same windowing, same precision, same logit adjustment; only the input representation differs | §1.5, sidecars banked to `docs/results/closing_data/v18_place_level/` |

Every number below traces to a result artifact inside the repository. Nothing is retyped from a
previous draft, and nothing rests on a file that lives only on the training host.

---

## 1 · What the measurements say

### 1.1 The two epoch-selection conventions do not tell the same story

The board scores each joint run two ways, and the paper reports one of them:

- **joint-best**: one served checkpoint per fold, chosen by the validation selector, both tasks read
  at that epoch. This is the deployable model.
- **diagnostic-best**: each task read at its own best epoch. This is a diagnostic upper bound; no
  single artifact achieves both numbers at once.

They differ materially, and the difference is one-sided (joint-best is always lower):

| dataset | category, joint-best − diagnostic-best | region, joint-best − diagnostic-best |
|---|---:|---:|
| Istanbul | −0.03 | −0.19 |
| Alabama | −0.11 | −0.42 |
| Arizona | −0.05 | −0.22 |
| Florida | −0.04 | −0.55 |
| Texas | −0.17 | −0.71 |
| California | −0.12 | −0.90 |

**Consequence: the verdict ladder changes with the convention.** Both ladders below use the same
paired data, the same tests, and the same Holm correction (m=6 within each task family); they differ
only in which epoch the joint model is read at.

### 1.2 The verdict ladder, both conventions

Deltas are joint minus dedicated, in points, averaged over 4 seeds x 5 folds (n=20).
"outperforms" = one-sided paired superiority, Holm-corrected within the task family, at the per-seed
footing. "matches" = TOST non-inferiority within a two-point margin.

> **A registration gap the author must close before this ladder is written.** The pre-registered
> protocol pins a two-point equivalence margin on the **region** axis only, and it explicitly forbids
> reusing a margin across axes. There is no registered equivalence margin for the **category** axis.
> Under convention (A) five of six category cells would be reported as "matches" on a margin that was
> never registered for that axis. Two honest routes: register a category-axis margin now, with its own
> justification and logged as a deviation, or report those five cells as **unresolved** (not superior,
> equivalence not tested) rather than as matches. **Do not silently borrow the region margin.**
> The region "matches" cells are unaffected: their margin is registered.

**Joint-best (the served checkpoint):**

| dataset | Δ category | verdict | Δ region | verdict |
|---|---:|---|---:|---|
| Istanbul | +0.08 | matches | −0.08 | matches |
| Alabama | −0.19 | matches | −0.87 | matches |
| Arizona | −0.00 | matches | −0.44 | matches |
| Florida | **+0.19** | **outperforms** | −0.16 | matches |
| Texas | −0.13 | matches | **+1.21** | **outperforms** |
| California | −0.00 | matches | **+1.06** | **outperforms** |

**Diagnostic-best (per-task best epoch):**

| dataset | Δ category | verdict | Δ region | verdict |
|---|---:|---|---:|---|
| Istanbul | +0.11 | outperforms | +0.11 | outperforms |
| Alabama | −0.08 | matches | −0.45 | matches |
| Arizona | +0.04 | outperforms | −0.22 | matches |
| Florida | +0.24 | outperforms | +0.39 | outperforms |
| Texas | +0.04 | outperforms | +1.91 | outperforms |
| California | +0.12 | outperforms | +1.95 | outperforms |

Every cell that is not an "outperforms" is within the two-point band: no cell anywhere on the board falls outside the
two-point equivalence margin on either task. The joint model is never worse than the dedicated
models by an amount the protocol treats as meaningful.

### 1.3 Region: the joint model's clearest result

At the two largest U.S. datasets the joint model outperforms the dedicated region model under
either convention, and the effect is large relative to its own dispersion:

| dataset | dedicated | joint (served) | Δ | 90% CI of the per-seed mean |
|---|---:|---:|---:|---|
| Texas | 64.94 | 66.15 | **+1.21** | [+1.13, +1.29] |
| California | 63.48 | 64.54 | **+1.06** | [+1.03, +1.08] |

All 20 of 20 (seed, fold) pairs favor the joint model at both datasets. Against the
protocol-matched Markov-1 region floor, the joint model clears the floor by +6.05 (Texas) and
+5.45 (California) points; the dedicated model clears it by +4.84 and +4.39.

### 1.4 What the shared-trunk ablations actually measure

Paired fold-by-fold at the same seed and the same folds, **both arms at the same 0.75 category
weight**, which is what the ablation drivers ran. The sign convention is **severed arm minus joint
arm**, so a negative value means the joint model (trunk present) scored higher:

| dataset | task | joint | severed | Δ (severed − joint) | folds severed higher | p |
|---|---|---:|---:|---:|---:|---:|
| Alabama | category | 27.38 | 27.37 | −0.02 | 3/5 | 0.94 |
| Alabama | region | 69.68 | 69.55 | −0.14 | 3/5 | 0.38 |
| Alabama | category (exchange neutralized) | 27.38 | 27.23 | −0.15 | 3/5 | 0.48 |
| Alabama | region (exchange neutralized) | 69.68 | 69.68 | −0.00 | 3/5 | 0.98 |
| Florida | category | 35.88 | 35.88 | +0.00 | 2/5 | 0.97 |
| Florida | region | 77.26 | 77.28 | +0.03 | 3/5 | 0.18 |

Test: paired t on the five matched folds, two-sided, at a single seed. The pre-registered rank test
cannot reach below 0.0625 at n=5, so the departure is logged rather than hidden.

**Every one of the six contrasts is unresolved.** The largest is 0.15 points and the smallest p is
0.18. At this footing the ablations neither credit the trunk nor convict it: they establish that the
effect, in either direction, is below what a single seed over five folds can detect.

Three facts bound what may be written from this:

1. Both ablation datasets sit in the "matches" column of §1.2 on region. Neither is a dataset where
   the joint model outperforms the dedicated region model.
2. The trunk's contribution at **Texas and California, the two datasets where the joint model does
   outperform the dedicated region model**, has been screened but not measured to board strength.
   A single-fold screen at seed 0 on the earlier representation exists, and it reads **against** a
   trunk attribution: severing the trunk moves region by −0.12 (Texas) and −0.10 (California), and
   removing the category task entirely moves California by −0.08, so the advantage survives both.
   That screen is one fold, one seed, on the earlier representation, which is why it licenses no
   published claim in either direction. It does, however, forbid asserting the opposite.
3. These arms ran before the current configuration was fixed, so on the category axis they are not
   the model the paper reports (their joint category values sit 1.7 to 3.3 points below the reported
   cells). On the region axis they are within 0.05 points of the reported values, which is why the
   region reading above is usable and a category reading from these arms would not be.

**The reason to decline the attribution is positive evidence, not absence of evidence.** At the two
datasets carrying the region result, the screen removed the trunk and then removed the category task
outright, and the region advantage moved by less than 0.15 points in every arm while category
collapsed to single digits, which confirms the task really was switched off. A screen that had the
resolution to see the causal hypothesis and did not see it is a stronger basis for declining to
assert it than an unresolved contrast would be. It is still one fold at one seed on the earlier
representation, so it licenses no published claim; it does foreclose asserting the converse.

So the supportable statements are: the shared trunk is a component of the architecture whose region
result is reported in §1.3; where its contribution has been probed, at two datasets sitting at
parity on region, no effect is resolvable in either direction; and at the two datasets carrying the
region result the only evidence available reads against a sharing explanation. The text
must not claim the trunk carries the category task, must not attribute the Texas and California
region gains to it, and must not describe it as doing nothing. **No causal attribution sentence for
the region gain survives this evidence.** The honest construction names the joint architecture as a
whole, with the trunk as one of its components, and leaves the decomposition open.

### 1.5 The representation contrast is materially smaller under the current recipe

Table 2 in the submitted paper contrasts the check-in-level and place-level representations. Its
place-level column was measured under a different training configuration than the one the paper now
reports. Re-measured under the current recipe (same folds, same head, same windowing, same epochs,
same precision, same logit adjustment; only the input representation differs), at seed 0:

| dataset | check-in level | place level | gap | folds favoring check-in level | p | submitted Table 2 |
|---|---:|---:|---:|---:|---:|---:|
| Istanbul | 35.35 | 29.07 | **+6.29** | 5/5 | 0.00003 | +28.09 |
| Alabama | 30.77 | 29.15 | **+1.62** | 5/5 | 0.0034 | +29.31 |
| Arizona | 34.51 | 31.93 | **+2.58** | 5/5 | 0.0004 | +27.63 |
| Florida | 37.36 | 37.13 | +0.23 | 5/5 | 0.067 | +39.62 |
| Texas | running | running |, |, |, | +37.47 |
| California | running | running |, |, |, | +37.95 |

Test: paired t on the five matched folds, two-sided, at seed 0. Both arms run the same head, the
same five folds, the same sliding windows, the same 50 epochs, the same precision, and the same
logit adjustment; only the input representation differs.

Under the current configuration the gap is **+0.23 to +6.29** where it has been measured, against
the **+27.6 to +39.6** the submitted table carries. The direction is consistent, every dataset has
all five folds favoring the check-in-level representation, and three of the four reach
significance. Florida does not: at +0.23 points with p = 0.067 it is the first dataset where the two
representations are, on this evidence, at parity.

**This is the single largest change to the paper's claims and it is not optional.** Table 2, the
abstract's "about 28 to 40 points" clause, and every sentence that leans on either must be rewritten
to the measured gap, and the claim itself must weaken from "transforms the task" to "a consistent
but modest advantage, clear at three of four datasets measured". Texas and California are building
and training now (§6); each rebuilds its place-level inputs, trains, scores, and releases the disk
before the next one starts.

---

## 2 · The one decision the author must make

**Which convention does the paper report?**

**Correction of record: the submitted paper already reports joint-best.** The author ruled this on
2026-07-18, and `tables/tbl3_results.tex` and `sections/06_results.tex` both state it. So option (A)
below is **no change of convention**, and option (B) would be a switch, in the direction that turns
five category cells from unresolved into wins.

- **(A) Keep joint-best, as the paper already does.** One served artifact, both numbers read from
  it; the claim is exactly the model a deployment would run. Cost: the category headline becomes
  "outperforms at Florida, within the two-point band elsewhere" rather than "outperforms
  everywhere".
- **(B) Switch to diagnostic-best.** Recovers a stronger category ladder, but the headline numbers
  would not be jointly achievable by one saved model, and the paper's existing disclosure sentence
  would have to be reversed. A reviewer will ask which single model produced both columns.

**Recommendation: (A), which is also the status quo.** It is the honest reading of a single-model
claim, the paper's whole argument is that one model serves both tasks, and the region result at the
two largest datasets survives it intact. The plan below is written for (A).

**What (A) costs beyond the category headline, which the author must see before ruling.** The
introduction, Table 3's row ordering, Figure 4 and a conclusion sentence all carry a scaling claim:
across the five U.S. states, the region gain grows with the number of regions. That ordering holds
under diagnostic-best and **fails under joint-best**, where California (8,501 regions, +1.06) sits
below Texas (5,265 regions, +1.21). Under (A) the claim must weaken to its surviving form: the two
datasets with the most regions are the two where the joint model outperforms the dedicated region
model, and the ordering within that pair does not follow region count.

---

## 3 · What does not change

- The four-level graph construction, the sliding-window protocol, the user-disjoint folds, the
  seeds, the metric definitions, the baselines and their roles.
- Section 2 (Related Work), Section 3 (Problem), and the dataset table.
- The figure inventory. Figure 4's numbers change; its design does not.

---

## 4 · File-by-file edit list

### `main.tex`, the abstract (four clauses go stale)
The abstract lives in `main.tex`, not in `sections/`. Four of its claims change:
1. "improves next-category prediction over a standard place embedding by about $28$ to $40$ points
   of macro-averaged F1" becomes the measured gap of §1.5.
2. "outperforms a dedicated category model on every dataset (about $+5$ to $+9$ macro-F1)" becomes
   the §1.2 category ladder.
3. "outperforms the dedicated region model on four of the six" becomes two of six under convention
   (A).
4. "At Istanbul ... the joint model is nevertheless ahead on region ($+0.19$ Acc@10, statistically
   supported)" inverts in sign under convention (A) and must be restated or dropped.
Keep the abstract's structure, its motivation-first opening, and its keyword list.

### `tables/tbl3_results.tex`, the main results table
- Replace every joint-model column with the joint-best values from §1.2.
- Replace the verdict marks so that each cell's mark is the one its own test earns.
- Rewrite the provenance comment block to name the current board, the convention, and n per cell.
- Keep the external baselines and floors as they are; they are unaffected.

### `tables/tbl2_substrate.tex`, the representation contrast
- Replace the place-level column and the Δ column with the re-measured values (§1.5).
- State in the caption that both arms run the same training configuration, folds, and windowing, and
  that the contrast isolates the input representation.
- If Texas and California do not land in time, the table simply covers the datasets it covers, and
  the caption names them. Write it as the table's stated scope, never as an account of what was or
  was not re-measured: directive 1 forbids the retrospective form.
- Re-check the two supporting controls in §6.1 against the new gap before reusing them. The
  feature-concatenation control raises the place-level arm by about two points at Alabama, which is
  larger than the new Alabama gap of +1.62; the sentence built on that control does not survive
  there and must be rescoped or dropped.

### `sections/06_results.tex`, rewritten
- §6.1: the representation subsection reports +1.62 at Alabama (and its siblings when they land),
  not a two-order-larger margin. The claim becomes "a consistent, measurable advantage under a
  matched recipe", which is what the data supports.
- §6.2: the joint-model subsection reports the §1.2 ladder. Category: outperforms at Florida, and
  within the two-point band elsewhere. Region: outperforms at Texas and California, within the band
  elsewhere.
  Every verb bound to its own test; no non-inferior cell upgraded to a win.
- §6.3: Istanbul keeps its role as the non-U.S. check; its numbers become the measured ones.
- Add one sentence, in passing, that the reported configuration follows a hyperparameter search over
  batch size and learning rate, run on both the joint and the dedicated category models. **Scope it
  honestly**: the dedicated region models use a fixed configuration and were not searched, and at
  Texas and California the joint configuration was transferred from the datasets where the search
  ran rather than validated there, which is the disclosure the postponed-work record requires.
  Write the sentence so it is true of all three arms, or name the ones it covers. This scoping is
  also what makes the Texas and California tuning hypothesis in the discussion coherent rather than
  self-contradictory: the hypothesis rests on exactly this transfer.

### `sections/07_discussion.tex`, rewritten
- Lead with the region result, which is where the joint model is strongest.
- Present the joint model as functional and producing the region result at Texas and California,
  naming the shared trunk as one component of the architecture that delivers it. Do **not** write a
  sentence that attributes the region gain to the trunk: §1.4 shows the trunk's isolated
  contribution is unmeasured at exactly the two datasets where the gain occurs, and the one
  resolved ablation (Florida region) favors the severed arm. State the decomposition as open, and
  name the Texas and California trunk ablation as the experiment that would settle it.
- Add the tuning hypothesis for the category gap at Texas and California: the joint model there
  inherits a configuration selected without a joint-specific search at those two datasets, and a
  dedicated search for the joint model at those datasets is the natural next step. Stated as a
  hypothesis, not a finding.
- Keep the limitations, adjusted to the new ladder.

### `sections/05_setup.tex`
- Update the metrics subsection to define the reported convention in one sentence.
- Add the sweep mention (one clause, cross-referenced from §6).

### `sections/04_method.tex` and `sections/03_problem.tex`
- Add the elapsed-time node features to the input description: alongside category, time of day, and
  day of week, each check-in node carries elapsed-time values measured up to that visit. One
  sentence in §4; the feature-width figure in the appendix table follows.

### `sections/01_introduction.tex` and `sections/08_conclusion.tex`
- Re-point the contribution bullets and the closing claims at the new ladder. The introduction
  currently promises a category win everywhere; under convention (A) it promises one model that
  matches both dedicated models everywhere and outperforms them where the region task is hardest.

### `figs/fig4_deltas.py` and `figs/fig4_deltas.tex`
- Regenerate from the new deltas. Same design, new inputs. The caption's ordering rationale follows
  the scaling claim and changes with it (§2).

### Sentences carrying magnitudes that move, wherever they sit
The external baselines and the floors are unchanged as values, but several sentences state the
paper's distance from them, and those distances move. Two are already false:

**The conclusion's "at least 4 Acc@10 points over the strongest region reference" fails.** Measured
against the per-dataset strongest external, under the served checkpoint:

| dataset | joint | strongest external | gap |
|---|---:|---|---:|
| Istanbul | 75.08 | 69.33 (ReHDM) | +5.75 |
| Alabama | 69.24 | 65.38 (ReHDM) | **+3.86** |
| Arizona | 59.04 | 53.00 (ReHDM) | +6.04 |
| Florida | 76.54 | 72.99 (STAN) | **+3.55** |
| Texas | 66.15 | 61.67 (STAN) | +4.48 |
| California | 64.54 | 58.52 (STAN) | +6.02 |

The floor is **+3.55**, not 4. Either restate the bound or drop the numeral. The companion claim of
"at least 33 macro-F1 points over POI-RGNN" on category needs the same re-derivation before reuse.

Also re-derive:
- the region gap over the Markov floor (now +4.07 to +10.02 under the served checkpoint);
- the one-point dispersion claim in `05_setup.tex`, which does not hold at Alabama under convention
  (A) and needs its number re-read from the battery.

Grep for each magnitude before assuming a sentence is safe; a sentence can name no table and still
carry a stale number.

---

## 5 · Dissertation follow-through (out of scope for this commit, listed so it is not lost)

`articles/dissertacao/src/chapters/5_mobiwac/` re-typesets this paper and is kept textually
identical where it reproduces it. Chapter 2's representation section and the Check2HGI appendix
both describe the check-in node's input width and must gain the elapsed-time columns. Chapter 6's
closing claims inherit the new ladder. These follow after the paper text is approved.

---

## 6 · Open items and their cost

| item | status | cost |
|---|---|---|
| joint-best at all 24 cells | **complete and verified** |, |
| statistical battery, both conventions | **complete** |, |
| place-level arm at Istanbul, Alabama, Arizona | **complete** |, |
| place-level arm at Florida, California, Texas | running: build, train, score, reclaim, one dataset at a time | ~6-10 h |
| trunk contribution at Texas and California | **not measured**, and it is the ablation that would license any attribution sentence for the region gain (§1.4) | ~22 h |

The disk was the binding constraint on the three largest datasets, at 38 GB free against a 9.5 GB
input for the largest. The running job therefore builds one dataset's place-level inputs, trains,
scores, and releases them before starting the next, so peak usage stays at one dataset. If it does
not finish in time, Table 2 reports the datasets measured under the matched recipe and says plainly
the scope it covers, which costs one sentence and no honesty.

---

## 7 · Author decisions needed before execution

1. **Convention**: (A) joint-best throughout, recommended, or (B) diagnostic-best headline with
   joint-best alongside.
2. **Table 2 scope**: rebuild the three largest datasets' place-level inputs, or report the
   datasets measured and disclose the rest.
3. **Anything in §4 to add, drop, or reorder** before the text is touched.
