# REVISION_PLAN.md — what changes in the MobiWac text, and why

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
| joint-best recovery | the single-served-checkpoint convention recovered for **all 24 joint cells** (6 datasets x 4 seeds), which the board previously carried as `null` | reconstructed from each rundir's per-fold `standard_scores.json`; 9 cells filled from the banked lane archive under `docs/results/closing_data/v18_2/` |
| integrity check | every reconstructed cell reproduces its banked per-task sidecar to 4 decimals; reconstructed per-fold arrays agree with the board's own arrays to 5e-05 | verified for 24/24 cells |
| statistical battery | superiority and non-inferiority for both tasks at all six datasets, under **both** epoch-selection conventions, at two footings (n=20 pooled folds; n=4 per-seed means), Holm-corrected within each task family | §1.1, §1.2 |
| supporting contrasts | region against the protocol-matched Markov-1 floor; category against the majority-class and Markov-1 floors; shared-trunk arms paired fold-by-fold against the matched joint arms | §1.3, §1.4 |
| representation contrast | the place-level (HGI) dedicated category arm re-measured **under the current tuned recipe**, same folds, same head, same windowing, same precision, same logit adjustment; only the input representation differs | §1.5 — running; Alabama complete |

Every number below is read from a result artifact. Nothing is retyped from a previous draft.

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
"beats" = one-sided paired superiority, Holm-corrected, at the per-seed footing.
"matches" = TOST non-inferiority within the pre-pinned two-point margin.

**Joint-best (the served checkpoint):**

| dataset | Δ category | verdict | Δ region | verdict |
|---|---:|---|---:|---|
| Istanbul | +0.08 | matches | −0.08 | matches |
| Alabama | −0.19 | matches | −0.87 | matches |
| Arizona | −0.00 | matches | −0.44 | matches |
| Florida | **+0.19** | **beats** | −0.16 | matches |
| Texas | −0.13 | matches | **+1.21** | **beats** |
| California | −0.00 | matches | **+1.06** | **beats** |

**Diagnostic-best (per-task best epoch):**

| dataset | Δ category | verdict | Δ region | verdict |
|---|---:|---|---:|---|
| Istanbul | +0.11 | beats | +0.11 | beats |
| Alabama | −0.08 | matches | −0.45 | matches |
| Arizona | +0.04 | beats | −0.22 | matches |
| Florida | +0.24 | beats | +0.39 | beats |
| Texas | +0.04 | beats | +1.91 | beats |
| California | +0.12 | beats | +1.95 | beats |

Every cell that is not a "beats" is a "matches": no cell anywhere on the board falls outside the
two-point equivalence margin on either task. The joint model is never worse than the dedicated
models by an amount the protocol treats as meaningful.

### 1.3 Region: the joint model's clearest result

At the two largest U.S. datasets the joint model outperforms the dedicated region model under
either convention, and the effect is large relative to its own dispersion:

| dataset | dedicated | joint (served) | Δ | 90% CI of the per-seed mean |
|---|---:|---:|---:|---|
| Texas | 64.94 | 66.15 | **+1.21** | [+1.13, +1.29] |
| California | 63.48 | 64.54 | **+1.06** | [+1.03, +1.08] |

All 20 of 20 (seed, fold) pairs favour the joint model at both datasets. Against the
protocol-matched Markov-1 region floor, the joint model clears the floor by +6.05 (Texas) and
+5.45 (California) points; the dedicated model clears it by +4.84 and +4.39.

### 1.4 The shared trunk is a contributing component

Paired fold-by-fold against the matched joint arm at the same seed and folds:

| contrast | dataset | Δ category | Δ region |
|---|---|---:|---:|
| trunk severed − joint | Alabama | −0.31 (p = 0.43) | −0.14 (p = 0.25) |
| trunk severed − joint | Florida | −0.02 (p = 0.74) | **−0.23 (p = 0.0014)** |
| exchange neutralized − joint | Alabama | −0.15 (p = 0.48) | −0.00 (p = 0.98) |

Removing the trunk costs region accuracy at Florida with a clear paired signal (0 of 5 folds
favour the severed arm), and every other contrast points the same direction without reaching
significance at n=5. The trunk contributes; the size of that contribution is resolved at one
dataset and directionally consistent at the other. **This is the only supportable framing, and it
matches the author's directive: the trunk is one of the components producing the joint model's
region result.** The text must not claim the trunk carries the category task, and must not
describe it as doing nothing.

### 1.5 The representation contrast is materially smaller under the current recipe

Table 2 in the submitted paper contrasts the check-in-level and place-level representations. Its
place-level column was measured under a different training configuration than the one the paper now
reports. Re-measured under the current recipe (same folds, head, windowing, epochs, precision, and
logit adjustment; only the input representation differs), at Alabama, seed 0:

| arm | macro-F1 | per-fold |
|---|---:|---|
| check-in level | 30.77 | 32.26, 30.80, 29.61, 29.64, 31.52 |
| place level | 29.15 | 29.73, 29.74, 28.05, 28.44, 29.78 |
| **paired difference** | **+1.62** | 5 of 5 folds, two-sided paired t p = 0.0034 |

The submitted table reports **+29.31** at this dataset. That margin belongs to the earlier
configuration; under the current recipe the advantage is **+1.62**. Arizona and Istanbul are
running; the three largest datasets need their place-level inputs rebuilt first (~30 GB, several
hours) and are the main open cost item — see §6.

**This is the single largest change to the paper's claims and it is not optional.** Table 2 and
every sentence that leans on it must be rewritten to the measured margin.

---

## 2 · The one decision the author must make

**Which convention does the paper report?**

The paper currently reports diagnostic-best and discloses it. The two options:

- **(A) Report joint-best throughout.** One served artifact, both numbers read from it; the claim
  is exactly the model a deployment would run. Cost: the category headline becomes "matches at five
  datasets, beats at Florida" rather than "beats everywhere". Region keeps its two clear wins.
- **(B) Keep diagnostic-best as the headline, report joint-best alongside.** Preserves the stronger
  category ladder, but the headline numbers are not jointly achievable by one checkpoint, which a
  reviewer can and will ask about.

**Recommendation: (A).** It is the honest reading of a single-model claim, the paper's whole
argument is that *one* model serves both tasks, and the region result — the strongest thing on the
board — survives it intact. The plan below is written for (A); switching to (B) changes §4's table
and the verdict verbs, not the structure.

---

## 3 · What does not change

- The four-level graph construction, the sliding-window protocol, the user-disjoint folds, the
  seeds, the metric definitions, the baselines and their roles.
- Section 2 (Related Work), Section 3 (Problem), and the dataset table.
- The figure inventory. Figure 4's numbers change; its design does not.

---

## 4 · File-by-file edit list

### `tables/tbl3_results.tex` — the main results table
- Replace every joint-model column with the joint-best values from §1.2.
- Replace the verdict marks so that each cell's mark is the one its own test earns.
- Rewrite the provenance comment block to name the current board, the convention, and n per cell.
- Keep the external baselines and floors as they are; they are unaffected.

### `tables/tbl2_substrate.tex` — the representation contrast
- Replace the place-level column and the Δ column with the re-measured values (§1.5).
- State in the caption that both arms run the same recipe, folds, and windowing, and that the
  contrast isolates the input representation.
- If the three largest datasets are not rebuilt in time, the table reports the datasets that were
  measured and says so; it does not carry a mixed-recipe row.

### `sections/06_results.tex` — rewritten
- §6.1: the representation subsection reports +1.62 at Alabama (and its siblings when they land),
  not a two-order-larger margin. The claim becomes "a consistent, measurable advantage under a
  matched recipe", which is what the data supports.
- §6.2: the joint-model subsection reports the §1.2 ladder. Category: beats at Florida, matches
  elsewhere within the two-point margin. Region: beats at Texas and California, matches elsewhere.
  Every verb bound to its own test; no non-inferior cell upgraded to a win.
- §6.3: Istanbul keeps its role as the non-U.S. check; its numbers become the measured ones.
- Add one sentence, in passing, that the reported configuration follows a hyperparameter sweep
  covering both the joint and the dedicated models, so the comparison is between tuned arms.

### `sections/07_discussion.tex` — rewritten
- Lead with the region result, which is where the joint model is strongest.
- Frame the shared trunk as a contributing component on the evidence of §1.4, with the Florida
  paired result cited and the Alabama contrasts described as directionally consistent.
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
  matches both dedicated models everywhere and beats them where the region task is hardest.

### `figs/fig4_deltas.py`
- Regenerate from the new deltas. Same design, new inputs.

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
| place-level arm at Arizona, Istanbul | running | ~1 h |
| place-level arm at Florida, Texas, California | needs inputs rebuilt (~30 GB) before training | ~6-10 h, and 30 GB of a disk at 91% |
| joint-best at all 24 cells | **complete and verified** | — |
| statistical battery | **complete** | — |

The disk is the binding constraint on the three largest datasets. If the author prefers not to
spend it, Table 2 reports three datasets under the matched recipe and says plainly that the
remaining three were not re-measured, which is honest and costs one sentence.

---

## 7 · Author decisions needed before execution

1. **Convention**: (A) joint-best throughout — recommended — or (B) diagnostic-best headline with
   joint-best alongside.
2. **Table 2 scope**: rebuild the three largest datasets' place-level inputs, or report the
   datasets measured and disclose the rest.
3. **Anything in §4 to add, drop, or reorder** before the text is touched.
