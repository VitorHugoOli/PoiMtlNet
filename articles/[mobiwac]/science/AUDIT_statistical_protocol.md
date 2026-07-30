# AUDIT REPORT — statistical-protocol integrity (MobiWac 2026)

**Scope.** Re-derivation of every claim in `articles/[mobiwac]/science/ISSUE_statistical_protocol.md`
against the committed tree, plus the pre-registered / executed / claimed reconciliation the issue asks for.
**No file was edited.** Verified 2026-07-25 on `main` (working tree at `/Users/vitor/Desktop/mestrado/ingred`).

Numbers below are quoted from artifacts or produced by re-running the repository's own test conventions
(`scripts/closing_data/superiority_wilcoxon.py`, `region_match_tost.py`) on committed per-fold arrays.
Nothing is recomputed from memory.

---

## 0 · Verdict on the issue file itself

| Issue claim | Verdict |
|---|---|
| D1 Wilcoxon -> paired t departure, logged in repo, undisclosed in paper | **CONFIRMED**, with two corrections (below) |
| D2 region superiority never pre-registered, in no multiplicity family | **CONFIRMED**, and understated: the family is also absent from the paper's own applied correction |
| D3 "fixed in an analysis plan during development" overstates the record | **CONFIRMED** |
| "No number and no verdict is in question" | **CONFIRMED** by independent re-derivation of all six datasets, both tasks |
| Only one protocol of record exists | **CONFIRMED** (path corrected, see §1) |
| The deviation log "is part of the released bundle" | **REFUTED** — the released bundle contains neither the protocol nor its deviation log (§1e). The issue file raised this as an open question (Step 1(e); §7 acceptance criterion 6) rather than asserting an answer; the answer is negative, and it is the most serious defect found. |
| R2 "Blocked at >=1 cell: Istanbul's per-fold category ceiling is not in the committed tree" | **CONFIRMED** and is the only blocker (§5) |
| R2 "p ~ 9.5e-07, 20/20 positive" | **CONFIRMED** by re-run (§5) |
| Section 3 calibration CIs (Ist +0.15..+0.23, FL +0.67..+0.76, TX +2.10..+2.13, CA +2.19..+2.21) | **CONFIRMED** to the printed precision |
| "Category superiority clears Holm at every dataset (worst adjusted p ~ 1e-06)" | **CONFIRMED** (worst Holm-adjusted p = 1.0e-06 at the reported footing) |
| Seed-level pairing is "the more conservative and defensible footing" | **PARTLY** — plausible and worth stating, but it is an argument, not an artifact; the repo records availability, not conservatism, as the reason (§2) |

---

## 1 · STEP 1 — What was PRE-REGISTERED

**Protocol of record.** Exactly one exists:
`docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md` (git-tracked; first committed
`c96c67e3`, 2026-06-21, header: "STATUS: PRE-REGISTERED. Commit this BEFORE the board unblinds").
The path REV-007 cites (`docs/studies/closing_data/STATISTICAL_PROTOCOL.md`) is the pre-rename location;
`a2cb7b4a` (2026-07-13) renamed it into `v17_completion/` (`git log --follow` shows R099). No second
protocol exists: `docs/research/evaluation_protocol_review.md`, `docs/archive/.../SPLIT_PROTOCOL.md` and
`.../MTL_ABLATION_PROTOCOL.md` are unrelated. `articles/[BRACIS]_Beyond_Cross_Task/STATISTICAL_AUDIT.md §0`
is the inherited grounding (three power regimes, n=5 ceiling) and registers TOST only for the substrate axis.

**Timing check (does "pre-registered" hold?).** Protocol committed 2026-06-21. The first board result JSONs
land 2026-06-22 (FL/CA region ceilings) and 2026-06-23 to 06-24 (AL/FL/CA/TX joint cells). The protocol
predates the unblind. The a-priori claim is sound *for what the document contains*.

**(a) Registered hypothesis families, per task and per dataset.** §1 table, family (A), verbatim:

> `cat: superiority (MTL > STL); reg: **non-inferiority** (MTL not worse than STL by more than delta_reg)`

The family is enumerated once more in §5.2, verbatim:

> "The headline family is small and fixed: {6 states} x {cat superiority, reg non-inferiority}."

So: **per task, not per dataset.** The assignment is uniform across the six datasets; no per-dataset
reasoning is recorded anywhere in the document. Family (B) is the separate baselines-vs-ours grid.

**(b) Registered test, pairing level, and n.** §2, verbatim:

> "Paired Wilcoxon signed-rank on the matched per-fold Deltas, **multi-seed pooled** (n=20 = 4 seeds x 5 folds ...)".

Pairing is per-fold and licensed by §4 ("Pair ONLY when both arms share the same folds"), which explicitly
names family (A) as paired by construction. For the region equivalence cells, §3.3 registers paired TOST at
delta_reg = 2 pp, reporting the TOST p and the 90% CI against the margin. §2 also pre-records the n=5 exact
Wilcoxon floor (0.0312 one-sided) as a known power ceiling; it does **not** anticipate an n=4 footing.

**(c) Registered multiplicity families, and exclusions.** §5.2, verbatim:

> "Apply Holm-Bonferroni **within the cat-superiority set** (6 states) and report the reg-non-inferiority TOST
> cells with their own delta_reg verdict (TOST cells are equivalence tests, not superiority tests, and are not
> pooled into the cat Holm family)."

§5.1 puts the baseline grid in its own Holm family, split by comparison arm. §5.3 excludes descriptive cells.
**The only registered superiority-correction family is the six-dataset category set.**

**(d) Is region superiority registered anywhere? NO.** Confirmed three ways: (i) grep of the protocol returns
no region-superiority family; §1's family (A) row and §5.2's family enumeration both pin region to
non-inferiority only; (ii) §7's DO/DON'T contract instructs region to be written "as TOST non-inferiority from
the start"; (iii) the original 2026-06-21 text (`git show c96c67e3:...`) is identical on both points, so this
is not a later deletion. The only place region superiority appears in the protocol is the **§8 entry dated
2026-07-18**, which is post-unblind and reports results, not a registration.

**(e) What the deviation log records, where, and whether it is released.**
§8 mandates that deviations be logged in `docs/studies/closing_data/log.md`. Three findings:

1. **The mandated location does not carry the deviation.** `log.md` contains no Wilcoxon-to-t entry. Its only
   related line is the 2026-07-13 PR-#63 record ("n=4 Wilcoxon floor 0.0625 disclosed, powered paired-t per
   protocol"), which is a merge note, not a deviation entry with cell / registered rule / deviation / reason.
2. **The actual deviation log lives elsewhere:** `v17_completion/stats_n20/RESULTS.md`, section
   "Deviation log (protocol §8)", three numbered entries (seed-level n=4 pairing; paired t alongside the
   registered Wilcoxon; Holm applied to the t family). It is complete and correctly reasoned. Its own header
   line 20 cites "protocol §8's powered-t deviation" — **§8 contains no such entry**; that forward reference is
   dangling. This is the issue file's D1 confirmed, and the ISSUE file's own §2.2 pointer is correct.
3. **The released bundle contains neither.** The paper's footnote 1 (`01_introduction.tex:22`) points at
   `github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac`. The `mobiwac` branch (local `689b0d6e`, remote
   `673fbd27`) ships **zero** files under `docs/`: no `STATISTICAL_PROTOCOL.md`, no `stats_n20/RESULTS.md`, no
   `log.md`, no `joint_best/`. It ships the three test scripts only. Therefore §5.3's sentence "The assignment
   and the margin were fixed in an analysis plan during development **and are released with the code**" is
   **UNSUPPORTED as written**: a reviewer following the footnote cannot read the plan. The released
   `README.md` §6 asserts "Pre-registered tests (margins and test families fixed before the final runs)" and
   labels `superiority_wilcoxon.py` as running "region superiority (FL/CA/TX)" under that heading, which
   repeats the unregistered claim in the artifact the reader can actually reach.

---

## 2 · STEP 2 — What was EXECUTED

**The reported footing (what Table III and Section 6.2 stand on).** Joint-best convention (one saved model per
fold at the `geom_simple`-selected validation epoch, `min_best_epoch=0`), n=20 = 4 seeds {0,1,7,100} x 5 folds
on both arms at all six datasets. Superiority is a **paired t on per-seed means, n=4**, with the 90% CI;
Holm m=6 across the six category cells; TOST at delta_reg = 2 pp for AL and AZ.

**Independent re-derivation.** I rebuilt both arms from the committed per-fold arrays and re-ran the
repository's own conventions. Sources: MTL joint-best per-fold at AL/AZ/FL/Istanbul from
`docs/studies/closing_data/joint_best/data/j1_results.json` (`per_run[].jb_{cat,reg}_folds`, 16 runs); at CA/TX
from `docs/results/closing_data/catx_v17_n20/joint_best/{california,texas}_s{0,1,7,100}.json`
(`joint_best.{cat,reg}_per_fold`, 8 runs). Dedicated category ceilings from
`v17_completion/cat_ceiling_sweep/sweep_results/<arm>_s<seed>.json` (`cat_per_fold`; AL = bs2048@0.005,
AZ/FL/CA/TX = bs8192@0.005 per `CEILINGS_N20_FINAL.md`). Dedicated region ceilings from
`docs/results/P1/region_head_*_ovl_stl_reg_{s0,topup_s1,topup_s7,topup_s100}.json`
(`heads.next_stan_flow.per_fold[].top10_acc` x 100). Istanbul category ceiling from the per-seed scalars
`v17_completion/h3_istanbul/step3_runs/cat_ceil_s{0,1,7,100}.txt`.

Region, joint-best, seed-level n=4 (reproduced vs printed in `06_results.tex`):

| Dataset | Delta reproduced | 90% CI reproduced | Paper prints | Verdict |
|---|---:|---|---|---|
| Istanbul | +0.1936 | (+0.154, +0.234) | +0.19; +0.15 to +0.23 | MATCH |
| AL | -0.4155 | (-0.631, -0.200) | -0.41; -0.63 to -0.20 | MATCH |
| AZ | -0.0031 | (-0.079, +0.073) | 0.00; -0.08 to +0.07 | MATCH |
| FL | +0.7124 | (+0.665, +0.760) | +0.71; +0.67 to +0.76 | MATCH (CI lower bound rounds 0.665 -> 0.67) |
| TX | +2.1124 | (+2.097, +2.128) | +2.10 to +2.13 | MATCH |
| CA | +2.2012 | (+2.190, +2.213) | +2.19 to +2.21 | MATCH |

Category, joint-best, seed-level n=4, Holm m=6 on the paired t:

| Dataset | Delta cat | seeds+ | t p (1-sided) | Holm-adj | reject | exact Wilcoxon p at n=4 |
|---|---:|---|---:|---:|:--:|---:|
| AL | +7.690 | 4/4 | 4.49e-07 | 1.0e-06 | yes | 0.0625 |
| AZ | +9.350 | 4/4 | 3.22e-07 | 1.0e-06 | yes | 0.0625 |
| FL | +5.332 | 4/4 | 6.87e-09 | 4.1e-08 | yes | 0.0625 |
| CA | +6.442 | 4/4 | 3.49e-07 | 1.0e-06 | yes | 0.0625 |
| TX | +7.446 | 4/4 | 2.52e-07 | 1.0e-06 | yes | 0.0625 |
| Istanbul | +8.584 | 4/4 | 1.61e-07 | 8.1e-07 | yes | 0.0625 |

Worst Holm-adjusted p = 1.0e-06, matching the value quoted in `CLAUDE.md §2`, `PAPER_PLAN.md §3`, and
`STATISTICAL_PROTOCOL.md §8`. The paper's "corrected p < 0.001" is true with four orders of magnitude to spare.
The n=4 exact one-sided Wilcoxon sits at its 0.0625 floor at every cell, confirming deviation-log entry #2.

TOST at delta_reg = 2 pp on the same arrays: AL p = 2.1e-04, AZ p = 4.7e-06, both CIs far inside +/-2 pp.
Istanbul and FL also pass TOST (9.2e-07, 4.2e-06); TX and CA do not, because their CIs sit **above** +2 pp,
which is the favourable direction and is exactly how `stats_n20/RESULTS.md §1b` records it.

**Correction to the issue file, D1.** The issue file says the departure was "forced by artifact availability"
(MTL per-fold sidecars A40-only). That was true when `stats_n20/RESULTS.md` was written (2026-07-13, LIMITS #2).
It is **no longer true today**: the joint-best re-score committed per-fold arrays for all six datasets on the MTL
side (`j1_results.json` + `catx_v17_n20/joint_best/`, both added 2026-07-13). The constraint has moved: the one
remaining gap is on the **dedicated** side at Istanbul (§5). Any remedy text that repeats "the per-fold values
are not available" would be stale.

**Correction to the issue file, D1 reason.** The repo's recorded reason for reporting the t is the **power floor**
(entry #2: at n=4 the exact Wilcoxon cannot clear alpha for any effect size), not the pseudo-replication argument
the issue file offers in its §3. The pseudo-replication reading is a defensible post-hoc justification and is
worth making in a response letter, but it is not what the artifact records, and a paper sentence should not
attribute it to the plan.

**Script text asserting an unregistered registration (the m1_stats_n20.py lead) — CONFIRMED, and it is not
alone.** Three occurrences, all git-tracked:

1. `docs/studies/closing_data/v17_completion/stats_n20/m1_stats_n20.py:333` prints
   `"superiority (the pre-registered reg-'beats' family, superiority_wilcoxon.py)"`. The phrase reaches the
   committed output at `stats_n20/m1_full_output.txt:83,91` (CA and TX). **UNSUPPORTED**: no such family is in
   the protocol.
2. `stats_n20/RESULTS.md §1b` repeats it: "the reg cells land in the pre-registered reg-**superiority** family
   (`superiority_wilcoxon.py` pins FL/CA/TX as 'the beats')". This states the script as the registration
   authority. **UNSUPPORTED.**
3. `scripts/closing_data/superiority_wilcoxon.py:1-6` is itself the source of the belief. Its docstring opens
   "Pre-registered superiority tests (STATISTICAL_PROTOCOL.md §2, §5.2)" and then lists
   "Region (FL/CA/TX, the 'beats'): superiority MTL reg > STL dedicated reg ceiling". §2 licenses the *test*
   for "any claim with a sign"; §5.2 does **not** enumerate a region-superiority family. The script was first
   committed `1e3449e6`, **2026-06-25** — three days after the FL/CA region ceilings landed (2026-06-22) and one
   day after TX (2026-06-24). The dataset selection FL/CA/TX was therefore made **after** those cells were
   readable. This is the concrete evidence that region superiority is post-hoc, and it is stronger than the
   issue file's argument from silence. **This script ships in the released bundle.**

---

## 3 · STEP 3 — What the PAPER claims

Every sentence in `articles/[mobiwac]/src/` asserting a test, a pre-registration, a correction, or a
significance verdict. Quotations are abbreviated where long; file:line is exact.

| # | Location | Quoted claim |
|---|---|---|
| P1 | `main.tex:80-86` (abstract) | "It outperforms a dedicated category model on every dataset ... and outperforms the dedicated region model on four of the six, matching it (statistically, within two points) on the other two." and "At Istanbul ... the joint model is nevertheless ahead on region ($+0.19$ Acc@10, statistically supported)." |
| P2 | `01_introduction.tex:31-36` | "In one forward pass, it outperforms a dedicated single-task category model on every dataset ... On region, it outperforms at four of the six datasets and stays statistically non-inferior within a two-point margin (TOST, a test of equivalence) at the remaining two." |
| P3 | `01_introduction.tex:40-41` | "matches the dedicated model (TOST, $\pm2$ percentage points, or pp) at the small region counts and outperforms it at the large region counts" |
| P4 | `05_setup.tex:42` | "We fix the assignment in advance, before reading any results: where the joint model was expected to outperform, we test superiority; where it was expected only to match the dedicated model, we test non-inferiority." |
| P5 | `05_setup.tex:42` | "The assignment and the margin were fixed in an analysis plan during development and are released with the code" |
| P6 | `05_setup.tex:42` | "superiority is tested with a paired $t$ on the per-seed means and reported with the 90\% confidence interval of the paired difference" |
| P7 | `05_setup.tex:42` | "On every dataset, we use four seeds for both models ($4\times5=20$ measurements) and pair the per-seed means ($n{=}4$ per dataset), with a Holm correction across the comparisons." |
| P8 | `05_setup.tex:42` | "The non-inferiority claim is that the joint model is no worse than the dedicated model by more than a two-point margin; we test it with the two one-sided tests (TOST) procedure." |
| P9 | `05_setup.tex:42` | "The two-point margin is fixed in advance as well" (+ the deployment rationale) |
| P10 | `05_setup.tex:42` | "The equivalence is well powered: the paired difference's standard deviation is 0.01 to 0.18 points across the datasets ... the intervals pass a margin as small as one point at Alabama and Arizona." |
| P11 | `06_results.tex:67-69` | "On region (Acc@10), the joint model outperforms the dedicated ceiling at Florida, Texas, California, and Istanbul, and stays a non-inferior match (TOST, $\pm2$~pp) at Alabama and Arizona." |
| P12 | `06_results.tex:74-78` | "All six datasets are measured with four seeds for both models, and the tests pair the per-seed means ($n{=}4$). On category, all four seeds favor the joint model on every dataset, and each gain is significant after a Holm correction across the six datasets (paired $t$, corrected $p<0.001$)." |
| P13 | `06_results.tex:78-89` | "On region, Alabama and Arizona are tested for equivalence (TOST, $\pm2$~pp) and pass with 90\% confidence intervals well inside two points ... At Florida and Istanbul, the 90\% intervals lie entirely above zero ... so the joint model outperforms there ... Texas and California exceed that margin." |
| P14 | `06_results.tex:121-125` | "the joint model outperforms the dedicated category ceiling by $+8.58$ macro-F1 ... and is slightly above the dedicated region ceiling at $+0.19$ Acc@10 ..., a gain with statistical support (the whole 90\% confidence interval of the paired difference lies above zero, and TOST, $\pm2$~pp, also passes)" |
| P15 | `tbl3_results.tex:30-33` (caption) | "Bold marks a statistically supported improvement over the dedicated model; in the region columns, $\uparrow$ marks that supported improvement and $\approx$ a non-inferior match (TOST, $\pm2$~pp ...)." |
| P16 | `tbl3_results.tex:52-53` (footnote) | "Joint and dedicated entries: four seeds $\times$ five folds; $\pm$: sd across seeds." |
| P17 | `07_discussion.tex:13` | (no test claim; states the qualitative reading only) |
| P18 | `08_conclusion.tex:11` | "On region, it outperforms the dedicated model on four of the six datasets and remains non-inferior (TOST, $\pm2$~pp) on the other two. The gains at Texas and California exceed the two-point margin; those at Florida and Istanbul are statistically supported but smaller." |
| P19 | `01_introduction.tex:22` (footnote 1) | "Code (model, representation, baselines, statistical tests): github.com/VitorHugoOli/PoiMtlNet/tree/mobiwac" (the referent of P5's "released with the code") |

---

## 4 · STEP 4 — Reconciliation

| # | Claim | Verdict | Deciding artifact / key |
|---|---|---|---|
| P1 | abstract verdicts | **MATCH** | Reproduced Deltas and CIs, §2 table. Abstract wording obeys the ledger's softened-TOST row. |
| P2 | contribution verdicts | **MATCH** | Same. Four of six outperform; AL/AZ non-inferior. |
| P3 | scaling + verbs | **MATCH** | Scoped to region counts, AZ not upgraded. |
| P4 | "We fix the assignment in advance ... where the joint model was expected to outperform" | **MISMATCH (partial)** | True per task (`STATISTICAL_PROTOCOL.md §1`, `§5.2`); the sentence's grammar ("where") reads per dataset, and no per-dataset assignment is recorded. For next-region the assignment recorded is non-inferiority at all six, which contradicts the four "outperforms" the paper then reports. |
| P5 | "fixed in an analysis plan during development and are released with the code" | **UNSUPPORTED** on the release half | `mobiwac` branch (`689b0d6e` / `673fbd27`) has 0 files under `docs/`; the protocol and its deviation log are not in the bundle footnote 1 points at. The "fixed during development" half is supported (protocol 2026-06-21 predates the 2026-06-22 unblind). |
| P6 | "superiority is tested with a paired $t$" | **MATCH as description, MISMATCH as disclosure** | The executed test is the paired t (`stats_n20/RESULTS.md`, reproduced §2). The registered test was the paired Wilcoxon (`§2`). The paper never says a substitution occurred. |
| P7 | four seeds, n=4 pairing, Holm "across the comparisons" | **MATCH numerically; UNSUPPORTED as scope** | n=4 seed-level pairing and Holm m=6 reproduce exactly. "across the comparisons" is unqualified and reads as covering every comparison in the paper; the applied Holm family is the six next-category cells only (`§5.2`), and the four next-region superiority claims sit in no family at all. |
| P8 | TOST, two-point margin | **MATCH** | `§3.2` delta_reg = 2 pp; AL/AZ pass, reproduced §2. |
| P9 | "margin is fixed in advance" | **MATCH** | `§3.2`, pinned pre-unblind with an axis-specific justification. |
| P10 | sd 0.01 to 0.18; intervals pass a one-point margin at AL/AZ | **MATCH** | Reproduced seed-level sd: CA 0.010, TX 0.013, Istanbul 0.034, FL 0.040, AZ 0.065, AL 0.183 (range 0.01 to 0.18). AL CI (-0.631, -0.200) and AZ CI (-0.079, +0.073) both inside +/-1. |
| P11 | region verdict list | **MATCH numerically, UNSUPPORTED in confirmatory status** | The four gains are real (CIs entirely above zero, reproduced). They are not in any registered family and carry no correction. |
| P12 | category Holm, corrected p<0.001 | **MATCH** | Worst Holm-adjusted p = 1.0e-06. |
| P13 | per-dataset CIs and verdicts | **MATCH** | All six reproduce to the printed precision. |
| P14 | Istanbul: CI above zero and TOST passes | **MATCH** | Reproduced: CI (+0.154, +0.234); TOST p = 9.2e-07. |
| P15 | caption: "statistically supported improvement" | **MATCH** | Every bolded and arrowed cell has a CI entirely above zero at the reported footing. |
| P16 | table footnote: four seeds x five folds | **MATCH** | 24 committed joint-best runs (6 datasets x 4 seeds), 5 folds each; ceilings likewise n=20. |
| P18 | conclusion verdicts | **MATCH** | Same as P11 numerically. |
| P19 | footnote 1 referent | see P5 | |

### The four questions, answered

**Q1. Is any reported number affected? NO — confirmed, not merely expected.** Every Delta, CI, p-value and
verdict in the paper reproduces from the committed joint-best arrays at the reported footing, including the
thinnest cell (Istanbul region, +0.1936, CI +0.154 to +0.234). The defect is entirely in the epistemic labels.
Two rounding notes, neither a defect: FL's reproduced CI lower bound is 0.665 (printed 0.67), and AZ's Delta is
-0.0031 (printed 0.00, and the ledger forbids upgrading AZ in either direction).

**Q2. Are the four region "outperforms" claims inside any registered or applied correction family? NO, on both
counts.** Registered: `§5.2` enumerates exactly one superiority family, the six next-category cells, and
explicitly excludes the region TOST cells from it; no region-superiority family exists in the protocol.
Applied: `stats_n20/RESULTS.md` and `m1_full_output.txt` apply Holm to the six category cells only; the region
superiority p-values are printed uncorrected. The paper's §5.3 "with a Holm correction across the comparisons"
therefore implies a coverage the analysis does not have.

*What correction would cost, if the author chooses to apply one (all uncorrected values reproduced above):*
folding the four region cells into the category family (m=10, seed-level paired t) gives worst adjusted
p = 7.2e-04 (Istanbul), every cell still rejecting at alpha = 0.05. Correcting the four region cells as their
own m=4 family leaves every one at adjusted p <= 7.2e-04 (Istanbul the largest; FL 4.9e-05, TX 9.4e-08, CA 5.0e-08). **No verdict moves under any correction scheme I tested.**
This is a cheap and complete defence, and it does not require the pre-registered test.

**Q3. Does the paper's "fixed in advance" sentence overstate what the protocol fixed? YES, in two ways.**
(i) Scope: the plan fixed the assignment **per task** (category -> superiority, region -> non-inferiority);
the sentence's "where the joint model was expected to outperform" implies a per-dataset a-priori judgement
that is nowhere recorded. (ii) Coverage: the plan did not cover next-region superiority at all, so the four
"outperforms" claims are outside the a-priori assignment the sentence advertises. A third, separable defect:
the same sentence's "released with the code" is false against the bundle footnote 1 names. Note also that this
sentence is **new** — `git log -S` dates "fixed in an analysis plan during development" to `158de7d1`
(2026-07-20), the same commit that switched superiority to the paired t. The pre-2026-07-20 text said only
"we test superiority with a paired Wilcoxon signed-rank test over the folds", which was closer to the
registration on the test name and made no release claim.

**Q4. Is the Wilcoxon-to-paired-t departure disclosed anywhere the reader can see? NO.** The paper names only
the paired t (P6) and never mentions the registered Wilcoxon, so a reader cannot detect a substitution. The
disclosure exists only in `v17_completion/stats_n20/RESULTS.md`, which is not in the release bundle; the
protocol's own mandated log (`docs/studies/closing_data/log.md`) does not carry it either, and neither file is
reachable from the paper. Worse, the released `README.md` §6 tells the reader the bundled
`superiority_wilcoxon.py` implements the pre-registered tests, which is the test the paper did **not** report.

---

## 5 · STEP 5 — Feasibility of the pre-registered test today

**Question.** For which datasets can a per-fold n=20 paired Wilcoxon be run on BOTH arms under the joint-best
convention, from the committed tree only?

**Answer: five of six for next-category (all but Istanbul); six of six for next-region.** The pre-registered
category family cannot be run at all six, so a complete pre-registered headline family is **not** available today.

Inventory (all paths git-tracked on `main`):

| Dataset | MTL joint-best per-fold, n=20 | Dedicated category per-fold, n=20 | Dedicated region per-fold, n=20 |
|---|---|---|---|
| AL | yes, `j1_results.json` | yes, `sweep_results/alabama_bs2048_lr0.005_s{0,1,7,100}.json` | yes, `P1/region_head_alabama_..._{s0,topup_s1,topup_s7,topup_s100}.json` |
| AZ | yes | yes, `arizona_bs8192_lr0.005_s{...}` | yes |
| FL | yes | yes, `florida_bs8192_lr0.005_s{...}` | yes |
| CA | yes, `catx_v17_n20/joint_best/california_s{...}.json` | yes, `california_bs8192_lr0.005_s{...}` | yes |
| TX | yes, `catx_v17_n20/joint_best/texas_s{...}.json` | yes, `texas_bs8192_lr0.005_s{...}` | yes |
| **Istanbul** | yes | **NO** | yes, `P1/region_head_istanbul_..._ovl_stl_reg_s{0,1,7,100}.json` |

**The exact missing artifact.** Istanbul's dedicated category ceiling exists in the committed tree only as four
per-seed scalars, `docs/studies/closing_data/v17_completion/h3_istanbul/step3_runs/cat_ceil_s{0,1,7,100}.txt`
(54.7063 / 54.8632 / 54.7705 / 54.6101; mean 54.7375, the board's 54.74). The per-fold arrays live in the
sidecar `stl_cat_ceiling_score.json` (key `cat_per_fold`) written by `scripts/closing_data/score_stl_cat_ceiling.py`
into each rundir `results/check2hgi_dk_ovl/istanbul/next_*_<pid>/` (per `h3_istanbul/run_step3_n20.sh:45-49`);
those rundirs are gitignored and are not on this machine. The two committed Istanbul category-ceiling JSONs are
different cells and must not be substituted: `istanbul_stride1_s0_stl_cat_ceiling.json` has per-fold values but
a mean of 53.20, and `istanbul_s0_stl_cat_ceiling.json` a mean of 52.10; neither matches the board's 54.74.

**Cost to obtain.** Two options. (a) Copy four `stl_cat_ceiling_score.json` sidecars off the run machine that
holds the Istanbul rundirs, if they still exist: minutes, no compute, no retraining. (b) Re-run the four
Istanbul dedicated category runs (`next_gru`, 5 folds, 50 epochs, seeds {0,1,7,100}, engine
`check2hgi_dk_ovl`) and re-score: GPU work of the order of the original H3 step-3 pass. Option (a) is the only
one worth doing before a camera-ready deadline.

**What the pre-registered test yields where it can be run** (paired one-sided Wilcoxon, exact, per-fold n=20,
joint-best arrays, conventions mirroring `superiority_wilcoxon.py`):

| Cell | Delta | folds positive | exact one-sided p |
|---|---:|---|---:|
| cat AL | +7.690 | 20/20 | 9.54e-07 |
| cat AZ | +9.350 | 20/20 | 9.54e-07 |
| cat FL | +5.332 | 20/20 | 9.54e-07 |
| cat CA | +6.442 | 20/20 | 9.54e-07 |
| cat TX | +7.446 | 20/20 | 9.54e-07 |
| cat Istanbul | — | — | **not runnable** |
| reg Istanbul | +0.194 | 20/20 | 9.54e-07 |
| reg FL | +0.712 | 20/20 | 9.54e-07 |
| reg TX | +2.112 | 20/20 | 9.54e-07 |
| reg CA | +2.201 | 20/20 | 9.54e-07 |

9.54e-07 = 1/2^20 is the exact n=20 floor; every runnable cell is at it. This confirms the issue file's
"p ~ 9.5e-07, 20/20 positive" for R2. Holm across the four region cells (m=4) gives adjusted p = 3.8e-06 each;
across the nine runnable cells (m=9), 8.6e-06 each. **Do not report a five-of-six category family as "the
pre-registered test": the registered family is six datasets, and a partial family is a different family.**

---

## 6 · STEP 6 — Remedy

### 6.1 Evaluation of R1 to R5, plus two options the issue file missed

Line costs are measured, not estimated: the §5.3 paragraph renders at 51.9 source characters per typeset line
(1,920 characters over 37 lines, IEEE two-column). Measured slack in the current 8-page build is **85.3 pt in
the last column, about 7 body lines** at the 11.9 pt pitch, so a small net addition does not overflow, but the
budget is genuinely tight and the author's rule is to fund any addition.

| Option | What it does | Line cost | Verdict |
|---|---|---:|---|
| **R1** minimal honesty repair | Restate what the plan fixed (per task), disclose the Wilcoxon-to-t departure with its reason, label the region gains as secondary. | +4.6 gross, **+2.2 net** with the funding trim below | **RECOMMENDED**, extended (see R6) |
| **R2** run the pre-registered test | Per-fold n=20 Wilcoxon on the joint-best arrays. | +1 to +2 if reported as one clause | **Not now.** Blocked at Istanbul category (§5); a five-of-six family is not the registered family. Also trades a documented, conservative deviation for a pseudo-replication objection the paper would then have to answer. Keep as camera-ready work if the sidecars are recovered. |
| **R3** report both footings | Seed-level headline plus per-fold corroboration. | +2 to +3 | **Not now**, same blocker as R2, and it spends lines to say twice what one clause says once. |
| **R4** full REV-007 manifest | One immutable analysis manifest, regenerate everything from it. | 0 in the paper; a repo work item | **Correct for the dissertation and the camera-ready**, and it is the only option that fixes the release-bundle defect. Not a text patch; see R7. |
| **R5** drop the region "outperforms" claims | | -2 to -3 | **Reject.** Understates a result with 20/20 folds positive and CIs far from zero, and collides with the verb law (the verb is licensed by a passing superiority test, which these cells have; what they lack is registration, which is a labelling matter). |
| **R6 (new)** R1 **plus** an applied correction for the region family | Report the four region superiority cells under their own Holm correction (m=4, adjusted p <= 4.9e-05 at the reported footing) and say so. | +0.6 on top of R1 | **RECOMMENDED as an author option.** Converts "in no family" into "in its own, disclosed family". Cheapest possible answer to Q2, and no verdict moves. |
| **R7 (new)** fix the released bundle | Add `STATISTICAL_PROTOCOL.md` + `stats_n20/RESULTS.md` (or a one-page `ANALYSIS_MANIFEST.md`) to the `mobiwac` branch, and correct `README.md` §6's "region superiority (FL/CA/TX)" line. | 0 paper lines | **REQUIRED whichever text option is chosen.** Without it, P5's "released with the code" stays false and the deviation stays invisible. This is repo work, not paper work. |

**Recommendation: R1 + R6 in the paper, R7 in the repository, R4 for the dissertation and camera-ready.**
R2 stays on the shelf until the Istanbul sidecars are recovered.

### 6.2 Exact replacement text

**Edit 1 — `articles/[mobiwac]/src/sections/05_setup.tex:42`.** Replace this span:

> We fix the assignment in advance, before reading any results: where the joint model was expected to
> outperform, we test \emph{superiority}; where it was expected only to match the dedicated model, we test
> \emph{non-inferiority}. The assignment and the margin were fixed in an analysis plan during development and
> are released with the code; superiority is tested with a paired $t$ on the per-seed means and reported with
> the 90\% confidence interval of the paired difference.

with:

> A written analysis plan, fixed during development and before any result was read, assigned one test to each
> task: superiority for next-category, non-inferiority for next-region, with the two-point margin pinned there.
> The plan assigned the tests per task, not per dataset, and it did not cover next-region superiority, so the
> four next-region gains of Section~\ref{sec:results-part2} are secondary results outside it. The plan
> registered a paired Wilcoxon signed-rank test; at four seeds its exact one-sided $p$ cannot fall below
> $0.0625$, so we report a paired $t$ on the per-seed means instead, with the 90\% confidence interval of the
> paired difference. The plan and this departure are released with the code.

% source: docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md §1 (family table), §2 (registered
% Wilcoxon, per-fold pooled), §3.2 (delta_reg = 2 pp), §5.2 (Holm within the six-dataset category set only);
% departure + 0.0625 floor: v17_completion/stats_n20/RESULTS.md, "Deviation log (protocol §8)" entries 1-3.

**Edit 2 — same paragraph, the Holm sentence.** Replace "with a Holm correction~\cite{holm1979} across the
comparisons." with:

> with a Holm correction~\cite{holm1979} across the six next-category comparisons and, separately, across the
> four next-region comparisons.

% source: STATISTICAL_PROTOCOL.md §5.2 (the registered category family, m=6, TOST cells excluded); the
% four-cell next-region family is applied here for the first time and is disclosed as such by Edit 1.

**Edit 3 (only if R6 is adopted) — `06_results.tex:77-78`.** Replace "after a Holm correction across the six
datasets (paired $t$, corrected $p<0.001$)" with:

> after a Holm correction across the six datasets (paired $t$, corrected $p<0.001$); the four next-region gains
> hold under their own Holm correction as well (corrected $p<0.001$).

% source: reproduced seed-level paired t, joint-best n=20 arrays, Holm m=4 over {Istanbul, FL, TX, CA}:
% adjusted p = 7.2e-04 (Istanbul), 4.9e-05 (FL), 9.4e-08 (TX), 5.0e-08 (CA). All < 0.001.

**Edit 4 — `tbl3_results.tex`, caption, after "(TOST, $\pm2$~pp; Section~\ref{sec:results-part2})".** Append:

> The next-region improvements are secondary results, outside the analysis plan
> (Section~\ref{sec:setup-metrics}).

This edit is optional; Edit 1 already carries the disclosure, and the caption is the tightest space in the paper.

**Compliance notes.** No em-dash; American English; no contractions; canonical task names throughout;
"dedicated" left untouched (the ledger's row); AZ not upgraded; the scaling claim untouched; the abstract and
the contribution list untouched, so the softened-TOST ledger row is not reopened. The verb "outperforms" stays
bound to the four region cells, which do have a passing superiority test; what Edit 1 adds is that the test was
not registered in advance. No number changes anywhere.

### 6.3 Line cost and the funding trim

| Item | Lines |
|---|---:|
| Edit 1 (712 chars replacing 475) | +4.6 |
| Edit 2 (Holm scope) | +0.6 |
| Edit 3 (optional, R6) | +0.6 |
| **Funding trim** (below) | **-2.4** |
| **Net with Edits 1+2** | **+2.8** |
| **Net with Edits 1+2+3** | **+3.4** |
| Measured slack in the current build | about 7 lines |

**The trim that funds it.** Delete from `05_setup.tex:42`: "For scale, with 520 to 8,501 regions, a random
top-ten guess includes the true region at most about two percent of the time." (125 characters, 2.4 lines).
It is a verbatim duplicate: `06_results.tex:35-36` already says "a random region top-ten guess is right at most
about two percent of the time (Section~\ref{sec:setup-metrics})", and §6.2 is where the reader needs the scale
anchor. Removing it in §5.3 leaves the forward reference in §6.2 pointing at the margin rationale, which still
stands there. If more room is wanted, Edit 4 can be dropped (it is optional) and the §5.3 power flourish
("giving power near 1.0 to declare equivalence when the true difference is near zero") can be shortened, but
neither is needed at the measured slack.

### 6.4 Parallel wording for dissertation Chapter 5

`articles/dissertacao/src/chapters/5_mobiwac.tex:355` carries the identical sentence and takes Edits 1 and 2
verbatim, with `\ref{sec:mobiwac:results-part2}` and `\ref{sec:mobiwac:setup-metrics}` substituted for the
paper's labels. The dissertation has no page pressure, so two additions are worth making there that the paper
cannot afford:

> The plan is `docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md` in the released repository;
> the departure from the registered test is recorded in its deviation log.

and, in Chapter 2, the reconciliation REV-007 item (i) and (ii) ask for: `2_fundamentals.tex:448-450` currently
says the paired Wilcoxon signed-rank test "is the test that licenses the verb outperforms", which contradicts
Chapter 5's paired $t$. **Flagged for the author, not drafted here:** that sentence is Chapter 2 frame prose and
sits outside this pass's scope, but it must move in the same edit wave or the dissertation contradicts itself
across chapters.

### 6.5 Response-letter version (two to four sentences)

> Our analysis plan, fixed during development and before any result was read, assigned superiority to
> next-category and non-inferiority to next-region, and pinned the two-point equivalence margin; it registered
> a paired Wilcoxon signed-rank test. Because the exact one-sided Wilcoxon cannot fall below 0.0625 at four
> seeds, we report a paired $t$ on the per-seed means, a departure recorded in the deviation log released with
> the code. The four next-region improvements are secondary results outside that plan: we now say so in
> Section V-C, and we report them under their own Holm correction, where every one of them holds at a corrected
> $p<0.001$. No reported estimate, interval, or verdict changes.

---

## 7 · [VERIFY] — open items I could not resolve from the artifacts

1. **Istanbul dedicated category per-fold arrays.** Not in the committed tree (§5). Whether the four
   `stl_cat_ceiling_score.json` sidecars still exist on the run machine is a question for the author. Until
   they are recovered, the pre-registered six-dataset category Wilcoxon cannot be run, and R2 stays blocked.
2. **Provenance of the §8 joint-best entry's numbers.** The entry dated 2026-07-18 prints the exact CIs the
   paper carries, and I reproduced every one of them from the committed arrays, so the values are sound. But no
   committed script emits that entry: `m1_stats_n20.py` reads the diag-best sources, and `score_joint_best.py`
   only scores cells. The generator of the joint-best statistics run is not in the tree. This does not affect
   any number; it is the concrete shape of REV-007's "no single manifest" and is what R7 would fix.
3. **Whether "released with the code" was ever true.** Both the local (`689b0d6e`) and remote (`673fbd27`)
   `mobiwac` branch tips ship no `docs/`, and the branch has one commit (2026-07-08), predating the sentence
   (2026-07-20). I found no evidence of a separate supplementary upload, but I cannot rule out that the author
   intends one (for example an EDAS supplementary file). If such an upload exists and contains the protocol,
   P5's second half becomes supported and only the departure disclosure is needed.
4. **Two ledger-adjacent questions for the author, flagged not decided.** (i) Edit 2 applies a Holm correction
   to a region family that the protocol does not register; disclosing a new applied family is a methodological
   choice, not a wording fix. (ii) The decisions ledger has no row on pre-registration language; Edit 1 changes
   how the paper describes its own plan, which is close enough to the verdict-verb row that the author should
   approve it explicitly rather than treat it as editorial.
5. **`m1_stats_n20.py` and `superiority_wilcoxon.py` docstrings.** Both assert a registration the protocol does
   not contain, and both are in the released bundle (§2). Correcting them is repo work outside this pass; I did
   not edit them. `stats_n20/RESULTS.md §1b` carries the same assertion and its header's "per protocol §8's
   powered-t deviation" is a dangling reference, since §8 has no such entry.


---

## 8 · ADDENDUM (2026-07-25, after author approval) — what was executed

The author approved R1 + R6 (paper), R7 (repository), and the dissertation mirror. During execution the
STEP 5 blocker dissolved, which changed the outcome for the better.

**Istanbul is no longer missing.** The four `stl_cat_ceiling_score.json` sidecars were on the A40
(`ssh:nespedgpu`) at `results/check2hgi_dk_ovl/istanbul/next_lr1.0e-04_bs2048_ep50_20260706_*_{3856035,
3861493,3866919,3872209}/`, tagged `h3ist_cat_s{0,1,7,100}`. Their per-seed means reproduce the committed
scalars exactly (54.7063 / 54.8632 / 54.7705 / 54.6101; n=20 mean 54.7375 = board 54.74), so **no GPU run
was needed** and the board cell is untouched. Committed at
`docs/studies/closing_data/v17_completion/h3_istanbul/step3_runs/cat_ceiling_perfold/`.
`stats_n20/RESULTS.md` LIMITS #2 is closed.

**R2 therefore ran, and the STEP 5 statement above is superseded.** The registered test at its registered
footing (per-fold n=20 paired one-sided Wilcoxon, protocol §2; Holm m=6, protocol §5.2), joint-best arrays,
now covers the complete six-dataset family: every cell Δ as reported, **20/20 folds positive**, exact
p = 9.5367e-07 (the n=20 floor 1/2^20), **Holm-adjusted 5.7220e-06, all reject at α = 0.05**. The four
next-region cells reject in their own m=4 family (adjusted 3.8147e-06). Generator with a 24/24
artifact-to-board gate: `stats_n20/m2_prereg_perfold.py`; output `m2_prereg_output.txt`.
**No verdict, estimate, or interval moved.** Q1 of §4 is unaffected and remains confirmed.

**Consequence for the remedy.** The recommendation in §6.1 to defer R2 no longer applies; R1 was extended
with one corroboration clause instead. The paper now names the departure *and* reports that the registered
test agrees, which answers the reviewer objection more completely than either option alone would have. The
pseudo-replication risk that made R2 unattractive is neutralized by reporting the seed-level $t$ as the
primary footing (the region CIs are computed on it) with the per-fold Wilcoxon as corroboration, rather
than swapping the primary.

**Line cost, measured against the built PDF.** My §6.3 estimate (+2.8 net) was wrong: the first build came
out at **9 pages**, because the projected "7 lines of slack" did not survive float reflow. Closing it took
six further de-duplication trims, each removing a fact stated elsewhere in the paper: the epoch-convention
tail clause ("so no claim depends on this choice"), the §6.2 "All six datasets are measured with four
seeds" sentence (§5.3 defines it), the §6.3 "both models average four seeds over five folds" clause, the
§5.3 power flourish ("giving power near 1.0 ..."), plus two tightenings of the new text. **Final build: 8
pages, 0 undefined references, 0 overfull boxes, bibtex clean.** Lesson for the next pass: on a flush
IEEE build, character-per-line arithmetic is not a page-count prediction; compile before promising a
budget.

**Where the record now lives (R7 done).** The `mobiwac` release branch commit `09b01923` adds
`analysis_protocol/` (protocol, deviation log, executed analysis, epoch-selection record, dedicated
ceilings, the registered-test output, the Istanbul per-fold arrays) plus `m2_prereg_perfold.py` and
`score_joint_best.py`, and rewrites README §6. The shipped script was verified from a clean
`git archive` checkout of the branch and reproduces every number. **The commit is local; it has not been
pushed.** §7 item 3 ("was 'released with the code' ever true?") is resolved as: it was not, and it now is.

**The protocol's own §8 gap is closed.** Entries **D-1 to D-4** are now in
`docs/studies/closing_data/log.md`, the location §8 mandates, and §8 links to them. `RESULTS.md:20`'s
dangling forward reference has a referent for the first time.

**Still open, unchanged from §7.** The seed-level joint-best statistics generator is still absent from the
tree (only `m2_prereg_perfold.py` covers the registered per-fold family), and the one-page
`ANALYSIS_MANIFEST.md` is still the remaining REV-007 work item. The three docstring assertions
(`superiority_wilcoxon.py`, `m1_stats_n20.py:333`, `RESULTS.md §1b`) are flagged in the shipped README and
in D-4 but were not edited: correcting a results record's own prose is a separate, claim-neutral pass.
