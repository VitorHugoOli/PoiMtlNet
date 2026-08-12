# REVISION_PLAN.md — what changes in the dissertation, and why

> **Status: PROPOSAL, revision 2 (after adversarial review). No chapter, table, or front-matter
> file has been modified.** `src_fix/` is a byte-identical branch of `src/` minus build output,
> committed at `de040d3c` as the tracking baseline. `src/` stays untouched until the author
> approves this document.
>
> **Read §1 (what the measurements say), then §2 (the rulings and directives), then §3–§9 (the
> file-by-file edit list), then §11 (what still needs an author decision).** Every number below
> was recomputed from the banked artifacts in this session; §10 is the source ledger.
>
> **Revision 2 changelog.** A critical review of revision 1 found three blocking gaps and eleven
> major ones; all were checked against the sources before being accepted, and the material ones
> were confirmed. The substantive changes: the project's own law files and gate suite carry the
> superseded ladder and must be updated *before* any prose (new §3.0); Appendix H describes the
> superseded bidirectional graph, not only a stale feature width (§6); the chapter preface's
> contract with the submitted manuscript is unresolved (§11.1); three region "matches" are
> statistically resolved deficits that the text must disclose rather than round to equality
> (§1.5); and four appendices plus two figures that revision 1 did not list are now in scope. Two
> of the reviewer's recommendations are declined with reasons (§12).

---

## 0 · What was audited to produce this plan

| step | what | where it lands |
|---|---|---|
| dissertation inventory | all 60 `.tex` files under `src/`, comment-stripped, swept for superseded values, verdict claims, the banned trunk adjective, feature-width statements, and elapsed-time coverage | §3–§9; 122 raw numeric hits across 22 files, plus the sites a value sweep cannot see because they carry no numeral. **31 edit targets in total: 26 dissertation source files, 2 figures, 4 law files, 1 gate script** (enumerated in §13) |
| law and gate sweep | `WRITING_LAW.md`, `GLOSSARY.md`, `NORTH_STAR.md`, `CLAUDE.md`, and `src_utils/check_audit_claims.py` swept for the superseded verdict ladder | §3.0 |
| verdict ladder | recomputed independently from `docs/results/closing_data/v18/joint_best_perfold.json` (24 cells, 6 datasets x 4 seeds) against the dedicated arms in `docs/studies/closing_data/v18/data/v18_results.json`; paired one-sided t on the four per-seed means, Holm within each task family (m=6), TOST at the registered two-point region margin | §1.1, reproduces the MobiWac tables exactly |
| representation contrast | recomputed from `docs/results/closing_data/v18_place_level/<state>_s0_cat_placelevel.json` against the check-in-level dedicated arm, paired t on the five matched folds | §1.2 |
| convention gap | diagnostic-best minus served-checkpoint, per dataset, to bound what the robustness sentence may claim | §1.3 |
| MobiWac reuse | the applied diff `da97ecf7..HEAD` over `articles/[mobiwac]/src_fix` (13 files, 671 insertions) read in full | §9 |
| Appendix F evidence | `src_utils/_round7/gradient_cosine_observations6.parquet` opened (4,650 rows, 7 datasets, 13 configurations) and every joint run directory in the v18 results tree swept for a non-empty `grad_cosine_shared` column | §1.4, §7 |

**No further experiment is required for Chapters 1–6 or for the tables.** Every cell the
dissertation reports exists, is banked per fold, and reproduces. There are **two execution
gaps, both in appendices**: Appendix F, ruled on by the author (§2.1) and running now, and
Appendix I, whose wider-dedicated arm was measured on a superseded engine and which therefore
cannot be brought current by editing numbers (§8, decision needed in §11.2).

---

## 1 · What the measurements say

### 1.1 · The verdict ladder (served-checkpoint convention, n = 4 seeds x 5 folds per cell)

Recomputed here from the per-fold artifact; matches `tables/tbl3_results.tex` in the MobiWac
working copy to the fourth decimal.

| task | dataset | dedicated | joint | Δ | Holm p | verdict |
|---|---|---:|---:|---:|---:|---|
| category | Istanbul | 35.34 | 35.42 | +0.08 | 0.181 | unresolved |
| category | Alabama | 30.78 | 30.59 | −0.19 | 1.000 | unresolved |
| category | Arizona | 34.57 | 34.57 | −0.00 | 1.000 | unresolved |
| category | **Florida** | 37.35 | **37.55** | **+0.19** | **0.011** | **outperforms** |
| category | Texas | 36.33 | 36.19 | −0.13 | 1.000 | unresolved |
| category | California | 35.63 | 35.63 | −0.00 | 1.000 | unresolved |
| region | Istanbul | 75.16 | 75.08 | −0.08 | 1.000 | non-inferior (TOST) |
| region | Alabama | 70.12 | 69.24 | −0.87 | 1.000 | non-inferior (TOST) |
| region | Arizona | 59.48 | 59.04 | −0.44 | 1.000 | non-inferior (TOST) |
| region | Florida | 76.69 | 76.54 | −0.16 | 1.000 | non-inferior (TOST) |
| region | **Texas** | 64.94 | **66.15** | **+1.21** | **0.00013** | **outperforms** |
| region | **California** | 63.48 | **64.54** | **+1.06** | **0.0000063** | **outperforms** |

Three consequences bind every edit below.

1. **The old ladder is gone.** "Category at all six, region at four of six" is false in both
   clauses. Category outperforms at one dataset; region outperforms at two.
2. **The five category nulls are NOT matches.** The two-point equivalence margin is
   pre-registered for the region axis only. A category cell that fails superiority is
   *unresolved*, and the text must say so rather than upgrade it (WRITING_LAW verdict ladder).
3. **Every region cell is at least non-inferior.** All six clear TOST at the registered margin,
   two of them by outperforming. This is what licenses the parity spine in §2.2.

### 1.2 · The representation contrast (seed 0, five matched folds)

| dataset | check-in level | place level (HGI) | Δ | folds favouring | paired p |
|---|---:|---:|---:|:--:|---:|
| Istanbul | 35.35 | 29.07 | +6.29 | 5/5 | 0.00003 |
| Alabama | 30.77 | 29.15 | +1.62 | 5/5 | 0.0034 |
| Arizona | 34.51 | 31.93 | +2.58 | 5/5 | 0.0004 |
| Florida | 37.36 | 37.13 | +0.23 | 5/5 | 0.067 |
| Texas | 36.32 | 35.33 | +0.99 | 5/5 | 0.00003 |
| California | 35.62 | 34.74 | +0.88 | 5/5 | 0.0005 |

The direction is unanimous at every dataset and every fold; the magnitude is between a quarter
of a point and six points, not the twenty-eight to forty points the current text claims. Florida
does not reach significance and must be reported as such.

### 1.3 · The convention gap (what the robustness sentence may claim)

Diagnostic-best minus served-checkpoint, averaged over four seeds: category +0.03 to +0.17,
region +0.19 to +0.90. The reported convention is therefore the conservative one on both axes,
and the text may say the ladder does not depend on the choice — but only within those bounds,
which are worth stating rather than waving at.

### 1.4 · Appendix F, the one ruled-on execution gap

The appendix's stated source of record is `gradient_cosine_observations6.parquet`. Opened this
session: 4,650 rows over seven datasets, but every row carries a configuration
(`T6_*`, `canonical`, `shipping_florida_mtl_ep50_seed42`) that predates the reported model. A
sweep of all 93 joint run directories in the v18 results tree found a non-empty
`grad_cosine_shared` column in exactly four, all seed 0, all before the settings freeze, and all
belonging to the loss-weight probe rather than the reported configuration. The appendix
therefore measures a model the dissertation no longer describes. Author ruling in §2.1.

### 1.5 · The direction inside the margin (this must be disclosed, not rounded away)

Non-inferiority is not equality. Testing the reverse direction on the same paired footing (one-sided
t on the four per-seed means, Holm within the region family) shows that three of the four region
"matches" are statistically **resolved in favour of the dedicated model**:

| dataset | Δ (joint − dedicated) | 90% CI | reverse-direction Holm p | reading |
|---|---:|---|---:|---|
| Alabama | −0.87 | [−1.00, −0.75] | 0.0016 | resolved deficit, inside the two-point margin |
| Florida | −0.16 | [−0.19, −0.13] | 0.0031 | resolved deficit, inside the margin |
| Arizona | −0.44 | [−0.62, −0.25] | 0.023 | resolved deficit, inside the margin |
| Istanbul | −0.08 | [−0.16, −0.00] | 0.14 | unresolved |

This does not weaken the parity spine: every one of these deficits clears the registered
two-point margin decisively, which is exactly what the pre-registered analysis asked. But the
text may not describe them as "no difference" or let a reader infer equality, because a committee
member running the obvious test will find the sign resolved. **Plan requirement:** every region
cell reported as non-inferior carries either its confidence interval or the direction of the
difference, so nothing resolved is left undisclosed.

One consequence for `05_setup.tex`: its claim that "at Alabama and Arizona, the intervals also
support a margin as small as one point" is false under the final data. TOST at one point passes
at Istanbul, Arizona, and Florida, and **fails at Alabama** (p = 0.052). The sentence is rewritten
to the datasets that support it, with Alabama named as the exception.

---

## 2 · The two rulings already given

### 2.1 · Appendix F (author, this session)

> "Re-run at the shipped configuration, but don't run for CA and TX since we wont have time."

Running now at Alabama, Arizona, Istanbul, and Florida: seed 0, five folds, fifty epochs, the
shipped joint command with per-epoch train diagnostics enabled. Georgia is dropped (no v18
engine, and building one would mean training a representation first). The appendix's coverage
therefore becomes **four of the six datasets Chapter 5 reports**, down from seven, and its
design (fold as the unit of independence, equivalence against the registered margin) is
unchanged. Consequences in §7.

One deviation is logged: the run sets `MTL_AMBIGUITY_STRICT=0`. The first attempt aborted at
Alabama fold 1 epoch 14 on the scored region metric's tie certificate, which fires when one
validation row out of roughly 766,000 sits in a tie group straddling the top-ten boundary. The
repository added that switch for exactly this case; every other guard, including the canon-recipe
and overlap-provenance checks under `MTL_STRICT=1`, stays armed. The gradient cosine is measured
on the training path and is not affected by validation tie semantics.

### 2.2 · The thesis spine (author, this session)

> "Parity plus a targeted win \[as the main resolution]... we can also \[use] the operational
> consolidation and scale conditional benefits as an argument for the text, but don't use them
> as the main resolution they are supportive ones, so we give more credibility."

So the resolution sentence is: **with a check-in-level representation and a redesigned sharing
topology, one joint model stays within the registered margin of two dedicated models on every
dataset and outperforms them where the region task is hardest.** The operational argument (one
model, one forward pass, both answers) and the scale observation (the two region wins are the two
largest region vocabularies) appear as *support*, after the resolution, each carrying its own
hedge. The scale observation must never be written as a law: California has more regions than
Texas and a smaller gain, and region count co-varies with corpus size across the states.

---

## 3 · Frame chapters

### 3.0 · The law files and the gate suite come FIRST

This is the prerequisite for every edit below, and revision 1 missed it. The project's own
fail-closed law still encodes the superseded ladder, so a compliant agent writing the corrected
text would be violating the writing law, and the mechanical gates would go red against correct
prose. Verified by grep this session:

| file | site | what it says now |
|---|---|---|
| `WRITING_LAW.md` | §3, line 200 | binds "outperforms" to "Istanbul/FL/TX/CA region; category everywhere" |
| `WRITING_LAW.md` | §7 checklist, line 308 | requires "outperforms 4 / matches AL–AZ" |
| `GLOSSARY.md` | §4 superiority row | same binding |
| `NORTH_STAR.md` | line 26 and §§1, 2, 6 | states the thesis as "category everywhere, region at four of six" |
| `CLAUDE.md` | decisions ledger | echoes the same ladder |
| `src_utils/check_audit_claims.py` | probe R10-blq2, line 416 | **requires the superseded sentence to be PRESENT** in `chapters/1_introduction.tex` |

**Order of work:** the author approves the new ladder wording; it lands in these five files; the
pinned probes are repointed to the new strings and sabotage-validated (a probe that cannot fail
is not a probe); only then does chapter prose change. The ladder is stated once, in
`WRITING_LAW.md` §3, and every other site points at it rather than restating it, so the next
change has one home.

### 3.1 · `content.tex` — Resumo and Abstract

| item | detail |
|---|---|
| current issue | The Abstract's claim block states the old ladder ("outperformed the dedicated models... in next-category prediction at every dataset"); the Resumo is its claim-parity pair and carries the same statement in Portuguese. |
| evidence | §1.1. |
| modification | Rewrite both claim blocks to the parity spine: joint model within the registered two-point margin on next region at all six datasets, outperforming at two; on next category, outperforming at Florida with the remaining five differences unresolved. Keep the two blocks a single claim in two languages, edited together, per WRITING_LAW §6. |
| MobiWac reuse | The paper's abstract was rewritten in the same direction (`eb24c64a`); the framing transfers, the wording does not, because the dissertation abstract also has to carry Chapters 3 and 4. |
| tables/figures | none |
| glossary | **Correction to revision 1, which claimed these were already registered.** A grep returns zero hits for `parity` and for `paridade`; only `não-inferioridade estatística` exists. Under the fail-closed rule the word may not enter prose or the Resumo until the author lands an entry. Either register `parity / paridade`, defined as the TOST result it summarizes, or write the spine without the word. |

### 3.2 · `chapters/1_introduction.tex`

| item | detail |
|---|---|
| current issue | L461–463 state the old ladder inside the contributions list. The research-question answer at L221 is phrased against it. |
| evidence | §1.1, §2.2. |
| modification | Rewrite the contribution bullet to the parity spine; restate the answer as a qualified yes whose condition is representation-and-topology, with the two region wins named. Add one clause pre-motivating the hyperparameter-sweep disclosure (directive 4) so Chapter 5's setup section is not the first mention. |
| MobiWac reuse | contributions list `eb24c64a`, adapted. |
| tables/figures | none |
| cross-refs | the ladder sentence is cited from Chapter 6; both must move together. |

### 3.3 · `chapters/2_fundamentals.tex`

| item | detail |
|---|---|
| current issue | (a) L1834 states the old ladder in the closing hinge. (b) §2.2's representation narrative implies the check-in level makes the category task learnable, a claim §1.2 no longer supports at that magnitude. (c) The Check2HGI description does not mention elapsed time, so the chapter defines a representation the later chapters do not use (directive 2). (d) L991–993 forward-references Appendix F as covering "every dataset measured there", which after §2.1 means four. |
| evidence | §1.1, §1.2, §1.4, directive 2. |
| modification | Rewrite the hinge to the parity spine. Soften the representation narrative to the measured effect: unanimous in direction at every dataset and fold, between a quarter point and six points in size, not significant at Florida. Add elapsed time to the check-in-level definition where the four node-feature groups are introduced, with its dimensionality, so §2.2 and the appendix agree. Rescope the Appendix F forward reference to four datasets. |
| MobiWac reuse | none directly; the paper has no fundamentals chapter. The elapsed-time wording comes from the paper's method section (`826aeaee`) and is adapted to a didactic register. |
| tables/figures | model-lineage table unchanged. |
| glossary | **proposes one entry**: `elapsed time` / `tempo decorrido` as a check-in-level node feature group. Fail-closed rule: the term may not enter prose until the author approves the entry. |

### 3.4 · `chapters/6_conclusion.tex`

| item | detail |
|---|---|
| current issue | Four separate sites: L36–38 (old ladder in the opening synthesis), L100–103 (old ladder with the "5.3 to 9.4 macro-F1 points" magnitude, which no longer exists), L128 (the monotone scaling claim), and the closing research-question answer. |
| evidence | §1.1, §2.2. |
| modification | Rewrite all four to the parity spine. The category magnitude sentence goes; in its place, the Florida win with its Holm-corrected p, and an explicit statement that the other five category differences are unresolved rather than matched. The scaling paragraph becomes the supporting observation of §2.2 with both hedges (non-monotone inside the pair, region count confounded with corpus size). Add the Texas/California hypothesis (directive 5) here as a limitation, in the cautious form the author specified. |
| MobiWac reuse | discussion and conclusion `c92a1c76`, `73f5ab09`, `0a27b356` — the hedged wording transfers well; the dissertation adds the cross-chapter synthesis the paper has no room for. |
| tables/figures | none |

---

## 4 · Chapter 5 (the re-typeset MobiWac chapter)

This chapter is where most of the numeric change lands. The MobiWac working copy has already
been revised against the same artifacts, so the reuse is heavy and deliberate — but the chapter
is a dissertation chapter, not a reprint: it keeps the didactic register, spells out the
protocol, and answers the dissertation's research question explicitly.

| file | current issue | modification | MobiWac source |
|---|---|---|---|
| `5_mobiwac.tex` (chapter opener) | L33 states the old category-everywhere claim | rewrite to the parity spine, one sentence | — |
| `5_mobiwac/01_introduction.tex` | L37 "+5 to +9 points" on every dataset; L39 "four of the six"; L42/L44 the monotone scaling contribution; L49 the +0.19 Istanbul region claim | rewrite the contribution list to the measured ladder; the scaling bullet becomes the hedged observation of §2.2 | `01_introduction.tex` in `eb24c64a` |
| `5_mobiwac/04_method.tex` | L35 the loss equation still reads 0.75/0.25 | restate as the shipped 0.50/0.50 split with the logit-adjustment note; this is a **configuration correction, not a numeric one**, and it must agree with Appendix H | `63268fea` |
| `5_mobiwac/05_setup.tex` | six sites, four of which revision 1 missed: (1) no elapsed-time feature; (2) no hyperparameter-sweep disclosure; (3) "the four next-region gains ... are secondary results" (now two); (4) Holm described as "across the six next-category comparisons and, separately, across the four next-region comparisons" (the final analysis is m=6 in both families); (5) "the standard deviation of the paired difference ranges from 0.01 to 0.18 points" (recomputed: 0.02 to 0.16); (6) "At Alabama and Arizona, the intervals also support a margin as small as one point" — **false**, TOST at one point fails at Alabama (p=0.052) and Arizona passes | add the elapsed-time node group with its width; add the sweep paragraph (directive 4) naming what was searched for the dedicated models and for the joint model, and stating plainly that Texas and California carry the joint configuration transferred from the smaller datasets; correct the four stale statements from the recomputed artifacts rather than from the old prose | `826aeaee`, `a43b3599`, and the corrected margin sentence at `[mobiwac]/src_fix/sections/05_setup.tex:47` |
| `5_mobiwac/06_results.tex` | 21 hits: the whole results narrative, both result tables' prose, the representation-gap magnitudes, the silhouette/purity sentence, the "learnable" framing, the 11-feature width | rewrite against §1.1 and §1.2; every verdict verb re-bound to the test that passed; the five category nulls reported as unresolved | `b11fd395`, `0ebbbceb`, `73f5ab09` |
| `5_mobiwac/07_discussion.tex` | L12–13 old ladder; L18–19 a shortlist claim resting on a superseded cell and carrying the stale numeral **65.69** (final California joint region is 64.54) | rewrite to the measured ladder; re-derive the shortlist sentence from the final board or drop it (§11.4) | `c92a1c76`, `0a27b356` |
| `5_mobiwac/08_conclusion.tex` | L26 the scaling claim; L28–29 the external-baseline margins ("at least 4 Acc@10", "at least 33 macro-F1") | rewrite the scaling clause; **recheck both external margins against the final table before restating** — they are derived numbers and may have moved | `73f5ab09` |
| `tables/mobiwac/results.tex` | 23 stale cells | replace wholesale from the MobiWac table of record, re-typeset to the dissertation's column conventions; keep the dissertation's own caption discipline | `tbl3_results.tex` |
| `tables/mobiwac/representation.tex` | 14 stale cells | same, from `tbl2_substrate.tex` | `tbl2_substrate.tex` |
| `figures/mobiwac/` (fig. 4 equivalent) | the delta figure is drawn from the superseded deltas | regenerate from the served-checkpoint deltas | `bb3a7278`, `fig4_deltas.py` |
| `figures/mobiwac/fig3_embquality*` **and** the silhouette/purity sentence in `06_results.tex` | the embedding-geometry panel is measured on the earlier data preparation (its own docstring sources the archived design-k engine), not on the representation the results use. This is the same defect this plan uses to condemn Appendix F, so it cannot be left uncorrected while that one is re-run | either re-measure the two geometry statistics on the final engines, or scope the figure and the sentence explicitly to the earlier preparation. **Author decision, §11.3.** The revised paper kept the old constants, so this is one place the dissertation should not simply follow it | — (divergence from `[mobiwac]/src_fix` is deliberate here) |

---

## 5 · Directive 3 — the shared trunk

The sweep found **no occurrence of the banned adjective** in any dissertation prose file, so
there is nothing to delete. The directive still binds the rewrite: where Chapter 5's discussion
and Chapter 6 explain the region result, the shared representation is described as one
contributing component, with the two region wins as its evidence, and without inflating that
into a transfer claim the ablations do not support. The gradient-orthogonality finding
(Appendix F) is the mechanism sentence and stays scoped to what it measures.

---

## 6 · Directive 2 — elapsed time, everywhere

| file | what changes |
|---|---|
| `chapters/2_fundamentals.tex` | the check-in-level definition gains the elapsed-time group |
| `chapters/5_mobiwac/04_method.tex` | the node-feature description gains it |
| `chapters/5_mobiwac/05_setup.tex` | the input description gains it |
| `chapters/apx_h_check2hgi_joint_model.tex` | **two arithmetic sites**: L125 "maps the 11 input features to 64 dimensions" and L428 "7 category indicators + 4 cyclic time values; width 11". Both become the elapsed-time-inclusive width. This is the one place where a stale number is a *dimensionality* error rather than a result. |

Consistency requirement: one width, stated identically at all four sites, and the arithmetic in
the appendix table must add up to it.

### 6b · Appendix H also describes a graph and a loss the study does not use

Revision 1 treated Appendix H as a feature-width fix only. Two further classes of error are in it,
and both are worse than a stale numeral because they describe a *different model*:

| site | current text | why it is wrong |
|---|---|---|
| `apx_h` L71 | consecutive visits "are connected in both directions" | the reported representation keeps only the forward edge, in training and at readout |
| `apx_h` L132 | a row "reflects the immediately preceding and following visits" | same |
| `apx_h` L378 | the loss equation, at the 0.75/0.25 split | the shipped objective is the 0.50/0.50 split with logit adjustment on the category head |
| `apx_h` L465 | "the supervised objective is the fixed 0.75/0.25 combination" | same |
| `5_mobiwac/04_method.tex` | the succession-edge sentence is silent on direction | must state the forward-only construction, so the two descriptions agree |

Consistency requirement: one loss statement and one edge-direction statement, each written once
and identical at every site. The *disclosure* question — how much of the reason for the
forward-only construction belongs in a dissertation that may not narrate its own history
(directive 1) — is an author decision, §11.5.

---

## 7 · Appendix F, after the re-run

| item | detail |
|---|---|
| current issue | Entire evidence base is off-configuration (§1.4); coverage claims seven datasets including Georgia. |
| evidence | the re-run now in flight (§2.1). |
| modification | Rebuild the appendix from the new measurements: four datasets, seed 0, five folds, fifty epochs at the shipped configuration. Restate the coverage paragraph, the unit-of-independence paragraph, the per-dataset statistics, the sign-test paragraph, the figure, and `tables/frame/cosine.tex`. Drop Georgia and the sentence that used it as an extra test. Rescope the Chapter 2 forward reference (§3.3) and the Chapter 5 related-work sentence at `02_related.tex:188–193`, which currently cites "four seeds each on four Gowalla states... and Georgia". |
| risk | The appendix's own closing note warns that a coverage change can falsify *verdicts*, not just counts, and names two sentences a numeral-grep would miss. **Every statistical verdict in the appendix is re-derived from the new data, not edited in place.** |
| tables/figures | `tables/frame/cosine.tex` rebuilt; `figures/fig_gradient_cosine.png` regenerated. |
| fallback | If the re-run yields fewer than four datasets, the appendix is rescoped to what landed and says so; it is not padded with the superseded rows. |

---

## 8 · Appendix I — the parameter-count control

L37–38 carry dedicated and joint values from the superseded board (56.82 / 70.60 / 64.54 /
77.05). The control's *design* is unaffected, but its numbers and its conclusion sentence must be
re-derived from the final board. **Open question for the author**: the wider-dedicated arm itself
was measured under the superseded configuration, so unless a v18-era wider arm exists, this
appendix cannot be brought to the final board by editing — it would need its own re-run. Flagged
rather than guessed; see §11.

---

## 8b · Four more files the value sweep could not see

These carry no superseded numeral, so a numeric grep returns nothing; each nevertheless asserts
something the final evidence contradicts.

| file | current issue | modification |
|---|---|---|
| `chapters/apx_a_contributions.tex` | the "reproducing the reported numbers" section points at the earlier reproducibility bundle (protocol, statistics scripts, results, "frozen together"). Following those pointers reproduces the numbers this revision removes | repoint every fold/seed/score/test pointer at the artifacts in §10, or state plainly which bundle corresponds to the reported numbers |
| `chapters/apx_d_ceiling.tex` | states a model-versus-benchmark gap "of four to six points" and attributes that reading to Chapter 5. Under the final data the dedicated check-in-level model at Florida is about 1.2 points above the label-history benchmark, not four to six | recompute every model-versus-benchmark distance; rewrite the cross-claim; time-index the screening record as development history rather than as a current statement |
| `chapters/apx_g_hgi_tuning.tex` | reports category F1 on a scale (roughly 0.74–0.82) that no longer sits anywhere near the final place-level column (roughly 0.29–0.37), with nothing telling the reader why | add a scoping sentence naming the configuration and preparation it was measured on, so the two scales cannot be read as comparable |
| `chapters/1_introduction.tex` L448–455 **and** `chapters/2_fundamentals.tex` L1804 | both assert the gradient-orthogonality result "on every dataset measured", citing Appendix F | rescope to the four datasets the re-run covers, once its results land (§7) |

## 8c · The frame's description of the published chapters

The published CBIC and CoUrb chapters are reproductions and their text does not change. But the
frame's *description* of them is dissertation prose, and one sentence no longer holds:
`chapters/3_cbic.tex` says "The two chapters that follow revise that verdict." Under the final
data, Chapter 5 partly **confirms** the CBIC category null rather than revising it, since the
category difference is unresolved at five of six datasets. The preface needs a reframe the author
approves: the later chapters answer the question under a corrected protocol and a different
representation, and they change the region verdict rather than overturning the category one.
Chapter 4's preface gets the same check.

## 9 · What is reused from MobiWac, and what is not

**Reused:** the served-checkpoint convention and its justification; the verdict ladder and both
result tables; the elapsed-time method wording; the hyperparameter-sweep disclosure; the
weakened scaling claim; the hedged discussion of the category nulls; the capacity-confound
limitation; the delta figure's construction.

**Not reused:** the paper's compression. The dissertation defines the convention where it is
first used, spells out why the served checkpoint is the honest reporting choice, connects the
Chapter 5 result back to the Chapter 3 null and the Chapter 4 diagnosis, and answers the research
question in the author's own frame. Where the paper says "the equivalence margin is
pre-registered for the region axis only" in a table note, the dissertation says it in the body,
because its reader has not read the protocol section three pages earlier.

**Consistency requirement:** after the revision, no claim may be stronger in the dissertation
than in the paper for the same measurement.

---

## 10 · Source ledger

| number in this plan | source of record |
|---|---|
| every joint cell, per fold | `docs/results/closing_data/v18/joint_best_perfold.json` (24 cells; 15 from rundir artifacts, 9 from the lane archive, 1 fold recomputed) |
| every dedicated cell, per fold | `docs/studies/closing_data/v18/data/v18_results.json` (`stl_cat_folds`, `stl_reg_folds`) |
| place-level arm | `docs/results/closing_data/v18_place_level/<state>_s0_cat_placelevel.json` |
| shipped joint configuration | `docs/studies/closing_data/v18/run_wave.sh`, `cell_joint()` |
| settings freeze and its rationale | `docs/studies/closing_data/v18/FINAL_SETTINGS.md` (author-approved 2026-08-09) |
| what v18 is | `docs/studies/closing_data/v18/METHODOLOGY.md` |
| head orthogonality | `docs/studies/closing_data/v18/LOSS_WEIGHT_PROBE.md` |
| Appendix F current basis | `articles/dissertacao/src_utils/_round7/gradient_cosine_observations6.parquet` |
| Appendix F new basis | the re-run of §2.1, harvested per fold with checksums |
| MobiWac applied revision | `da97ecf7..HEAD` over `articles/[mobiwac]/src_fix` |

Recomputations performed this session (not read from prose): the verdict ladder with Holm and
TOST, the representation contrast with its paired tests, the convention gap. All three reproduce
the MobiWac tables, which is the independent check that the paper's numbers are traceable.

---

## 11 · Open questions for the author

### 11.1 · What does Chapter 5 say it reproduces? (blocking)

The chapter preface states that it reproduces the article submitted to MobiWac 2026, under
review, and that "No result, claim, or conclusion was altered"; `apx_b_errata.tex` states that
corrections are applied to the submitted source as well and "the two texts stay identical". After
this revision, every headline number in Chapter 5 differs from the submitted version of record.
The preface, the errata appendix, and the status wording must be settled together. The honest
options: the chapter reproduces the manuscript **as revised after the final evaluation**, and says
so; or it reproduces the submitted version and carries an erratum block. This is a claim about the
work, which is the failure class the guardrails flag as the most common — so it is the author's
call, not mine.

### 11.2 · Appendix I (recommendation: cut and fold)

Answered from the artifacts: the wider-dedicated arm was measured on the `check2hgi_dk_ovl`
engine, not the final one, so the appendix cannot be corrected by editing. Beyond that, its
premise has evaporated. The control exists to test whether parameter count explains a category
advantage of seven to eight points; under the final data the category advantage is +0.19 at one
dataset and unresolved elsewhere, so there is almost nothing left to explain. **Recommendation:**
cut the appendix and fold one sentence into the Chapter 5 discussion noting that the joint model
carries more parameters and that this confound is not separated. Alternative: re-run the wider arm
on the final engine at two datasets (roughly one dedicated cell each) to defend a paragraph whose
question no longer has force.

### 11.3 · The embedding-geometry figure

Re-measure the two geometry statistics on the final engines, or scope the figure and its sentence
to the earlier preparation? Re-measuring is cheap (it reads exported representations, no
training); scoping costs nothing but leaves a reader comparing two preparations.

### 11.4 · The Chapter 5 shortlist sentence

`07_discussion.tex:18–19` rests on a superseded cell and carries the numeral 65.69. Re-derive from
the final board, or drop the sentence?

### 11.5 · How much of the forward-only construction is explained?

Appendix H and Chapter 5's method section must describe the forward-only graph (§6b). Directive 1
forbids narrating history. The two can be satisfied together by describing the construction as the
design — "each visit sees only the visits that precede it" — with the leakage rationale stated as
a design principle rather than as a correction. Confirm that reading.

### 11.6 · Closed since revision 1 (no decision needed, recorded for the audit)

The external-baseline margins were recomputed from the final table rather than left as a promise.
The dissertation's current claims are both false: the region margin over the strongest external
reference is **at least 3.55** points, not "at least 4", and the category margin over the strongest
external is **at least 3.06** macro-F1 points, not "at least 33". Both figures match the revised
paper, and both sentences are rewritten to them.

### 11.7 · Anything in §3–§8 to add, drop, or reorder before any file is touched.

---

## 11b · Exit gates (what "done" means)

The revision is not complete until all of these pass, in this order:

1. a fresh numbers ledger mapping every numeral in the document to an artifact in §10, with its convention (metric, selection rule, n);
2. the Resumo and Abstract claim-parity audit;
3. a claim-strength diff against `[mobiwac]/src_fix`, confirming no dissertation claim is stronger than the paper's for the same measurement;
4. the register, hard-phrasing, and idiom sweeps of the writing law;
5. `check_audit_claims.py` green **after** its probes are repointed, each sabotage-validated;
6. a full build in both modes with zero errors, zero undefined references or citations, and the list of tables and figures regenerated;
7. a read of the complete diff, hunk by hunk, against this plan.

## 12 · Reviewer recommendations declined, with reasons

An adversarial review produced twenty findings. Most are adopted above. Two are declined, and one
is adopted in a weaker form than proposed; recording them keeps the disagreement auditable.

1. **Declined: requiring the Florida category effect size to travel with every mention of the
   verdict.** The reviewer is right that +0.19 macro-F1 is statistically solid and practically
   small, and right that a banca will ask what it buys. But the remedy of attaching the magnitude
   to every mention would make the abstract and the introduction unreadable. Adopted instead: the
   magnitude appears wherever the verdict is *first* stated in each chapter, and the discussion
   says plainly that the category axis is where joint training neither helps nor hurts much. A
   reader who meets the claim anywhere meets its size within the same section.
2. **Declined: reporting the representation contrast one-sided to match §1.1.** The reviewer notes
   that Florida's contrast is significant one-sided (0.034) and not two-sided (0.067), and that the
   plan mixes conventions. The mixing is real and is now stated explicitly. But the contrast is a
   *descriptive* comparison of two input representations with no pre-registered direction, and
   switching it to one-sided after seeing that it would flip Florida from null to significant is
   exactly the choice the guardrails forbid. It stays two-sided, and Florida stays reported as not
   reaching significance, with the unanimous fold direction stated alongside.
3. **Adopted in weaker form: the arc claim.** The reviewer argues the thesis subtitle's emphasis on
   the check-in-level representation now rests on a smaller effect than the frame promises. That is
   true of the magnitude and false of the direction: the contrast favours the check-in-level
   representation at every dataset and in every one of the thirty matched folds. The frame is
   therefore rewritten to claim consistency rather than size, and §3.0 puts the thesis sentence in
   `NORTH_STAR.md` on the edit list. A stronger retreat is not warranted by the evidence.

---

## 13 · The complete edit list (reconciles the count in §0)

**Law and process (4), before any prose — see §3.0**

1. `WRITING_LAW.md` (§3 verdict binding, §7 checklist)
2. `GLOSSARY.md` (§4 superiority row; the parity entry of §3.1; the benchmark rows of §8b)
3. `NORTH_STAR.md` (thesis sentence and arc, §§1, 2, 6)
4. `CLAUDE.md` (decisions ledger)

**Gate script (1)**

5. `src_utils/check_audit_claims.py` (probes pinned to superseded strings, repointed and sabotage-validated)

**Frame chapters (5)**

6. `content.tex` (Resumo and Abstract, edited together)
7. `chapters/1_introduction.tex` (contributions, research-question answer, cosine bullet)
8. `chapters/2_fundamentals.tex` (hinge, representation narrative, elapsed time, two cosine sites)
9. `chapters/6_conclusion.tex` (four ladder sites, scaling paragraph, Texas/California hypothesis, Appendix I paragraph)
10. `chapters/3_cbic.tex` and `chapters/4_courb.tex` prefaces (frame prose only; the reproduced chapter text is untouched)

**Chapter 5 (8)**

11. `chapters/5_mobiwac.tex` (opener, and the preface contract of §11.1)
12. `chapters/5_mobiwac/01_introduction.tex`
13. `chapters/5_mobiwac/02_related.tex` (the cosine-coverage sentence)
14. `chapters/5_mobiwac/04_method.tex` (loss equation, edge direction, elapsed time)
15. `chapters/5_mobiwac/05_setup.tex` (six sites, §4)
16. `chapters/5_mobiwac/06_results.tex`
17. `chapters/5_mobiwac/07_discussion.tex`
18. `chapters/5_mobiwac/08_conclusion.tex` (scaling clause, both external margins)

**Appendices (7)**

19. `chapters/apx_a_contributions.tex` (reproducibility pointers)
20. `chapters/apx_b_errata.tex` (the identity contract, §11.1)
21. `chapters/apx_d_ceiling.tex` (benchmark distances and the Chapter 5 cross-claim)
22. `chapters/apx_f_cosine.tex` (rebuilt from the re-run, §7)
23. `chapters/apx_g_hgi_tuning.tex` (scoping sentence)
24. `chapters/apx_h_check2hgi_joint_model.tex` (feature width, edge direction, loss, §6 and §6b)
25. `chapters/apx_i_parameter_count_control.tex` (subject to §11.2)

**Tables (3) and figures (2)**

26. `tables/mobiwac/results.tex`
27. `tables/mobiwac/representation.tex`
28. `tables/frame/cosine.tex`
29. `figures/mobiwac/fig4_deltas` (regenerated)
30. `figures/mobiwac/fig3_embquality` (subject to §11.3)
31. `figures/fig_gradient_cosine.png` (regenerated with Appendix F)

---

## 14 · Author rulings, round 2 (all seven open questions closed)

Recorded verbatim in substance, with the resolution each one gets.

### 14.1 · The ladder and the category axis — RESOLVED WITH A DERIVED BOUND

The ruling was "agora o MTL empata com o dedicated, temos que ser assertivo nisso", approved for the
region axis. The category axis could not carry it, because the two-point equivalence margin is
pre-registered for region only. The resolution does not choose a new margin and does not need one.

**The device.** Two one-sided tests at margin *d* pass at the five percent level exactly when the
ninety percent confidence interval for the difference lies inside the interval from minus *d* to
plus *d*. So rather than picking a margin and testing against it, the smallest margin the data
support is **read off the interval**. Nothing is chosen, so there is nothing to justify and no
deviation to log. The document already uses this language in Appendix F ("equivalent to zero
within a margin of five hundredths"), so the reader meets a familiar device.

**What it gives, computed from the banked artifacts:**

| axis | per-dataset bound | simultaneous over all six (Bonferroni) | registered margin |
|---|---:|---:|---:|
| next category | 0.334 pp | **0.489 pp** | none registered |
| next region | 1.287 pp | **1.372 pp** | 2 pp (passes) |

So an assertive, fully test-bound sentence is available on **each axis, with its own bound**. The
two must never be merged into one number:

- **Next category: "every difference is equivalent to zero within half a point."** Verified by
  running the tests explicitly at the read-off margin, where all six category cells pass, and by
  confirming that they fail just below it, which is what makes the bound tight rather than
  comfortable.
- **Next region: "every difference stays within the two-point margin registered before any result
  was read."** The derived bound here is **1.372 pp** simultaneous, tighter than the registered
  margin but roughly three times the category bound. Half a point is *false* on this axis: Texas
  (+1.21) and California (+1.06) are outperformances and Alabama is −0.87.

The strongest correct single sentence spanning both tasks is therefore that no difference exceeds
the registered two-point margin anywhere, with the category differences an order tighter than that
at half a point.

**What still holds.** The bound is a statement about magnitude, not about sign, so the §1.5
disclosure stays: on region, Alabama (−0.87), Arizona (−0.44) and Florida (−0.16) are resolved
deficits in the reverse direction, inside the margin; on category, Florida (+0.19) is the one
Holm-surviving advantage, and the per-dataset intervals at Istanbul, Alabama and Texas exclude
zero before multiplicity correction. Every "equivalent" claim travels with the direction.

### 14.2 · Wording — plain surface, technical name glossed once

"parity / paridade" is dropped. The recurring readable surface is **the margin itself**:
"stays within the two-point margin" in English, "permanece dentro da margem de dois pontos
registrada antes de qualquer resultado ser lido" at first use in Portuguese, with
"não inferioridade estatística (teste TOST)" glossed there once and used thereafter only where the
test is being named. On the category axis the surface is the bound: "the difference is bounded
within half a point". Never "empata", "semelhante", "a par", or "matches" bare, since each asserts
equality without a test behind it. `GLOSSARY.md` gains a surface row registering the plain phrase
as the surface of the TOST verdict, and `WRITING_LAW.md` §3's verdict ladder names it, so the next
agent cannot reintroduce "matches".

### 14.3 · Chapter 5 preface — "manuscrito revisado após a avaliação final"

The preface says the chapter reproduces the manuscript as revised after the final evaluation; the
submission status wording stays "submitted, under review". `apx_b_errata.tex`'s "the two texts stay
identical" contract is rewritten to say that the revision supersedes the submitted numbers and may
be sent to the venue.

### 14.4 · Appendix I — measure first, then decide (calibration running)

Kept, not cut. The author's recollection of the original control was checked against the record and
differs: it ran at **Alabama and California** (not Florida), at **four seeds by five folds, twenty
fitted models per arm** (not five), across five arms, on the superseded `check2hgi_dk_ovl` engine.
Estimates from scaled wall times gave 0.7–1.7 h (Alabama, winning arm), 9–26 h (both datasets), or
23–64 h (full reproduction), and the author chose to replace the estimate with a measurement: one
fold at Alabama, hidden 672, on the final engine, which is running. Two things are settled
regardless of the number: the matched widths must be **re-derived against the final joint parameter
count** before any arm runs, and the appendix's **framing changes either way**, because it was
written to rule out parameter count as the explanation for a seven-to-eight point category
advantage that is now 0.19 points at one dataset. Its stored verdict string is written against the
superseded board and is not reusable. If the refreshed wider arm beats the joint model on category,
that is reported.

### 14.5 · Figures — full inventory first (done; rebuild list below)

| figure | prints where | data source | verdict |
|---|---|---|---|
| `fig1_dataflow.pdf` | Ch.5 related work | hand-drawn TikZ; its only data claim is the node-feature list "category, hour, weekday" | **stale**: must gain elapsed time |
| `fig2_model.pdf` | Ch.5 method | hand-drawn TikZ; labels the trunk "bidirectional cross-attention" (correct: that is the attention, not the graph) | check dimensions against the shipped configuration during the Appendix H sweep |
| `fig3_embquality.pdf` | Ch.5 results | its own docstring: the archived `design_k` engine, five states, Georgia excluded | **stale**: re-measure on the final engines |
| `fig4_deltas.pdf` | Ch.5 results | the dissertation copy predates the paper's rebuild (Jul 23 vs Aug 11) | **stale**: regenerate from the served-checkpoint deltas |
| `fig_gradient_cosine.png` | Appendix F | the superseded cosine parquet | **stale**: regenerate from the re-run |
| `check2hgi_flow.tex` | Appendix H | hand-drawn; numerals inspected and are all TikZ geometry, not data | no data defect; check the two 64-dimension labels in the sweep |
| `joint_model_flow.tex` | Appendix H | hand-drawn; no data-bearing labels found | no defect found |
| `cbic_mtlnet_arch.png`, `courb/arquitetura_modelo.png`, `courb/distribuicao_estados.png` | Ch.3, Ch.4 | published-chapter figures | **do not touch** |

The `[mobiwac]/src_fix/figs` copies were audited alongside: `fig1`/`fig2` exist there only as TikZ
sources, and `fig3` carries the same stale geometry constants as the dissertation copy, so on that
one figure the dissertation deliberately diverges from the paper rather than following it.

### 14.6 · The shortlist sentence — recompute, then choose in the author's order

Kept if the evidence supports it. The sentence rests on two quantities: the share of visits whose
true next region is among the ten shortlisted (California, 65.69, superseded; the final board gives
64.54 for that cell) and the geographic spread of the shortlist (3 to 8 km against 17 to 176 km).
The second is the harder problem: its record shows it was computed on the superseded preparation at
four datasets with a per-sample prediction dump, and per-sample predictions are not serialized by
default. The dump flag still exists in the evaluation path, so the measurement is reproducible, but
it requires a re-run with the flag set rather than a rescore. Order of preference, as instructed:
fix it from a recomputation; failing that, make the mobility point another way (the Acc@10 shortlist
share alone, without the distance claim); remove only as a last resort.

### 14.7 · Appendix H — full legacy sweep, not a fix list

Every architectural and training statement in the appendix is verified against the shipped
configuration: input width, edge direction, loss weights, embedding and hidden dimensions, head
counts, dropout, learning rates, scheduler, epochs, folds, selector, and the region-transition
prior. The forward-only construction is described as the design, with the rationale as a design
principle, per directive 1.
