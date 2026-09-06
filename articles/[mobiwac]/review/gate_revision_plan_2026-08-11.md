# Adversarial advisor — pre-execution gate on `src_fix/REVISION_PLAN.md`

> Persona 14 (`reviewers/14_adversarial_advisor.md`), Common protocol `reviewers/README.md`.
> Read-only. Scope: the plan before it is applied, assuming a competent but literal writer.
> Law consulted: `[mobiwac]/GLOSSARY.md` (wins for this paper), `WRITING_LAW.md`,
> `AGENT_GUARDRAILS.md` §1-§3, `PAPER_PLAN.md` §3, `ERRATA.md`.
> Evidence re-derived from `docs/studies/closing_data/v18/data/v18_results.json`,
> `docs/results/closing_data/{v18,v18_probe,v18_trunk,v18_2,markov_floor_stride1}/`,
> `v18/{METHODOLOGY,FINAL_SETTINGS,PRECISION_CAVEAT,POSTPONED,GAPS,LOSS_WEIGHT_PROBE}.md`,
> `v17_completion/STATISTICAL_PROTOCOL.md`.

---

## (a) Verdict

**DO NOT EXECUTE AS WRITTEN.** The plan's arithmetic is sound and its verdict ladder reproduces the
board exactly (all 24 dedicated and diagnostic-best cells match `v18_results.json` to four decimals).
Its instincts on the trunk are right. But it is not yet an executable edit list: it rests one of its
two load-bearing evidentiary conclusions on a confounded contrast, it omits the file that carries the
abstract, it does not tell the author that its recommended convention destroys a headline claim, and
several of its instructions, applied literally, would produce prose that violates the paper's own law
or asserts something the evidence contradicts.

Eight items are blocking. Fix those, add the missing files to §4, and re-issue the plan; the
structure underneath is worth keeping.

---

## (b) Top three

**T1 (F1, BLOCKER)** The one trunk contrast the plan calls "resolved" compares two different
training configurations. The severed Florida arm ran at `--category-weight 0.75`; the joint arm it is
paired against ran at 0.50. Weight-matched, the effect is +0.023 (p = 0.21, 4/5 folds), not +0.23
(p = 0.0014, 5/5). The plan forbids an attribution sentence partly on the strength of that number.

**T2 (F2, BLOCKER)** Convention (A) breaks the region-count scaling claim, and §2 does not say so.
Under joint-best the U.S. region deltas ordered by region count are −0.874, −0.437, −0.156, +1.206,
+1.057: California, with the most regions, is below Texas. The claim is contribution 3 of the
introduction, the ordering rationale of Table 3 and Fig. 4, and a sentence in the conclusion.

**T3 (F3, BLOCKER)** §6.1's other numbers are not in the edit list, and one of them now refutes its
own claim: the feature-concatenation control raises the place embedding by +2.0 at Alabama against a
new place-to-check-in gap of +1.62.

---

## (c) Findings, ranked

### F1 · BLOCKER · The Florida trunk contrast is loss-weight-confounded

> "The single contrast that *is* resolved runs the other way: at Florida, severing the trunk
> **raises** region accuracy by 0.23 points on all five folds." — `REVISION_PLAN.md` §1.4, L116-119
> (table row L112: `| Florida | region | 77.05 | **77.28** | **+0.23** | **5/5** | **0.0014** |`)

The two arms differ in two things, not one. `run_trunk_florida.sh:43` runs the severed arm with
`--mtl-loss static_weight --category-weight 0.75`; the joint comparand is `v18_probe/FL_cw050.json`,
the 0.50 arm, which the plan's own CSV labels `trunk severed vs joint (arm B, 0.50)`. The shipped
configuration is 0.50 (`FINAL_SETTINGS.md` family (c); `run_wave.sh:209`).

`LOSS_WEIGHT_PROBE.md` §5 measures that second difference directly at Florida: region 0.50 − 0.75 =
−0.202, negative on 5/5 folds, p = 0.001. It accounts for essentially the whole contrast.
Re-derived here from the per-fold arrays (the reconstructed 0.75 arm means 77.2575 against the banked
wave-1 cell's 77.2552, so the reconstruction is sound):

| Florida region, severed minus joint | Δ | folds severed higher | paired t p |
|---|---:|---:|---:|
| as the plan pairs it (severed @0.75 vs joint @0.50) | **+0.227** | 5/5 | **0.0014** |
| weight-matched (severed @0.75 vs joint @0.75) | **+0.023** | 4/5 | **0.21** |

Alabama is unaffected: −0.138 against arm A, −0.142 against arm B. So the eighth row is the only one
that moves, and it is the only one the plan treats as evidence.

*Direction:* re-pair the Florida row against the 0.75 joint arm, or drop the row. Then §1.4's
"resolved only once, in the direction of the severed arm" is false and the paragraph's conclusion
becomes "no contrast at either ablation dataset resolves in either direction", which is a cleaner
basis for the same refusal to attribute.

### F2 · BLOCKER · Convention (A) kills the scaling claim; §2 does not price it

> "Cost: the category headline becomes \"matches at five datasets, beats at Florida\" rather than
> \"beats everywhere\". Region keeps its two clear wins." — §2, L166-168

That is the whole cost as the author is being shown it. It is incomplete. Ordered by region count:

| convention | AL | AZ | FL | TX | CA | strictly rising? |
|---|---:|---:|---:|---:|---:|---|
| joint-best (A) | −0.874 | −0.437 | −0.156 | +1.206 | +1.057 | **no** |
| diagnostic-best (B) | −0.450 | −0.221 | +0.391 | +1.912 | +1.953 | yes |

Under (A) the monotone claim fails at the top of the range. Affected sites: `01_introduction.tex:37-44`
("the region result moves monotonically with the region count ... with the state with the most regions
(California) showing the largest region gain"), `06_results.tex:63-66`, `08_conclusion.tex:11`,
`figs/fig4_deltas.tex` caption, and the ordering rationale of both result tables. It is also the claim
`GLOSSARY.md` §1 scopes explicitly ("scoped to the five U.S. states").

*Direction:* put this in §2 as part of the (A)/(B) trade before the author rules, and name the
replacement claim under (A): the region gain is negative or negligible below about 4,700 regions and
positive above it, without a monotone ordering.

### F3 · BLOCKER · §6.1's controls are not in the edit list, and one now refutes its claim

> "§6.1: the representation subsection reports +1.62 at Alabama (and its siblings when they land),
> not a two-order-larger margin." — §4, L213-216

That is the only §6.1 number the plan names. The subsection contains four more claim-bearing blocks
anchored to the superseded check-in-level column, all of which move with it:

| site | current text | status under v18 |
|---|---|---|
| `06_results.tex:19-27` | feature-concat "raises the place embedding by only $+2.0$, $+1.7$, and $+0.8$ ... under a tenth of the place-to-check-in gap at each state" | AL: +2.0 against a gap of **+1.62**. The control now **exceeds** the gap. AZ: +1.7 against +2.58 (66%). FL: unmeasured. |
| `06_results.tex:19-27` | "The gain therefore comes from the hierarchical per-visit representation, not from contextualization alone or feature injection." | falsified at Alabama by the row above |
| `06_results.tex:15-19` | CTLE frozen: "our representation ahead by $+37.8$, $+37.0$, and $+28.7$" | largest available new gap is +6.29 |
| `06_results.tex:15-19` | CTLE end-to-end at FL: "$33.45$ ... far below the check-in-level representation's $75.15$" | the 75.15 comparand no longer exists; FL v18 dedicated cat is 37.35 |
| `06_results.tex:12-14` | silhouette $\approx 0.57$ vs $0.00$; kNN purity $\approx 0.98$ vs $0.78$ | **UNVERIFIED — blocked on** a v18 embedding-geometry measurement; none found under `docs/results/embedding_eval/` |

*Direction:* §4 must enumerate these five sites with a per-site ruling. The feature-concat sentence in
particular cannot be repaired by re-stating a number: at Alabama the control's conclusion inverts, so
the site needs either a re-measured control on the v18 engine or removal of the Alabama instance with
the claim scoped to where it still holds.

### F4 · BLOCKER · The edit list omits `main.tex` and `figs/fig4_deltas.tex`

§1.5 names "the abstract's \"about 28 to 40 points\" clause" in prose, but §4, the file-by-file list,
never names the file the abstract lives in. `main.tex:66-87` also carries, under (A):

- "outperforms a dedicated category model on every dataset (about $+5$ to $+9$ macro-F1)" — becomes
  one dataset of six;
- "outperforms the dedicated region model on four of the six, matching it (statistically, within two
  points) on the other two" — becomes two and four;
- "Across the five U.S. states, the region gain grows with the number of regions" — see F2;
- "At Istanbul ... ahead on region ($+0.19$ Acc@10, statistically supported)" — becomes −0.079, a match;
- the Fig. 1 and Fig. 2 captions (L103-118), which are `main.tex` text.

`figs/fig4_deltas.tex` is also missing. The plan lists `figs/fig4_deltas.py` only:

> "`figs/fig4_deltas.py` — Regenerate from the new deltas. Same design, new inputs." — §4, L251-252

The caption is a separate file and both of its clauses are false under (A): "The category gain
(macro-F1) is positive at every dataset; the region gain (Acc@10) rises across the five U.S. states
and is also positive at Istanbul."

*Direction:* add `main.tex` and `figs/fig4_deltas.tex` to §4 with their own bullets.

### F5 · BLOCKER · Category becomes a TOST claim on an axis with no pre-registered margin

> "\"matches\" = TOST non-inferiority within the pre-pinned two-point margin." — §1.2, L79-80

The two-point margin was pinned for one axis only. `STATISTICAL_PROTOCOL.md` §0 is explicit that a
margin is per-axis and "MUST NOT be silently reused", and §3.2 pins δ_reg = 2 pp for the
MTL-vs-STL **region** axis with a justification specific to Acc@10 over thousands of region classes.
No δ_cat exists. Macro-F1 at 30 to 37 is not the same scale as Acc@10 at 59 to 77, so two points does
not carry over by assertion.

This also falsifies a paragraph the plan does not list. `05_setup.tex:38-42` currently states:

> "A written analysis plan, fixed during development and before any result was read, assigned a
> superiority test to next-category prediction and a non-inferiority test to next-region prediction."

Under (A), five of six category cells become non-inferiority claims outside that plan, and the Holm
sentence's counts ("across the six next-category comparisons and, separately, across the four
next-region comparisons") no longer describe the analysis. That paragraph is the subject of the
2026-07-25 `ERRATA.md` entry; rewriting it without care re-opens a correction that was fought for once.

*Direction:* §4 needs a `05_setup.tex` bullet covering the analysis-plan paragraph, not only "define
the reported convention in one sentence". The author must decide whether δ_cat = 2 pp is adopted as a
new, disclosed, post-hoc margin, or whether the category cells are reported as "no measured
difference" without an equivalence claim.

### F6 · BLOCKER · "The externals are unaffected" is true of the numbers and false of the prose

> "Keep the external baselines and floors as they are; they are unaffected." — §4, L203

The external cells do not move. Three prose sentences that quantify the distance to them do, because
the joint model moved. Re-derived from `markov_floor_stride1/*.json` and the Table 3 external columns:

| site | current text | under (A) | under (B) |
|---|---|---|---|
| `06_results.tex:88-92` | Markov-1 floor "exceeds it by $4.9$ to $10.3$ points on all six datasets" | 4.07 to 10.02 | 4.61 to 10.21 |
| `08_conclusion.tex:11` | "by at least 4 Acc@10 points over the strongest region reference" | min **3.55** (FL) | min 4.10 |
| `08_conclusion.tex:11` | "by at least 33 macro-F1 points over POI-RGNN" | min **3.06** (FL) | min 3.10 |

The ordering claim ("above every external baseline on both tasks") survives at every dataset under
both conventions. Only the magnitudes fail.

*Direction:* reword the instruction to "the external cells keep their values; every sentence that
states a distance to them is recomputed", and list the three sites.

### F7 · BLOCKER · The sweep sentence contradicts `FINAL_SETTINGS.md` and undoes a logged narrowing

> "Add one sentence, in passing, that the reported configuration follows a hyperparameter sweep
> covering both the joint and the dedicated models, so the comparison is between tuned arms." — §4, L220-222

`FINAL_SETTINGS.md` family (b), dedicated next-region, opens: "**Never swept.** This study tuned the
category recipe only". So "covering both models" is true on the category axis and false on the region
axis. "The comparison is between tuned arms" would therefore be a false statement about the region
comparison, and it would contradict the sentence the author already owns at `07_discussion.tex:75-78`:
"while both sides of the region comparison run a fixed configuration" — the scope narrowing logged in
`ERRATA.md` (2026-08-05, item 2) as a mitigation that had been made to sound stronger than its
evidence.

The same bullet also loses a disclosure. `V18_RESULTS.md` disclosure 2: the dedicated-category
comparator's per-state learning rate was selected on this same cross-validation while the joint
model's category learning-rate axis was measured null, so Δcat is "a **conservative** estimate of any
MTL advantage, and an optimistic one of any MTL deficit". Under (A) the category column is deficits
and matches, so the bias now runs **against** the paper's reading and must be disclosed in that
direction.

*Direction:* scope the sweep clause to the category comparison, keep the region-side fixed-configuration
sentence verbatim, and carry the deficit-direction disclosure.

### F8 · BLOCKER · The Table-2 fallback wording is banned process narration

> "the table reports the datasets that were measured and says so; it does not carry a mixed-recipe row."
> — §4, L209-210
> "Table 2 reports the datasets measured under the matched recipe and says plainly that the rest were
> not re-measured, which costs one sentence and no honesty." — §6, L277-279

"The rest were not re-measured" states that a previous measurement exists. That is author directive 1
(no retrospective references) and `WRITING_LAW.md` §1, which bans process narration absolutely and
names this exact sub-class: "if the sentence would be false or pointless once the circumstance
changes, or if it explains why something is missing rather than stating what is present, cut it".

The information is not lost by cutting it. A three-dataset table with a scoped claim ("at Istanbul,
Alabama, and Arizona") states what is present and asserts nothing about six.

*Direction:* replace with "the table reports the datasets it covers and every claim reading it names
that scope", and delete "and says so" / "were not re-measured" from both sites.

### F9 · BLOCKER · Only half the representation change is mandated, and the unmandated half falsifies a limitation

Directive 2 asks for the elapsed-time node feature. `METHODOLOGY.md` §1 defines v18 as two changes:
the elapsed-time columns **and** a consecutive-visit graph that "keeps only `src < tgt`, in **training
and at readout**". The plan mandates the first and is silent on the second.

The second is the one with textual consequences. `07_discussion.tex:79-82` (the fourth limitation,
added 2026-08-05 on the author's own ruling and probe-guarded as `R13-leak4th`) reads:

> "Fourth, each visit node in the representation graph is linked to the visit that follows it, and
> because category is a node input feature the vector of an earlier visit could absorb the category
> of the next one."

On a forward-only graph that describes a build that no longer exists. `05_setup.tex` carries the
matching disclosure. Left unedited, the paper discloses a threat it has designed out; edited without
care, it becomes a retrospective reference banned by directive 1.

It is stateable without retrospection, as a property of the present construction ("each visit node
links forward in time only, to the visit that follows it"), which is why this is a fixable blocker
rather than a conflict between the directives. But the plan must say so, and must rule on what
happens to the limitation count, which `ERRATA.md` records as functioning as the paragraph's own
inventory.

*Direction:* add a `07_discussion.tex` bullet and extend the `05_setup.tex` bullet; state the
forward-only edge as a design property; re-rule the limitation count.

### F10 · MAJOR · The claim registry still encodes the old verdict map and the plan does not amend it

`AGENT_GUARDRAILS.md` §3 C1 makes `PAPER_PLAN.md` §3 the governing whitelist, and `GLOSSARY.md` §1
carries the region wording as law: "the joint model **outperforms** at **Istanbul (+0.19 ...),
Florida (+0.71 ...), Texas (+2.11 ...), and California (+2.20 ...)**" and "matches ... at
**Alabama ... and Arizona**". `ERRATA.md`'s "Constraints to preserve" repeats it. Under (A) that set
becomes {Texas, California}; Istanbul and Florida move to matches.

A literal writer working from the plan and then checking the law will find the law contradicts the
plan and, per `L2`, the law wins. The plan needs an explicit instruction that `GLOSSARY.md` §1, §6,
`PAPER_PLAN.md` §3 and `ERRATA.md`'s constraint list are amended in the same commit, with the author
signing the amendment. Note "never upgrade Arizona" survives unchanged and should be restated, not
dropped, when the list is rewritten.

### F11 · MAJOR · "Has never been measured" overstates the gap, and omits evidence that cuts against the trunk

> "The trunk's contribution at **Texas and California ... ** has never been measured. That ablation
> is the deferred P4 experiment." — §1.4, L124-127

`docs/studies/closing_data/v18/region_1fold_triage/FINDING.md` records single-fold arms at both
states: severing the trunk moves region by −0.099 (CA) and −0.120 (TX), and at California deleting
the category task entirely moves it by −0.077. Its own reading: the hypothesis that the trunk carries
the +2 pp "is refuted by a test with power for it", while a sub-0.15 pp contribution stays unresolved.

That is a materially different statement from "never measured", and it cuts against the author's
directive 3 rather than for it: the plan tells the writer to name the trunk as a contributing
component of the architecture delivering the region result, at the two datasets where a powered
screen disfavours exactly that. The plan's refusal to write a causal sentence is right; its evidence
summary understates how little room there is.

Two caveats the triage's own record requires when it is cited: the arms ran on the v17 substrate and
the pre-retune recipe.

*Direction:* replace "has never been measured" with the accurate statement (large-effect hypothesis
disfavoured by a single-fold screen; a sub-0.15 pp contribution unresolved; five-fold ablation
deferred), and keep the refusal to attribute.

### F12 · MAJOR · §3 protects a Related Work sentence the new ladder falsifies

> "Section 2 (Related Work), Section 3 (Problem), and the dataset table." — §3, L192

`02_related.tex:47-50` ends: "this paper introduces the next-region task and adds the check-in-level
representation, **on which sharing helps instead of hurting**
(Section~\ref{sec:results-part2})". Under (A) sharing matches at five of six on category and at four
of six on region. The sentence is also the one self-positioning delta `GLOSSARY.md` §9.3 permits, so
it cannot simply be deleted; it has to be re-scoped.

The same file, at L99-105, carries "Measured during development on the same joint architecture (**on
an earlier preparation of the data**)". That parenthetical is already the class directive 1 bans, and
it becomes more conspicuous once the representation has changed underneath it.

*Direction:* remove Section 2 from the "does not change" list and add a bullet for these two sites.

### F13 · MAJOR · The plan drops the honesty device the new ladder needs in three more places

`06_results.tex:57-60` already carries the pattern for a negative-but-non-inferior cell: "At Alabama,
the whole interval lies below zero, a small but statistically significant deficit, still well within
the two-point margin." Under (A) that situation now holds at Alabama category (−0.188, CI
[−0.334, −0.043]), Texas category (−0.131, CI [−0.186, −0.076]), Alabama region (−0.874,
CI [−1.003, −0.746]), Arizona region (−0.437), Florida region (−0.156) and Istanbul region (−0.079,
CI [−0.156, −0.002]).

The plan's §6.2 instruction says only "Every verb bound to its own test; no non-inferior cell upgraded
to a win". Upgrading is not the only failure available: reporting six negative point estimates as
"matches" without the deficit sentence understates them.

*Direction:* mandate the existing sentence pattern for every cell whose interval lies entirely below
zero, and name the cells.

### F14 · MAJOR · The tuning hypothesis is unscoped and drops a mandated disclosure

> "Add the tuning hypothesis for the category gap at Texas and California: the joint model there
> inherits a configuration selected without a joint-specific search at those two datasets" — §4, L227-231

Two problems. First, at California the joint-best category delta is **−0.004**. There is no gap to
hypothesize about; a paragraph explaining it is a claim about noise, and `GAPS.md` §7 item 1 records
the practical-significance floor as an open author decision precisely for cells of this size. Texas
(−0.131) is the only site where the hypothesis has an object.

Second, `POSTPONED.md` P6 attaches a standing disclosure to any large-state category number:
"wherever a large-state category number is reported it must say the recipe was **transferred, not
validated**". That applies to both arms: `FINAL_SETTINGS.md` family (a) grades the large-state
dedicated tier `[1f] TX` only. So the asymmetry the hypothesis implies (the joint model untuned, the
dedicated model tuned) is not what the record shows at these two datasets.

*Direction:* scope the hypothesis to Texas, state the magnitude, and carry the transferred-not-validated
disclosure for both arms.

### F15 · MAJOR · The plan sends an edit to a file that does not exist, and to one that must not take it

> "One sentence in §4; the feature-width figure in the appendix table follows." — §4, L243-244

This paper has no appendix and no feature-width table. `main.tex` ends at the bibliography; the
`02_related.tex` comment block states it directly ("this article has no appendix"). The instruction is
a dangling pointer a literal writer would either ignore or invent.

The same bullet names `sections/03_problem.tex`. `PAPER_PLAN.md` §9.3 fixes that section's scope:
"Problem section describes the problem only, zero solution mechanics (no window sizes, no models)".
Node features are solution mechanics.

*Direction:* drop the appendix clause and drop `03_problem.tex` from the bullet; the node-feature
sentence belongs in `04_method.tex` §4.1 alone.

### F16 · MAJOR · Net length is positive against a budget that is already over

`ERRATA.md` records the current build at **9 pages**; `PAPER_PLAN.md` §11 budgets 8 (10 with a fee).
The plan adds a convention sentence (§5.3), a sweep clause (§5.3 and §6), an elapsed-time sentence
(§4.1), a tuning-hypothesis passage (§7), a restructured trunk paragraph (§7), a Table-2 scope
sentence, and the deficit sentences of F13. It names no compensating cut, and persona 14's contract
requires the net-size call.

Rough estimate: +12 to +18 lines of running text before the F3 and F13 additions, against zero
identified savings. The one natural saving the plan creates and does not claim: under (A) the
verdict prose simplifies, and Fig. 4's ordering rationale (F2) may no longer earn its space.

*Direction:* add a §4 line naming what shrinks, and decide the 8-versus-10-page variant before the
text is touched rather than at typesetting.

### F17 · MAJOR · UNVERIFIED — the representation contrast has no artifact in the repository

§1.5's three completed rows (Istanbul +6.29, Alabama +1.62, Arizona +2.58) are the basis of "the
single largest change to the paper's claims". I could not verify them. The path named in the review
brief, `docs/results/closing_data/v18_place_level/`, does not exist; a repository-wide search for the
values 29.0685 / 29.1481 / 31.9278 and for the deltas 6.2854 / 1.6173 / 2.5802 returns nothing outside
the plan itself; no directory matching `*place*level*` exists anywhere in the tree.

What *does* verify: the check-in-level column of that contrast is the v18 seed-0 dedicated cat cell at
all three datasets, exactly (`v18/{istanbul,alabama,arizona}_s0_cat.json` = 35.3539 / 30.7654 / 34.508).
So the contrast's own arm is real and traceable; only the place-level arm is unlocatable.

**UNVERIFIED — blocked on** the place-level sidecars being committed under `docs/results/closing_data/`.
Per `AGENT_GUARDRAILS.md` N1/N3, Table 2 cannot be rewritten from a number that exists only in the plan.

### F18 · MINOR · Register defects in the plan text that would transfer to the page

Four classes, all in instruction text a writer would work from:

- **"beats"** as the superiority verb, throughout, including inside the §6.2 instruction ("Category:
  beats at Florida, matches elsewhere"). `GLOSSARY.md` §1: "The superiority verb is \"outperforms\"
  (never \"beats\" / \"wins\")."
- **"margin" for the Part-1 gap**: "reports +1.62 at Alabama ..., not a two-order-larger margin"
  (§4, L214). `GLOSSARY.md` §4 reserves "margin" for the TOST margin; the representation difference is
  a **gap**. The plan uses the reserved word in the instruction for the exact sentence where the
  reservation bites.
- **British spelling**: "All 20 of 20 (seed, fold) pairs **favour** the joint model" (§1.3, L145) and
  the §1.5 column header "folds **favouring** check-in level", which is the kind of string that gets
  copied into a caption. `WRITING_LAW.md` §1 bans it.
- **em-dashes** throughout, banned by `GLOSSARY.md` §6 and `WRITING_LAW.md` §1.

*Direction:* one pass over the plan's own prose before it is handed to a writer, or an explicit
"instruction text is not model prose" banner at the top.

### F19 · MINOR · Two instructions are retrospective in form

"not a two-order-larger margin" (§4, L214) and "The claim becomes \"a consistent, measurable advantage
under a matched recipe\"" (§4, L215-216) both define the new claim by contrast with the old one. The
second is also an "X, not Y" construction, which `GLOSSARY.md` §7 permits only where it scopes a claim
and lists five ledger-mandated keeps; a sixth would need sign-off.

*Direction:* state the target claim positively and let the number carry the size.

### F20 · MINOR · Internal inconsistency about how much of Table 2 exists

§0 says the representation contrast is "§1.5 — running; Alabama complete". §1.5 shows three complete
rows. §6 lists "place-level arm at Istanbul, Alabama, Arizona | **complete**". §4 then says "reports
+1.62 at Alabama (and its siblings when they land)", which reads as Alabama-only.

A literal writer following §4 writes a one-row table when three rows exist.

*Direction:* make §0 agree with §1.5 and §6, and state the row count §4 should assume.

### F21 · NIT · The 24/24 integrity check is not independently reproducible from the repository

> "every reconstructed cell reproduces its banked per-task sidecar to 4 decimals; reconstructed
> per-fold arrays agree with the board's own arrays to 5e-05 | verified for 24/24 cells" — §0, L18

Nine `joint_best_score.json` artifacts exist under `v18_2/modal_runs/*/*_joint/`, and all nine
reproduce their `v18/` sidecar's diagnostic-best cell exactly, which is a real check that passes. The
other fifteen cells' rundirs are not in the repository (no `standard_scores.json` exists anywhere
under `docs/`), so the claim is not falsifiable here. Not a defect in the plan; a note that the
sentence asserts more than a reader of this repository can confirm.

---

## (d) What holds — do not touch

1. **The refusal to write a causal attribution sentence for the region gain** (§1.4). Correct on the
   evidence, correct against directive 3 read honestly, and consistent with the 2026-07-28 and
   2026-08-05 `ERRATA.md` rulings that already withdrew component attribution once. It survives F1 and
   F11 intact; both findings make it *better* supported, not weaker.
2. **The three bounds §1.4 places on the text**: no claim that the trunk carries the category task, no
   attribution of the Texas and California region gains to it, and no description of it as doing
   nothing. That is the exact needle directive 3 asks for.
3. **The recommendation of (A) on single-model grounds.** "The paper's whole argument is that *one*
   model serves both tasks" is the right reason, independent of what the ladder does. F2 changes the
   price, not the principle.
4. **§3's protected list**, minus Section 2 (F12): the four-level graph construction, the sliding-window
   protocol, the user-disjoint folds, the seeds, the metric definitions, the baselines and their roles,
   and Table 1. None of Table 1's columns depend on the representation.
5. **The battery's cell values.** All 24 dedicated cells and all 12 diagnostic-best joint cells
   reproduce `v18_results.json` to four decimals; the dedicated column is correctly restricted to
   seeds that have a joint cell (`stl_cat_paired`). The plan's numeric spine is sound.
6. **The per-seed footing (n = 4) as the superiority footing** (§1.2). It matches `05_setup.tex` as
   written and avoids the fold-pooling anti-conservatism that `V18_RESULTS.md` discloses. Keep it, and
   do not let the fold-level columns of the battery reach the page.
7. **The Table-2 caption instruction** that both arms run the same recipe, folds and windowing and that
   the contrast isolates the input representation. That is the sentence which makes the contrast
   defensible.
8. **The plan's own honesty posture** in §1.4 and §1.5: it reports a result that costs the paper its
   headline and does not soften it. That is the standard the rewrite has to hold.

---

## (e) Open questions only the author can answer

1. **Convention, re-asked with the full price.** (A) or (B), now knowing that (A) also costs the
   region-count scaling claim (F2), contribution 3 of the introduction, and Fig. 4's ordering rationale.
2. **The practical-significance floor** (`GAPS.md` §7 item 1, still open). Is a +0.04 reportable as a
   direction? Under (B) the ladder awards "outperforms" to Arizona (+0.043) and Texas (+0.042) on
   category. Under (A) the tuning hypothesis is asked to explain −0.004 at California. The same
   decision governs both.
3. **Does the category axis get a margin?** If (A), five category cells become equivalence claims on an
   axis with no pre-registered δ (F5). Adopt δ_cat = 2 pp as a disclosed post-hoc margin, report those
   cells as "no measured difference" without an equivalence claim, or something else. This also decides
   how `05_setup.tex`'s analysis-plan paragraph is rewritten without re-opening the 2026-07-25 erratum.
4. **Table 2 scope.** Three datasets or six. If three, does contribution 1 of the introduction still
   stand as written ("across all six datasets"), or does it re-scope to the three measured?
5. **The forward-only edge** (F9). May it be stated as a design property of the representation? If yes,
   the fourth limitation is rewritten and the limitation count is re-ruled. If no, the paper discloses a
   threat its build no longer carries.
6. **The §5.2 integrity numbers.** The train-users-only audit deltas (region −0.33 to +0.01, category
   0.00 to +0.29) and the 67-to-87-percent coverage figure were measured on the previous substrate and
   have not been re-run. `GLOSSARY.md` §9.4 forbids cutting the leak audit below its evidence floor.
   Re-measure, or carry them with a scope the author signs off on?
7. **Page budget** (F16): 8-page cut or the 10-page fee variant, decided now.
8. **The submission itself.** The paper is under review at MobiWac. A revision of this magnitude is not
   a camera-ready correction. Does this rewrite target the review response, the dissertation's Chapter 5,
   or both, and does the dissertation's §5 follow-through (`REVISION_PLAN.md` §5) still wait on it?
