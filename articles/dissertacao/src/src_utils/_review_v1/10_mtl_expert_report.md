# MTL Expert Review — Dissertation v1 (persona 10)

> Reviewer: MTL domain expert (persona `10_mtl_expert.md`). Read-only.
> Scope: Ch.2 (MTL fundamentals), Ch.3 (CBIC), Ch.4 (CoUrb), Ch.5 (MobiWac), MTL claims in Ch.1/Ch.6.
> Default prior: a tuned fixed-weight scalarization matches specialized MTL optimizers
> (Kurin 2201.04122; Xin 2209.11379; Royer 2310.08910; Hu 2308.13985).
> Sources of truth per reviewers/README §Sources. Numbers traced, never recomputed unless noted.
>
> STATUS: COMPLETE (2026-07-23).

## Files reviewed (read in full)
- src/chapters/2_fundamentals.tex — §2.3 MTL is the core of scope; §2.4 evaluation
- src/chapters/3_cbic.tex — the published null result (arc foundation)
- src/chapters/4_courb.tex — representation ablation (within-MTL)
- src/chapters/5_mobiwac.tex — the joint-model win (highest MTL content)
- src/chapters/1_introduction.tex — MTL/arc claims
- src/chapters/6_conclusion.tex — the consolidated MTL answer + capacity baseline
- Traced against: docs/studies/closing_data/{RESULTS_BOARD,perhead_lr_n20,joint_best/JOINT_BEST_RESULTS,v17_completion/CEILINGS_N20_FINAL}.md;
  storyline/audit/capacity_baseline_experiment.md; articles/[mobiwac]/src/sections/*.tex (to establish what is inherited vs introduced).

## Overall verdict: **SOUND-WITH-CORRECTIONS**

The MTL content is, at the level of the science, in strong shape and unusually well defended against
the field's 2022–2026 skeptical turn: the dissertation's own finding (a tuned fixed weighting won; the
balancers did not) *aligns* with the field null, and the text cites the skeptic literature (Kurin, Xin)
rather than only pro-balancer work. The freeze control, the parameter-count disclosure, the
capacity-matched baseline, the joint-best convention with its robustness bound, the test-bound region
verbs, and the task-pair confound concession are all present where the claims live. The verdict is not
"sound" because of **one BLOCKER**: Chapter 5 states that the CBIC prior work studied *next-category and
next-region* and *observed negative transfer*, which is false on both counts and contradicts Chapters 1,
3, 4, and 6 of this same document. It is inherited verbatim from the MobiWac version of record, so it is
a framing error the coletânea format newly exposes, and it is fixable through the sanctioned errata
mechanism, not an integrity problem with the experiments. Three MAJOR items (a cross-chapter joint-value
inconsistency, a mechanism sentence that over-generalizes the gradient-cosine measurement, and a
loss-shaping concordance gap) should be fixed before the advisor sees the document.

## Top 3 findings
1. **[BLOCKER] Ch.5 attributes a next-region task and an observed negative-transfer result to CBIC,
   which had neither** — contradicts Ch.1/Ch.3/Ch.4/Ch.6 (Finding 1).
2. **[MAJOR] The joint Alabama next-category score is 64.51 in Ch.5's headline table but 64.54 in Ch.6's
   capacity discussion** — the joint-best/diagnostic-best distinction (lens 6) blurs across chapters
   (Finding 2).
3. **[MAJOR] "A balancer therefore has no conflict to resolve" over-generalizes a directional (cosine)
   measurement to magnitude-based balancers** — the single most likely MTL-examiner objection; tighten and
   cite Elich (Finding 3).

## Ranked findings

### Finding 1 — BLOCKER (lens 12; attack Q12; NORTH_STAR §6 signed-off addition (a))
**Ch.5 states CBIC studied next-category + next-region and observed negative transfer. Both are false, and
both contradict the rest of the dissertation.**

Quotes (src/chapters/5_mobiwac.tex):
- L44: *"Prior work observed exactly this for next-category and next-region~\cite{silva2025mtlnet}"*
  (the "this" = "shared parameters can converge to a compromise ... helping one while hurting the other").
- L140: *"Our earlier work~\cite{silva2025mtlnet} established this two-task setup and observed negative
  transfer (sharing hurt one task)"* (in §5.2.3, titled "Predicting the next category and the next region").

Why it is false:
- CBIC (Ch.3, L34–35) studies **(1) POI Category Classification [static] and (2) Next-POI Prediction =
  "Predicting the category of the next POI"**. There is **no region task anywhere in Ch.3**. CoUrb (Ch.4,
  L24–25) studies the same two. **MobiWac (Ch.5) is the first chapter to add next region** — the chapter
  itself claims this novelty ("the first work to treat fine-grained region as an end target", L58).
- CBIC's *result* was a parity null, not an observed negative transfer. Ch.3 reports the MTL/Single
  difference is *"largely comparable ... without a clear, consistent, and significant advantage"* and only
  **hypothesizes** *"Subtle Negative Transfer"* ("We hypothesize", Ch.3 L358–360). Ch.4's recap states it
  correctly: MTLnet *"performed on par with the dedicated single-task models at a higher training cost"*
  (Ch.4 §4 mtlnet-recap). Ch.5 upgrades that hedged hypothesis into an observed fact ("sharing hurt one task").

Contradiction surface inside the dissertation (this is what makes it a BLOCKER, not a MINOR): the correct
framing is stated four times elsewhere — Ch.1 §1.2 (*"the first two studies paired next category prediction
with the static classification of a place's category"*), Ch.1 §1.2 arc (*"the task pair therefore evolved
... from static category classification plus next category in the first two to next category plus next
region in the last, and this dissertation names that evolution plainly"*), Ch.3's task definitions, Ch.4's
task definitions, and Ch.6 limitation 6 (*"Chapters 3 and 4 paired static category classification with next
category, while Chapter 5 pairs next category with next region"*). An examiner who reads Ch.3 then reaches
Ch.5 L140 has a one-line kill-shot: *"Chapter 3 has no region task — how did it observe negative transfer
on next region?"* This directly violates the signed-off addition NORTH_STAR §6(a) ("named plainly, never
narrated as one experiment on a constant pair").

Provenance / fix path: inherited verbatim from articles/[mobiwac]/src/sections/01_introduction.tex L17 and
02_related.tex L48–49. In the standalone paper, CBIC is only a citation and the slip is hard to check; in
the coletânea, CBIC is the adjacent Chapter 3. Suggested direction (author, not applied): correct the two
sentences so the negative-transfer observation is attributed to CBIC's *actual* pair (static category
classification + next category) — or to Caruana-style MTL in general — and state the region task enters in
this chapter; align the CBIC characterization with Ch.4's "on par" (or supply the measured basis if
"sharing hurt one task" is to stand). Because this departs from the version of record, log it in
articles/[mobiwac]/ERRATA.md and Appendix B per NORTH_STAR §4/§5(7).

### Finding 2 — MAJOR (lens 6; guardrails N5)
**The same joint result (Alabama, next-category, n=20) is 64.51 in Ch.5 and 64.54 in Ch.6 — the joint-best
vs diagnostic-best conventions are blurred across chapters.**

- Ch.5 Table 3 (`tab:mobiwac:results`) reports joint Alabama category = **64.51 ±0.09**. This is the
  **joint-best** value (author ruling 2026-07-18 to report joint-best; JOINT_BEST_RESULTS.md L32:
  `AL | ... | 64.54 ±0.10 | 64.51 ±0.09 | −0.04`, where 64.54 is diag-best, 64.51 is joint-best).
- Ch.6 §6.2 capacity paragraph quotes the joint as **64.54** ("56.16 macro-F1, against 56.82 for the
  dedicated model ... and 64.54 for the joint model") and again ("reaches 56.16 ... 64.54"). 64.54 is the
  **diagnostic-best** value, inherited from the capacity record (capacity_baseline_experiment.md §5.3:
  "joint v17 = 64.54 (n=20)").

Impact: the numeric verdict is unaffected (the capacity gap is +7.72 with 64.54 or +8.35 with 64.51), so
no *conclusion* moves. But lens 6 requires the joint-best/diagnostic-best distinction to *never blur*, and
here the flagship table reports one convention while the Conclusion silently uses the other for the same
cell. A number auditor comparing Ch.5 Table 3 to Ch.6 finds joint-AL = 64.51 ≠ 64.54. Suggested direction:
in Ch.6 use 64.51 to match Ch.5's reported convention (gap becomes +8.35), or state explicitly that the
capacity comparison is against the diagnostic-best joint value and why. Hand the exact numeric
reconciliation to persona 06 (number auditor).

### Finding 3 — MAJOR (lens 2; attack Q6; must-cite Elich 2311.04698)
**"A balancer therefore has no conflict to resolve" generalizes a directional (cosine) measurement to all
balancer families; magnitude-based balancers act on an axis the cosine does not measure.**

Quote (Ch.5 §5.2.4, L180–187): *"none of the balancers that we tried ... improved on a tuned fixed task
weighting in our model. The reason is visible in the gradients. ... the cosine similarity between the
next-category and next-region updates on the shared trunk averages +0.001 ... A balancer therefore has no
conflict to resolve"*.

The measurement itself is exemplary and honestly scoped (four seeds, three of six datasets, per-dataset
means within ±0.003, development-time, earlier data preparation, "a finding for this pair of tasks, not a
general rule"). The problem is the inference verb. Cosine ≈ 0 establishes there is no *directional*
conflict — which cleanly explains why the gradient-*surgery* family (PCGrad, CAGrad, Nash-MTL) found
nothing to project away. It does **not** establish "no conflict to resolve" for the *magnitude/loss-scale*
family (GradNorm, uncertainty weighting), which target gradient-norm imbalance, an axis a cosine cannot
see (Elich et al., arXiv:2311.04698: angular conflict is not the whole story; magnitude differences
dominate and Adam already partially normalizes scale). The honest reason those balancers did not help is
that **the tuned fixed 0.75/0.25 weighting already sets the task scale** — a static intervention on exactly
the axis GradNorm/UW would adjust. The empirical backstop ("none of the balancers we tried improved") makes
the *conclusion* sound regardless, so this is a tighten-and-cite fix, not a retraction. Suggested direction:
scope the sentence to directional conflict ("no *directional* conflict for a gradient-surgery method to
resolve; the fixed weighting already fixes the task scale"), and cite Elich (currently absent everywhere —
Kurin + Xin are present, which meets the skeptic-block minimum, but Elich is the specific anchor for a
gradient-conflict mechanism claim). Same sentence recurs in Ch.6 §6.2 ("essentially orthogonal gradients:
sharing stopped hurting") — that phrasing is safer and can stay.

### Finding 4 — MAJOR (attack Q9; lens 3 confound)
**Ch.2 says the pipeline uses class-weighted cross-entropy; Ch.5 says the joint model uses plain unweighted
cross-entropy and that class-weighting was tested and rejected. The dedicated arms' loss shaping is never
stated, leaving the MTL-vs-STL loss-shaping parity undocumented.**

- Ch.2 §2.4 (L409–410): *"The training pipeline counters the same imbalance with class-weighted
  cross-entropy."* §2.4 presents itself as the protocol "used throughout".
- Ch.5 §5.2 method-model (L248): *"$L_{cat}$ and $L_{reg}$ are plain unweighted cross-entropy losses ...
  Class-weighting, tested on both outputs, lowered both region accuracy and category macro-F1."*

Two problems: (i) a document-wide concordance contradiction on a method fact (class-weighted vs unweighted)
that lands in the fundamentals chapter, my primary scope; (ii) a Q9 confound-disclosure gap — Ch.5 states
the *joint* model's loss shaping (unweighted) but never states the *dedicated* single-task arms use the same
unweighted CE. If MTL and STL arms differed in loss shaping, the transfer comparison would be confounded.
It is very likely both arms are unweighted (the class-weighting rejection is disclosed, and the method
comment in the source reads "both outputs plain unweighted cross-entropy"), so this is probably a disclosure
gap rather than a real confound — but the text should say so at the dedicated arm, and Ch.2 must not assert
the opposite of what the headline model does. Suggested direction: reconcile Ch.2 §2.4 to the actual
practice (state that the joint model uses unweighted CE, class-weighting having been tested and rejected;
if class-weighting was used only in CBIC/CoUrb, scope the Ch.2 sentence to those chapters), and add one
clause to Ch.5 confirming the dedicated arms use identical unweighted CE. Hand the document-wide
reconciliation to personas 04 (concordance) and 06/09.

### Finding 5 — MINOR (attack Q1, Q8; lens 7)
**The balancer sweep budget is not stated, and the Nash-MTL solver used in the MobiWac sweep is not
identified relative to the CBIC-era solver bug.**

Ch.5 §5.2.4: *"none of the balancers that we tried, including [PCGrad, Nash-MTL], improved on a tuned fixed
task weighting"*. The fixed weight was "tuned once on validation"; the balancers' tuning budget is not
given (defaults or swept?). Asymmetry favors the null, so risk is low and the finding aligns with the field
prior — but a pro-balancer examiner (Kurin) will ask. Separately: Ch.3's preface contains the Nash-MTL
solver-bug containment (the CBIC-era NashMTL collapsing to [1,1]); Ch.5 does not say whether its Nash-MTL
run used the fixed solver. If it used the buggy one, "Nash did not beat fixed weighting" is partly
confounded — though the 0.75/0.25 fixed weight differs from a collapsed-to-[1,1] Nash, and the cosine
mechanism is solver-independent, so the conclusion survives either way. Suggested direction: one clause on
the balancer budget ("run at their published defaults" or "swept over X"), and a half-sentence confirming
the corrected Nash-MTL solver was used.

### Finding 6 — MINOR (must-cite canon; lens 3)
**Ch.2 §2.3 attributes the negative-transfer definition to a task-grouping paper and omits the canonical
anchor.**

Ch.2 §2.3: negative transfer is introduced via *"joint training can hurt as easily as it helps depending on
the pairing \cite{standley2020tasks}"*. Standley (1905.07553) supports the *pairing-dependence* claim but is
not the definitional source; the canonical negative-transfer anchor Zhang et al. (arXiv:2009.00909, "which
tasks should be learned together") is absent, as noted in the section's own ledger ("Zhang2020 REMOVED").
The informal definition given ("joint training can leave a task worse off than its single-task model") is
correct. Also absent from the skeptic block: Hu (2308.13985, theory), Royer (2310.08910), Elich
(2311.04698) — Kurin + Xin are present, meeting the essential minimum. This is a thin frame chapter, so the
guidance is: flag the gap, do not demand padding. Adding Zhang 2009.00909 (one cite) at the
negative-transfer definition and Elich where the cosine mechanism is discussed (Finding 3) would close the
two that matter.

### Finding 7 — MINOR (lens 7; NORTH_STAR §4)
**Ch.3's Nash-MTL preface caveat is vaguer than the repo record supports.**

Ch.3 preface: *"The chapter's preference for the Nash-MTL optimizer is likewise a conclusion of the time,
weakened by a later finding about the optimizer implementation"*. This satisfies the NORTH_STAR §4
containment ("the chapter preface may note the later finding"), and the body's Nash-MTL "consistently
yielded a better overall performance" claim is thereby contained. NORTH_STAR §4 only requires "may note", so
this passes — but "a later finding about the optimizer implementation" is soft; a reader cannot tell the
finding was a solver collapse to equal weights. Optional sharpening (author's call): name it as "a
solver-implementation bug that collapsed the balancer toward equal weighting", which strengthens the honesty
rather than weakening the chapter.

### Finding 8 — MINOR (lens 4/5; attack Q7)
**The freeze control (category-gain attribution) is run on 3 of 6 datasets; the category gain is claimed on
all six. The region win (4/6) has no symmetric per-direction control.**

Ch.5 §5.2.2: *"the full category gain survives at Alabama, Arizona, and Florida ... We therefore attribute
the category gain to a stronger shared trunk"* — reported "as a finding, not a hypothesis". The scope (3
named datasets) is stated in the sentence, so this is not hidden; but the attribution is generalized to the
category gain at all six, and Istanbul/TX/CA are untested by the freeze control. Separately, the freeze
control gives the *region-does-not-teach-category* direction only; the region win at 4/6 has no symmetric
control, and the text correctly does **not** claim "category teaches region" (it credits the private spatial
path + trunk semantic context), so there is no silent reverse-direction claim (lens 5 satisfied) — but the
region gain's source is less dissected than the category gain's. Suggested direction: soften "as a finding"
to name its scope ("at the three datasets where the control was run"), which Ch.6 already does correctly and
Ch.5 could mirror.

### Finding 9 — NIT
Ch.5 §5.2.2: *"A reader used to multi-task learning expects the harder task ... to teach the easier one"* is
a loose characterization of MTL expectations (the standard expectation is shared-representation
regularization, not specifically harder→easier transfer). It motivates the freeze control adequately; no
action needed beyond awareness.

## Credibility signals present (what an MTL expert will trust here)
- **Scalarization-first skepticism cited, not dodged.** Ch.2 §2.3 and Ch.5 §5.2.4 both cite Xin
  (2209.11379) and Ch.2 cites Kurin (2201.04122); the text states the fixed-weight baseline is "a serious
  competitor" and the finding *aligns* with the field null rather than claiming a balancer win.
- **Gradient-cosine measured and fully scoped** (four seeds, three datasets, ±0.003, development-time,
  "not a general rule") — not asserted "not shown".
- **Parameter-count disclosed as cost** ("4.2M at Alabama against 1.1M ... operational rather than
  arithmetic"); no compute-saving claim anywhere (F3 guard held, incl. Ch.1).
- **Freeze control** attributes the category gain to the trunk, explicitly refuting "the region task
  teaching the category one".
- **Capacity-matched dedicated baseline** (Ch.6 §6.2) closes the parameter-count alternative: a 4.2M-param
  dedicated category model reaches 56.16 vs 56.82 tuned-narrow, recovering none of the joint gain.
- **Joint-best convention named + diagnostic-best robustness bound** ("at most 0.06 / 0.11 points, so no
  claim depends on this choice").
- **Conservative asymmetry:** dedicated category models tuned per-dataset; the joint model uses one fixed
  config across all six and still wins.
- **Region verbs bound to tests; AZ never upgraded** ("interval centered on zero, so we report a match, not
  a gain").
- **Task-pair confound conceded** (Ch.6 limitation 6) with the fixed-pair ablation named as future work.
- **CBIC null time-indexed** and **Nash-MTL solver-bug contained** in the Ch.3 preface.
- **CoUrb capacity confound acknowledged** (192-d vs 64-d input; §4 embedding-integration calls for a
  dimensionality-equalized control).

## Unstated defenses (facts the repo holds but the text does not fully carry)
- **The cosine is directional-only, and the fixed weight handles the scale axis.** The capacity record and
  the field literature support this reasoning, but the chapter's mechanism sentence asserts "no conflict to
  resolve" without it (Finding 3). Stating it converts a weak spot into a defense.
- **The dedicated arms use the same unweighted CE as the joint model** (implied by the source method
  comment and the class-weighting rejection) — but Ch.5 never states the dedicated arms' loss shaping, so
  the loss-shaping parity that defeats the Q9 confound is undocumented (Finding 4).
- **The capacity sweep's fairness scope** (capacity_baseline_experiment.md §5.3: the wide arm got a
  3-recipe sweep vs the ceiling's wider grid; the 0.55 pp spread ≪ the 8.4 pp gap, so a wider grid cannot
  change the verdict) — Ch.6 carries the numbers and the "partial California, fifteen of twenty" honesty,
  but not the fairness-scope caveat. Worth one clause whenever the capacity number is quoted.

## Out-of-scope handoffs (noted, not judged by this persona)
- **Statistics/leakage** → persona 09: n=4 paired t / TOST power; development-seed contamination (Q10:
  recipe decisions made on the reported seeds?); the A4 leak-audit prose and the region-transition-prior
  per-fold rebuild.
- **Numbers** → persona 06: the exact 64.51/64.54 reconciliation (Finding 2); the Ch.6 capacity numbers
  (56.16/56.82/64.54) against al_capmatch_summary.json; CoUrb 15/21 + 1 tie and +20.2–22.0 pp.
- **Citations** → persona 05: `\cite{nash}` vs the ledger's `navon2022nashmtl` — verify the key resolves in
  the global bib; the DGI triple-key consolidation.
- **Concordance** → persona 04: the Ch.2↔Ch.5 class-weighted/unweighted contradiction (Finding 4) as a
  document-wide pass; the CBIC-task-pair reconciliation (Finding 1) across Ch.1/3/4/5/6.
- **POI/novelty** → persona 11: "first work to treat fine-grained region as an end target of equal
  standing".

## Open questions only the author can answer
1. Which joint Alabama next-category value is canonical for the frame — **64.51** (joint-best, Ch.5
   headline) or **64.54** (diagnostic-best, Ch.6 capacity comparison)? (Finding 2)
2. Do the **dedicated single-task arms use unweighted cross-entropy identical to the joint model**? Confirm
   for the Q9 parity disclosure. (Finding 4) And what is the source of Ch.2 §2.4's "class-weighted
   cross-entropy" statement — CBIC/CoUrb only, or a stale generalization?
3. Was **MobiWac's Nash-MTL run with the corrected solver** (post the CBIC-era [1,1] collapse)? (Finding 5)
4. Do you want the **CBIC negative-transfer misattribution corrected via ERRATA.md** (silent fix in the
   dissertation text + Appendix B listing), given it is inherited from the version of record? (Finding 1)

---
## Working notes (raw; superseded by the ranked findings above)

### Ch.2 §2.3 (MTL fundamentals) — working notes

**Scalarization skepticism (lens 1): PRESENT and correctly positioned.** §2.3 cites the
skeptic block: lin2022rlw (RLW competitive), xin2022domtl (optimizers often do not beat tuned
fixed weights), kurin2022scalarization (unitary scalarization matches/improves). Closes with:
"a fixed-weight baseline is a serious competitor, and a balancer earns its place only by
outperforming it." This is the field's null, stated correctly. Text engages Kurin AND Xin, not
only pro-balancer work. GOOD.

**Balancer canon coverage: essentially complete.** Present: caruana1997multitask, ruder2017,
kendall2018 (UW), chen2018 (GradNorm), sener2018 (MGDA), liu2019 (DWA), yu2020pcgrad, liu2021cagrad,
navon2022nashmtl (Nash), senushkin2023aligned, liu2023famo, lin2022rlw, kurin2022, xin2022,
vandenhende2022, standley2020, MMoE/PLE/DSelect-k/cross-stitch all present.
CAGrad "with convergence guarantees" — accurate. FAMO "constant time/space" — accurate.

**Canon GAPS (flag, do not demand padding — thin frame chapter):**
- Skeptic block INCOMPLETE: Hu (2308.13985, theory), Royer (2310.08910), Elich (2311.04698,
  gradient-conflict mechanism) NOT cited. Kurin+Xin present = essential minimum met. Elich is
  the one that matters IF Ch.5 makes a gradient-cosine mechanism claim (check Ch.5).
- Negative-transfer canonical anchor Zhang 2009.00909 NOT cited; §2.3 attributes negative
  transfer to standley2020tasks. Standley supports "joint training can hurt depending on
  pairing" (true) but is a task-grouping paper, not the definitional source. The informal
  definition given ("worse off than its single-task model") is CORRECT. MINOR.
- Task grouping TAG (2109.04617), surveys Zhang&Yang (1707.08114), Crawshaw (2009.09796) absent.
  Frame chapter; not blocking.

**Nash-MTL cite-key mismatch:** prose uses `\cite{nash}`; ledger references `navon2022nashmtl`
[ERRATA: consolidate nash double-key]. Verify `nash` resolves in the global bib (citation-auditor
scope; flag for cross-check). Description of Nash-MTL is accurate.

**MTLnet null (lens 12): time-indexed correctly.** "does not outperform the dedicated single-task
models, a result that holds for that configuration" — verb bound, "beat" removed, time-capsule
framing present. GOOD.

**Structured-sharing lineage (lens): NOT overclaimed.** Joint model "adopts the principle...
realized with cross-attention... though it realizes it with cross-attention rather than expert
gating" — no false PLE/MoE descent. GOOD.

**§2.4 evaluation:** TOST two-point margin, paired Wilcoxon bound to "outperforms", Holm,
user-disjoint StratifiedGroupKFold, majority floor + Markov floor + single-task ceiling all
present. song2010limits 93% correctly scoped as next-LOCATION bound, explicitly NOT a ceiling on
category macro-F1 / region Acc@10. Delta_m (maninis2019) as relative multi-task change. GOOD for
MTL-relevant evaluation framing.

**§2.5 relevance / hinge:** win verb explicitly bound to tests ("by paired superiority tests,
outperforms... on next category everywhere and next region at four of six datasets, and matches...
by non-inferiority testing... at the other two"). AZ/AL never upgraded. GOOD. (Number four-of-six
+ margin to be confirmed against Ch.5 board — cross-chapter, verify in Ch.5 read.)

### Ch.5 (MobiWac) — working notes (the win; highest-MTL-content chapter)

STRONG CREDIBILITY SIGNALS (lenses 1-8 largely satisfied):
- **Scalarization-first (L1):** §5.2.4 "joint training with a fixed loss weighting is standard
  practice, not itself our contribution... gradient-balancing methods... rarely improve on a
  well-tuned fixed weighting with two tasks [xin2022domtl]. We confirm this: none of the
  balancers that we tried, including [PCGrad, Nash-MTL], improved on a tuned fixed task
  weighting." Null-aligned, cites Xin. GOOD.
- **Gradient mechanism (L2):** §5.2.4 cosine +0.001 CITED WITH FULL SCOPE: "four seeds each on
  three of our six datasets, per-dataset means within ±0.003", "measured during development on
  an earlier preparation of the data", "directional conflict only", "a finding for this pair of
  tasks, not a general rule." Matches NORTH_STAR §6 N3 verbatim. Scope honestly stated. GOOD.
- **Capacity (L4):** §5.method-model parameter disclosure "4.2 million parameters at Alabama
  against 1.1 million for the two dedicated models combined (5.2 against 2.0 at California)...
  operational rather than arithmetic." No compute-saving claim (F3 guard held). GOOD.
- **Freeze control (L3/L4):** §5.results-part2 "We freeze the region pathway at the start of
  training so it can neither learn nor teach the category task, yet the full category gain
  survives at Alabama, Arizona, and Florida (within 0.3 of the joint model)... We therefore
  attribute the category gain to a stronger shared trunk, not to the region task teaching the
  category one." Gain-as-transfer explicitly refuted. GOOD.
- **Checkpoint honesty (L6):** §5.results-part2 joint-best convention named + diagnostic-best
  robustness bound "at most 0.06 (category) and 0.11 (region) points, so no claim depends on this
  choice." Selector a priori (geom_simple). Convention named at the table. GOOD.
- **STL tuned (Q2):** dedicated category "tuned per dataset over batch size and learning rate";
  joint uses ONE fixed config across all six -> joint wins DESPITE per-dataset-tuned dedicated
  category. Conservative. GOOD.
- **Per-task reporting (L8):** Table 3 per-task columns + Fig 4 signed deltas; geom-mean used
  ONLY for selection/cascade, never as headline. GOOD.
- **Region verbs bound to tests:** outperforms Ist/FL/TX/CA (90% CI > 0), matches AL/AZ (TOST
  ±2pp); AZ never upgraded ("interval centered on zero, we report a match, not a gain"). GOOD.

POTENTIAL FINDINGS (to rank):
- **[MAJOR? mechanism over-generalization]** §5.2.4 "A balancer therefore has no conflict to
  resolve" generalizes from a DIRECTIONAL (cosine) measurement to ALL balancers. Elich et al.
  (2311.04698 — in my canon, NOT cited anywhere) show angular conflict is not the whole story;
  magnitude/scale differences dominate and are what GradNorm/uncertainty-weighting address, and
  the fixed 0.75/0.25 weighting IS itself a static scale intervention. The cosine cleanly
  explains the PCGrad-family null (PCGrad acts on direction), but magnitude-based balancers
  target a different axis the cosine does not measure. The EMPIRICAL "none improved" backs the
  conclusion regardless, so this is "tighten the mechanism sentence + acknowledge the scale axis
  (and that the fixed weight handles it) + cite Elich", not "claim is wrong."
- **[CHECK / possible MAJOR confound — Q9 loss-shaping parity]** §5.method-model: joint model
  uses "plain unweighted cross-entropy" (class-weighting "lowered both region accuracy and
  category macro-F1"). But Ch.2 §2.4 states "The training pipeline counters the same imbalance
  with class-weighted cross-entropy." CONTRADICTION to resolve: do the DEDICATED single-task
  models use class-weighted CE while the joint uses unweighted? If loss-shaping differs across
  MTL vs STL arms, the transfer comparison is confounded (Q9). MUST verify against Ch.3/Ch.4 and
  the code. Flag concordance (Ch.2 vs Ch.5) either way.
- **[MINOR — balancer tuning budget, Q1]** "none of the balancers we tried improved" — budget
  for the balancers not stated (were they tuned or run at defaults?). Fixed weight was tuned
  once. Asymmetry favors the null but a pro-balancer examiner (Kurin) would ask. Since the
  finding is null-aligned, low risk; state the balancer sweep budget.
- **[MINOR — freeze control scope]** trunk attribution "reported as a finding, not a hypothesis"
  but freeze control run on 3/6 datasets (AL/AZ/FL); category gain claimed on all six. Scoped in
  the sentence (names the 3), not hidden; note generalization to Ist/TX/CA untested.
- **[MINOR — region-side mechanism asymmetry, Q7/L5]** category gain has the freeze control
  (region does not teach category); the REGION win (4/6) has no symmetric control. Text avoids
  claiming "category teaches region" (credits private spatial path + trunk semantic context),
  so no silent reverse-direction claim — but the region gain's source is less dissected than the
  category gain's. Per-direction affinity evidence is one-directional.
- **[CHECK — Nash-MTL solver bug carryover]** Ch.3 has the Nash-MTL solver-bug containment
  (collapsed to [1,1]). Ch.5's "Nash-MTL didn't improve" — did the MobiWac sweep use the fixed
  or buggy solver? If buggy, "balancers don't help" is partly confounded; but the cosine
  mechanism is solver-independent so the conclusion survives. Verify + note.
- **[MINOR framing]** "A reader used to MTL expects the harder task... to teach the easier one"
  is a loose characterization of MTL expectations (standard expectation is shared-representation
  regularization, not specifically harder->easier). Motivates the freeze control fine; light nit.

HANDOFFS (out of scope): n=4 seeds for paired t (persona 09); dev-seed contamination Q10
(persona 09); leak-audit prose (persona 09); novelty "first fine-grained region as end target"
(persona 11).

### Ch.3 (CBIC null) — working notes
- Preface: time-capsule done well. "conclusions of the time, for the configuration... with a place-level
  embedding and hard parameter sharing, MTL did not consistently improve on the dedicated single-task
  models. Ch.4/5 revise... The chapter's preference for the Nash-MTL optimizer is likewise a conclusion
  of the time, weakened by a later finding about the optimizer implementation." NASH-MTL CONTAINMENT
  PRESENT (NORTH_STAR §4). GOOD.
- CBIC TASKS (confirmed, line 34-35): (1) POI Category Classification [STATIC], (2) Next-POI Prediction =
  "Predicting the CATEGORY of the next POI". => CBIC pair = {static category classification, next-category}.
  NO next-region task anywhere in Ch.3.
- Nash-MTL body claim (§3.4, "consistently yielded a better overall performance... lower combined
  multi-task loss"): reproduced published claim; contained by preface. Preface caveat is vague (does not
  say the finding was solver collapse to [1,1]) but NORTH_STAR §4 only requires "may note" -> acceptable.
  MINOR: could be more specific.
- Errata (b) wall-time fixed ("about 2.3 times", was "almost four times"); errata (c) MFLOPs fixed
  ("table does not show a higher cost"). GOOD.
- INTERNAL TENSION (reproduced): "Rationale for Hard Parameter Sharing / Computational Efficiency /
  edge devices / Jetson" oversells efficiency that the chapter's own convergence result (MTL 2.3x wall
  time) refutes; chapter self-corrects in §Convergence. MINOR (in original). Ch.1 must NOT inherit
  "MTL is efficient" (F3 guard) — check Ch.1.
- Negative transfer: CBIC data = "largely comparable, within std, no consistent advantage" +
  HYPOTHESIZES "Subtle Negative Transfer" (hedged, "We hypothesize"). Honest. It does NOT cleanly
  "observe negative transfer (sharing hurt one task)".

### Ch.4 (CoUrb) — working notes
- CoUrb TASKS (confirmed, lines 24-25): (1) POI Category Classification [static], (2) Next-POI Prediction
  = next-category. NO region task. => neither CBIC nor CoUrb studied next-region.
- Preface: Item-6 floor PRESENT ("isolates the representation effect with MTLNet as its only baseline;
  does not revisit the MTL-vs-single-task question, which Ch.5 reopens"). Sample-stratified split
  disclosed as weaker. GOOD.
- CoUrb is a within-MTL representation ablation (MTLNet vs ST-MTLNet, all MTL). No MTL-vs-STL claim.
  Preface claim accurate.
- §4 mtlnet-recap: recaps CBIC as "performed ON PAR with the dedicated single-task models at a higher
  training cost." <-- CAREFUL, CORRECT recap of the null. Contrast with Ch.5's "negative transfer
  (sharing hurt one task)". DISSERTATION CONTAINS BOTH characterizations of the SAME CBIC result =>
  internal inconsistency (feeds the BLOCKER below).
- CAPACITY CONFOUND DISCLOSURE (§4 embedding-integration): "difference in input dimensionality may
  influence part of the observed gains... an additional experimental control equalizing the
  dimensionality... would allow validating... not only from the increase in input dimensionality."
  EXCELLENT capacity-confound acknowledgment (192-d vs 64-d). Credibility signal. Lens 4.
- Nash-MTL used (inherited), no superiority claim -> solver-bug caveat absent but LOW risk (Nash is a
  constant across baseline+variants; does not threaten the representation conclusion). MINOR/context.
- Win-count numbers are the AUDITED set (15/21 + 1 tie; +20.2..22.0 pp). Errata applied. GOOD.

### *** CROSS-CHAPTER BLOCKER (candidate top finding): Ch.5 misattributes next-region to CBIC ***
Ch.5 twice claims prior work (silva2025mtlnet = CBIC) studied next-category + next-region and observed
negative transfer. CBIC (Ch.3) and CoUrb (Ch.4) BOTH study {static category classification, next-category};
NEITHER has a region task. MobiWac (Ch.5) is the FIRST to add next-region.
- Ch.5 L44 (§5.1 intro): "Prior work observed exactly this [compromise, helping one while hurting the
  other] for next-category and next-region~\cite{silva2025mtlnet}". FALSE: CBIC had no next-region task.
- Ch.5 L140 (§5.2.3): "Our earlier work~\cite{silva2025mtlnet} established this two-task setup [next-cat +
  next-region] and observed negative transfer (sharing hurt one task)". FALSE: CBIC did not establish the
  next-region setup, and reported parity (not a clean "sharing hurt").
- Ch.5 §5.2.1 recap "the first joint model for this task pair" reinforces the conflation (softer).
VIOLATIONS: (1) NORTH_STAR §6 signed-off addition (a): "the pair evolved... named plainly, NEVER narrated
as one experiment on a constant pair." (2) Factual: CBIC has no region task -> contradicts Ch.3's own task
definitions (concordance failure the dissertation makes MORE visible by placing Ch.3 before Ch.5).
(3) Amplifies CBIC's hedged null ("on par", per Ch.4) into factual "negative transfer (sharing hurt one
task)" -> also contradicts Ch.4's careful recap. Examiner kill-shot: "Ch.3 has no region task; how did it
observe negative transfer on next-region?" SEVERITY: BLOCKER.
Suggested direction (author, not applied): in Ch.5 L44/L140, attribute negative transfer to CBIC's ACTUAL
pair (static category classification + next-category) OR to caruana-style MTL generally, and state the
task pair evolved to next-category+next-region in this chapter. Align the CBIC characterization with Ch.4's
"on par" (or add the measured basis if "sharing hurt one task" is to stay).
