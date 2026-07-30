# ERRATA — MobiWac 2026 (this folder's article)

> **Article:** "Predicting the Next Category and Region of a Visit: A Check-in-Level Multi-Task
> Study on Mobility Data." MobiWac 2026, EDAS #1571313639. **Submitted / under review** — status
> wording is always "submitted, under review", never "published/accepted".
>
> **Purpose.** Living errata record for this folder. The MobiWac source is the version of record
> for the dissertation's Ch.5 and its claim discipline is inherited verbatim. As of this session no
> citation/number defects are known here (the paper's own bib was the verified donor set). Points
> below are the discipline constraints to preserve during adaptation, plus a placeholder for future
> errata.

## Constraints to preserve during adaptation (not defects — guardrails)

- **Status wording:** "submitted to MobiWac 2026, under review". Never "published" or "accepted".
- **Claim discipline (inherited):** region verbs bound to tests — "outperforms" only where a paired
  superiority test licenses it (Istanbul / FL / TX / CA), "matches" only under TOST non-inferiority
  (AL / AZ); never upgrade AZ. Scaling claim scoped to the five U.S. states. Cascade is "a tie at equal cost".
- **Never-cite lists:** the STAN v4-collapse numbers, the ReHDM v2 row, and VOID cells must not be cited.
- **Re-sync point:** `[mobiwac]/src/` is refined by the author in parallel; re-sync Ch.5 before the final gate.

## Known errata
- **Statistical-protocol labels (2026-07-25, author-approved; fixed in source).** §5.3 claimed the
  analysis plan fixed the test assignment "in advance" in a form that read per dataset, said the plan
  was "released with the code", and named only the paired $t$. Against the protocol of record
  (`docs/studies/closing_data/v17_completion/STATISTICAL_PROTOCOL.md`) three things were wrong: the plan
  fixed the assignment **per task**, not per dataset; it registered a paired **Wilcoxon**, and the
  reported $t$ was an undisclosed departure (the exact one-sided Wilcoxon $p$ floors at 0.0625 at n=4);
  and it did **not** register next-region superiority, so the four region "outperforms" claims sat in no
  correction family. The release branch also shipped no `docs/`, making "released with the code" false.
  **Every number, interval, and verdict was re-derived from the committed arrays and none changed.**
  Fixed: §5.3 now states what the plan fixed, discloses the departure with its reason, and labels the
  four region gains as secondary results outside the plan; the Holm sentence names both families; §6.2
  reports the region family's own Holm correction. The registered test was then run at its registered
  footing for the first time at all six datasets (Istanbul's per-fold ceiling recovered from the A40):
  next-category Holm m=6 all reject, worst adjusted p = 5.72e-06, 20/20 folds positive; region m=4
  adjusted 3.81e-06. §5.3 and §6.2 report it alongside the $t$. The duplicated scale sentence was cut
  from §5.3 to fund the additions; the paper stays at 8 pages. Mirrored verbatim in dissertation Ch.5,
  with Ch.2's contradicting "the Wilcoxon licenses outperforms" sentence reconciled. Audit of record:
  [`science/AUDIT_statistical_protocol.md`](science/AUDIT_statistical_protocol.md); review entry
  REV-007 in `articles/dissertacao/src_utils/dissertation_review.md`. Should accompany the next revision
  sent to the MobiWac review.
- **CBIC misattribution (2026-07-24, author-approved; fixed in source).** The introduction
  (`sections/01_introduction.tex`) and related work (`sections/02_related.tex`) described the prior
  CBIC work (`silva2025mtlnet`) as having studied *next-category and next-region* and as having
  *observed* negative transfer. Both are wrong against the CBIC record: CBIC paired **static POI
  category classification** with **next-category** prediction (there is no region task in CBIC), and
  it **hypothesized** negative transfer to explain a parity null, it did not observe it. This paper
  is the first to add the region task. Corrected in both sections to "our earlier work reported no
  consistent multi-task advantage for the paired category tasks and attributed it, in part, to this
  effect" and "our earlier work paired static category classification with next-category prediction
  and found no consistent multi-task gain; this paper introduces the next-region task ...". The
  correction is claim-neutral for this paper's own contributions (it strengthens the novelty claim,
  it does not weaken any result) and should accompany the next revision sent to the MobiWac review.
  Surfaced by the dissertation review suite (persona 10 BLOCKER / persona 14 B.1); mirrored in the
  dissertation Ch.5 and its Appendix~B.

### Corrections applied in the source during review (2026-07-27)

The dissertation's Chapter 5 reproduces this article. Where a correction is minor enough to fold into
a paper still under review, the author's standing instruction is to apply it **in both texts** rather
than declare an erratum, so the two stay identical. Two such corrections were applied here:

1. **The balancer sentence (`src/sections/02_related.tex`).** It read "none of the balancers that we
   tried, including the two named above, improved on a tuned fixed task weighting", which reads as a
   two-optimizer test. The screen of record is
   `docs/results/mtl_improvement/T4_full_screen.json`: **nineteen** loss and gradient balancers at two
   datasets. Against equal weighting at Alabama (53.57 macro-F1) **two** methods are above it,
   `nash_mtl` 54.25 (+0.68) and `scale_norm` 53.76 (+0.19), and neither holds that position elsewhere
   (`nash_mtl` loses the lead at Florida; `scale_norm` gains on category there while its region score
   collapses, 35.47 against 62.53). The sentence now states the count and names both exceptions with
   what each gives up. No result changes; the claim becomes narrower and checkable.
2. **A third limitation (`src/sections/07_discussion.tex`).** The paragraph named two limits and did
   not name the consequence of the two-way split the paper already discloses: epoch selection consults
   the fold the score is read on, so absolute scores are optimistic. The added limit states that, gives
   the two reasons the joint-versus-dedicated comparison is affected far less (identical rule on the
   same folds; the dedicated arm receives the wider search), declines to claim exact cancellation, and
   discloses that the four seeds reuse one fixed fold partition. Section reference remapped to this
   paper's own `sec:setup-windows` label; wording changed from "this chapter's claims" to "the paper's
   claims".

Verified after applying: paper rebuilds at **9 pages**, 0 undefined references, 0 undefined citations,
0 overfull boxes, 0 errors.

### Corrections applied in the source during review (2026-07-28)

Four more, under the same standing instruction: applied in both texts rather than declared as errata,
so the paper and the dissertation's Chapter 5 stay identical.

1. **The balancer screen's scope (`src/sections/02_related.tex`).** The sentence reporting the
   nineteen-balancer screen stated neither the repetition count nor which two datasets, and its phrase
   "including the two named above" resolved to the two **balancers** of the preceding sentence, PCGrad
   and Nash-MTL, not to a pair of datasets, so no reader could recover the screen's pool. Measured
   2026-07-28: neither state was named anywhere earlier in that file. The screen ran at registry
   defaults, at **one seed**, on **Alabama and Florida**
   (`docs/results/mtl_improvement/T4_audit_and_verdict.md`: "the full screen at registry DEFAULTS, seed
   0, AL+FL", restated under "Caveats / scope" as "Single-seed (seed0) screen ... FL+AL"), and those two
   states are the only two top-level keys of `T4_full_screen.json`, each holding the nineteen arms the
   sentence counts. The sentence now reads "at their default configurations at a single seed on two
   datasets, Alabama and Florida, including the two methods named above". The literal "seed 0" stays out
   of the prose per `GLOSSARY.md` §3, which gives "a single seed" as the phrasing. No count, number, or
   verb changed.
2. **The gradient-cosine measurement pool (`src/sections/02_related.tex`), a parity divergence closed.**
   The paper read "four seeds each on three of our six datasets"; the dissertation had been corrected in
   `dccf45d2` and the paper had not, so the two texts disagreed two lines from the sentence in item 1.
   The paper's wording was wrong in two ways at once. The measurement pools **four** Gowalla states, the
   only four keys of the `states` object in
   `docs/results/mtl_improvement/R0_matched_metric_bar.json` (alabama, arizona, georgia, florida, each
   with seeds 0/1/7/100), and `scripts/mtl_improvement/plot_grad_cosine.py` reads its run directories
   out of that same file and names the same four. **Georgia is not one of the six datasets this study
   reports**, so "three of our six" both undercounted the pool and implied every state in it is one we
   report. Read from the JSON, not recomputed. The correction is in the study's favor: the finding
   replicates on a state the paper never reports. Rendered here as "which this study does not otherwise
   use", the same substitution class as the earlier "this chapter's claims" to "the paper's claims".
3. **The trunk attribution (`src/sections/06_results.tex` and `src/sections/07_discussion.tex`).**
   Section 6 attributed the category gain to "a stronger shared trunk" and closed hedging with "We
   report this attribution as a finding, not a hypothesis"; Section 7 opened with "the shared trunk
   carries the semantic context that lifts the next-category task". The freeze control cannot support a
   component attribution: it fixes the **region** pathway, so it eliminates the region-teaches-category
   reading but leaves the category stream's own encoder, the per-stream feed-forward blocks, and the
   added depth untouched. A development arm that tests the trunk directly disagrees, at Florida over
   five folds and fifty epochs: category macro-F1 68.36 ± 0.74 with cross-attention on against
   68.32 ± 0.67 off, a difference of −0.04 ± 0.13 that a paired Wilcoxon cannot separate from zero
   (W+ = 5, p = 0.6250), quoted from
   `docs/findings/archive/F50/F50_T1_5_CROSSATTN_ABSORPTION.md`. That arm ran on an earlier
   configuration and its own record reads the null as a compensation effect, so it is not evidence
   against the trunk either, and neither text presents it as such. **The negative result is kept as a
   finding** (the gain is not the region task teaching the category one, which the control does
   establish); only the component attribution is withdrawn. Section 7's opening sentence now states the
   result with each verb bound to its test, "outperforms" the dedicated category model at all six
   datasets and, on region, "outperforms" at four and "matches" within the two-point margin at the other
   two, and says in one clause that the locus is unsettled. Arizona is not upgraded and no number is
   added.
   **One declared divergence between the two texts, deliberate.** The dissertation's Chapter 5 states
   the disconfirming ablation with its numbers and its two limits, because it has the room; this paper
   does not. What is identical is the **strength** of the claim, which is the property the parity rule
   protects: neither text now names a component as the source. Recorded so the asymmetry is not read
   later as an oversight.
4. **"Applied identically" in the limitations paragraph (`src/sections/07_discussion.tex`).** The
   mitigation of the epoch-selection bias said "the selection rule is applied identically to both
   models on the same folds". Section 6 states otherwise: "each dedicated model at its task's best
   epoch, and the joint model at the epoch selected by its joint validation score (the geometric mean of
   the two task metrics)". The procedure is shared; the objective is not. Because the word modifies a
   **mitigation**, overstating it understates a limitation, which is the same class of defect as
   overstating a result. Now: "the selection rule is the same for both models on the same folds, an
   epoch chosen on validation and read on that fold, with each model selected on its own validation
   objective (Section~\ref{sec:results-part2})". "On the same folds" is kept, since it is what makes the
   comparison paired, and the following sentence declining to claim exact cancellation is untouched.

Verified after applying: paper rebuilds at **9 pages**, 0 undefined references, 0 undefined citations,
0 overfull boxes, 0 LaTeX errors, measured in an isolated copy of the source tree.

**What deliberately stayed an erratum in the dissertation** (Appendix B, Table on claim-scope
corrections), because neither can be folded into a paper under review: the representation-integrity
paragraph, whose added fourth ground cites the dissertation's label-history benchmark appendix, and the
freeze-control restatement, which cites the dissertation's own results table.

### Corrections applied in the source during review (2026-07-30)

One, under the same standing instruction, and it is the author's ruling on the dissertation tracker's
item 2.11 (his words: promote the caveat in both texts, so both read "four wins plus two statistically
non-inferior").

1. **The §6.2 subsection lead (`src/sections/06_results.tex`).** The lead read "One model outperforms
   or matches the dedicated models on both tasks." Two properties made it worth changing. It collapses
   the four-of-six / TOST partition that the same section states precisely eleven lines later ("the
   joint model outperforms the dedicated ceiling at Florida, Texas, California, and Istanbul, and stays
   a non-inferior match (TOST, ±2 pp) at Alabama and Arizona"), and "matches" is TOST language, so a
   reader meeting the lead first receives the region result as a win at all six. The lead now reads:
   "One model outperforms the dedicated models on next category at all six datasets, and on next region
   it outperforms them at four of the six and is statistically non-inferior within a two-point margin
   at the other two." **No number, test, or verdict changes**, and Arizona is not upgraded; the verbs
   are the ones §6.2 already licenses. Applied to the dissertation's `chapters/5_mobiwac/06_results.tex`
   in the same pass and verified textually identical afterward with a comment-stripping sweep across
   line wraps.

   **Measured before editing, because the tracker's premise was wider than the defect.** Item 2.11
   records that the frame prose states the region result without the TOST caveat at nine sites. Swept
   over all 54 live `.tex` files of the dissertation and the 16 of this paper, comments stripped and
   matching across line wraps: **15 sites** state "four of six" or an equivalent, and **14 of the 15
   already pair it with the caveat**. A second sweep, for region-claim sentences that omit the
   partition altogether, found **exactly two** defects: this lead, and the dissertation's own
   consolidated-answer sentence in Chapter 6. Both are fixed. The nine-site count in the tracker was
   an inventory of where the claim *appears*, not of where the caveat is *missing*.
