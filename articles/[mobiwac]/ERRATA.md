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

**What deliberately stayed an erratum in the dissertation** (Appendix B, Table on claim-scope
corrections), because neither can be folded into a paper under review: the representation-integrity
paragraph, whose added fourth ground cites the dissertation's label-history benchmark appendix, and the
freeze-control restatement, which cites the dissertation's own results table.
