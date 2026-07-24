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
