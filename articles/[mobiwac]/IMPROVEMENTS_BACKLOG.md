# MobiWac 2026: Post-draft improvements backlog (ranked)

> **What this is.** The paper is a complete, compiling 9-page draft with real authors and a sourced leak rebuttal.
> This file ranks every remaining strengthening we can still do, by **impact × feasibility**, so we pick the next
> move deliberately. Each item says what it buys, what it costs, how to run it, and whether it is a submission
> blocker (none are — the draft is submittable today; these raise the ceiling). Companion runbooks:
> [`LEAK_AUDIT_EXTEND_HANDOFF.md`](LEAK_AUDIT_EXTEND_HANDOFF.md), [`STAN_REFOOTING_HANDOFF.md`](STAN_REFOOTING_HANDOFF.md).
>
> Reviewer context (the 3-agent panel): the #1 recurring attack is **single-seed n=5** (so P1 below is the single
> highest-leverage item); the leak rebuttal is now sourced (P2 only widens it); clarity is in good shape after the
> applied fixes.

---

## TIER A — data runs that move the verdict (heaviest, highest impact)

### P1. Multi-seed top-up to n=20 (MTL + STL), seeds {1,7,100} — THE top lever
**Buys:** kills the single biggest reviewer objection. At n=5 each cell sits at the Wilcoxon floor (p=0.031) and
per-cell Holm cannot clear 0.05; at **n=20** (4 seeds × 5 folds) the real effects reach sub-1e-4, per-cell Holm
clears, and "provisional" becomes paper-grade. It also retires the pooled-fold pseudoreplication workaround (we can
report proper per-state significance) and removes the seed-0 development-bias caveat.
**Cost:** the big one. Champion-G **MTL** + the **STL** category and region **ceilings**, at all 6 states ×
seeds {1,7,100} × 5 folds, fp32 (Ampere bf16 grad-NaN), matched scorer. This is the `closing_data` board protocol;
reuse the H100/A40 drivers. Per-fold log_T must be rebuilt **per seed** (`compute_region_transition.py --per-fold
--seed {S}`) or region leaks across the split. Istanbul is already n=20.
**Acceptance:** all §6.2 cells re-reported at n=20; per-state paired Wilcoxon (cat) + TOST (reg) with Holm across
the 6-cell family; update §5.3/§6.2 prose to drop "provisional/seed-0/no-per-cell-Holm" and the pooled-fold
fallback. Recompile.
**Scope discipline:** if compute is tight, prioritize the **large states (FL/CA/TX)** first (where seed-0 bias is
worst, +3..+8 pp) and the **region beats** (the FL +0.57 cell most needs multi-seed confirmation).

### P2. Extend the transductive-leak audit beyond AL+FL — see `LEAK_AUDIT_EXTEND_HANDOFF.md`
**Buys:** answers the reviewers' "leak audit is only 2 of 6 states". AZ runs as-is (~3h/fold, CPU); CA/TX/Istanbul
need a shapefile entry added first. The gate is already ON NULL at AL+FL, so this is coverage, not correctness.
**Cost:** AZ ~1 day CPU; CA/TX/Istanbul a small code add + heavier rebuild. **Non-blocking.**
**Acceptance:** add the AZ (then a large-state) row to `A4_RESULTS.md` and extend the §5.2 state list.

### P3. Faithful STAN at Florida — see `STAN_REFOOTING_HANDOFF.md`
**Buys:** fills the one empty STAN cell in Table 3 (currently "FL not yet available"); HMT-GRN already covers FL, so
this is completeness, not a gap in the claim.
**Cost:** one converged faithful-STAN run at FL (the slow-backward fix + audit fixes are already in). CA/TX stay
footnoted infeasible-at-scale.
**Acceptance:** FL faithful-STAN cell filled in Table 3; confirm it sits below our joint reg (77.28).

---

## TIER B — cheap data adds that harden the tables (low compute, real payoff)

### P4. Bridging metrics (region Acc@1/@5/MRR; category Acc@1) — already computed in most runs
**Buys:** external anchors. 7-class macro-F1 and tract Acc@10 have no published reference scale; adding Acc@1/@5/MRR
(region) and Acc@1 (category) lets a reader calibrate "is 65.66 Acc@10 good?" and connects to the next-POI
literature (protocol review §4.3). Several reviewers wanted the metric made interpretable.
**Cost:** mostly re-tabulation from existing JSONs (these k-values are usually already in the result files); a small
supplementary table or extra columns. Little or no new training.
**Acceptance:** add a metrics-ladder column/footnote (or a supplementary table) for the headline cells; state the
random-top-10 floor (<1%) and the category majority floor inline.

### P5. ReHDM at CA/TX/Istanbul (post-deadline update)
**Buys:** completes the ReHDM reference row (currently AL/AZ/FL, CA/TX footnoted infeasible, Istanbul deferred).
**Cost:** heavy (faithful ReHDM is ~75–120 h/state at scale); Istanbul needs the FSQ→mahalle adapter. Low priority;
HMT-GRN + faithful STAN already carry the region-external story.

### P6. HGI-Istanbul build → add the Istanbul row to Table 2
**Buys:** Table 2 (substrate contrast) currently omits Istanbul (no place-level HGI build). A build lets Istanbul
join the representation comparison, making Part 1 six-state like Part 2.
**Cost:** one HGI build at Istanbul (CPU) + one HGI-overlap cat-STL cell. Moderate. Optional (the footnote is honest).

---

## TIER C — text / presentation (no new runs; do any time)

### P7. Reviewer text fixes still open (the ones not yet applied)
- **"ceiling" terminology** (clarity reviewer): a literal reader reads "beats the ceiling" as a contradiction. Either
  add one justifying clause at first use ("we call it a ceiling because joint training is expected to land below it")
  or replace with "dedicated single-task model" throughout. (Decision pending; framing-adjacent.)
- **Abstract jargon density**: the first region sentence stacks "non-inferior / two-point margin / TOST". The
  glossary requires the TOST phrase in formal claims, so keep it once, but consider simplifying the abstract's
  instance to "matches (statistically, within two points)" and defining TOST in §5.3. (User call — glossary tension.)
- **TOST phrase repetition** (~10×): define once, then use "matched (TOST, ±2 pp)" in later instances. Pure trim.
- **§6.2 metric calibration**: one clause each ("Acc@10 over thousands of regions, random top-10 < 1%"; "macro-F1
  over 7 classes, majority floor ~X"). Partially done in §5.3; mirror into the first §6 result.

### P8. Bibliography / submission hygiene
- **Restore `IEEEtran.bst`** for the final build (we fell back to `ieeetr` because IEEEtran.bst is absent locally;
  it is present on Overleaf / full TeX Live). Restore `\bibliographystyle{IEEEtran}` at submission.
- `references.bib`: `lin2021ctle` has both `volume` and `number` (cosmetic bibtex warning); drop one.
- De-anonymize check (single-blind, names already in) + a final em-dash / codename sweep over all rendered text.

### P9. Page budget (only if you leave the 10-page fee variant)
Currently 9 pages (fine for the fee variant you chose). If you later want 8: `\small` the bibliography (page 9 is
only the last ~7 refs), trim the densest §2/§6 passages, or shrink one full-width figure. ~1 page is reclaimable
without cutting content.

---

## Priority order (one line)
**P1 (n=20 multi-seed)** ≫ P2 (leak AZ) > P3 (STAN FL) > P4 (bridging metrics) > P7/P8 (text + bib hygiene) >
P5/P6 (ReHDM CA/TX, HGI-Istanbul). Everything below P1 is polish; **P1 is the one item that changes a reviewer's
verdict.** None are submission blockers — the draft is submittable today; the deadline check is the only true gate.
