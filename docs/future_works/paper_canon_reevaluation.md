# Future Work — Paper-canon §0.1 multi-seed re-evaluation under the F1-fix selector

**Date drafted:** 2026-05-20
**Source studies:** [`docs/studies/canonical_improvement/`](../studies/canonical_improvement/) (Tier 6 CORRECTION 2026-05-19), [`docs/studies/mtl-protocol-fix/`](../studies/mtl-protocol-fix/) (sequencing decision 2026-05-20)
**Sequencing:** **deferred until after MTL-architecture revisit (`mtl_architecture_revisit.md`) lands**, because the paper-canon re-evaluation only carries weight if it includes whatever MTL arch / loss balancing the next round of work surfaces.

## What's deferred

Re-running the v11 paper-canon n=20 multi-seed numbers (`RESULTS_TABLE.md §0.1`) under the F1-fix selector for all five states (AL, AZ, FL, CA, TX) and pairing them with the existing `joint_canonical_b9` numbers in a 2-panel reg table.

The selector fix itself (one-line code change at `src/training/runners/mtl_cv.py:679`) is in scope for `mtl-protocol-fix`. The **5-state paper-canon multi-seed re-evaluation** is what's deferred.

## Why deferred

1. **MTL architecture / loss balancing work is queued next.** Any paper-canon revision should reflect whichever (substrate, MTL arch, selector) combination becomes the new shipping recipe — not the current shipping under a fixed selector that may itself shift.
2. **Re-running the multi-seed canon under a single change buys little.** A 2-row update (current vs F1-fix) without the MTL-arch revision context will require a second revision shortly. Pay the publication cost once.
3. **The `mtl-protocol-fix` study produces F1-fix numbers at single-seed=42 first**, which is enough to (a) inform paper §Discussion that the gap exists, and (b) feed into the MTL-arch work as a baseline.

## Acceptance criterion

When picked up:

1. The shipping stack has stabilised after MTL-architecture revisit.
2. The F1-fix code change has landed and been validated.
3. `scripts/canonical_improvement/analyze_t64_selectors.py` (or its successor) is applied to every v11 paper-canon run dir in `results/check2hgi/{state}/mtlnet_*/` to produce per-task disjoint + joint_geom_simple numbers.
4. `RESULTS_TABLE.md §0.1` is extended to a 3-column reg block: `joint_canonical_b9` (legacy), `joint_geom_simple` (new), `per-task disjoint` (substrate capacity).
5. Paired Wilcoxon at n=20 on the new selectors confirms statistical significance.
6. Paper-side update: revise BRACIS submission's reg numbers OR file a supplementary table; the existing canon stays internally consistent under its declared protocol.

## Cost (estimated)

Zero-retraining. ~2-4 GPU-h of analysis re-runs + a few hours of doc/paper sweep.

## Live docs the work would touch

- `docs/results/RESULTS_TABLE.md §0.1` — primary update target
- `docs/NORTH_STAR.md` — selector-limitation banner can be downgraded to "resolved"
- `docs/CONCERNS.md` C21 — closure path
- `docs/CLAIMS_AND_HYPOTHESES.md` CH23-B — promote from §Discussion-only to §Paper-headline if statistical significance holds at n=20
- `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md` — reg-number revision

## Pointers

- Source-of-truth dual-selector numbers: `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}`
- Analysis tool: `scripts/canonical_improvement/analyze_t64_selectors.py`
- Run dirs holding existing per-epoch val CSVs (zero retraining): `results/check2hgi/{alabama,arizona,florida,california,texas}/mtlnet_lr1.0e-04_bs2048_ep50_*/`
