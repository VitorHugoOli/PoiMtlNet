# N5 — CH14 / CH10 / P0.2 run-or-retire decision proposals

**Date:** 2026-04-28
**Source:** Original-plan gap audit (see `review/2026-04-28_post_phase1_overview_and_gap_audit.md §4`).
**Goal:** Clean the ledger before camera-ready by either executing each item or formally retiring it with an explicit memo.

---

## CH14 — fclass-shuffle ablation (P0.6)

**Original goal:** Unconditional fclass-shuffle ablation on AL single-task `next_poi` to confirm the model isn't relying on a shortcut feature in the category label channel.

**Where it lives:** `archive/phases_original/P0_preparation.md §P0.6`, `results/P0/ch14_audit.md` (mentioned, never executed).

**Status:** Code inspection done; experiment never run.

### Run-or-retire analysis

- **Cost to run:** ~30 min on M4 Pro. Implementation: shuffle the `fclass` column in input data, re-train STL `next_gru` cat 1f×50ep on AL, compare F1 to non-shuffled baseline. If shuffle-F1 ≥ baseline-F1 → shortcut confirmed (bug); if shuffle-F1 ≪ baseline-F1 → no shortcut, claim solid.
- **Reviewer risk:** Medium-low. A reviewer asking "have you ruled out trivial shortcuts in fclass?" is plausible for a paper with an MTL contribution. Without this ablation, the rebuttal is "we didn't run it because the cat F1 head-invariance result (CH16) shows substrate-dominance, which would not happen if shuffling killed performance."
- **Task-pair caveat:** Original CH14 was on `next_poi`. The pivot to `next_category` (see `N2a_task_pivot_memo.md`) means the shuffle target should be the *category label channel in input embeddings*, not the prediction target. For Check2HGI substrate, this would mean shuffling `fclass` in the check-in data before computing embeddings, then re-training. That's a heavier ~3h pipeline run.

**Recommendation:** **RETIRE** with a CONCERNS.md note. Rationale:
- The substrate finding (CH16, head-invariant) implicitly rules out trivial fclass shortcut (a shortcut would be head-coupled).
- The per-visit context mechanism (CH19, ~72% of substrate gap from POI-pooled counterfactual) does not depend on fclass; it depends on visit-time context.
- Re-running on the new task pair is ~3h of compute that adds little.
- If a reviewer asks → rebuttal is straightforward via CH16 + CH19 logic.

**If the user prefers to run:** simplest scope is a 30-min STL `next_gru` cat 1f×50ep on AL with `fclass` shuffled in the input parquet. Compare cat F1 to baseline. Document outcome in `research/CH14_FCLASS_SHUFFLE_FINDINGS.md`.

---

## CH10 — Optimiser ablation (P5.2)

**Original goal:** Compare NashMTL vs equal_weight vs CAGrad on the joint loss. Run on AL + FL.

**Where it lives:** `archive/phases_original/P5_ablations.md §P5.2`.

**Status:** Not done in formal-ablation form. Some PCGrad-vs-static comparison exists for AL (research/ATTRIBUTION_PCGRAD_VS_STATIC.md), but no full grid. The current champion (H3-alt) uses `static_weight(category_weight=0.75)` — so an optimiser ablation comparing NashMTL/equal_weight/CAGrad against static_weight is what's missing.

### Run-or-retire analysis

- **Cost to run:** ~3-5h MPS (3 optimisers × AL 5f×50ep + FL 5f×50ep). Could go to 1-fold smoke screen for ~1h to detect headline-level differences.
- **Reviewer risk:** Medium. MTL-optimiser space is well-known (NashMTL, PCGrad, GradNorm, FAMO are all live in the literature). A paper claiming a champion config without an optimiser comparison is vulnerable to "did you try CAGrad / FAMO?"
- **Confound with H3-alt:** H3-alt ALREADY changes the optimiser via per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3`). The "loss-side weighting" choice (static vs NashMTL etc.) is somewhat orthogonal but interactive. F49 Layer 2 actually shows that loss-side ablations under cross-attn are tricky → re-running NashMTL etc. under H3-alt regime could give surprising results.
- **Existing ATTRIBUTION_PCGRAD_VS_STATIC** finding: "static_weight ≈ PCGrad on AL GETNext (Acc@10 Δ=−0.17)" — already documented PCGrad equivalence. Extending to NashMTL + CAGrad fills the gap.

**Recommendation:** **PARTIAL RUN** — narrow 1-fold AL screen on `static_weight` (current) vs NashMTL vs equal_weight under the H3-alt LR regime, ~1h MPS. Promote to 5-fold only if 1f shows a meaningful gap. Cost-effective insurance against reviewer pushback.

**If user prefers full retire:** add a CONCERNS.md §C18 entry referencing existing ATTRIBUTION_PCGRAD_VS_STATIC + framing static_weight as a "calibrated default" for cross-attn under per-head LR; cite OneCycleLR mismatch with NashMTL's quadratic solver as a stability rationale.

---

## P0.2 — Label round-trip spot check

**Original goal:** Take 20 sampled `placeid` values; verify they survive the canonical `placeid → poi_idx → category` mapping intact.

**Where it lives:** `archive/phases_original/P0_preparation.md §P0.2`.

**Status:** Not documented as run.

### Run-or-retire analysis

- **Cost to run:** ~10 min one-off Python script.
- **Reviewer risk:** Very low. This is a defensive sanity check, not a paper-level claim.
- **Implicit verification:** Every Phase-1 + F49 + B3 run that produced sensible cat F1 numbers (≥35) implicitly relied on correct `placeid → category` mapping. If labels were corrupted, no model would clear the Markov-1 baseline.

**Recommendation:** **RETIRE** with one-line CONCERNS.md note: "Label integrity is implicitly validated by all Phase-1 + B3 + F49 runs achieving above-baseline cat F1; explicit P0.2 spot check skipped."

**If user wants to run:** trivial script — sample 20 random `placeid` from raw checkins parquet, look up category via the canonical mapping, manually verify against a Google Maps lookup or the original Foursquare POI metadata.

---

## Summary

| Item | Recommendation | Cost if run | Cost if retired |
|------|---------------|-------------|-----------------|
| CH14 fclass-shuffle | **RETIRE** with CONCERNS note | ~30 min (1f) or ~3h (full) | 5 min memo |
| CH10 optimiser ablation | **PARTIAL RUN** (1f AL screen) | ~1h MPS | 5 min memo |
| P0.2 label round-trip | **RETIRE** with one-line note | ~10 min | 1 min memo |

**Total cost if accepting recommendations:** ~1h MPS (CH10 1f screen) + ~10 min documentation.

**Decision required from user:** confirm or override each. I'll execute / write the retirements once SSD access is restored.
