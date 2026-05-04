# Task-pair pivot memo: `{next_poi, next_region}` → `{next_category, next_region}`

**Date:** 2026-04-28
**Status:** Draft — to be merged into either `PAPER_STRUCTURE.md §1` (Methods preamble) OR a new appendix paragraph in `NORTH_STAR.md §Validation status`.
**Audit-trail purpose:** Make explicit a foundational decision that the current paper-facing docs assume silently. Without this memo, a reader of `NORTH_STAR.md`, `OBJECTIVES_STATUS_TABLE.md`, or `CLAIMS_AND_HYPOTHESES.md` cannot reconstruct *why* the original task pair (P0–P2 plan, archived) was retired.

---

## Original framing (P0–P2, archived in `archive/phases_original/`)

The original phase plan targeted the joint task pair **`{next_poi, next_region}`**:
- `next_poi` — POI-granularity prediction over the full POI vocabulary (~tens of thousands of placeids per state)
- `next_region` — coarser census-tract-level prediction (~1K classes for AL, ~4.7K for FL)

CH01 (P2 headline) was: *"MTL improves both heads over single-task on `next_poi` Acc@10 and `next_region` Acc@10."*

CH02 was the bidirectional-no-negative-transfer gate.

## Current scope (post-pivot, all post-B3 work)

The active task pair is **`{next_category, next_region}`**:
- `next_category` — 7-class flat classification (Food, Outdoors, Travel, …)
- `next_region` — same as before (~1K–4.7K census-tract classes per state)

The CLAUDE.local.md branch-scoped context records this scope explicitly: *"next-POI ranking at the POI granularity is out of scope for this study."*

## Why the pivot

Three concerns drove the retirement of `next_poi`:

1. **POI-granularity sparsity.** Per-state placeid vocabularies have a long tail (>50% of POIs receive ≤2 visits across the entire dataset). User-grouped 5-fold CV leaves many POIs with zero in-fold support; macro-F1 collapses to noise; Acc@10 against tens-of-thousands of classes is dominated by the trivial floor.

2. **Industry-aligned positioning.** The external baselines we measure against (POI-RGNN, MHA+PE, HMT-GRN, MGCL) report results on **flat category prediction** OR **region prediction**, not POI-id ranking at full vocabulary. The literature has converged on the coarser-granularity comparison; reporting `next_poi` would have no head-to-head competitor to cite.

3. **Substrate framing alignment.** Check2HGI's contribution is per-visit context (CH19 — ~72% of substrate gap on cat). That contribution shows up cleanly on a 7-class category task because per-visit semantics distinguish, e.g., "this Starbucks visit was during commute" from "this Starbucks visit was during weekend leisure" → different category transition probabilities. At POI granularity, the contribution is dominated by long-tail noise.

## What was preserved

- **`next_region`** — unchanged from the original plan; it remained the auxiliary reg-side task throughout.
- **All Phase-1 substrate validation** (CH16, CH17, CH18, CH19) and **F49 architecture attribution** (CH20, CH21) operate on the new task pair.
- **Single-task baselines** (F21c, F37) are matched to the new pair: STL `next_gru` for cat, STL `next_getnext_hard` for reg.

## What was retired (silently before this memo)

- **CH01 original** ("MTL `next_poi` Acc@10 > STL") — superseded by CH18 (substrate-specific MTL) + CH21 (interactional architecture × substrate).
- **CH02 original** (bidirectional no-negative-transfer gate on `next_poi`) — replaced by the AZ-side hard variant Wilcoxon (research/B5_AZ_WILCOXON.md) on the new pair.
- **CH04** (2× Markov-region floor on `next_poi`) — retired by C08 reframing (absolute pp improvement over Markov-1-region instead of multiplicative ratio).
- **`next_poi` infrastructure** in `src/data/inputs/` — never built (the original P0.3 deliverable).

## Citation in the paper

Suggested Methods-section sentence (place near task definitions):

> "Following pilot experiments showing that POI-granularity prediction was dominated by long-tail sparsity at the user-grouped 5-fold level (>50% of POIs with ≤2 visits across the entire dataset), we report the joint pair `(next_category, next_region)`. The auxiliary `next_region` task is unchanged from earlier framings; the primary task `next_category` (7 flat classes) replaces `next_poi` ranking. This aligns with the granularity at which prior work (POI-RGNN, MHA+PE, HMT-GRN) reports comparable results."

## Decision required from user

- Where to place this memo: PAPER_STRUCTURE.md §1 (preferred — it's a Methods-level decision) OR a separate `docs/studies/check2hgi/SCOPE_DECISIONS.md`?
- Whether to rename CH01/CH02 or formally archive them (see `N2b_ch15_rename.md` for a parallel decision on CH15).
