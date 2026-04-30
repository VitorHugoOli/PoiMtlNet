# CH15 ID-reuse resolution

**Date:** 2026-04-28
**Issue:** The identifier `CH15` is used for two different scientific claims across the doc history. Reviewers cross-referencing claim IDs will encounter contradictory content.
**Status:** Draft proposal — pick one of options A/B/C and apply.

---

## The two CH15s

### CH15-original (P0.7, in `archive/phases_original/P0_preparation.md`)

> "Transductive-leakage audit: a held-out-user fold Check2HGI retrain produces single-task `next_poi` numbers within σ of the canonical Check2HGI run."

This was a P0 preflight gate intended to detect data-side leakage in the embedding training pipeline. Audit was procedurally documented but never executed (the next_poi task was retired before P0.7 ran — see `N2a_task_pivot_memo.md`).

### CH15-current (in `CLAIMS_AND_HYPOTHESES.md` and `OBJECTIVES_STATUS_TABLE.md` post-Phase-1)

> "Reframed: HGI > Check2HGI on next_region under STL STAN was a head-coupled artefact, not pure substrate. Under matched MTL reg head (`next_getnext_hard`), Check2HGI ≥ HGI everywhere (AL TOST non-inferior, AZ +2.34 pp p=0.0312)."

This is a *substrate-comparison reframing* finding from Phase-1 Leg II / C16 head-agnostic sweep. Confirmed 2026-04-27.

The two findings are about **different tasks** (`next_poi` vs `next_region`) and have **different scientific content**. They share an ID only by accident of doc-history reuse.

---

## Why this matters

1. A reviewer who reads `CLAIMS_AND_HYPOTHESES.md §CH15` sees the head-coupled finding. If they then look up `archive/phases_original/P0_preparation.md` they see something completely different. Audit trail is broken.
2. Any cross-doc grep on `CH15` returns mixed-context hits.
3. Future paper revisions or reviewer-rebuttal docs may inadvertently conflate the two.

---

## Three resolution options

### Option A — rename the current finding to **CH15b** or **CH15-revised**

- Pros: Minimal churn; existing `CLAIMS_AND_HYPOTHESES.md` references stay valid with a single search-replace.
- Cons: A "b" variant suggests a refinement of the same claim, which it isn't.
- Update sites: `CLAIMS_AND_HYPOTHESES.md`, `OBJECTIVES_STATUS_TABLE.md`, `SUBSTRATE_COMPARISON_FINDINGS.md`, `NORTH_STAR.md` substrate caveat note, `SESSION_HANDOFF_2026-04-27.md`, `CONCERNS.md §C16`. ~6 files.

### Option B — rename the current finding to a **next free ID** (e.g. **CH22**)

- Pros: Clean separation; no ambiguity; respects the original CH15 even if archived.
- Cons: Slightly bigger doc-update churn (must also update OBJECTIVES_STATUS_TABLE row mapping); the chosen CH-number must not collide with anything in flight.
- Current max CH ID: CH21 (joint claim, headline) per `CLAIMS_AND_HYPOTHESES.md`.

### Option C — keep CH15 for the current finding; add a redefinition note

- Pros: No doc-update churn; minimal user effort.
- Cons: Audit-trail problem persists; reviewer who reads archive first will be confused.
- Implementation: a single sentence at the top of `CLAIMS_AND_HYPOTHESES.md §CH15`:
  > "**Note:** This claim ID was originally allocated to a transductive-leakage audit on the `next_poi` task (see `archive/phases_original/P0_preparation.md §CH15`). That audit was never executed; the `next_poi` task was retired (see `SCOPE_DECISIONS.md`). The ID is reused here for the head-coupled substrate reframing."

---

## Recommendation

**Option A** (CH15 → CH15b for the current finding). Reasoning:
- Smallest edit footprint — single search-replace across ~6 files.
- The "b" suffix signals "second occurrence under this number" which is the actual situation.
- Avoids the cleanup risk of Option B (orphan archived references that still say CH15).
- Doesn't leave audit-trail debt like Option C.

If user prefers a fresh ID for cleanliness, **Option B (CH22)** is the alternative.

## Files needing edits (Option A)

```
docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md
docs/studies/check2hgi/OBJECTIVES_STATUS_TABLE.md
docs/studies/check2hgi/NORTH_STAR.md
docs/studies/check2hgi/CONCERNS.md (§C16 reference)
docs/studies/check2hgi/SESSION_HANDOFF_2026-04-27.md (§0.1 substrate findings)
docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md
docs/studies/check2hgi/PAPER_STRUCTURE.md (if it references CH15)
```

Search pattern: `\bCH15\b` → `CH15b`. Verify each hit is the substrate finding and not a stray archive cross-reference.

## Decision required from user

Pick A / B / C — I'll apply once SSD permission is restored.
