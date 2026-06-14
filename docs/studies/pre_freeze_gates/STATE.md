# pre_freeze_gates — STATE

**Status:** SCAFFOLDED, not launched · **Machine:** A40 · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Level / blocking
- Level 1 (pre-freeze gates). Blocks: `closing_data` P2 FREEZE. Runs after/with `mtl_frontier` on the A40.

## Gate queue
| Gate | Type | State | Verdict |
|---|---|---|---|
| A2 feature-concat control | interpretation | not started | — |
| A4 transductivity bound | disclosure | not started | — |
| Overlapping-windows adopt/keep | base change (effect already validated AL) | not started | — |

## Conventions
- A2/A4 are interpretation/disclosure gates — they change what the paper *claims*, not the frozen numbers,
  but still must resolve pre-freeze so the RUN_MATRIX carries the right caveats.
- Overlapping-windows default = KEEP non-overlap (internal consistency). ADOPT only with user sign-off on
  the full-base rebuild, and only pre-freeze. It is a base change, not an MTL lever (rising-tide rule).

## Decisions log
- 2026-06-14 — scaffolded. A2 from `baseline_gap_analysis.md` Tier-1; A4 from `evaluation_protocol_review.md §4.1`;
  overlapping-windows decision points at the validated memo `docs/future_works/overlapping_windows.md`.
