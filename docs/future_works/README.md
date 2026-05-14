# docs/future_works/ — Forward-looking notes

Notes that capture work-to-do-later. Not in `docs/studies/` (which holds *active* research tracks) and not in `docs/archive/` (which holds *closed* work). This is the staging ground for things deferred to future iterations.

## Convention

Each file is a self-contained memo with at minimum:
- **Date drafted**
- **What's deferred and why** (clear scope)
- **Concrete acceptance criterion** if it ever gets picked up (what does "done" look like?)
- **Pointers to live docs** the work would touch (NORTH_STAR.md, CLAIMS_AND_HYPOTHESES.md, articles/[BRACIS]_*, etc.)

When a future-work item is picked up:
- If small: do the work + delete the memo (or `git mv` to `docs/archive/future-works-done-YYYY-MM-DD/`).
- If large enough to be its own study: spawn a `docs/studies/<name>/` folder for it; remove the memo here.

## Current items

| Memo | What's deferred | Why deferred |
|---|---|---|
| [`task_pivot_memo.md`](task_pivot_memo.md) | Merge the task-pair pivot rationale (`{next_poi, next_region}` → `{next_category, next_region}`) into `docs/NORTH_STAR.md` Validation status section, OR into a `articles/[BRACIS]_*/src/sections/` paragraph. | The pivot itself is reflected in live docs (NORTH_STAR uses the new task pair); only the *historical why* is documented standalone here. Merging into NORTH_STAR is a polish item that needs careful wording for paper-facing reviewers. Deferred to next pass. |

## Sibling folders

- [`../studies/`](../studies/) — *active* research tracks (canonical_improvement, merge_design, hgi_category_injection)
- [`../archive/`](../archive/) — *closed* studies and snapshots (fusion-study, pre-promotion plans, paper-closure, etc.)
- [`../findings/`](../findings/) — paper-supporting per-experiment findings (read-only F-trail)
