# Stride-1 (overlapping-window) leak re-audit — STATIC-ANALYSIS pass

> Created 2026-06-18 (A40 pre-freeze, branch `study/pre-freeze-a40`). Scope: the Lane-2 leak
> re-audit checklist (`HANDOFF_A40_PREFREEZE §2` / `EXECUTION_PLAN §4b`). This pass closes the
> **structural / code-level** component of the four fold paths by reading the fold creators; the
> **empirical** run-validation (build stride-1 sequences, measure Acc@10/F1 deltas) remains
> Lane-2-full and GPU-gated. Anchored to Luca et al. (ML 2023) — leak surface = a sample whose
> features overlap a sample in the other split.
>
> The overlap memo's stride-9 CLEAN verdict does **not** cover stride-1; this is the gap it left.

## The structural invariant that decides (a) and (b)

`src/data/inputs/core.py::generate_sequences` builds windows **within a single user's history**
(`places_visited` is one user's check-in list) and the `stride` only changes the *density* of
windows per user — never which user a window belongs to. The fold creators key on **`userid`**:

- `_create_check2hgi_mtl_folds` / `_create_mtl_folds` → `StratifiedGroupKFold(groups=userids)`
  (`folds.py:1021`, `:717`).
- `_create_single_task_folds` NEXT branch → `StratifiedGroupKFold(groups=userids)` (`folds.py:629`,
  the `FOLD_LEAKAGE_AUDIT` 2026-04-17 fix).

Because the grouping key is `userid` and `userid` is **stride-invariant**, *all* of a user's
windows — however much denser stride-1 makes them — land in the **same** fold. A train/val
straddle would require a single user's windows to split across folds, which `StratifiedGroupKFold`
forbids by construction. **⇒ user-disjointness holds identically at stride-1 and stride-9.**

## Verdict per path

- [x] **(a) MTL `StratifiedGroupKFold(userid)`** — **PASS by construction under stride-1.** Grouping
  key stride-invariant; denser windows stay co-located per user. No new leak surface from overlap.
- [x] **(b) STL-NEXT `StratifiedGroupKFold(userid)`** — **PASS by construction under stride-1.** Same
  argument; covers the board's STL `next_gru` next-category ceiling and the STL next-region ceiling.
- [x] **(c) STL-CATEGORY plain `StratifiedKFold`** — **NOT A STRIDE-1 LEAK SURFACE (re-scoped).** The
  handoff worry ("per-(POI,window) rows straddle the cat fold boundary") presupposes a *windowed*
  category task. The code says otherwise: the plain-`StratifiedKFold` carve-out (`folds.py:633`) is
  the **flat POI-level** category classifier — **one row per POI** (`placeid, category, emb…`),
  no window, no sequence (`FOLD_LEAKAGE_AUDIT` line 108: "categorise a POI, not predict a user's
  next action"). A POI is wholly in train or val; **stride is a no-op for it.** Moreover the
  closing_data board's category metric is **`next_category` (macro-F1) via `next_gru`**
  (`RUN_MATRIX §2a`) — a *sequence* task that routes through the user-grouped path (a)/(b), **not**
  the flat-POI carve-out. So (c) is doubly moot: the carve-out is stride-invariant **and** not a
  board cell. ⚠ **Guard:** if a *windowed* category-STL is ever added to the board, it MUST use
  `StratifiedGroupKFold(userid)` — do not let a windowed task reach `folds.py:633`.
- [ ] **(d) second-dataset E2 chronological per-user split** — **OPEN (the genuinely dangerous one).**
  A *per-user chronological* 80/10/10 cut is not user-grouped (the same user spans train+val+test by
  design), so a stride-1 window straddling a boundary shares 8/9 check-ins across the cut — a real
  leak surface that code-inspection alone cannot clear. Requires the empirical re-audit on
  `scripts/second_dataset/build_chrono_split.py` output (Mac track; `EXECUTION_PLAN §1a`). Note this
  is exactly why §6 recommends **deferring** the E2/A5 chrono bridge unless cheap.

## What remains for Lane-2-FULL (GPU-gated, not closed here)

1. **Empirical reproduction** of the overlap effect at **FL + one small/mid state, multi-seed**, KD-off,
   **MIN_SEQUENCE_LENGTH held at 5** (isolate overlap vs the AL prior; `EXECUTION_PLAN §1b/§4-axis3`).
   If weak at FL-scale → STOP for the user (a finding in itself; AL +9.8 but FL +1.3 saturation warning).
2. **(d)** the chrono-split leak re-audit above.
3. Full-base rebuild cost estimate under stride-1 (~7.5–8.4× more sequences at CA/TX).

**Bottom line:** the *structural* leak question for the main-board fold paths (a)/(b)/(c) is **closed
clean** for stride-1 — overlap does not open a within-board user-disjointness hole. The base change
still gates on the **empirical** FL-scale reproduction (does overlap actually help at scale?) and on
the **(d)** chronological-split surface, neither of which is a fold-creator code property.
