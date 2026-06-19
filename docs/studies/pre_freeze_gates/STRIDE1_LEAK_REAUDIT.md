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
- [x] **(d) second-dataset E2 chronological per-user split** — **CLOSED CLEAN at stride-1**
  (independently re-verified + adversarially checked 2026-06-19 on real FL Gowalla; workflow `wf_37d016d2`).
  The worry — a per-user chronological 80/10/10 cut is not user-grouped, so a stride-1 window straddling
  a boundary shares 8/9 check-ins across the cut — is real ONLY for the *naive* window-then-split design
  (~11.5% of windows straddle). **The code does not do that:** `build_chrono_split.py:219` windows each
  80/10/10 portion **separately** (`[_sequences_within_portion(...) for s in SPLITS]`), so a window's 9
  inputs + target are always co-located in one split → **0 / 1.35M boundary-straddling windows** at
  stride-1 (empirically reproduced twice; matches the on-disk 1.378M stride-1 count). E2/A5 is also
  DEFERRED + off the BRACIS critical path (no train/eval consumer reads the chrono artifacts; in no freeze
  hash manifest) → zero champion-byte-identity / board-number risk. **Caveat for any future promotion:**
  the in-code `leak_check_1` (line 156) is only a *partial* backstop — it silently passes ~33% of a naive
  regression on large val/test portions; the construction invariant at line 219 is the actual guard. IF
  E2 is ever promoted on-board at stride-1, REQUIRE (i) an explicit stride at line 139 (today silently
  stride-9 via `stride=None`) and (ii) a tightened per-window per-check-in span assertion replacing
  `leak_check_1`. **All four stride-1 leak paths (a/b/c/d) are now closed for the freeze.**

## What remains for Lane-2-FULL (GPU-gated)

1. ✅ **DONE (2026-06-19)** — overlap effect reproduced at **FL + AL multi-seed** → ADOPT supported
   (`LANE2_OVERLAP_VALIDATION.md`; FL cat +3.64 / AL cat +8.12; not weak enough to STOP).
2. ✅ **DONE (2026-06-19)** — **(d)** chrono-split leak re-audit CLOSED CLEAN (above).
3. Full-base rebuild cost estimate under stride-1 (~7.5–8.4× more sequences at CA/TX) — **P3 (board build)**.

**Bottom line:** ALL FOUR stride-1 leak paths (a/b/c/d) are now **closed clean**. The main-board fold
paths (a)/(b)/(c) are structurally leak-free under stride-1 (StratifiedGroupKFold(userid) is
stride-invariant); path (d) — the second-dataset chronological split — was empirically re-audited
(2026-06-19) and is leak-free by construction (within-portion windowing). The base change's remaining
gate is purely the **empirical** FL-scale reproduction (Lane-2-FULL: does overlap help at scale? →
answered YES, `LANE2_OVERLAP_VALIDATION.md`). No open leak surface remains for the freeze.
