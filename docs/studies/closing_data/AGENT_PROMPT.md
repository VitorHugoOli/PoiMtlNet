# closing-data — study briefing (DRAFT scaffold, 2026-06-12 — NOT YET LAUNCHED)

> **Status: SCAFFOLD (re-scoped 2026-06-12 per user).** Created at the close of `mtl_improvement`.
> This study is the **experimental engine for the NEW paper**: it regenerates the full results base
> — STL baselines re-run, MTL champion, and every relevant experiment from the BRACIS paper — at
> **ALL states × 4 seeds {0,1,7,100} × 5 folds**, once, under the final frozen recipe. The new
> paper's *story* is defined in a follow-up effort and may not use every cell — closing-data is
> story-agnostic and errs on the side of completeness (cells can be dropped later; they can't be
> cheaply un-run later). Do not start execution until the user launches it. **Entry point for a
> returning agent: `HANDOFF.md`** (state of play, arrival checklist, traps) — then this file, then
> `PLAN.md`, then append to `log.md`.

## Mission

1. **Re-evaluate every study** in `docs/studies/log.md` — confirm each closure is sound and harvest
   anything promotable that was parked/orphaned (P1a).
2. **Inventory the BRACIS experimental suite** (`TABLES_FIGURES.md` T1–T5 + `RESULTS_TABLE.md`
   §0.1–0.6 + the baselines strategy) into **`RUN_MATRIX.md`** — per cell: RE-RUN / REUSE /
   STORY-DEPENDENT, with exact run specs (P1b). This is the ledger Phase 3 executes.
3. **Run the pre-freeze gates** — the cheap tests that could still change the recipe (currently
   one: the aligned-pairing test inherited from `mtl_improvement`) (P0).
4. **FREEZE the final recipe + protocol** — canon version (v16 = champion G today; v17 only if a
   gate promotes), **substrate = v14 or its blessed successor at launch (user decision 2026-06-12;
   re-check `CANONICAL_VERSIONS.md`)**, seeds {0,1,7,100} × 5 folds (n=20) for every cell,
   matched-metric fp32-parity scoring, selector (P2 — hard barrier). External baselines: RE-RUN
   under this new regime (no reuse of BRACIS-era lighter numbers). Execution splits across three
   machines (H100 6h / A40 unmetered / M4 Pro local) — see `PLAN.md §Machine allocation`.
5. **Regenerate the full experimental base ONCE** (P3): CA/TX substrate builds + seeded log_T for
   all reporting seeds; **STL baselines re-run** (per-task ceilings, composite, external baseline
   engines per the matrix); **champion G at all states**; the remaining BRACIS-suite cells; one
   full-board matched-metric re-score with per-cell provenance.
6. **Hand off the results base** to the new-paper story effort + sync the canonical docs (P4).

## Why this study exists

CA/TX and the full-board re-runs were deliberately NOT done during the improvement studies — every
recipe change would have forced re-running the most expensive builds and the whole comparison board
(CA 8.5k / TX 6.5k regions, multi-day large-state compute; ~n=20 per cell across 6 states). This
study is the single, final spend that the NEW paper draws from. Corollary: **no improvement work
happens here.** If a gate or the re-evaluation surfaces a promotable lever, it becomes a scoped
go/no-go decision for the user — not ad-hoc tuning inside this study.

## Inherited state (where this study starts from)

- **Champion/recipe**: G = canon **v16**, the `scripts/train.py --task mtl` default
  (`docs/studies/mtl_improvement/FINAL_SYNTHESIS.md` §2; exact command `CHAMPION.md §3`).
- **The bar**: matched-metric G−ceiling scoring (FULL `top10_acc`, fold-paired, seeds {0,1,7,100})
  per `r0_matched_rescore.py` — every number this study reports uses it.
- **Inherited items** (full specs in `PLAN.md`): the X1 aligned-pairing pre-freeze gate; the CA/TX
  majors (INDEX `#T6-1` card); the scale checks to fold into the majors (HSM at 8.5k/6.5k;
  `next_conv_attn` FL-only cat lever); the cross-study harvest candidates.

## Hard rules (lessons from C25–C28 — non-negotiable)

1. **Pin `--canon` in every script**; explicit flags override the bundle.
2. **Matched metric, matched seeds, matched folds, matched eval precision** for every comparison.
3. **Seed-tagged per-fold log_T freshness preflight** before any run that uses
   `--per-fold-transition-dir` (stale log_T silently inflates reg — CLAUDE.md warning).
4. **PID-suffixed rundir capture + per-run seed echo in manifests**; never `ls -dt | head` under
   concurrency (C28). Re-verify any multi-seed cell whose manifest rows share a timestamp.
5. **Assert the mechanism fired** before trusting any null (C28: dead codepaths produce confident
   nulls — e.g. check aux non-None, α/β trajectories).
6. **Development seed 42 develops; {0,1,7,100} report.** Paper-grade = multi-seed + a falsification
   attempt.
7. Commit with explicit pathspec (the repo pre-stages unrelated `articles/*`).

## Read first

1. This file → `PLAN.md` (phases + cards) → `log.md` (chronology, append-only).
2. `docs/studies/mtl_improvement/FINAL_SYNTHESIS.md` — the predecessor's outcome + corrections
   registry (cite the RIGHT claims).
3. `docs/results/CANONICAL_VERSIONS.md` — version pins; `docs/studies/log.md` — the cross-study
   outcomes registry (Phase 1 walks every row).
