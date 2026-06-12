# closing-data — study briefing (DRAFT scaffold, 2026-06-12 — NOT YET LAUNCHED)

> **Status: SCAFFOLD.** Created at the close of `mtl_improvement` per the user's plan: *after all
> improvement studies are evaluated and closed, open one final study that re-checks everything and
> then spends the heavy compute ONCE against the final frozen recipe.* Do not start execution until
> the user launches it. Read this file, then `PLAN.md`, then append to `log.md`.

## Mission

1. **Re-evaluate every study** in `docs/studies/log.md` — confirm each closure is sound and harvest
   anything promotable that was parked/orphaned. Output: a "nothing left on the table" verdict or a
   short pre-freeze addendum list.
2. **Run the pre-freeze gates** — the cheap tests that could still change the recipe (currently one:
   the aligned-pairing test inherited from `mtl_improvement`).
3. **FREEZE the final recipe** as a canon version (v16 = champion G today; v17 only if a gate
   promotes) — model, heads, loss, schedule, seeds {0,1,7,100}, 5-fold protocol, matched-metric
   scoring, per-state recipe variants if any.
4. **Run the majors ONCE**: CA/TX substrate builds + champion + STL ceilings — the only remaining
   scale gap (the recorded prediction says the C25 margins are LARGEST there).
5. **Produce the final cross-state tables** for the paper/thesis and sync the canonical docs.

## Why this study exists

CA/TX were deliberately NOT run during the improvement studies — every recipe change would have
forced a re-run of the most expensive builds (CA 8.5k / TX 6.5k regions, multi-day large-state
compute). This study is the single, final spend. Corollary: **no improvement work happens here.**
If a gate or the re-evaluation surfaces a promotable lever, it becomes a scoped go/no-go decision
for the user — not ad-hoc tuning inside this study.

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
