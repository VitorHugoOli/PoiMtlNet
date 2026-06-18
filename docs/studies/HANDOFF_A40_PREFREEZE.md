# HANDOFF — A40 pre-freeze lanes (G0.1 · overlapping-windows · canonical-v14 prep)

> Created 2026-06-17. Machine: **A40 (unmetered CUDA workhorse)**, now free after `mtl_frontier` R4–R9.
> These are the three lanes that gate the `closing_data` P2 freeze (the reading lane — B1–B5 + RUN_MATRIX —
> runs separately, no GPU). Run them in parallel where independent; honor the sequencing note in Lane 3.
>
> **Read first:** [`closing_data/FREEZE_READINESS.md`](closing_data/FREEZE_READINESS.md) (why these gate the
> freeze + the cross-cutting audit constraints), [`closing_data/PLAN.md`](closing_data/PLAN.md) (gate specs),
> [`closing_data/M0_P3_PLAN.md`](closing_data/M0_P3_PLAN.md) (substrate inventory). Honor `CLAUDE.md` (the
> canonical recipe + stale-log_T preflight) and C28 (PID-suffixed rundirs, per-run seed echo, no `ls -dt|head`).
> Branch `study/pre-freeze-a40`; do **not** commit to `main`. Each lane closes with a verdict + a gate-ledger
> row in `PRE_FREEZE_PROGRAM.md` + `closing_data/PLAN.md`, and **STOP for the user** on any recipe-changing
> or base-changing result.

## Lane 1 — G0.1 aligned-pairing (the lone mandatory recipe-changing gate) — DO FIRST

- **Question:** the MTL cross-attn trained on **randomly-paired** windows the whole improvement study; the
  roll-probe proved the published numbers pairing-safe but is **circular** against "mixing is learnable under
  aligned pairing." A positive changes the recipe → must precede the freeze.
- **Spec (from `PLAN.md §G0.1` + `docs/results/mtl_improvement/X_SERIES_FINDINGS.md §X1`):** ONE shared
  permutation for BOTH MTL train loaders — a single sampler / joint dataset (same seed on two
  `DataLoader(shuffle=True)` is **NOT** enough; see `src/data/folds.py:1054-1080`). Champion **G** recipe
  EXACTLY (the `NORTH_STAR.md` invocation), v14 substrate, geom_simple selector. Start **AL + FL seed 0** vs
  the R0 matched-metric bar.
- **Gate:** ≥0.3 pp either head → expand to {0,1,7,100} → **STOP for the user** (recipe → v17). Null → v16
  freezes; the "wins without per-sample mixing" wording is fully earned. Either way write the G0.1 ledger row.
- **Leak/precision:** per-fold per-seed train-only priors; **stale-log_T freshness preflight** before any
  `--per-fold-transition-dir` run (and note AL's `design_k` log_T is currently stale — rebuild it, Lane 3).

## Lane 2 — Overlapping-windows ADOPT/KEEP (base-change decision)

- **Question:** adopt overlapping (strided) windows or keep the non-overlap canon? Effect validated at AL
  only (`docs/future_works/overlapping_windows.md`). This is a **base change** — ADOPT forces a full base
  rebuild + a clean leak re-audit + baseline re-match.
- **Spec:** reproduce the AL overlapping-window effect at ≥1 more state (FL decides scale, as everywhere);
  **re-run the window leak-audit on the overlapping surface** (overlapping windows change the leak surface —
  user-disjoint folds must still hold, and no target may appear in another sample's context across the
  train/val boundary); estimate the full-base rebuild cost (every base + every seeded log_T regenerates).
- **Gate (base change — USER SIGN-OFF):** produce an ADOPT/KEEP **recommendation** with the leak re-audit
  result + cost; the user makes the call (default KEEP non-overlap). If ADOPT, the regime baselines must match
  the adopted stride and **Lane 3's log_T builds must wait for it**. Write the ledger row.

## Lane 3 — Canonical-v14 prep + log_T hygiene (M0a) — substrate now, log_T after Lane 2

The audit's 🔴 blocker: the board needs **ONE** canonical v14 per state, built on a **fixed machine + fixed
build seed**, with a **hash manifest** every consumer verifies against (the three studies' local rebuilds are
non-identical → absolutes are not cross-comparable; see `FREEZE_READINESS.md §2`).

- **Substrate (windowing-INDEPENDENT — start NOW):** build the genuine missing v14 substrates — **CA + TX**
  (`scripts/canonical_improvement/regen_emb_t3.py --state {S} --encoder resln` + the design_k/mae flags per
  `CANONICAL_VERSIONS.md §v14`, then `postbuild_design_substrate.sh check2hgi_design_k_resln_mae_l0_1 {st}`);
  **verify/sync GE** (champion G was validated at GE in `mtl_improvement` → its v14 likely already exists —
  sync, don't rebuild, if so). **Pin `--seed` explicitly** on every build and record it. Emit a **hash
  manifest** (sha256 of each `output/check2hgi_design_k_resln_mae_l0_1/{state}/…` artifact) so P3 and the
  three audited studies can be checked for identity. Per-state deps first: POI2Vec teacher + Delaunay POI-POI
  graph + region artifacts at CA/TX/GE. **Measure the first build's wall-time before promising the rest.**
- **log_T (windowing-DEPENDENT — WAIT for Lane 2):** rebuild the currently-**stale AL** `design_k` log_T now
  (it's needed for any AL re-run regardless), but **defer the full seeded {0,1,7,100} log_T build** at every
  state until the overlapping-windows decision lands — if overlap is adopted, every log_T regenerates.
- **Shared utility (small code task — flag if not done elsewhere):** a freshness-assert helper
  (`log_T mtime > next_region.parquet mtime`) wired into every `--per-fold-transition-dir` consumer
  (`a4_eval.py` and `p1_region_head_ablation.py` currently lack it).

## What is NOT in scope here
P3 (the full base regeneration — all states × 4 seeds × 5 folds) is **post-freeze** and not started until P2
commits. The H100's metered 6 h is reserved for the CA/TX builds **only if** the A40 can't absorb them
overnight (measure first; the A40 is unmetered, so prefer it). The reading lane (B1–B5 + RUN_MATRIX) is being
driven separately and does not need the A40.
