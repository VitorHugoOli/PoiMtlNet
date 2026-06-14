# HANDOFF — closing-data (read this FIRST, then `AGENT_PROMPT.md` → `PLAN.md` → `log.md`)

> **You are here (2026-06-12): SCAFFOLDED, NOT LAUNCHED.** The plan is at **v2** with all four
> pre-launch questions resolved by the user. The one remaining launch gate is **user sign-off on
> `RUN_MATRIX.md` (the P1b inventory output) together with the P2 freeze commit**. The reading-heavy
> phases (P1a/P1b) and the P0 gate are ready to start whenever the user says go — they need no GPU
> beyond G0.1 and no story decision.

---

## 1. What this study is (one paragraph)

The **experimental engine for the NEW paper** (successor to the BRACIS submission). It regenerates
the entire results base ONCE under a frozen recipe: **STL baselines RE-RUN + MTL champion G + every
relevant BRACIS-suite experiment, at ALL states × 4 seeds {0,1,7,100} × 5 folds (n=20/cell)**. The
paper's *story* is defined by a separate follow-up effort and may not use every cell — this study is
story-agnostic and errs on completeness (cells can be dropped later; they can't be cheaply un-run).
It is **NOT an improvement study**: any promotable lever found en route becomes a user go/no-go
gate, never ad-hoc tuning.

## 2. State of play

| item | status |
|---|---|
| `PLAN.md` | **v2** — phases P0→P4; decisions block on top; machine allocation defined |
| Pre-launch questions | **RESOLVED** (user 2026-06-12): substrate = **v14 or newer blessed base at launch**; baselines **re-run under the new regime, full n=20**; studies assumed settled (P1a verifies); no timeline pressure |
| Launch gate | user sign-off on `RUN_MATRIX.md` + the P2 freeze commit |
| Executed so far | **nothing** — no runs, no matrix, no freeze |
| Predecessor | `mtl_improvement` **CLOSED 2026-06-12** — read its [`FINAL_SYNTHESIS.md`](../archive/mtl_improvement/FINAL_SYNTHESIS.md) before touching anything (esp. its **corrections-and-retractions registry**: several superseded claims float in older docs — cite only the corrected ones) |

## 3. What to do when you arrive (in order)

1. **Re-read the launch context**: this file → `AGENT_PROMPT.md` (mission + hard rules) → `PLAN.md`
   (the phases) → `log.md` (decisions trail) → `../archive/mtl_improvement/FINAL_SYNTHESIS.md` §2/§5/§8.
2. **Check for drift since scaffolding**: `docs/results/CANONICAL_VERSIONS.md` (is v14 still the
   blessed base, or did a successor land? is v16 still the champion recipe?), `docs/studies/log.md`
   (any study re-opened?), and `git log` on `scripts/train.py` + `src/configs/canon.py` (defaults
   moved?). The plan's freeze (P2) takes whatever is blessed AT LAUNCH, not what was blessed at
   scaffold time.
3. **Start P1a** (cross-study re-eval sweep — `merge_design` first) and **P1b** (the BRACIS-suite
   inventory → `RUN_MATRIX.md`). Both are reading-only and parallelizable.
4. **Run G0.1** (the aligned-pairing pre-freeze gate — the only P0 item; ~2 runs + a small code
   change; spec below).
5. **STOP at the freeze**: present `RUN_MATRIX.md` + `PHASE1_VERDICT.md` + the G0.1 result to the
   user; the P2 freeze is THEIR commit. Only then does P3 (the heavy spend) start.

## 4. The inherited specs you must not re-derive

- **Champion / recipe**: G = canon **v16** = the bare `scripts/train.py --task mtl` default
  (`--canon` selector; explicit flags override). Exact config + reproduce command:
  `../archive/mtl_improvement/CHAMPION.md §2-3`. Bare `--canon v16` runs dev seed 42 — always pass `--seed`
  for reporting runs.
- **Scoring**: the matched-metric method (FULL `top10_acc` both sides, fold-paired, fp32-parity) —
  `scripts/mtl_improvement/r0_matched_rescore.py` is the reference implementation; the R0 bar
  (`docs/results/mtl_improvement/R0_matched_metric_bar.json`) is the current 4-state baseline.
- **G0.1 aligned-pairing gate**: the MTL cross-attn trained on randomly-paired windows for the whole
  improvement study (two independent shuffled loaders, `src/data/folds.py:1054-1080`); the roll
  probe proved the numbers pairing-safe but is CIRCULAR against "mixing is learnable under aligned
  pairing." Fix = ONE shared permutation for both train loaders (same seed on two
  `DataLoader(shuffle=True)` is NOT enough). Run G at AL+FL seed0 vs the R0 bar; promote gate
  ≥0.3pp either head → multi-seed → STOP for user (recipe → v17). Full spec + circularity analysis:
  `docs/results/mtl_improvement/X_SERIES_FINDINGS.md §X1` correction banner.
- **The BRACIS suite to inventory (P1b)**: `articles/[BRACIS]_Beyond_Cross_Task/TABLES_FIGURES.md`
  (T1 dataset stats / T2 substrate ablation / T3 MTL-vs-STL / T4 Δm / T5 external baselines +
  figures) and `docs/results/RESULTS_TABLE.md §0.1–0.6`, plus `docs/PAPER_BASELINES_STRATEGY.md`.
  Disposition per cell: RE-RUN / REUSE / STORY-DEPENDENT, with exact run specs.
- **Scale checks folded into P3** (from mtl_improvement): HSM-vs-flat at CA 8.5k / TX 6.5k
  (STL-level, cheap — the FL null doesn't cover that band); `next_conv_attn` FL-only cat lever
  (only if P1a promotes it). Recorded prediction to test at CA/TX: **C25 margins largest there**.

## 5. Decisions already made — do NOT re-ask the user

1. Substrate = **v14** (or its blessed successor at launch). Single-substrate board.
2. External baselines = **RE-RUN at the full new protocol** (n=20). Never reuse BRACIS-era numbers.
3. Protocol = ALL states × seeds {0,1,7,100} × 5 folds for every cell; matched-metric scoring;
   selector `geom_simple`.
4. CA/TX belong HERE (they were deliberately never run in the improvement studies).
5. The new paper's story is **out of scope** — a separate follow-up effort picks from this study's
   board. Park STORY-DEPENDENT cells; don't argue them.

## 6. Machine allocation (user-defined — respect the metering)

- **H100 — 6 hours TOTAL, metered.** Default purchase: the CA/TX v14 substrate builds (the critical
  path). HARD RULE: do not start the H100 until its exact job list is written down AND timed on the
  A40 (extrapolate from first epochs). If one build > ~5h: CA on H100, TX on A40. If builds don't
  fit at all: flip — builds to the A40 (slow, unmetered), H100 takes the most expensive large-state
  run waves. Zero exploratory spending on this box.
- **A40 — unmetered workhorse.** The full run board (M1 baselines / M2 champion / M3 suite cells)
  at AL/AZ/GE/FL + whatever large-state work the H100 didn't take.
- **M4 Pro 32GB (local, MPS) — prep + scoring lane.** Fold freezing, seeded log_T builds, input
  generation, aggregation/re-score (M4), doc settling; small-state runs only as overflow (MPS
  caveats: no AMP, fp32, slower — `docs/infra/README.md` per-machine guidance).
- Every machine keeps its own manifest (PID-suffixed rundirs + per-run seed echo); merged at M4.

## 7. Traps that have already burned this project (C25–C28 — verify, don't trust)

1. **Objective↔metric mismatch (C25)**: any loss/weighting change must be checked against the
   reported metric before anything else.
2. **Dead-codepath nulls (C28)**: never accept a null without proof the lever FIRED (aux non-None,
   α/β trajectories, weight logs).
3. **Manifest rundir races (C28)**: never `ls -dt | head` under concurrency; re-verify any
   multi-seed cell whose manifest rows share a timestamp.
4. **Stale log_T**: per-fold seed-tagged `region_transition_log_seed{S}_fold{N}.pt` must be fresher
   than `next_region.parquet` — rebuild per seed; the preflight is mandatory (CLAUDE.md).
5. **Dev-seed contamination**: seed 42 develops, {0,1,7,100} report. Large-state seed-42 overshoots
   by up to +8pp.
6. **Circular probes**: a model trained under a degenerate regime is invariant to what an eval-time
   probe measures — ask what the probe has power against (the X1 lesson).
7. **Metric/precision parity**: matched metric, matched seeds, matched folds, matched eval
   precision (fp32) on BOTH sides of every comparison.
8. Commit with explicit pathspec (the repo pre-stages unrelated `articles/*`); pin `--canon` in
   every script.

## 8. Pointer map

| need | where |
|---|---|
| Mission + hard rules | `AGENT_PROMPT.md` (this folder) |
| Phases + machine plan + resolved decisions | `PLAN.md` (v2) |
| Decisions trail | `log.md` (this folder, append-only) |
| Predecessor outcome + corrections registry | `../archive/mtl_improvement/FINAL_SYNTHESIS.md` |
| Champion config + do-not-retry table | `../archive/mtl_improvement/CHAMPION.md` |
| Version pins / what's blessed | `docs/results/CANONICAL_VERSIONS.md` |
| Cross-study registry | `docs/studies/log.md` + `docs/studies/README.md` |
| Concerns C25–C28 | `docs/CONCERNS.md` |
| Infra per machine | `docs/infra/README.md` |
| Paper-side coordination (author-owned) | `../archive/mtl_improvement/PAPER_UPDATE.md` + `articles/[BRACIS]_Beyond_Cross_Task/` |
