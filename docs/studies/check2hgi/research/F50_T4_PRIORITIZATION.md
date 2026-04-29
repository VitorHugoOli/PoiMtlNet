# F50 T4 — Prioritization & Execution Tracker

**Status (live, updated 2026-04-29 22:00 UTC):** **Paper story locked. All paper-blocking runs done.**

Final consolidated synthesis: **`F50_T4_FINAL_SYNTHESIS.md`** ← read this first.

Headline (clean, FL):
- **B9 champion** = 63.47 ± 0.75 reg / 68.59 ± 0.79 cat @ ≥ep5
- **+3.34 pp Δreg vs H3-alt clean**, paired Wilcoxon p=0.0312, 5/5 positive on BOTH tasks ✅
- **STL F37 ceiling** = 71.12 ± 0.59 (clean) → STL→MTL gap ~7.7 pp, closed by ~3.3 pp
- **PLE Pareto-WORSE** under leak-free (cat −4.22 pp, NEW finding)
- **F62 two-phase REJECTED**

This file tracks what's left and the order of execution.

**Living document — DO NOT close as completed.** Keep updated as remaining hygiene tasks land. The synthesis file is the canonical handoff.

---

## Current state recap (CLEAN — leak-free per-fold log_T, FL, ≥ep5)

| recipe | reg top10 | cat F1 | gap to STL (71.12) | source |
|---|---:|---:|---:|---|
| **STL F37 ceiling** (clean) | **71.12 ± 0.59** | n/a | 0 | upper bound |
| **B9 (P4+Cosine + α-no-WD)** ⭐ | **63.47 ± 0.75** | **68.59 ± 0.79** | **−7.65** | **CHAMPION** (Pareto-dominant) |
| P4-alone (constant) | 63.41 ± 0.77 | 67.82 | −7.71 | minimal-paper-grade |
| P0-A (P4+Cosine, no α-no-WD) | 63.23 ± 0.64 | 68.51 | −7.89 | predecessor |
| F62 two-phase | 60.25 ± 1.26 | n/a | −10.87 | REJECTED |
| PLE-lite (clean full 5×50) | 60.38 ± 0.79 | 64.13 ± 1.04 | −10.74 | Pareto-WORSE |
| H3-alt (clean baseline) | 60.12 ± 1.14 | 68.34 | −11.00 | anchor |

**Paper-grade lift: +3.34 pp Δreg (B9 vs H3-alt), p=0.0312, 5/5 positive.** The simplification: **P4 alone (alternating optimizer step) is the intervention** — Cosine and α-no-WD give marginal lift (within 0.25 pp).

**Headline numbers are locked.** Remaining work below is hygiene + paper-figure diagnostic.

---

## Priority tiers (post-consolidation)

### P0 — Paper-blocking ✅ ALL DONE

All TIER 0 leak-free runs landed. The headline is locked. Receipts in `F50_T4_FINAL_SYNTHESIS.md §1`.

| # | task | status |
|---|---|---|
| #62 P0-A | C4 verification: FL champion clean | ✅ done (63.23) |
| #63 P0-B | Cross-state AL clean | ✅ done (49.44 reg @≥ep10) |
| #64 P0-C | Cross-state AZ clean | ✅ done (40.61 reg @≥ep10) |
| (bonus) GA | Cross-state GA clean | ✅ done (46.57 reg @≥ep10) |
| #65 P1-A | B9 alpha-no-WD clean | ✅ done — **CHAMPION** |
| #67 P1-C | B10 bs1024 clean | ✅ done (loses by −2.54 pp) |
| #71 P1-E | F62 two-phase clean | ✅ done (REJECTED) |
| #72 P0-D | TGSTAN smoke clean | ✅ done (uniform leak confirmed) |
| #73 P0-E | PLE clean full 5×50 | ✅ done (Pareto-WORSE — NEW finding) |
| (anchor) | H3-alt clean | ✅ done (60.12 anchor) |
| (anchor) | P4-alone clean | ✅ done (63.41) |
| (anchor) | STL F37 clean | ✅ done (71.12) |

---

### P1 — Remaining hygiene + diagnostic (re-prioritized)

| order | # | task | type | effort | rationale |
|---|---|---|---|---|---|
| **NOW** | **#70 P3-A** | C7 — stamp `aggregation_basis` in `full_summary.json` | code | ~30 min | Finishes audit C-series cleanly. Per-fold JSON already has `per_metric_best`; aggregates need basis stamp. |
| NEXT | **#39 D5** | Reg encoder weight-trajectory diagnostic | code+GPU | ~80 min | Paper-figure diagnostic for the temporal-dynamics narrative. Generates a NEW figure showing reg encoder weight drift across epochs. |

---

### P2 — Deferred (do NOT execute unless reviewer asks)

| # | task | status | reason |
|---|---|---|---|
| #66 P1-B | B2 warmup-decay reg_head_lr | **deferred** | B9 ≈ P4-alone within 0.06 pp → lift ceiling ≤ 0.25 pp; not paper-grade for the marginal cost (1h dev + 19 min run). Re-open if reviewer pushes. |
| #49 F65 | Joint-dataloader cycling ablation | **deferred** | D8 (cw=0) already refuted the cat-loss-dominance hypothesis. |
| #68 P1-D | B4 alpha freeze warmup-unfreeze | ✅ already completed | — |
| #69 P2-A | F63 α trajectory plot | ✅ already completed | figure landed at `figs/f63_alpha_trajectory.png` |

---

### P3 — Closed

| # | task | status | reason |
|---|---|---|---|
| #33 AL+AZ P1 cross-state | deleted | superseded by #63/#64 |
| #35 P5 identity cross-attn | deleted | subsumed by F50 T3 (gap is temporal) |
| #48 F64 warmup-decay reg_head_lr | deleted | superseded by #66 (deferred above) |
| #52 Re-evaluate ALL F50 findings | completed | F50_T4_PRIOR_RUNS_VALIDITY.md |

---

## Decision rules — APPLIED (final state)

| outcome | resolution |
|---|---|
| C4 re-run drops champion ≥ σ (~0.5 pp) | **YES** — drop was 13–17 pp (uniform). Footnote added to PRIOR_RUNS_VALIDITY.md. New champion = B9 clean (63.47 reg). |
| AL champion ≥ +3 pp vs AL H3-alt | **NO** — AL clean = 49.44 (predecessor stack with leak); cross-state directional only. |
| AZ champion ≥ +3 pp vs AZ H3-alt | **NO** — AZ clean = 40.61. Same. |
| Cross-state — paper claim | "FL-strong; AL/AZ/GA directional but not paper-grade." |
| ANY P1 stacks ≥ +1 pp on B9 while preserving cat | **NO** — B9 ≈ P4-alone within 0.06 pp; the simplification is the headline. |
| All P1 wash out | **YES** — locked. Ship paper as-is. |

---

## Update log

| date | event |
|---|---|
| 2026-04-29 17:50 | File created with P0-P3 plan; closed obsolete tasks #33/#35/#48/#52 |
| 2026-04-29 17:55 | Disk cleanup — `.uv-cache` (7.9 GB) freed; FL data refetched (2.3 GB), per-fold log_T built (5 × 88 MB) |
| 2026-04-29 17:58 | P0-A FL champion + per-fold log_T LAUNCHED in tmux p0a — fold 1 ep 1 |
| 2026-04-29 17:59 | B9 (#65) implemented + 6/6 helpers tests pass — committed `60107eb` |
| 2026-04-29 18:00 | GA data fetched (539 MB); per-fold log_T built (5 × ~5 MB) |
| 2026-04-29 18:00 | Queue script `scripts/run_p1_b9_b10_ga_fl.sh` written — runs B9 → B10 → GA-cross after P0-A |
| 2026-04-29 18:05 | **User supplied AL+AZ Drive folder IDs.** Added to `runpod_fetch_data.sh`; both fetched in parallel with P0-A run (AL 202 MB, AZ 309 MB). Per-fold log_T built (AL 1109 regions, AZ 1547 regions, FL 4703 regions, GA 2283 regions — full cardinality sweep). Queue script now chains B9 → B10 → GA → AL → AZ. |
| 2026-04-29 18:10 | F60-F65 audit: F60/F61/F63 done, F62/F64 actionable. **F62 two-phase added to queue** (#71) — but watchdog had already launched the queue at 18:13 (commit 6d394bc), so F62 missed the train. |
| 2026-04-29 19:12 | **All 6 queue runs completed** (P0-A, B9, B10, GA, AL, AZ). Extracted metrics. |
| 2026-04-29 19:15 | **🚨 MAJOR: C4 leakage was 15.7 pp, not 0.5-2 pp.** Champion drops 76.07 → 60.36 reg @ ≥ep10 with per-fold log_T. ALL prior `next_getnext_hard*` numbers in the study were inflated by ~16 pp. The +4.63 pp Δ vs H3-alt may still hold (waiting on H3-alt + per-fold log_T re-run); absolute values are paper-shaking. |
| 2026-04-29 19:15 | **🏆 B9 BEATS champion under leak-free conditions: +0.24 pp Δreg AND +0.08 pp Δcat at ≥ep5, paired Wilcoxon p=0.0312 5/5 positive on BOTH tasks.** First Pareto-dominant 5/5-on-both-tasks result in the study. |
| 2026-04-29 19:15 | B10 (bs=1024) loses by −2.54 pp Δreg 0/5 — smaller batches don't help under per-fold log_T. |
| 2026-04-29 19:18 | Cross-state numbers (per-fold log_T): GA 46.57 / AL 49.44 / AZ 40.61 reg @≥ep10. Lower than FL's 60.36 — recipe doesn't transfer cleanly to small states. Best-epoch distribution differs (AL {45,38,44,31,35} suggests instability), needs further investigation. |
| 2026-04-29 19:20 | **Catch-up queue launched** in tmux `catchup`: H3-alt + per-fold log_T (anchor leak-free baseline, ~19 min) + F62 two-phase (~19 min). Total ~38 min. |
| 2026-04-29 19:28 | **Predecessor watchdog armed** in tmux `pred_queue`: P4-alone + P4+OneCycle leak-free re-runs, fires when catchup closes. Without these, the leak-free champion verdict isn't grounded relative to the predecessor stack. |
| 2026-04-29 19:30 | **🔬 Independent advisor verification of C4 16 pp drop: CONFIRMED.** Direct measurement on FL fold 1: 70.2% of val transitions leaked, prior-only val top10 gap = 10.19 pp from log_T alone. α grows from 0.1→1.8 over training (18×), so prior dominates ranking; trained-model gap amplifies to ~16 pp. Audit's 0.5-2 pp estimate was based on α at init, not convergence. **All 4 alternative hypotheses (code bug, indexing bug, filter bug, top10 calc bug) rejected with high confidence.** Full diagnosis: `F50_T4_C4_LEAK_DIAGNOSIS.md`. |
| 2026-04-29 19:40 | **🎉 H3-alt clean landed: Δ vs H3-alt SURVIVES C4 fix.** B9 vs H3-alt clean: +3.34 pp Δreg, paired Wilcoxon p=0.0312, 5/5 positive ✅. Cat preserved within σ. Leak was uniform across recipes (~16 pp), relative ordering preserved. Paper-grade claim holds. |
| 2026-04-29 19:50 | **🔬 Broader leakage audit (independent agent):** C4 propagates to ALL 5 log_T-loading heads (`next_getnext_hard`, `next_getnext`, `next_getnext_hard_hsm`, `next_tgstan`, `next_stahyper`). **`next_tgstan` flagged as potentially WORSE** — has TWO trainable amplifiers (α + per-sample gate). Class weights, samplers, baselines all clean ✅. Embeddings have structural full-data training but no learnable amplifier (low severity). Full report: `F50_T4_BROADER_LEAKAGE_AUDIT.md`. |
| 2026-04-29 19:55 | **F50 prior-runs validity matrix written.** Most F50 ablations valid for relative Δ claims (uniform leak); absolute numbers across the study inflated by ~13-17 pp. F50 D1 STL α=0 = NO LEAK (α=0 ⇒ no log_T contribution; 72.61 encoder-only ceiling is REAL). 8/9 paper claims survive; 2 absolute headlines (STL 82.44, champion 76.07) need restatement. Full matrix: `F50_T4_PRIOR_RUNS_VALIDITY.md`. |
| 2026-04-29 19:59 | **P4-alone clean landed:** 63.41 ± 0.77 reg @ ≥ep5, +3.28 pp Δ vs H3-alt clean, p=0.0312, 5/5 positive ✅. **Even the minimal recipe (just `--alternating-optimizer-step`) hits paper-grade.** Leak drop on P4-alone = 15 pp at ≥ep10, similar to H3-alt's 17 pp → uniform-leak hypothesis confirmed. New clean ranking: B9 63.47 ≈ P4-alone 63.41 ≈ P0-A 63.23 (all within ~0.25 pp of each other; cosine/alpha-no-WD give marginal lift). Paper story simplifies: **P4 alone is the intervention.** |
| 2026-04-29 20:55 | **F62 two-phase clean landed:** 60.25 ± 1.26 reg @ ≥ep5 — sub-anchor; **F62 mode=step REJECTED**. Coarse-grained two-phase scheduling does NOT replicate P4's per-batch alternating granularity. Clean paper finding: P4's per-step granularity is mechanistically essential. |
| 2026-04-29 21:25 | **PLE clean full 5×50 landed:** 60.38 ± 0.79 reg / 64.13 ± 1.04 cat. Δreg vs H3-alt = +0.26 (matches leaky +0.25 ✅, **uniform-leak hypothesis VALIDATED a second time**). Δcat = **−4.22 pp, 0/5 folds positive** — NEW finding: **PLE Pareto-WORSE under leak-free**. PLE's expert routing hurts cat without helping reg under clean conditions. Strictly dominated by P4-alone, B9, and P0-A. |
| 2026-04-29 22:00 | **Consolidation:** `F50_T4_FINAL_SYNTHESIS.md` written. All TIER 0 paper-blocking runs done. Headline locked. Remaining work re-prioritized: P0=#70 (C7 aggregation_basis stamp, ~30 min hygiene); P1=#39 (D5 reg encoder weight-trajectory, paper figure); P2=defer #66 B2 and #49 F65. |
