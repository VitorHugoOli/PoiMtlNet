# F50 T4 — Prioritization & Execution Tracker

**Status (live, updated 2026-04-29 19:20 UTC):** **Champion changed under C4-correction.**

Previous (leaky log_T): P4+Cosine = 76.07 reg / 68.51 cat. **DEPRECATED — inflated by ~16 pp.**

New (leak-free per-fold log_T):
- **B9 (P4+Cosine + alpha-no-WD)** = **63.47 ± 0.75 reg / 68.59 ± 0.79 cat** @ ≥ep5
- Paper-grade Pareto-dominant over P0-A (P4+Cosine without B9): +0.24 pp reg AND +0.08 pp cat, paired Wilcoxon p=0.0312 5/5 positive on BOTH tasks ✅
- Awaiting H3-alt + per-fold log_T to know whether champion-vs-H3-alt Δ survives

This file tracks what's left and the order of execution.

**Living document — DO NOT close as completed.** Keep updated as tasks land. When all P0+P1 are done, fold the synthesis into `F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md §6.5` and archive this file.

---

## Current state recap

| state | reg top10 @≥ep10 | cat F1 | gap to STL (82.44) | source |
|---|---:|---:|---:|---|
| H3-alt CUDA REF | 71.44 ± 0.76 | 68.36 ± 0.74 | −10.99 | predecessor |
| P4 alone | 75.48 ± 0.75 | 68.20 ± 0.69 | −6.96 | first paper-grade |
| **P4 + Cosine** ⭐ | **76.07 ± 0.62** | **68.51 ± 0.88** | **−6.37** | **committed champion** |
| P4 + OneCycle | 77.52 ± 0.53 | 66.52 ± 2.29 ⚠ | −4.92 | reg-only-optimal (Pareto-trade) |
| STL ceiling (F37) | 82.44 ± 0.38 | n/a | 0 | upper bound |

**The 6.37 pp residual is the next target.**

---

## Priority tiers

### P0 — Paper-critical (blocks the headline claim)

| # | task | type | effort | status | when | parallel? |
|---|---|---|---|---|---|---|
| **#62** P0-A | C4 verification: FL champion with `--per-fold-transition-dir` | GPU | 25 min | **🟡 in flight** (tmux p0a) | NOW | — |
| **#63** P0-B | Cross-state validation: P4+Cosine at AL (1109 regions) | GPU | 19 min | **🟢 unblocked** — AL data + per-fold log_T ready | after #62 (in queue) | sequential on GPU |
| **#64** P0-C | Cross-state validation: P4+Cosine at AZ (1547 regions) | GPU | 19 min | **🟢 unblocked** — AZ data + per-fold log_T ready | after #63 (in queue) | sequential on GPU |
| (bonus) | Cross-state validation: P4+Cosine at Georgia (2283 regions) | GPU | 19 min | pending | after #62 (in queue) | sequential on GPU |

**Why P0:**
- **#62 C4 verification** — every `next_getnext_hard*` number in the study (ALL P4 variants, H3-alt, A1-A6) carries 0.5-2 pp val→train leakage in the GETNext graph prior. If actual Δ > σ when we re-run with per-fold log_T, we need a footnote on every figure.
- **#63/#64 cross-state** — the champion was found at FL only. Without AL+AZ, the paper can only claim "fix works at FL". With both, it's "the recipe is portable across region cardinalities".

**Total wall-time on GPU: ~60 min sequential.** Code dev for P1-A (#65) and P1-B (#66) can happen in parallel (CPU-bound).

---

### P1 — Close the remaining 6.37 pp gap (tier-B from brainstorm, ROI-ordered)

| # | task | type | effort | expected lift | status | dependencies |
|---|---|---|---|---|---|---|
| **#65** P1-A | B9: weight_decay exempt α | code+GPU | 30 min dev + 19 min run | +1-3 pp | ✅ committed `60107eb`, run queued | independent |
| **#66** P1-B | B2 (was F64): warmup-decay LambdaLR on reg_head only | code+GPU | 1h dev + 19 min run | +3-6 pp | pending | independent |
| **#67** P1-C | B10: `--batch-size 1024` (2× α steps per epoch) | GPU CLI | 19 min | +1-3 pp | run queued | independent |
| **#68** P1-D | B4: alpha freeze warmup-then-unfreeze | code+GPU | 1h dev + 19 min run | +2-4 pp | pending | independent |
| **#71** P1-E | F62 two-phase step-schedule (orthogonal mechanism vs P4) | GPU CLI | 19 min | +0-5 pp | run queued | independent (code shipped `5550789`) |

**Strategy:** B9, B10 are quick CLI/cheap dev — do first. B2 is the highest expected lift but takes 1h dev. B4 has freeze_alpha plumbing already from F50 D1; just needs the epoch-boundary hook.

**All four are independent of each other and of P0 results** — can be developed in parallel with P0 GPU runs.

---

### P2 — Mechanism narrative (paper figures)

| # | task | type | effort | status | dependencies |
|---|---|---|---|---|---|
| **#69** P2-A | F63 α trajectory plot (the smoking-gun figure) | analysis | 30 min | pending | needs P0+P1 data to plot |
| **#39** D5 | Reg encoder weight-trajectory diagnostic | code+GPU | 80 min | pending | independent diagnostic |

**Strategy:** Hold P2-A until P0 + P1 land — that's when we have all the trajectories worth plotting. D5 is a diagnostic and can run anytime.

---

### P3 — Hygiene + remaining audit

| # | task | type | effort | status | rationale |
|---|---|---|---|---|---|
| **#70** P3-A | C7 finalization: stamp aggregation_basis | code | 30 min | pending | finishes the audit |

---

### P4 — Deferred / closed

| # | task | status | reason |
|---|---|---|---|
| #33 AL+AZ P1 cross-state | deleted | superseded by #63/#64 (champion cross-state, not P1 cross-state) |
| #35 P5 identity cross-attn | deleted | subsumed by F50 T3 (gap is temporal, not architectural mixing) |
| #48 F64 warmup-decay reg_head_lr | deleted | superseded by P1-B (#66) which stacks on champion instead of replacing |
| #52 Re-evaluate ALL F50 findings | completed | done via posthoc tool + F50_CORRECTED_SCOREBOARD.md |

---

## Parallelization plan

**The constraint:** one GPU. Code dev and analysis are CPU-bound and can run in parallel with GPU jobs.

```
Wall-clock timeline (assuming GPU starts when this file is committed):

t=0      ┌──────────────────────────────────────────────────────────┐
         │ GPU                                                       │
         │   [#62 C4 verify: FL ~25 min]                             │
         │     [#63 AL champion: ~19 min]                            │
         │       [#64 AZ champion: ~19 min]                          │
         │                                                           │
         │ CPU (parallel)                                            │
         │   [#65 P1-A B9 weight-decay-α dev: ~30 min]               │
         │   [#66 P1-B B2 warmup-decay-reg-head dev: ~1h]            │
         │   [#68 P1-D B4 alpha-freeze-warmup dev: ~1h]              │
t=60     └──────────────────────────────────────────────────────────┘

t=60     ┌──────────────────────────────────────────────────────────┐
         │ GPU (P0 done, queue P1 runs)                              │
         │   [#65 B9 run: ~19 min]                                   │
         │     [#67 B10 bs1024 run: ~19 min]                         │
         │       [#66 B2 run: ~19 min]                               │
         │         [#68 B4 run: ~19 min]                             │
         │                                                           │
         │ CPU (parallel)                                            │
         │   [#69 F63 α trajectory plot: ~30 min]                    │
         │   [#70 C7 aggregation_basis: ~30 min]                     │
t=140    └──────────────────────────────────────────────────────────┘

t=140    [synthesis: update F50_T3 §6.5 with P0+P1 numbers,
          decide if a new champion emerges, push, archive this file]
```

**Estimated total wall-time: 2h 20min** if everything goes smoothly. Realistic: 3-4h with debug/iteration.

---

## Decision rules (post-execution)

After P0+P1 lands:

| outcome | action |
|---|---|
| C4 re-run drops champion ≥ σ (~0.5 pp) | Add footnote to all P4 numbers. Re-rank if drop > Δ to predecessor. |
| AL champion ≥ +3 pp paired Wilcoxon vs AL H3-alt | Cross-state portability claim ✅ |
| AZ champion ≥ +3 pp paired Wilcoxon vs AZ H3-alt | Cross-state portability claim ✅ |
| Either AL or AZ misses +3 pp | Paper claim becomes "FL-strong, cross-state directional but not paper-grade" |
| ANY P1 stacks ≥ +1 pp on top of P4+Cosine while preserving cat | Update champion |
| All P1 wash out | Lock champion + ship paper as-is |

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
