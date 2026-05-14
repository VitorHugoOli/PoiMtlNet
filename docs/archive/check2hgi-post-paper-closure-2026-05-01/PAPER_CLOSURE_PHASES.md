# Paper Closure — Phased Execution Plan

**Created:** 2026-05-01
**Branch:** `worktree-check2hgi-mtl`
**Goal:** close the check2hgi MTL paper end-to-end on a free H100 (80 GB).

This document is a temp working tracker so we don't lose context across sessions.
It records (a) the phases we agreed to run, (b) the recipes, (c) the expected
outputs, and (d) the analysis that follows.

---

## Phase 0 — F51 Tier 3 optimizer/scheduler sweep (B9 base, FL)

**Why:** reviewer-rebuttal insurance. HANDOVER.md:94 marks this "not paper-blocking;
only run if reviewer asks." We're running it first now to bank a clean negative
result (or a small positive) before paper-closure phases consume the H100 budget.

**Pre-flight:** ✅ FL seed=42 per-fold log_T already rebuilt with seeded naming.
✅ CLI flags `--weight-decay`, `--adam-eps`, `--max-grad-norm`, `--eta-min` patched
into `scripts/train.py` (4-line additions, all 152 training tests pass).

**Recipe base:** B9 champion at FL, 5f×30ep (smoke), seed=42.

**Sweep matrix (15 smokes total):**

| knob | values | notes |
|---|---|---|
| weight_decay | {0.0, 0.01, 0.1, 0.2} | B9 default 0.05 |
| max_grad_norm | {0.5, 2.0, 5.0, 0.0=disabled} | B9 default 1.0 |
| eta_min (cosine floor) | {1e-5, 1e-4} | B9 default 0.0 |
| OneCycle pct_start | {0.1, 0.4, 0.5} | requires `--scheduler onecycle` (B9 uses cosine) |
| AdamW eps | {1e-7, 1e-6} | B9 default 1e-8 |

**Promotion rule:** any smoke that beats B9 reg-best by >0.5 pp at ≥ep5 → promote
to 5f×50ep paper-grade (single extra run).

**Parallelism:** 3-way on H100 (small model, batch 2048; B9 uses ~6-10 GB/job).

**ETA:** ~50-60 min wall.

**Script:** `scripts/run_f51_tier3_sweep.sh`.

**Status:** [x] complete (2026-05-01) — **clean negative**: all 15 smokes within
±0.5 pp of B9 (fold-5 best reg = 47.65 – 48.19); no promotion to 50ep. B9 is
locally optimal in the optimizer/scheduler axis too. Best knob: `pct_start=0.5`
(48.19, +0.31 pp, below threshold). Caveat: parallel runs collided in
result-dir naming (minute-granular); fold-5-best from train logs is comparable
across smokes but not canonical 5-fold-mean. Sufficient for "locally flat".

---

**[x] Phase 1 complete** (2026-05-01). 12/12 runs landed (after one retry pass for
17 OOM/script-bug failures — diagnosed and fixed in `PAPER_CLOSURE_RESULTS_2026-05-01.md §5`).

## Phase 1 — Cross-state P3 + STL ceilings (the paper closure)

**Why:** P0 paper-blocker per audit + `PAPER_PREP_TRACKER.md §2.1`.
Without CA + TX cross-state evidence, FL's −16.16 pp architectural cost is
state-idiosyncratic — a known reviewer attack surface (`§4 Risk Register`).

**Pre-flight:** ✅ CA + TX seeded per-fold log_T built.
✅ FL seeds {0, 1, 7, 100} per-fold log_T built (for the FL STL reg multi-seed extension).

**Run matrix (12 paper-grade runs at 5f×50ep, seed=42 except where noted):**

| # | State | Arm | Recipe |
|---|---|---|---|
| 1 | CA | MTL B9 | per-head LR + cosine + α-no-WD + alt-SGD |
| 2 | CA | MTL H3-alt | per-head LR (1e-3/3e-3/1e-3) + scheduler constant |
| 3 | TX | MTL B9 | (same as #1 at TX) |
| 4 | TX | MTL H3-alt | (same as #2 at TX) |
| 5 | CA | STL cat `next_gru` | matched-head ceiling, no log_T |
| 6 | CA | STL reg `next_getnext_hard` | matched-head ceiling, per-fold log_T |
| 7 | TX | STL cat | |
| 8 | TX | STL reg | |
| 9-12 | FL | STL reg multi-seed | seeds {0, 1, 7, 100} (extends F37's n=1 leak-free baseline) |

**Parallelism:** 2-way (heavy states; CA reg head = 8501 regions, fits 80 GB).

**ETA:** ~75-90 min wall.

**Status:** [ ] not started · [ ] running · [ ] complete

---

**[x] Phase 2 complete** (2026-05-01). 16/16 runs landed (8 STL reg multi-seed + 8 MTL B9 multi-seed at AL/AZ).

## Phase 2 — Multi-seed symmetry on AL + AZ (P0 + P1)

**Why:** without this, the architectural-Δ scale curve {AL +6.48 / AZ −6.02 / FL /
CA / TX} has FL+CA+TX with multi-seed error bars but AL+AZ with a single seed →
reviewer-fatal asymmetry. CA+TX single-seed is OK because they're scale-curve
*endpoints*; AL+AZ are the *favorable-architectural-Δ anchors* of a comparison
the paper headlines.

**Pre-flight:** ✅ AL + AZ seeded per-fold log_T built for seeds {0, 1, 7, 42, 100}.

**Run matrix (16 paper-grade runs at 5f×50ep):**

| group | state × seed × arm | runs |
|---|---|---|
| **P0** STL reg multi-seed | {AL, AZ} × {0, 1, 7, 100} × `next_getnext_hard` | 8 |
| **P1** MTL B9 multi-seed   | {AL, AZ} × {0, 1, 7, 100} × B9 | 8 |

**Parallelism:** 4-way on H100 (AL ~1109 regions, AZ ~1547 regions — small).

**ETA:** ~45-60 min wall.

**Status:** [ ] not started · [ ] running · [ ] complete

---

## Phase 3 — Zero-compute analysis (after Phases 1-2 land)

1. ✅ **`results/RESULTS_TABLE.md`** — paper-closure header added 2026-05-01 with
   leak-free Δ table; full v6 body rewrite deferred until paper-side decisions land.
2. ⏸ **`scripts/analysis/f50_delta_m.py`** re-run — deferred (CH22 Δm scoreboard
   needs leak-free re-extraction; one-shot script run, ~5 min).
3. ✅ **Paired-Wilcoxon JSON** at all 5 states, both tasks —
   `research/PAPER_CLOSURE_WILCOXON.json` via `scripts/analysis/paper_closure_wilcoxon.py`.
4. ✅ **`NORTH_STAR.md`** — paper closure banner added with scale-curve Δ table.
5. ⏸ **Regen** `research/figs/f63_alpha_trajectory.png` — deferred (no new α-logs).
6. ✅ **`HANDOVER.md`, `OBJECTIVES_STATUS_TABLE.md`, `PAPER_PREP_TRACKER.md`,
   `FOLLOWUPS_TRACKER.md`** — all annotated with the leak-free reframe banner.

**Status:** [x] complete (Phase 3 zero-compute items 1, 3, 4, 6 done 2026-05-01;
items 2 and 5 deferred but not paper-blocking — CH22 Δm rerun is ~5 min when needed).

---

## Cancellation log

- **2026-05-01 00:10 UTC:** Phase 1 + 2 launched together via
  `scripts/run_paper_closure_h100.sh`. User cancelled after ~2 min to insert
  Phase 0 (Tier 3) first. All child procs SIGKILL'd; GPU clean. Existing log_T
  pre-flight survives — both sets are still good.

---

## Quick recipe cheatsheet

**B9 (champion):** `--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 --scheduler cosine
--max-lr 3e-3 --alternating-optimizer-step --alpha-no-weight-decay
--mtl-loss static_weight --category-weight 0.75 --gradient-accumulation-steps 1
--per-fold-transition-dir <state-output-dir>`

**H3-alt (anchor):** B9 minus `--alternating-optimizer-step`, `--alpha-no-weight-decay`,
`--scheduler cosine`; replace with `--scheduler constant`.

**STL cat ceiling:** `scripts/train.py --task next --model next_gru --max-lr 3e-3`.

**STL reg ceiling:** `scripts/p1_region_head_ablation.py --heads next_getnext_hard
--input-type region --region-emb-source check2hgi --override-hparams d_model=256
num_heads=8 transition_path=... --per-fold-transition-dir ... --max-lr 1e-3`.
