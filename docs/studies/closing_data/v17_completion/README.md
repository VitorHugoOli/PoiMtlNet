# v17_completion — the track to finish the MobiWac paper board at v17

> **New track (2026-07-01; re-allocated 2026-07-08).** The paper switched its headline to **v17** (= v16 + bs8192 +
> per-head cat-lr 1e-3; `DEFAULT_CANON`, `f54a04de`). This track holds the remaining runs + analysis to make the
> **whole board v17**. ⚠ **The H100 lane is GONE (no access, 2026-07-08)** — machine split is now:
> **A40 = ALL GPU work (incl. the former-H100 H1) · M2 Pro = no-GPU analysis.**
> Board SSOT: [`../RESULTS_BOARD.md §1`](../RESULTS_BOARD.md). Paper close-out: [`../../../articles/[mobiwac]/CLOSER_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/CLOSER_HANDOFF.md).

## Where v17 stands (what's DONE)
- **v17 MTL n=20 at AL / AZ / FL** — DONE (`../perhead_lr_n20.md`): AL 64.54 / 69.80, AZ 65.84 / 59.56, FL 79.85 / 77.42.
- **v17 MTL n=20 at CA / TX** — ✅ **DONE 2026-07-11 (A1)** (`catx_v17_seed0_5f/RESULTS.md` + `docs/results/closing_data/catx_v17_n20/`, fp32): CA 77.052±0.006 / 65.693±0.017, TX 77.239±0.014 / 67.062±0.007 — n=20 confirms seed-0.
  **Finding: v17's cat lever is a STATE-SIZE trade** — wins small/mid (+0.2…+2.5), costs ~0.28 cat at CA/TX (real, not
  a bf16 artifact — TX board is fp32), reg-neutral+ everywhere. Kept board-wide.
- **STL cat + reg ceilings (n=20, best-vs-best)** — ✅ **DONE 2026-07-03 (AZ corrected 2026-07-08), 5 Gowalla states →
  [`CEILINGS_N20_FINAL.md`](CEILINGS_N20_FINAL.md).**
  Cat re-tuned best-vs-best (per-state max: AL `bs2048@0.005`; AZ/FL/CA/TX `bs8192@0.005`); reg topped up to n=20.
  Δcat = AL +7.72 / **AZ +9.40** / FL +5.34 / CA +6.45 / TX +7.45 (MTL beats cat ceiling everywhere; CA/TX now n=20).
  Δreg = AL −0.31 / AZ +0.10 / FL +0.72 / CA +2.20 / TX +2.11 (matches small, beats large; CA/TX now n=20).
  ⚠ AZ correction: the published 56.24 wasn't the per-state max — the n=20 `bs8192@0.005` arm is 56.43 (Δ +9.40, not
  +9.60); two higher n=10 screens (57.04/56.93) pending a cheap top-up (see A1-az in the A40 queue). **Istanbul: done via H3 (below).**
  > Prior audit note (kept for the record): the *old* `dk_ovl` board reg ceiling was seed-0 (n=5), not n=20 — the
  > "already n=20" claim conflated it with the v14 substrate (~9 pp different). Now genuinely n=20 on `dk_ovl`; reg is
  > seed-invariant (n=20 vs seed-0 diff < 0.13 pp), so the seed-0 verdict was correct — the top-up just makes the paired
  > test rigorous. (MTL cat/reg at CA/TX are now **n=20** too — A1 done 2026-07-11; the ceilings were already n=20.)
- **H3 · Istanbul rebuilt on `dk_ovl` + v17 (n=20)** — ✅ **DONE 2026-07-06 → [`h3_istanbul/RESULTS.md`](h3_istanbul/RESULTS.md).**
  Same substrate identity as the 5 Gowalla states (cross-substrate caveat RETIRED). **Δcat +8.59 / Δreg +0.28 — beats
  BOTH** (reg flips positive vs the old stride-1-GCN base's −0.52). Baselines re-footed: CTLE-SC +28.73 / HGI +28.09 hold.
- **P6 · Istanbul cascade** — ✅ DONE 2026-07-07 (v17): Δjoint −0.22 ≈ tie, parallel ≥ cascade on both heads
  (`RESULTS_BOARD §1b`). CA/TX cascade still open.
- **ReHDM (corrected v4 code)** — **AL ✅ re-run on v4: 65.38 ± 1.08** (the −6 pp scare was an init bug; audit
  `research/baselines/rehdm/REHDM_AUDIT_CHANGES.md`); **CA/TX 🔄 RUNNING on the A40** (~22 h, interleaved, resumable).
  ⚠ The old cited AL/AZ/FL row (66.06/54.65/65.68) is **v2-code** — AZ/FL need the cheap v4 re-run so the paper row
  is version-uniform (see A2 in the queue).
- **Faithful STAN — DONE at ALL 6 (CA/TX closed-as-partial, user decision 2026-07-08):** AL 60.72 / AZ 49.86 /
  FL 72.99 / Istanbul 61.86 (full 5f) + **TX 61.67 (folds 0–3) / CA 58.52 (folds 0–1) — the citable-final numbers,
  fold counts disclosed** (deadline; both clear the best-simple floor 54.9/52.1 ✅ and sit below our MTL reg; the
  "infeasible" footnote is DROPPED, replaced by the n-folds disclosure). Remaining 4 folds = optional post-deadline
  robustness (→ [`stan_catx/STATUS.md`](stan_catx/STATUS.md), resumable, ~2.6 h/fold).
  HMT-GRN (6 states), ReHDM (AL/AZ/FL), CTLE (FL), CSLSL tie, floors — all in.
- **DEFAULT_CANON = v17**; v16 still via `--canon v16`; §0.1/v11 frozen bundle unaffected.

## What's LEFT (the run inventory — updated 2026-07-08, post-PR #58; A40 = all GPU)

| ID | Run / analysis | Machine | Status | Cost | Blocks |
|----|---|---|---|---|---|
| **A1 (ex-H1)** | **CA/TX v17 MTL n=20, seeds {1,7,100}** — MIGRATED from the H100 (lane gone) | **A40** | ✅ **DONE 2026-07-11** (`catx_v17_n20/`: CA 77.052/65.693, TX 77.239/67.062; n=20 confirms seed-0) | fp32 serial ~4.9/6.3 h/cell, 6 cells | firmed the large-state Δ → **M1-full** unblocked (re-run `m1_stats_n20.py`) |
| **A1-az** | AZ cat-ceiling screen top-up: `bs2048@{0.0025,0.0075}` × seeds {7,100} | A40 | **DROPPED (user 2026-07-08; see A40.md)** — AZ ceiling stays 56.43/Δ+9.40; the paper carries a visible sensitivity clause (§6.2) | — | — |
| **A2** | ReHDM v4: **CA/TX 🔄 RUNNING** (~22 h, resumable) + **AZ/FL v4 re-run** (version-uniform row; AL done 65.38) | A40 | running / open | AZ/FL ≈ ~25–60 min/state | ReHDM paper row (v4-uniform) |
| **A3** | Faithful STAN CA/TX | A40 | ✅ **CLOSED-AS-PARTIAL** (user 2026-07-08): TX 61.67 (4/5) / CA 58.52 (2/5) = citable-final, fold counts disclosed | 0 h (remaining folds optional post-deadline) | Table-3 cells FILL NOW with the n-folds footnote |
| **A4** | CA/TX cascade coverage (P6) | A40 | open (optional) | ~2 × seed-0 5f | nothing (coverage) |
| **M1** | v17 stats: Wilcoxon + TOST + per-cell Holm — **partial NOW at AL/AZ/FL/Istanbul** (fully n=20 both sides); CA/TX join after A1 | **M2 Pro** | **partially unblocked** | hours, no GPU | drops "provisional" where n=20 is complete |
| **M2** | A4 transductive-leak audit → CA/TX/Istanbul | M2 Pro (CPU) | open | ~3 h/fold | nothing (coverage) |
| **M3** | Bridging-metrics re-score (reg Acc@1/@5/MRR; cat Acc@1) | M2 Pro | open | short (needs saved logits) | nothing (coverage) |
| **M4** | STAN precision-mix disclosure + v4-collapse guard (now incl. the CA/TX v6-p10 partials) | M2 Pro | open (doc) | doc | STAN hygiene |
| **M5** | Stale-doc fixes + submission mechanics | M2 Pro | open (doc) | doc | submission |
| **J1** | **Joint-best re-score of the board rundirs** via `scripts/closing_data/score_joint_best.py` (Table 3 = per-task diag-best, disclosed §6.2; joint-best = the single-checkpoint lane — see [`../JOINT_BEST_SCORING.md`](../JOINT_BEST_SCORING.md)): perhead_lr_n20 AL/AZ/FL, catx_v17_seed0_5f (+ CA/TX n=20 after A1), h3_istanbul | A40 (CPU-only) | ✅ **DONE 2026-07-09** → [`../../closing_data_v2/`](../../closing_data_v2/) (served checkpoint reproduces Table 3 within ≤0.11 pp; no verdict changes; CA/TX still seed-0 pending A1) | minutes | camera-ready / response letter — NOT the submission |

**The critical path to a paper-grade v17 board:** **A1 (CA/TX MTL n=20, A40)** → **M1 full** (all 6 datasets n=20,
drop "provisional" everywhere). The ceilings + Istanbul are already done; **M1-partial can run NOW** on
AL/AZ/FL/Istanbul. A2/A3 fill the two baseline rows (ReHDM v4-uniform, STAN CA/TX); everything else is coverage.

**A40 queue order (one card, serialize):** finish **A3 STAN** (~10 h, already partial) and **A2 ReHDM CA/TX** (running)
→ **A1 CA/TX MTL n=20** (~1.5 d, the verdict-changer) → A2-az/fl (cheap) → A4 (optional). (A1-az DROPPED, user 2026-07-08.)

## The two handoffs (H100.md is decommissioned — kept as a pointer)
- **[`A40.md`](A40.md)** — ALL GPU work, queue order above.
- **[`M2PRO.md`](M2PRO.md)** — no-GPU analysis (M1-partial NOW; M2–M5 anytime; M1-full after A1).

> ⚠ **Recipe discipline (all cells):** engine `check2hgi_dk_ovl` (gated stride-1 overlap, MIN_SEQ=10), heads
> `next_gru`(cat) + `next_stan_flow_dualtower`(reg, prior-OFF), `geom_simple` selector, matched scorer. v17 = add
> `--batch-size 8192` + `--onecycle-per-head-lr` (cat/reg/shared 1e-3/3e-3/1e-3). Use `--canon none` + explicit recipe
> under `MTL_STRICT=1`. **No-fold-collapse check:** reg best-epoch must land late (not ≤~5), 0 skipped-step storms.
>
> 🔧 **PR #57 (pipeline_audit) param updates — apply to all cells:**
> - **Precision = fp32 board-wide** (`MTL_DISABLE_AMP=1`, the board invariant — bf16 costs ~1 pp at large C, fp16
>   NaN-collapses CA/TX). `p3_board.sh` now exports it; **auto-fp32 now also covers eval**, so a separate
>   `MTL_DISABLE_AMP_EVAL` is no longer needed (a bare large-C run previously scored val in fp16 → rank-tie optimism).
> - **Perf:** add `MTL_NO_TRAIN_DIAGNOSTICS=1` (P4 lever, ~9 % wall at AL, byte-identical eager) to any run.
> - `--only-fold` / `--only-folds` now work **under `--canon`** (the bundle's `--folds 5` no longer trips the mutual
>   exclusion) — fan-out can use `--canon v17` directly.
> - Champion path re-verified **leak-clean + default-preserving** (ga=1, fp32, no-freeze) — the committed board cells
>   stand; no re-execution.
