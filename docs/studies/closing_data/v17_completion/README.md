# v17_completion — the track to finish the MobiWac paper board at v17

> **New track (2026-07-01).** The paper switched its headline to **v17** (= v16 + bs8192 + per-head cat-lr 1e-3;
> `DEFAULT_CANON`, `f54a04de`). This track holds the remaining runs + analysis to make the **whole board v17**, split
> across the three machines per the user rule: **n=20 → H100 · the rest fast→slow → A40 · simple/no-GPU analysis → M2 Pro.**
> Board SSOT: [`../RESULTS_BOARD.md §1`](../RESULTS_BOARD.md). Paper close-out: [`../../../articles/[mobiwac]/CLOSER_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/CLOSER_HANDOFF.md).

## Where v17 stands (what's DONE)
- **v17 MTL n=20 at AL / AZ / FL** — DONE (`../perhead_lr_n20.md`): AL cat 64.54 / reg 69.80, AZ 65.84 / 59.56,
  FL 79.85 / 77.42. (These are the bs8192 + per-head cat-lr rows; recipe-identity confirmed via `run_n20_perhead.sh`.)
- **v17 MTL seed-0 5f at CA / TX** — DONE (`../catx_v17_seed0_5f/RESULTS.md`, fp32, clean): CA cat 77.04 / reg 65.69,
  TX cat 77.23 / reg 67.07. **Finding: v17's cat lever is a STATE-SIZE trade** — wins small/mid (+0.2…+2.5), costs
  ~0.28 cat at CA/TX (real, not a bf16 artifact — TX board is fp32), reg-neutral+ everywhere. Kept board-wide.
- **STL region ceiling** — already n=20 at all 6 states (do NOT re-run).
- **Baselines** — HMT-GRN (6 states), faithful STAN (AL/AZ/FL/Istanbul; CA/TX footnote-infeasible), ReHDM (AL/AZ/FL),
  CTLE (FL), CSLSL tie, floors. All verified against source JSONs.
- **DEFAULT_CANON = v17**, v16 still via `--canon v16`; §0.1/v11 frozen bundle unaffected.

## What's LEFT (the run inventory → the 3 handoffs)

| ID | Run / analysis | Machine | n | Cost | Blocks |
|----|---|---|---|---|---|
| **H1** | CA/TX v17 MTL n=20, seeds {1,7,100} | **H100** | n=20 | fast (bf16 or fp32; ~10–15 min/seed) | large-state Δcat significance |
| **H2** | **STL cat ceiling re-tuned to v17** (bs8192 + cat-lr 1e-3), all 5 states × {0,1,7,100} | **H100** (A40-offloadable, trivial) | n=20 | trivial (~100 s/run) | the **pairing** — Δcat = MTL − this ceiling |
| **H3** | **Istanbul rebuild on `check2hgi_dk_ovl` + v17** (substrate build → MTL + ceilings, n=20; re-foot baselines) | **H100** | n=20 | heavy (full board regen) | removes the cross-substrate caveat |
| **A1** | *(fallback)* CA/TX v17 n=20 on the A40 | A40 | n=20 | ~1.5 d serial | H1 backup |
| **A2** | ReHDM-faithful CA/TX/Istanbul | A40 | own | ~75–120 h/state | nothing (footnote-OK) |
| **M1** | n=20 re-score + Wilcoxon + region TOST + per-cell Holm + drop "provisional" prose | **M2 Pro** | — | hours, no GPU | lifts the n=5 label once H1/H2 land |
| **M2** | A4 transductive-leak audit → CA/TX/Istanbul | M2 Pro (CPU) | — | ~3 h/fold | nothing (coverage) |
| **M3** | Bridging-metrics re-score (reg Acc@1/@5/MRR; cat Acc@1) | M2 Pro | — | short (needs saved logits) | nothing (coverage) |
| **M4** | STAN precision-mix disclosure (S1) + v4-collapse guard | M2 Pro | — | doc | STAN hygiene |
| **M5** | Stale-doc fixes + submission mechanics (EDAS upload, deadline, Germano edits) | M2 Pro | — | doc | submission |

**The critical path to a paper-grade board:** H1 + H2 (n=20 MTL + its paired ceiling at CA/TX and all states) → M1
(re-run the two pre-registered tests, drop "n=5 provisional"). H3 (Istanbul rebuild) removes the last caveat.
Everything in the A2/M2–M3 tier is coverage that changes no verdict.

## The three handoffs
- **[`H100.md`](H100.md)** — the n=20 completion (H1 CA/TX MTL, H2 STL-cat ceiling, H3 Istanbul rebuild).
- **[`A40.md`](A40.md)** — the rest, fast→slow (STL-cat offload, CA/TX n=20 fallback, ReHDM).
- **[`M2PRO.md`](M2PRO.md)** — simple analysis (stats re-runs, A4-leak, bridging, STAN disclosure, stale-docs, submission).

> ⚠ **Recipe discipline (all cells):** engine `check2hgi_dk_ovl` (gated stride-1 overlap, MIN_SEQ=10), heads
> `next_gru`(cat) + `next_stan_flow_dualtower`(reg, prior-OFF), `geom_simple` selector, matched scorer. v17 = add
> `--batch-size 8192` + `--onecycle-per-head-lr` (cat/reg/shared 1e-3/3e-3/1e-3). Use `--canon none` + explicit recipe
> under `MTL_STRICT=1` (auto-v17 pins the v14 substrate and hard-fails the dk_ovl wrong-substrate guard). Large-C:
> **fp32** (`MTL_DISABLE_AMP=1`) or clean bf16 (`MTL_AUTOCAST_BF16=1` + `MTL_DISABLE_AMP_EVAL=1`); never bare-fp16.
> **No-fold-collapse check:** reg best-epoch must land late (not ≤~5), 0 skipped-step storms.
