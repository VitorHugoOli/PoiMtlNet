# v17_completion — the track to finish the MobiWac paper board at v17

> **New track (2026-07-01).** The paper switched its headline to **v17** (= v16 + bs8192 + per-head cat-lr 1e-3;
> `DEFAULT_CANON`, `f54a04de`). This track holds the remaining runs + analysis to make the **whole board v17**, split
> across the three machines (user allocation 2026-07-01): **H100 = only H1 (the heavy CA/TX n=20) · A40 = everything
> else GPU (H2 + H3 + ReHDM), fast→slow · M2 Pro = simple no-GPU analysis.**
> Board SSOT: [`../RESULTS_BOARD.md §1`](../RESULTS_BOARD.md). Paper close-out: [`../../../articles/[mobiwac]/CLOSER_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/CLOSER_HANDOFF.md).

## Where v17 stands (what's DONE)
- **v17 MTL n=20 at AL / AZ / FL** — DONE (`../perhead_lr_n20.md`): AL 64.54 / 69.80, AZ 65.84 / 59.56, FL 79.85 / 77.42.
- **v17 MTL seed-0 5f at CA / TX** — DONE (`../catx_v17_seed0_5f/RESULTS.md`, fp32): CA 77.04 / 65.69, TX 77.23 / 67.07.
  **Finding: v17's cat lever is a STATE-SIZE trade** — wins small/mid (+0.2…+2.5), costs ~0.28 cat at CA/TX (real, not
  a bf16 artifact — TX board is fp32), reg-neutral+ everywhere. Kept board-wide.
- **STL region ceiling** — already n=20 at all 6 states (do NOT re-run).
- **Faithful STAN** — DONE + citable (AL 60.72 / AZ 49.86 / FL 72.99 / Istanbul 61.86); **CA/TX now queued as an
  attempt** (A3, the last A40 task — `FAITHFUL_STAN_FINDINGS.md` estimates ~1.5–2 h/state bf16+compile; the earlier
  "infeasible" footnote was over-conservative). HMT-GRN (6 states), ReHDM (AL/AZ/FL), CTLE (FL), CSLSL tie, floors — all in.
- **DEFAULT_CANON = v17**; v16 still via `--canon v16`; §0.1/v11 frozen bundle unaffected.

## What's LEFT (the run inventory → the 3 handoffs)

| ID | Run / analysis | Machine | n | Cost | Blocks |
|----|---|---|---|---|---|
| **H1** | CA/TX v17 MTL n=20, seeds {1,7,100} | **H100** (only job) | n=20 | fast (~10–15 min/seed) | large-state Δcat significance |
| **H2** | **STL cat ceiling re-tuned to v17** (bs8192 + cat-lr 1e-3), all 5 states + Istanbul × {0,1,7,100} | **A40** | n=20 | trivial (~100 s/run) | the **pairing** — Δcat = MTL − this ceiling |
| **H3** | **Istanbul rebuild on `check2hgi_dk_ovl` + v17** (substrate build → MTL + ceilings, n=20; re-foot substrate-bound baselines) | **A40** | n=20 | moderate (small state, 520 regions) | removes the cross-substrate caveat |
| **A2** | ReHDM-faithful CA/TX/Istanbul | A40 | own | ~75–120 h/state | nothing (footnote-OK) |
| **A3** | **Faithful STAN CA/TX** (the last A40 task) | A40 | n=5 | ~1.5–2 h/state (bf16+compile) | nothing (coverage; drops the STAN-infeasible footnote if it clears Markov) |
| **M1** | n=20 re-score + Wilcoxon + region TOST + per-cell Holm + drop "provisional" prose | **M2 Pro** | — | hours, no GPU | lifts the n=5 label once H1+H2 land |
| **M2** | A4 transductive-leak audit → CA/TX/Istanbul | M2 Pro (CPU) | — | ~3 h/fold | nothing (coverage) |
| **M3** | Bridging-metrics re-score (reg Acc@1/@5/MRR; cat Acc@1) | M2 Pro | — | short (needs saved logits) | nothing (coverage) |
| **M4** | STAN precision-mix disclosure (S1) + v4-collapse guard | M2 Pro | — | doc | STAN hygiene |
| **M5** | Stale-doc fixes + submission mechanics (EDAS upload, deadline, Germano edits) | M2 Pro | — | doc | submission |

> **STAN** — faithful STAN is done at AL/AZ/FL/Istanbul; **CA/TX are now an attempt** (A3, last A40 task, ~1.5–2 h/state
> bf16+compile — worth trying per user; coverage, changes no verdict). Plus the **M4** doc disclosure (precision/version
> mix + v4-collapse guard). If A3 clears the Markov floor it fills the Table 3 CA/TX STAN cells; else the footnote stands.

**The critical path to a paper-grade board:** **H1 (H100) ∥ H2 (A40)** both land → **M1 (M2 Pro)** re-runs the two
pre-registered tests and drops "n=5 provisional". **H3 (A40)** removes the last (cross-substrate) caveat. Everything in
the A2 / M2–M3 tier is coverage that changes no verdict.

**Sequencing across machines:** H1 (H100) and H2/H3 (A40) run **in parallel**. On the M2 Pro, **M4/M5 (doc, submission)
and M2/M3 (coverage) are independent — do them anytime, now included** — but **M1 (the re-score/stats payoff) must wait
for H1 + H2 to land** (it consumes their JSONs). So: M2 Pro is *mostly* last, but not idle until then.

## The three handoffs
- **[`H100.md`](H100.md)** — **only H1** (CA/TX v17 MTL n=20). The H100 is reserved for the one heavy large-C job.
- **[`A40.md`](A40.md)** — **H2 (STL-cat re-tune) → H3 (Istanbul rebuild) → ReHDM**, fast→slow; + H1 fallback.
- **[`M2PRO.md`](M2PRO.md)** — simple analysis (M1 stats after H1+H2; M2–M5 anytime).

> ⚠ **Recipe discipline (all cells):** engine `check2hgi_dk_ovl` (gated stride-1 overlap, MIN_SEQ=10), heads
> `next_gru`(cat) + `next_stan_flow_dualtower`(reg, prior-OFF), `geom_simple` selector, matched scorer. v17 = add
> `--batch-size 8192` + `--onecycle-per-head-lr` (cat/reg/shared 1e-3/3e-3/1e-3). Use `--canon none` + explicit recipe
> under `MTL_STRICT=1`. Large-C: **fp32** (`MTL_DISABLE_AMP=1`) or clean bf16; never bare-fp16. **No-fold-collapse
> check:** reg best-epoch must land late (not ≤~5), 0 skipped-step storms.
