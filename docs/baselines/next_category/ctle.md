# CTLE — Context and Time Aware Location Embeddings

## Source
- **Paper:** Lin, Wan, Guo, Lin. *Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectories for User Next Location Prediction.* AAAI 2021.
- **Reference impl:** authors' release (MLM + masked-hour pre-training over trajectories); our build `scripts/baselines/build_ctle_substrate.py` (substrate-column) + `scripts/baselines/ctle_e2e.py` (native end-to-end).

## Why this is a baseline (not our model)
CTLE is **the closest representation competitor**: like Check2HGI it produces a *contextual, per-visit* embedding (one vector per check-in, conditioned on time + neighbors), so it is the baseline that decides contribution-1's novelty. The reviewer panel made it a **non-negotiable gate** (`REVIEW_PANEL.md` required-change #2): novelty C1 "collapses to CTLE on a hierarchical graph" unless we **score CTLE leak-clean and show Check2HGI beats it on next-category attributable to the hierarchy**. The decisive form is the **substrate-column (SC)** comparison — CTLE's embedding fed to *our* matched head vs Check2HGI's embedding fed to the *same* head — which isolates the representation at matched capacity.

## What's faithful, what's adapted
- **Faithful to paper:** per-fold MLM + masked-hour pre-training of the CTLE transformer; 64-d contextual per-visit vector.
- **Adapted because our task / data differ:**
  - **SC variant:** the pre-trained CTLE encoder is frozen and emitted as a per-visit 64-d column, then read by our `next_gru` (cat) / `next_stan_flow` (reg) head — identical heads/folds/windowing to the champion, so only the substrate axis varies.
  - **Leak-clean:** pre-trained **per fold on that fold's TRAIN users only** (the original `--folds 1` recipe leaked 81.8 % of val users into the pre-train corpus; closed via per-fold staging + `--only-fold`).
  - **Windowing:** gated stride-1 overlap, `min_seq=10` (`--stride 1`), row-aligned to the board base (the `--stride` fix landed in PR #30).

## Variants we run
- `sc` — substrate-column (frozen CTLE embedding → our matched head). **The W3 gate.**
- `e2e` — native end-to-end (CTLE's own transformer fine-tuned with the heads); a second, honest read (see note).

## Results — board (seed 0 × 5 folds, gated stride-1 overlap, MPS fp32)

**SC (the headline — substrate isolation, next-category macro-F1):**

| State | CTLE-SC cat | Check2HGI-SC cat | **Δcat (ours − CTLE)** | status |
|---|---:|---:|---:|---|
| Alabama | 17.77 ± 1.47 | 55.59 ± 1.78 | **+37.82** | ✅ leak-clean 5f |
| Arizona | 19.30 ± 0.95 | 56.31 ± 1.61 | **+37.01** | ✅ leak-clean 5f |
| Florida / California / Texas | — | — | — | → CUDA (`../../studies/closing_data/BASELINE_H100.md`), pending |

**Read:** the substrate (representation) drives next-category — **Check2HGI beats CTLE-SC by ≈ +37 pp** at matched capacity, which is exactly the W3-gate evidence ("Check2HGI > CTLE attributable to the hierarchy"). Source: `docs/results/closing_data/baseline_compare/{alabama,arizona}_{ctle,check2hgi_sc}.json`; consolidated in `docs/studies/closing_data/RESULTS_BOARD.md §4`.

- **Next-region is NOT a CTLE-SC claim.** The SC *region* numbers were invalid (substrate-bypass + shared prior + stale log_T) and are quarantined (`_reg_status=INVALID_PENDING_RERUN`). Region's substrate-isolation story is weak anyway (contextual baselines tie us on region); the article's region baselines are native-E2E (HMT-GRN/STAN) + Markov-1, not SC.
- **`e2e` note:** native CTLE-E2E cat ≈ 21.2 > frozen CTLE-SC 17.75 — this is a *within-CTLE* frozen-vs-fine-tuned effect (CTLE never ingests our vectors in either setup), so it only shows "frozen-SC undersells deep models" and justifies running E2E. It is **not** evidence that a baseline benefits from our substrate (do not claim that).
