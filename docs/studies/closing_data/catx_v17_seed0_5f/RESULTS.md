# CA/TX v17 seed-0 5-fold — results (A40, 2026-07-01)

Champion **v17** (= v16 + bs8192 + per-head cat-lr 1e-3 via `MTL_ONECYCLE_PER_HEAD_LR`), seed 0 × 5 folds,
engine `check2hgi_dk_ovl` (gated stride-1 overlap), **fp32** (`MTL_DISABLE_AMP=1`), matched scorer
`a40_score_matched.py`. Serial on the A40 (2-wide infeasible: 2×~27 GB VRAM > 46; fold-construction RAM guard).

## Cells

| state (regions) | v17 cat macro-F1 | board cat | Δcat | v17 reg top10 | board reg | Δreg | board prec |
|---|---|---|---|---|---|---|---|
| **CA** (8501) | **77.04 ± 0.20** | 77.33 | −0.29 | **65.69 ± 0.30** | 65.66 | **+0.03** | bf16 |
| **TX** (6553) | **77.23 ± 0.12** | 77.51 | −0.28 | **67.07 ± 0.45** | 67.02 | **+0.05** | fp32 |

Per-fold (diag-best epoch):
- **CA cat** [76.95, 77.28, 76.70, 77.17, 77.11] · **CA reg** [65.40, 65.56, 65.65, 65.57, 66.28]
- **TX cat** [77.42, 77.06, 77.19, 77.29, 77.17] · **TX reg** [66.98, 67.36, 66.24, 67.27, 67.50]

Runs clean: 0 NaN/OOM, swap 0 throughout, healthy late best-epochs (48–50). CA wall ~4.9 h, TX ~6.3 h.

## Finding — v17's per-head cat-LR is a STATE-SIZE trade, not a strict board-wide win

- **Small/mid states** (AL/AZ/FL, n=20 `perhead_lr_n20.md`): cat **+0.99 / +2.45 / +0.17** — clear wins.
- **Largest states** (CA/TX): cat **−0.28 / −0.29**; **reg ties/beats** (+0.03 / +0.05).

**This is NOT a bf16 artifact.** TX's board cell is **fp32** (clean same-precision, same seed, same 5 folds) and
still shows −0.28 cat (~2× its fold-std 0.12) — matching CA's −0.29. So the large-state cat dip is **real and
consistent**. Mechanism read: lowering cat-LR fixes cat *overshoot* at small states (exposed by the bigger batch),
but the two largest states have enough data that the cat head wanted the higher LR → slight cat underfit. reg is
unaffected everywhere. (Credit: user flagged the CA-only-decrease asymmetry, which led to isolating this via TX.)

## Decision (user, 2026-07-01): **keep v17 board-wide**

Accept the small large-state cat cost (~0.28 pp, reg-neutral) for the large small-state gains + single-champion
simplicity. **v17 stays `DEFAULT_CANON`.** The large-state cat trade is documented here + in the board callout so
it travels with the numbers.

## Next
Seeds **{1,7,100} → H100** (`run_catx_v17_n20_h100.sh` + `CATX_V17_N20_H100_HANDOFF.md`) to complete CA/TX at
n=20. The H100 fp32 n=20 will firm up the large-state Δcat significance (paired vs the v16 board).
