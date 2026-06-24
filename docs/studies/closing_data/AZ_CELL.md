# AZ cell — champion-G MTL fp32 vs STL ceilings (gated overlap, seed 0 × 5f)

> H100 board lane (`study/board-h100`), 2026-06-23. AZ is a small state (1547 regions) → fp32 (no precision
> gate; small states standardized on fp32 per the AL-gate user decision). All artifacts FRESH on the v14
> overlap substrate `check2hgi_dk_ovl`, seed 0, 5 folds. MTL scored fp32 by `h100_score_matched.py`; STL reg
> ceiling = p1 `next_stan_flow` a0 (TRUE fp32); STL cat ceiling = `next_gru` (`score_stl_cat_ceiling.py`).

## Result (4 dp)
| AZ (seed0, 5f, gated overlap, fp32) | value | ceiling | Δ vs ceiling |
|---|---|---|---|
| MTL cat macro-F1 | **63.3875** | STL cat **57.1305** (5f, `next_gru`) | **+6.26 (beats)** |
| MTL reg FULL top10 | **59.3360** | STL reg **59.40** (5f, p1 fp32) | **−0.06 (matches)** |

Per-fold:
- MTL cat: [65.2657, 62.2935, 64.6314, 62.7745, 61.9722] (epochs 21,22,19,25,20)
- MTL reg: [62.6472, 58.9686, 59.3046, 57.6669, 58.0925] (epochs 27,30,27,28,33)
- STL cat ceiling: [59.2361, 56.0525, 57.9549, 56.4685, 55.9406] (epochs 47,43,49,47,15)
- STL reg ceiling Acc@10: [62.82, 59.46, 59.59, 57.42, 57.70] (best_ep 46,48,40,40,47); AGG 59.40 ± 2.15

## Reading
Same headline pattern as AL under correct (fp32) precision: **MTL beats the cat ceiling by a wide margin
(+6.26)** and **matches the reg ceiling (−0.06)** — the "MTL sacrifices reg" gap is essentially closed at AZ
(tighter than AL's −0.18). Supports the central claim that correct-precision MTL does not trade reg for cat.

## Artifacts
- MTL rundir: `results/check2hgi_dk_ovl/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260623_174233_87207` (`h100_matched_score.json`, tag `arizona_fp32`; 43 min/5f); env `MTL_DISABLE_AMP=1` (fp32), `--canon none` + explicit class-weight flags (see AL_PRECISION_GATE.md recipe note).
- STL reg ceiling: `docs/results/P1/region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_s0.json` (16 min/5f).
- STL cat ceiling: `results/check2hgi_dk_ovl/arizona/next_lr1.0e-04_bs2048_ep50_20260623_181527_96788/stl_cat_ceiling_score.json` (full 5f; supersedes the prior 4/5 partial 57.43).

## Status
**AZ cell COMPLETE.** No further AZ runs.
