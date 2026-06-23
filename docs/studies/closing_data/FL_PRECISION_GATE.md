# FL precision gate — bf16 vs fp32 (champion-G MTL, gated overlap, seed 0 × 5f) — COMPLETE

> H100 board, 2026-06-23. Companion to `AL_PRECISION_GATE.md`. FL re-run under correct precision (the prior
> fp16 FL MTL reg 75.42 is VOID — fp16-overflow harness, `CA_MTL_DIVERGENCE.md`). Both arms resumed per-fold
> across studio restarts (`resume_fl_fold.sh`). fp32-scored by `h100_score_matched.py`.

## Result (full 5f)
| metric | fp32 | bf16 | Δ (bf16−fp32) | STL ceiling | Δ vs ceiling (fp32 / bf16) |
|---|---|---|---|---|---|
| cat macro-F1 | **79.8247** ± 0.51 | **80.0691** ± 0.48 | +0.2444 | 75.147 | **+4.68 / +4.92 (beats)** |
| reg FULL top10 | **77.2760** ± 0.77 | **77.2954** ± 1.11 | +0.0194 | 76.7123 | **+0.56 / +0.58 (BEATS)** |

Per-fold reg fp32 [77.68,77.39,76.32,76.55,78.44] / bf16 [77.68,77.44,76.33,75.93,79.09].
Per-fold cat fp32 [79.38,79.98,79.75,79.30,80.72] / bf16 [79.61,80.21,80.04,79.59,80.90].

## Verdict
- **bf16 ≈ fp32** (reg Δ+0.02 within the 0.05pp rule; cat Δ+0.24, bf16 marginally higher). Sign flips vs AL
  (bf16 −0.12 reg there) → no systematic quality gap, all within fold noise. **Speed identical** (data/launch-
  bound, GPU ~8-25%). bf16 validated equivalent → safe as the CA fp16-overflow fix.
- **FL MTL beats BOTH ceilings** (fp32 reg +0.56 / cat +4.68). The fp16 "MTL sacrifices reg" (−1.29) was a
  harness artifact; correct precision **reverses** it. FL cell kept = **fp32** (small/mid = fp32 user decision).

## FL CELL — COMPLETE (fp32)
MTL cat **79.8247** / reg **77.2760** vs ceilings cat 75.147 / reg 76.7123 → Δcat **+4.68**, Δreg **+0.56**.
Rundirs: fp32 `...mtlnet_..._173805_85991`, bf16 `..._174254_87133`. Scores in `docs/results/closing_data/h100/florida_s0_mtl_{fp32,bf16}_5f_matched_score.json`.
