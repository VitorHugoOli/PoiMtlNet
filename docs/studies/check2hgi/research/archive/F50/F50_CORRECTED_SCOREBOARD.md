# F50 RE-EVALUATION — corrected selector

**Generated:** 2026-04-29. **Selector:** top10-best epoch within ≥ep5 (delayed-min).
**Reference:** CUDA H3-alt baseline `_0153`. **Acceptance:** Δreg ≥ +3 pp at paired Wilcoxon p<0.05.

## Master scoreboard (corrected vs original)

| variant | F1-sel reported (ORIG) | top10-sel ≥ep5 (CORRECTED) | Δ correction | Δ vs corrected H3-alt | n+/n- | Wilcoxon p_> | verdict (corrected) |
|---|---:|---:|---:|---:|:-:|:-:|:-:|
| H3-alt CUDA (ref) | 73.61 | 74.72 | +1.11 | — | — | — | reference |
| **T1.2 HSM** | 70.60 | 69.56±1.98 | -1.04 | -5.16 | 0/5 | 1.0000 | tied/negative |
| **T1.3 FAMO** | 74.23 | 74.05±0.46 | -0.17 | -0.66 | 1/4 | 0.9688 | tied/negative |
| **T1.4 Aligned-MTL** | 73.50 | 75.16±1.01 | +1.67 | +0.45 | 4/1 | 0.0625 | tied/positive |
| **P1 no_crossattn** | 73.40 | 74.43±0.57 | +1.03 | -0.29 | 2/3 | 0.6875 | tied/negative |
| **P2 detach-K/V** | 73.55 | 74.66±0.62 | +1.11 | -0.06 | 2/3 | 0.5938 | tied/negative |
| **P3 cat_freeze@10** | 73.31 | 74.58±0.78 | +1.27 | -0.14 | 2/3 | 0.7812 | tied/negative |
| **P4 alt-SGD** | 74.57 | 78.55±0.63 | +3.98 | +3.83 | 5/0 | 0.0312 | **✅ PASS +3pp** |
| **PLE-lite** | 74.72 | 74.97±0.58 | +0.25 | +0.25 | 3/2 | 0.4062 | tied/positive |
| **Cross-Stitch def** | 73.73 | 74.98±0.67 | +1.25 | +0.26 | 3/2 | 0.4062 | tied/positive |
| **Cross-Stitch detach** | 69.56 | 74.72±0.35 | +5.16 | +0.01 | 3/2 | 0.5000 | tied/positive |
| **MTL cw=0.50** | 74.14 | 75.00±0.84 | +0.87 | +0.29 | 4/1 | 0.0938 | tied/positive |
| **MTL cw=0.25** | 74.55 | 75.10±0.70 | +0.54 | +0.38 | 4/1 | 0.0938 | tied/positive |
| **MTL cw=0.0 (D8)** | 74.06 | 75.01±0.66 | +0.95 | +0.30 | 3/2 | 0.1562 | tied/positive |
| **D3 reg_enc_lr=3e-2** | 73.44 | 74.61±0.34 | +1.17 | -0.11 | 2/3 | 0.6875 | tied/negative |
| **D3 reg_enc_lr=1e-2** | 69.79 | 74.62±1.25 | +4.82 | -0.10 | 2/3 | 0.5938 | tied/negative |
| **D6 reg_head_lr=3e-2** | 74.12 | 14.85±32.20 | -59.28 | -59.87 | 0/5 | 1.0000 | tied/negative |

## Verdict changes under corrected selector

| variant | original verdict (F1-sel) | corrected verdict (top10-sel ≥ep5) | change |
|---|---|---|---|
| **PLE-lite** | directional positive (Δreg +1.11) | tied (Δreg +0.25) | downgraded — was selector artifact |
| **Cross-Stitch detach** | catastrophic fold-4 collapse Δreg=-4.05 | tied with H3-alt | downgraded — was selector artifact |
| **P4 alt-SGD** | tied (Δreg +0.96, p=0.0938) | **✅ PASS +3pp (Δreg +3.83, p=0.0312)** | **upgraded** — paper-grade fix |
| T1.4 Aligned-MTL | tied (Δreg -0.11) | tied (Δreg +0.45, n+=4/5) | borderline; not paper-grade |
| All others | unchanged qualitatively | unchanged qualitatively | uniform +3.5 pp shift to baseline+config |

## Implications for past F-experiments

- **F37 STL=82.44 pp** ✓ unaffected (STL pipeline has its own selector)
- **F49 architectural Δ pattern** — uses `diagnostic_best_epochs` (F1-selector) on MTL side; per-state architectural Δ values likely off by ~+/-1-3 pp under corrected selector. NEEDS RE-COMPUTATION.
- **CH18 'FL architectural cost = 8.83 pp'** — CORRECTED to ~5.3 pp (much smaller).
- **CH22 Δm computation** — used cat F1 (F1-best-epoch) + reg MRR (at F1-best-epoch). MRR has same selector mismatch as top10. CH22 verdicts may shift but Pareto signs likely hold.
- **OBJECTIVES_STATUS_TABLE** numbers (e.g., 'MTL H3-alt FL = 71.96 reg top10 joint-best') — affected; corrected per-task-best top10 = 77.16; joint-best is a separate metric, also worth checking.
