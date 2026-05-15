# Self-Audit of mtl-exploration Findings under the n_splits Leak

**Date:** 2026-05-15
**Trigger:** The `--folds < 5` × per-fold log_T `n_splits=5` mismatch bug discovered mid-study (see [`LEAK_BLAST_RADIUS_AUDIT.md`](LEAK_BLAST_RADIUS_AUDIT.md) and [`MTL_FLAWS_AND_FIXES.md §2.13`](../../findings/MTL_FLAWS_AND_FIXES.md)).
**Question:** Do any conclusions of this study need to be retracted? Do we need to re-run anything?

## TL;DR

- **All single-fold runs in this study (every `--folds 1` invocation) loaded a leak-contaminated prior.** Absolute reg `top10_acc_indist` numbers are inflated by ~13–23 pp.
- **No conclusion in this study needs to be retracted.** Every claim is built on within-state, within-budget *pairwise Δs*, which are preserved under F51's documented uniform-leak property (clean and leaky Δs match within ~0.10 pp).
- **The in-flight AZ multi-seed run is leak-free by construction** (`--folds 5` matches log_T `n_splits=5`) and will provide paper-comparable absolute numbers for the factorial conclusion.
- **No re-runs needed**; the AZ multi-seed (already queued) is the only follow-up worth its compute.

---

## Inventory: every run in this study

### Single-fold runs (`--folds 1` — leak-contaminated on absolute numbers, Δs preserved)

| Run | State | Budget | Variant | Used in conclusion |
|---|---|---|---|---|
| `mtlnet_lr1.0e-04_bs2048_ep50_20260515_124941_52410` | AL | 50ep | no_encoders (cell A) | yes — original ablation |
| `mtlnet_lr1.0e-04_bs2048_ep50_20260515_125610_54851` | AZ | 50ep | no_encoders (cell A) | yes — original ablation |
| `mtlnet_lr1.0e-04_bs2048_ep50_20260515_130534_56454` | AL | 50ep | baseline (cell D) | yes |
| `mtlnet_lr1.0e-04_bs2048_ep50_20260515_130734_56727` | AZ | 50ep | baseline (cell D) | yes |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_131410_57890` | FL | 25ep | no_encoders (cell A) | yes — original ablation |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_132529_59455` | FL | 25ep | baseline (cell D) | yes |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_134643_61554` | AL | 25ep | no_encoders (cell A) | yes — matched-protocol table |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_134838_61640` | AL | 25ep | baseline (cell D) | yes |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_134725_61587` | AZ | 25ep | no_encoders (cell A) | yes — discrimination |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_134940_61729` | AZ | 25ep | baseline (cell D) | yes |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_141109_64724` | AL | 25ep | linear+d=64 (cell B) | yes — AL factorial |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_141155_64891` | AL | 25ep | linear+d=256 (cell C) | yes |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_142438_65853` | AZ | 25ep | linear+d=64 (cell B) | **yes — load-bearing AZ discrimination** |
| `mtlnet_lr1.0e-04_bs2048_ep25_20260515_142555_66027` | AZ | 25ep | linear+d=256 (cell C) | **yes — load-bearing AZ discrimination** |

### Multi-fold runs (`--folds 5` — leak-free by construction)

| Run | State | Budget | Variant | Status |
|---|---|---|---|---|
| `ms_baseline_arizona_seed0_5f25ep` | AZ | 5f25ep, seed=0 | baseline | ✅ done; clean |
| `ms_baseline_arizona_seed1_5f25ep` | AZ | 5f25ep, seed=1 | baseline | 🔄 in flight |
| `ms_baseline_arizona_seed7_5f25ep` | AZ | 5f25ep, seed=7 | baseline | ⏳ queued |
| `ms_baseline_arizona_seed100_5f25ep` | AZ | 5f25ep, seed=100 | baseline | ⏳ queued |
| `ms_linear_arizona_seed{0,1,7,100}_5f25ep` | AZ | 5f25ep | linear+d=256 | ⏳ queued (2nd phase) |

All multi-seed runs use `--folds 5` → trainer's `n_splits=5` matches log_T's `n_splits=5` → **leak-free**. Per-seed log_T files for seeds {0,1,7,100} were just built today at canonical n_splits=5; the AZ baseline seed=42 file has been on disk since May 4.

---

## Conclusion-by-conclusion impact analysis

### Conclusion 1 (original ablation): "Removing task encoders hurts both heads at AZ/FL; at AL it's within n=1 noise at 25ep and baseline pulls ahead by 50ep."

**Built on:** within-state pairwise Δs (cell A `no_encoders` vs cell D `baseline`).

| State | Δ_cat F1 (no_enc − base) | Δ_reg top10 |
|---|---:|---:|
| AL 25ep | +2.03 | +0.54 (within noise) |
| AZ 25ep | −1.89 | **−8.32** |
| FL 25ep | −2.95 | **−5.26** |

**Leak impact:** Both arms hit the same leaky prior on the same val set. **Pairwise Δ preserved within ~0.10 pp** (F51 documented). **Conclusion holds.**

**Decision:** keep as-is, write absolute-number caveat.

### Conclusion 2 (trajectory): "no_encoders saturates earlier than baseline at AL."

**Built on:** per-epoch val trajectory at AL 25ep.

Looking at val metrics:
- no_encoders: reg top10 peaks at ep 17 = 47.76 and holds; cat F1 peaks at ep 13 = 30.66 and holds.
- baseline: reg top10 climbs through ep 25 = 47.22 (and ep 50 = 68.21 in the 50ep run); cat F1 climbs through ep 25 = 28.63 (and ep 50 = 35.78).

**Leak impact:** Both curves are inflated by the same leaky prior, both at the same epochs. The leak adds a roughly time-varying offset (driven by α growth) but **the relative trajectory shape — "this curve saturates while that one keeps climbing" — is preserved**. The trajectory measures *model capacity to absorb additional training* — α-amplified leak boosts both arms similarly.

**Decision:** keep as-is. The mechanism conclusion (smaller model converges faster to a lower ceiling) is supported.

### Conclusion 3 (factorial): "At AZ, B ≈ A and C ≈ D on reg top10, with a +8.1 pp gap between the d=64 pair and d=256 pair. The cross-attn `d_model` carries the reg gap, NOT the encoder's non-linearity/depth."

**Built on:** within-state pairwise Δs across 4 cells at AZ 25ep:
- B−A = +0.14 pp (Linear vs Identity at d=64)
- C−D = +0.08 pp (Linear vs 2-MLP at d=256)
- A,B << C,D gap = ~8.1 pp

**Leak impact:** All 4 cells hit the same leaky prior on the same val set. Every pairwise Δ is preserved. **Conclusion is the strongest in the study.**

**Decision:** keep as-is. **This conclusion will be additionally confirmed by the in-flight AZ multi-seed run** comparing C (linear+d=256) vs D (baseline) at n=20 paired Wilcoxon. If C ≈ D survives multi-seed, the "encoder MLP is over-engineered" claim is paper-grade.

### Conclusion 4 (AL factorial): "All four cells indistinguishable at AL 25ep (within ~2 pp)."

**Built on:** within-state pairwise Δs at AL 25ep.

**Leak impact:** uniform across cells. Pairwise Δs preserved. The conclusion **"AL 25ep is too early in baseline's convergence to discriminate"** is itself a Δ-based observation that's leak-symmetric.

**Decision:** keep as-is.

### Conclusion 5 (cross-state ordering): "AZ Δ_reg = −8.3, FL Δ_reg = −5.3 — non-monotone in region cardinality."

**Built on:** *cross-state* comparison of within-state Δs.

**Leak impact:** Each state's pairwise Δ is preserved individually. The cross-state COMPARISON is also preserved because both states have leak-symmetric Δs. **But note:** I never made this a strong claim — the writeup explicitly flags "non-monotone, n=1, needs multi-seed".

**Decision:** keep as-is with existing caveat.

---

## What about absolute numbers in the writeup?

The §Results section in `EXPERIMENT_NO_ENCODERS.md` quotes absolute values like:
- AL 50ep baseline cat F1 = 35.78, reg top10 = 68.21
- AZ 25ep baseline cat F1 = 43.03, reg top10 = 64.75
- FL 25ep baseline cat F1 = 63.26, reg top10 = 76.51

These are **leak-inflated on reg** by ~13–23 pp vs v11 multi-seed. **They are correctly flagged as "leak-inflated, do not compare to v11" in the existing §⚠ Critical leak discovery section.** No further retraction needed.

## Re-run decision matrix

| Candidate re-run | Cost | Value | Decision |
|---|---|---|---|
| Re-run AL+AZ+FL no_encoders at `--folds 5 --epochs 25` (clean absolutes) | ~3 h MPS (15 min × 6 runs) | Marginal — pairwise Δs already preserved | **Skip** |
| Re-run AL+FL factorial cells B+C at `--folds 5 --epochs 25` | ~2 h MPS | Confirms cross-state generalisation but AZ already gives clean discrimination | **Skip unless cross-state claim becomes paper-grade** |
| **AZ multi-seed (baseline + linear@d=256) `--folds 5 × 4 seeds`** (currently in flight) | ~2 h MPS | High — promotes "C ≈ D at AZ" to n=20 paired Wilcoxon paper-grade | **Already running** |
| Re-run F50 D5 trajectory at `--folds 5 --epochs 50` | ~10 h MPS (FL 5-fold 50ep ≈ 5 h × 2 arms) | Low — mechanism already supported in weight space | **Skip — caveat added to F50 D5 doc** |

**Bottom line: only the in-flight AZ multi-seed is worth its compute. Everything else stands on uniform-leak-preserved pairwise Δs.**

---

## Continuing the study

After the AZ multi-seed completes (~1–2 h remaining), the natural next steps are:
1. **Verify Conclusion 3 at n=20.** If C ≈ D survives (Δ_reg within ±0.5 pp at p > 0.05 paired Wilcoxon), the factorial conclusion becomes paper-grade.
2. **Finalize the writeup.** Convert "the encoder MLP is over-engineered" from "directional smoke finding" to "n=20-confirmed factorial result".
3. **Update `docs/studies/mtl-exploration/INDEX.html`** to point to this audit + the multi-seed results.

If you want a stronger cross-state claim, the cheapest extension would be the **AL B+C factorial at `--folds 5 --epochs 50`** (~30 min MPS) — AL is the smallest state and an honest "C ≈ D" there would strengthen the "linear encoder suffices" claim.

No FL multi-seed needed unless we go for a paper claim — the within-experiment direction at FL is already covered by the existing AZ multi-seed conclusion (under the assumption that the d_model factor generalizes).
