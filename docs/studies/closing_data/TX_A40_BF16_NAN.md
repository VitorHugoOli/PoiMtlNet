# TX champion-G MTL bf16 NaN-collapse on the A40 — device-class root cause + fp32 fix

> A40 board lane (`study/board-a40`). Companion to [[CA_MTL_DIVERGENCE]] (the fp16-overflow story) and
> [[BOARD_H100_FINDINGS]] / [[TX_CELL]] (the H100's clean TX bf16 run). Root-caused by a 5-agent investigation
> (workflow `wf_9dcc2181-9c7`, 415k tokens, 104 tool-uses, 2026-06-24).

## TL;DR
The A40's TX champion-G MTL **bf16** run NaN-collapsed (74,812 skipped optimizer steps from ep33), producing a
**VOID** Δreg = −2.37. The **H100 ran the byte-identical cell clean** (reg 67.13, **beats** the 64.96 ceiling
+2.17, 0 NaN). The investigation cleared every transferable cause — **same code, same recipe, byte-identical
substrate/engine/log_T, same torch**. The only uncontrolled axis is the **GPU device class**: an A40-**Ampere**
bf16 *backward-pass* NaN gradient (finite loss — NOT the fp16 forward overflow that [[CA_MTL_DIVERGENCE]] fixed)
that the H100-**Hopper** float trajectory does not cross under identical config. **The fix is true fp32**
(`MTL_DISABLE_AMP=1`): the A40 fp32 run finished **CLEAN — 0 skips, reg 67.02 beats the 64.96 ceiling +2.06**,
cat 77.51 (+7.56). **MTL beats BOTH heads**, matching the H100 (reg 67.13). The bf16 −2.37 / fp16 −2.41 are
confirmed **VOID** Ampere-bf16 collapse artifacts. **CELL CLOSED.**

## Symptom (A40, `scripts/closing_data/a40_task2_tx_mtl_bf16.sh`)
- champion-G MTL on `check2hgi_dk_ovl`, texas, seed 0, 5f, **bf16** (`MTL_AUTOCAST_BF16=1`), compiled+tf32.
- First non-finite: **epoch 36 batch 69, `grad_norm=NaN` with a FINITE `loss=1.79`** — a backward-pass NaN
  gradient, not an fp16 logit overflow (which would NaN the loss). Deterministic in all 5 folds.
- With `MTL_STRICT=1` the non-finite guard (`mtl_cv.py:guard_finite_step`) **hard-aborts** (first attempt died
  at fold1 ep36). With it OFF the guard **skips** the bad batch and continues → **74,812 skips** (onset ep33,
  escalating to ~6000/epoch by ep41-49). The skip-guard kept the shared backbone alive (cat healthy) but
  **starved the reg head** of late-epoch training.
- Result: reg **best-epochs [4,50,5,4,4]** (peaks at ep4-5 then degrades to ~5% val); diagnostic-best captured
  the pre-collapse ep4 peak → reg **62.5892**, **Δreg = −2.3705** vs the fp32 ceiling 64.96 (≈ the older fp16
  run's −2.4095). cat survived at **75.8661**.

## What the investigation RULED OUT (nothing to port / rebuild)
| candidate | verdict | evidence |
|---|---|---|
| missing H100 code/commit | **NO** | `git log origin/main..origin/study/board-h100 -- src/` is **empty**; PR #35 is 100% data/docs/tooling |
| missing bf16 fix | **NO** | the bf16 autocast fix (`958d71f3`) is **already on main**; the A40 log printed the bf16 banner |
| recipe difference | **NO** | flag-for-flag identical to the H100's `launch_tx_s0.sh` / `board_h100_mtl.sh` |
| bad/degenerate v14 substrate | **NO** | A40 on-disk substrate is **byte-identical** to the frozen V14 manifest, finite |
| mis-built overlap engine | **NO** | `next_build_provenance.json`: min_seq=10, stride=1, emit_tail=false (correctly gated) |
| log_T leak (stride-9 vs dk_ovl) | **NO** | inert under prior-OFF (`freeze_alpha=True alpha_init=0.0` + `--log-t-kd-weight 0.0`); also byte-identical |
| torch version | **NO** | both `2.11.0+cu128` |

## Root cause — A40-Ampere bf16 backward-pass NaN
A finite-loss `grad_norm=NaN` is a NaN entering the **backward** graph — most-supported mechanism: a degenerate
attention/STAN softmax row as the representation drifts in the OneCycle **anneal tail** (peak ~ep15, onset ep33
on the decay slope). bf16's fp32 exponent range rules out the 65504 forward overflow; the residual exposure is
the masked-softmax path (`src/models/next/next_stan/head.py` `_STANAttention`; the cross-attn MHA B-side is
already zero-fill-guarded, `models/mtlnet.py`). **Ampere (A40 GA102) vs Hopper (H100 GH100) realize different
bf16 tensor-core accumulation / `--compile`+tf32 kernel selection**, a small numerical drift that tips the A40
attention backward into NaN at ep33 where the H100 stays finite.

## The H100 ran the same cell clean (the trustworthy number)
H100 TX bf16 folds 1-2 (`[[TX_CELL]]`, autonomous per-fold): cat **77.69 / 77.31**, reg **66.94 / 67.32**,
**late** best-epochs [49,50], monotonic reg climb to ep50, **0 NaN** → reg **67.13 beats the 64.96 ceiling
+2.17**. Had the A40's NaN occurred there, the H100's `MTL_STRICT=1` would have aborted it too — so the H100
did not run a *better* path, it rode a *different, clean* per-device float trajectory.
> ⚠ Caveat: the H100 TX is committed at only **2/5 folds** (no final 5-fold score yet).

## The fix — true fp32 (`MTL_DISABLE_AMP=1`) — ✅ CONFIRMED
`scripts/closing_data/a40_task2_tx_mtl_fp32.sh`. Full 23-bit mantissa removes the bf16-rounding NaN.
**Final result (seed 0, 5f, 0 non-finite skips)** → `docs/results/closing_data/a40/tx_ba2_fp32_s0.json`:

| metric | MTL (fp32) | per-fold | best-ep | ceiling | Δ |
|---|---|---|---|---|---|
| reg FULL top10 | **67.0243** | [66.94, 67.33, 66.14, 67.23, 67.49] | [48,49,49,50,50] | 64.9597 | **+2.06 (BEATS)** |
| cat macro-F1 | **77.5115** | [77.73, 77.38, 77.43, 77.55, 77.47] | [50,50,50,49,50] | 69.9492 | **+7.56 (BEATS)** |

**0 skips**, every fold a healthy **late** best-epoch — fp32 sailed clean through the ep33–50 zone where bf16
NaN-collapsed. reg 67.02 ≈ the H100 clean bf16 (67.13), confirming the collapse was **Ampere-bf16-specific /
precision-borne**. The contingency (degenerate-softmax guard / lower reg-LR) was **NOT needed**; no code change.
Board-consistent: fp32 is the non-CA small/mid-state decision ([[AL_PRECISION_GATE]]).
> ⚠ The driver's auto-verdict string *"FLAG re-scope (>2pp)"* is a **sign-agnostic** threshold check —
> Δreg = **+2.06 is a SURPLUS (MTL beats)**, the opposite of a deficit; it does **NOT** trigger a reg-claim re-scope.

## TX cell status (seed 0, 5f, gated overlap) — ✅ CLOSED
| piece | value | source |
|---|---|---|
| STL reg ceiling (fp32, p1) | **64.9597** | committed `e1aa4003` (clean, reused) |
| STL cat ceiling (next_gru) | **69.9492 ± 0.21** | committed `50d9611d` |
| MTL reg FULL top10 (fp32) | **67.0243** → **Δreg +2.06 (BEATS)** | fp32 run, 0 skips ✓ |
| MTL cat macro-F1 (fp32) | **77.5115** → **Δcat +7.56 (BEATS)** | fp32 run ✓ |
| **verdict** | **MTL beats BOTH ceilings (reg +2.06, cat +7.56)** — the −2.4 "reg re-scope" was the VOID Ampere-bf16 artifact | |

## Operational lessons (carry forward)
1. **bf16 MTL is NOT cross-GPU portable at large-C scale.** A40-Ampere and H100-Hopper diverge under
   byte-identical config (A40 grad-NaN @ep33 vs H100 clean). Large-state bf16 cells on **Ampere** should use
   **true fp32** to avoid the backward-NaN. Reinforces the governing rule: paired comparisons stay within one
   device class.
2. **A skip-heavy run is a collapse artifact, not a clean number.** `MTL_STRICT=1` aborts on the first NaN
   (fail-loud, correct for a *clean* run); OFF lets a degenerate run *complete* but the diagnostic-best then
   captures a pre-collapse peak (the A40 −2.37, like the VOID CA −5.23). Always read the **skip count** before
   citing an MTL reg number.
3. **The A40 bf16 −2.37 and fp16 −2.41 are VOID** collapse-contaminated artifacts. The real TX MTL reg is the
   clean run (A40 fp32 / H100 bf16), which beats/matches the ceiling.
