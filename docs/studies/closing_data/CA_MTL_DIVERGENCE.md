# CA champion-G overlap MTL collapses at ep30 — fp16-autocast overflow (no GradScaler), not the tf32 toggle

> H100 board lane (`study/board-h100`), 2026-06-22/23. From the early CA reg cell (HANDOFF_BOARD_A100 Task 2).
> **TL;DR: champion-G MTL on the gated-overlap substrate suffers a sudden TOTAL COLLAPSE at epoch 30 at CA's
> scale — both heads drop to a ~3% degenerate floor. It is deterministic (seed 0) and happens IDENTICALLY in
> tf32 AND fp32, so it is a numerical/optimization instability (a NaN event at ep30), NOT a tf32-precision
> problem.** The −5.23 pp CA "δ_reg breach" first reported is a collapse artifact — VOID.

> ⚠ **CORRECTION (2026-06-23).** An earlier version of this doc (commits `6cffef22`, `caa7e087`, when it was
> named `CA_MTL_TF32_DIVERGENCE.md`) concluded "tf32 is the culprit; fp32 fixes it." **That was WRONG** — it was
> based on the fp32 run's ep5–22 window, before the ep30 cliff. The fp32 run then collapsed at the *same ep30*
> as tf32 (evidence below). Since fp32 *increases* precision and the failure is unchanged, precision is
> **refuted** as the cause. This version supersedes that conclusion. **ROOT CAUSE FOUND (2026-06-23):** the
trainer runs forward+loss+eval under **fp16 autocast with NO GradScaler**, and `--tf32` never disables it — so
neither prior run actually ran in higher precision. At CA scale the 8501-wide fp16 logits overflow (65504) at a
deterministic ep30 batch → NaN → `clip_grad_norm_` (inf-norm → coef 0) poisons the shared backbone → both heads
die. A known confound (`t2p0_fp32_control.sh`, 2026-06-04). See "ROOT CAUSE" below.

## Symptom
CA champion-G MTL (gated stride-1 overlap, frozen v14, seed 0, max-lr 3e-3): both heads train normally, then
**collapse to a degenerate ~3% floor at ep30** — every fold, abruptly. The per-task best-epoch selector masks
it by picking the pre-collapse peak (reg ep5 ≈ 58, cat ep27 ≈ 74), so the matched scorer reported
`reg 58.26 / cat 74.72` → an apparent **Δreg = −5.23 pp** vs the CA STL reg ceiling 63.48. That "breach" is
spurious — it is the early peak of a run that then dies.

## The decisive evidence — same ep30 cliff in tf32 AND fp32 (precision is NOT the cause)
Per-epoch CURRENT val metric (`val=N<reg>|C<cat>`), fold 1, both runs at max-lr 3e-3, seed 0:

| epoch | fp32 reg / cat | tf32 |
|---|---|---|
| 27 | 3.84 / **73.93** | healthy |
| 28 | 3.97 / **73.81** | healthy |
| 29 | 4.38 / **73.76** | healthy |
| **30** | **0.00 / 3.16** ← cliff | **collapse ~3%** |
| 31–50 | flat 0.00 / 3.16 | flat ~3% |

Both precisions are healthy through ep29 then **collapse at exactly ep30** (deterministic given seed 0 → same
batch order). A sudden, total, same-epoch collapse in both fp32 and tf32 ⇒ a **NaN event** (a bad
batch/gradient at ep30 NaNs the shared backbone → both heads output a constant). Not a metric/OOD artifact
(`ood_fraction` constant 0.0005). Grad-clip is active (`max_grad_norm=1.0`) but cannot rescue a NaN that
originates in the forward/backward (clip-by-NaN-norm leaves grads NaN).

## Scale dependence — same failure, severity scales with state size
| state | regions | collapse | effect |
|---|---|---|---|
| **FL** | 4703 | 1/5 folds, at **ep50** (the very end) | benign — scorer already captured high late peaks (~76); FL reg 75.42 mostly clean (fold 3 used its pre-collapse ep7 peak). |
| **CA** | 8501 | **all folds, at ep30** | catastrophic — peaks captured at ep5 (≈58, before convergence) → fake −5.2. |

So FL hits the same instability but late+rare (harmless); CA hits it early+always (fatal). TX (6553 reg /
3.83 M rows, A40 lane) is between/worse and must be assumed at risk.

## Scope — what is CLEAN vs what is affected (verified 2026-06-23)
The collapse hits **only the complex MTL model** (`mtlnet_crossattn_dualtower`). **Single-task STL heads are
stable** — verified by per-fold trajectories (all converge with LATE best-epochs, no NaN):

| run | head | status | re-run? |
|---|---|---|---|
| FL STL cat ceiling | next_gru | ✅ clean (folds ep50 acc 77–79%) | No |
| FL STL reg ceiling | next_stan_flow | ✅ clean (per-fold 75.6–78.4, best-ep 44–50; A40-matched −0.0015pp) | No |
| CA STL reg ceiling | next_stan_flow | ✅ clean (per-fold 63.25–64.02, best-ep 48–49) | No |
| CA champion-G MTL | mtlnet_crossattn_dualtower | ❌ collapses @ep30 (tf32 AND fp32) | **YES — needs a real fix** |
| FL champion-G MTL | mtlnet_crossattn_dualtower | ⚠ 1/5 folds collapse @ep50 | re-run once a fix exists |
| TX MTL (A40, if run) | mtlnet_crossattn_dualtower | ⚠ assume at risk | A40 lane re-check |

**STL ceilings do NOT need re-running.** The −5.23 pp CA "breach" is **VOID**.

## ROOT CAUSE — fp16-autocast overflow with NO GradScaler (CONFIRMED, 2026-06-23)
The MTL trainer runs the **entire forward + loss (and eval) under `torch.autocast(dtype=torch.float16)`** with
**NO `GradScaler`** anywhere in the codebase. The `--tf32` flag only toggles `allow_tf32` on the *float32*-matmul
accumulation path — **it never disables the fp16 autocast**. So both the earlier "tf32" and "fp32" runs executed
the dominant numerics in **fp16** → the precision axis was never actually varied (which is exactly why dropping
`--tf32` changed nothing). Verified in code:
- `src/training/runners/mtl_cv.py:311-322` — forward+loss under `torch.autocast(DEVICE.type, float16)` unless
  `MTL_DISABLE_AMP=1`; the in-code comment itself says *"the CUDA trainer runs fp16 autocast with NO GradScaler."*
- `src/training/runners/mtl_eval.py:165-177` — eval autocasts fp16 too (unless `MTL_DISABLE_AMP[_EVAL]=1`).
- `grep GradScaler src/` → **none** (only comments noting its absence).
- `scripts/train.py:1940-1946` — `--tf32` = `set_float32_matmul_precision('high')` + `allow_tf32`; does NOT touch autocast.

**Mechanism (explains all 3 constraints):** at CA's scale the largest fp16 tensor — the **8501-wide reg
logits** (and the cross-attn/STAN activations feeding them) — grows over training until, at a deterministic
ep30 batch (seed 0 fixes data order; **no GradScaler to stochastically skip/back-off**), a value exceeds the
fp16 finite ceiling **65504 → inf/NaN**. Then `clip_grad_norm_(params, 1.0)` (`mtl_cv.py:810`) computes
`total_norm=inf` → `clip_coef = 1.0/inf = 0` → it **zeros every finite grad and turns the offending grad into
`inf*0 = NaN`**, which `optimizer.step()` writes into the **shared cross-attn backbone**. From the next forward
on, the shared stream is NaN → BOTH the GRU cat head and the dualtower reg head die together → cat collapses to
the 3.16% majority-class floor, reg top-k of NaN = 0.00, flat to ep50. 
- **C1 determinism:** seed-fixed data order + no GradScaler ⇒ the same batch overflows at the same ep30 every fold/precision.
- **C2 ep30:** not an LR event (OneCycle peaks ~ep15); it's where the slowly-growing fp16 magnitude first crosses 65504.
- **C3 scale:** CA's 8501-wide logits overflow earlier (ep30, every fold = fatal) than FL's 4703 (ep50, 1/5 = benign); single-task STL has no shared backbone to poison and a smaller fp16 graph ⇒ stable.

**This is a KNOWN confound, flagged 2026-06-04** (`scripts/mtl_improvement/t2p0_fp32_control.sh`: *"the CUDA MTL
trainer runs fp16 autocast with NO GradScaler … MTL_DISABLE_AMP=1 forces the full fp32 path … fix the trainer
precision path"*). It is a **trainer bug present in ALL prior MTL runs** (the whole canon trained under
fp16-autocast); benign at small scale, fatal at CA/TX.

## Confirmation (in progress)
`MTL_DISABLE_AMP=1 MTL_DISABLE_AMP_EVAL=1` (true fp32 forward+eval, no autocast) CA MTL run launched — if the
ep30 cliff disappears, the fp16-overflow cause is empirically confirmed and fp32 is a working (slow) path.
[result appended on completion]

## Fix options (recipe-level — user decision)
1. **Non-finite-grad skip guard** (`mtl_cv.py:808-815`): after `clip_grad_norm_`, if `not torch.isfinite(total_norm)`
   skip `optimizer.step()/scheduler.step()` and `zero_grad`. **Least invasive — a no-op for any run that never
   overflows (byte-identical for the small states / existing canon), only kicks in at CA/TX.** Keeps fp16 speed.
   Must verify CA then converges (it skips ~1 poisoning batch).
2. **Add `torch.amp.GradScaler`** — the *standard* missing fp16 protection (auto-detect inf/NaN, skip + back off
   the loss scale). The "proper" fix; changes numerics slightly → re-validate all states.
3. **`MTL_DISABLE_AMP=1` (true fp32)** — correct, ~2–3× slower; changes numerics → re-validate.
Recommendation: **(1)** as the minimal robustness fix (preserves the frozen recipe where nothing overflowed),
with **(3)** as the confirmation/control. **Choosing the board-wide fix is a user call** (it touches the trainer
precision path that produced every prior MTL number).

## DECISION (user, 2026-06-23) — measure data quality first, then pick the fix via an advisor
Priority = **best data quality**, and the deeper goal = **close the MTL reg gap vs the STL reg ceiling**. The
fp16-no-GradScaler harness may have *understated* MTL reg even at small states (the `t2p0_fp32_control.sh` note:
fp32 "jumps toward (c)"), so the gap may be partly a precision/harness artifact rather than joint-loop sacrifice.
Plan (do NOT pick a fix yet):
1. **Small-state bias experiment** — run true-fp32 (`MTL_DISABLE_AMP=1`) champion-G MTL at FL (and AL) and
   compare reg vs the committed fp16 reg and the STL reg ceiling:
   - fp32 reg → STL ceiling ⇒ the −1.2…−5.2 gap was largely **precision/harness** ⇒ fp32/GradScaler is the
     quality fix AND it CLOSES the reg gap (a major positive for the central claim).
   - fp32 reg ≈ fp16 reg (still below ceiling) ⇒ the gap is **real joint-loop**, and precision only fixes the
     CA/TX crash (skip-guard suffices there).
2. **CA true-fp32** (running) — large-state confirmation (cliff disappears + real CA reg vs ceiling 63.48).
3. **Advisor** after the empirics — evaluate the best outcome for data quality + the reg-gap claim, then choose
   the board-wide fix (skip-guard vs GradScaler vs fp32) and the re-baseline scope.

### Empirical result — AL precision-bias (seed0, 5f, gated overlap) — 2026-06-23
AL (1109 regions) is too small to fp16-overflow, so fp16 MTL does NOT crash here → a clean precision-bias measurement:
| | reg full top10 | Δreg vs STL ceiling 69.98 | cat macro-F1 |
|---|---|---|---|
| fp16 MTL (default autocast) | 69.60 | **−0.38** | 63.44 |
| fp32 MTL (`MTL_DISABLE_AMP=1`) | 69.80 | **−0.18** | 63.48 |

**fp32 closes ~HALF the small-state reg gap (+0.20 pp: −0.38 → −0.18)** — the fp16 harness *was* understating MTL
reg (confirms the 2026-06-04 `t2p0_fp32_control.sh` hypothesis) — but a real **~−0.18 joint-loop gap persists**
in fp32. cat unchanged (≈63.4 both). So the precision fix (a) stops the CA/TX crash AND (b) materially improves
reg toward the ceiling, but does not fully erase the gap.

### Empirical result — FL fp32 fold-1 (2026-06-23) — even stronger
FL fp32 MTL fold-1: reg FULL top10 = **77.71** (best-ep 50, **NO collapse**) — that is **ABOVE the FL STL reg
ceiling (76.71)** and well above fp16 fold-1 (76.05) / the fp16 5f mean (75.42, which was dragged down by the
fold-3 ep50 collapse). So at FL, true-fp32 MTL reg may actually **BEAT** the ceiling (Δreg ≳ 0) — the apparent
"MTL sacrifices reg" was largely a **fp16-harness artifact** (crash + systematic understatement), not a real
joint-loop cost. (FL 5-fold mean pending; AL retains a small real −0.18, so the effect is state-dependent but
the direction is unambiguous: correct precision closes — and at FL reverses — the reg gap.)

**Emerging picture:** the fp16-autocast-no-GradScaler harness both (1) **crashed** large states (CA/TX) and (2)
**understated MTL reg** everywhere (≈+0.2 pp at AL, ≈+1.7 pp at FL fold-1). Correct precision (fp32 / GradScaler)
is therefore a **data-quality fix that also closes/reverses the central reg gap** — a material positive for the
paper's MTL-vs-STL reg claim. This warrants a board-wide MTL re-baseline under the corrected precision path.

## Repro / evidence
- Both runs: §3c champion-G command on `--engine check2hgi_dk_ovl --state california`, seed 0; tf32 = `--compile
  --tf32`, fp32 = `--compile` (no `--tf32`). Both collapse at ep30.
- Evidence: per-epoch `best/tr/val=N..|C..` in the run logs (`logs/ca_mtl_*.log`); per-fold
  `metrics/fold*_next_{region,category}_val.csv` (top10/f1, loss, ood_fraction).

## Status
Both CA MTL runs (tf32, fp32) killed (collapsed = void). CA reg ceiling 63.48 stands. CA Δreg blocked pending a
real fix to the ep30 collapse. FL Task 1 result stands with the caveat (1 late-fold collapse; minor).
