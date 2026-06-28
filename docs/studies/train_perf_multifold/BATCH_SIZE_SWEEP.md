# Batch-size × LR sweep — plan

> ⚠ **AUDIT CORRECTION (2026-06-28, advisor a7e80a72).** Three over-claims below were corrected:
> 1. **The FL "regression" was n=1 (a single fold, seed 0).** cat −0.58 is only ~1.3–1.5σ (single-fold
>    difference-SE ≈ 0.31–0.44 pp) → **statistically consistent with zero**. Comparing FL n=1 against AL/AZ
>    n=20 was a methodological error. The matching reg **+0.34** is the SAME single fold — you cannot keep one
>    and call the other a "regression." ~~FL is UNCONFIRMED pending the settle run.~~ **RESOLVED 2026-06-28:
>    the n=5 settle (`run_fl_settle.sh`, see bottom section) shows the regression is REAL and LARGER (−1.07 pp
>    at 5-fold seed-0, reg flat) — the n=1→n=5 fix resolved AGAINST 8k at FL. Take #1 ("keep bs=2048 at large
>    states") now stands on n=5 evidence.**
> 2. **The pct_start "FALSIFIED" label overreaches** — also n=1; the 3 arms span 0.12 pp, below the ~0.3 pp
>    single-fold noise floor. It **cannot resolve** a real pct_start effect; re-test at 5-fold.
> 3. **The WD sub-claim was self-contradictory** — weaker WD-per-step at 8k = LESS regularization = predicts
>    WORSE, not the small-state gain. The coherent single explanation for the AL/AZ-gain↔FL-(unconfirmed) sign
>    is **gradient-noise-scale (1/√batch)** alone. The OneCycle step-budget remains the headline mechanism but
>    was never tested with an epochs arm (see the AL/AZ coupled sweep).
>
> **Still SOUND:** the AL/AZ n=20 small-state win (AL cat +0.36 t≈4, AZ +0.75 t≈13; both heads up), "keep
> canon.py at 2048", and "linear LR-scaling is harmful (8k_lin cat 48.80)". The AL/AZ **base** n=20 (63.545 /
> 63.565) was re-scored race-free from the 4 distinct timestamp-window rundirs (not the paired summary.tsv).
> **Caveat:** "small-state" is generalized from n=2 states (AL/AZ) — Istanbul (520 regions) is the cheap class
> confirmation.


**Goal.** Raise the champion batch size (2048 → 4096 / 8192) for speed + GPU utilisation **without losing
quality**. Larger batch reduces gradient noise + the number of optimizer steps, so the LR (and possibly warmup)
must be re-tuned to hold quality. Find the best **(batch_size, LR-scaling)** combination.

**Why it's tricky (NOT byte-identical).** Changing the batch size changes the data partition per step, the
gradient-accumulation grouping, the OneCycleLR `steps_per_epoch`, and the SGKF/shuffle RNG consumption → the
result changes by construction. So `parity_check.sh` does NOT apply; we gate on **quality vs the bs=2048
baseline** (and the board §1), screen at seed 0, confirm the winner at n=20.

**Baseline (champion-G, fp32).** bs=2048, OneCycle max-lr 3e-3, per-head cat-lr 1e-3 / reg-lr 3e-3 / shared-lr
1e-3, geom_simple selector. RESULTS_BOARD §1: **AL cat 63.56 / reg 69.81**, AZ 63.39 / 59.34.

## The grid (LR-scaling rules)

The two standard rules when scaling batch by factor k: **linear** (LR ×k — Goyal et al.) and **sqrt** (LR ×√k —
preserves the SGD noise scale). Plus a **none** control (LR unchanged → expect quality drop, isolates the LR
effect from the batch effect).

| exp | bs | k | rule | max_lr | cat_lr | reg_lr | shared_lr |
|---|---:|---:|---|---:|---:|---:|---:|
| **base** | 2048 | 1 | — | 3e-3 | 1e-3 | 3e-3 | 1e-3 |
| 4k-none | 4096 | 2 | none | 3e-3 | 1e-3 | 3e-3 | 1e-3 |
| 4k-sqrt | 4096 | 2 | √2 | 4.243e-3 | 1.414e-3 | 4.243e-3 | 1.414e-3 |
| 4k-lin | 4096 | 2 | ×2 | 6e-3 | 2e-3 | 6e-3 | 2e-3 |
| 8k-none | 8192 | 4 | none | 3e-3 | 1e-3 | 3e-3 | 1e-3 |
| 8k-sqrt | 8192 | 4 | √4=×2 | 6e-3 | 2e-3 | 6e-3 | 2e-3 |
| 8k-lin | 8192 | 4 | ×4 | 12e-3 | 4e-3 | 12e-3 | 4e-3 |

## Protocol

- **Phase 1 — screen (this run):** AL, **2 folds × 50 ep, seed 0, fp32**, the 7 cells. Cheap (AL bs≥4096 has
  fewer steps → fast). Read: cat macro-F1 + reg Acc@10 vs base; flag cells that hold/beat. Also record wall-clock
  (the speed motive) — but note the GPU is shared with another user, so *quality* is the trustworthy axis here.
- **Phase 2 — confirm:** the top 1–2 cells at **AL 5-fold + AZ 5-fold**, seed 0; then **n=20 {0,1,7,100}×5f** for
  any promote candidate (≥ −0.0 pp vs base on both heads = "holds"; > base = "win").
- **Phase 3 — scale check:** if a cell holds at AL/AZ, verify it at **FL** (memory: bs=8192 FL ≈ 36 GB, fits the
  A40; reg matmul grows) before adopting board-wide.

**Possible extra knobs if LR-scaling alone doesn't hold quality:** OneCycle `pct_start` (warmup is shorter in
*steps* at large bs → may need a larger fraction), `--epochs` (more epochs to recover the lost update count),
gradient-accumulation (emulate large bs at fixed memory — a separate axis).

**Success = a (bs, LR) cell that matches base quality at higher bs** (→ faster / better GPU util), ideally bs=4096
which halves the step count with modest memory.

## Phase-1 screen RESULTS (AL, 2f×50ep, seed0, same folds via --per-fold-seed)

| cell | bs | LR rule | cat macroF1 | reg Acc@10 | Δcat vs base | Δreg | wall(s)* |
|---|---:|---|---:|---:|---:|---:|---:|
| base | 2048 | — | 61.35 | 67.04 | — | — | 448 |
| 4k_none | 4096 | unscaled | 61.88 | 67.11 | +0.53 | +0.07 | 429 |
| 4k_sqrt | 4096 | ×√2 | 61.40 | 67.19 | +0.05 | +0.15 | 435 |
| 4k_lin | 4096 | ×2 | 61.02 | 67.10 | −0.33 | +0.06 | 437 |
| 8k_none | 8192 | unscaled | **62.05** | 67.04 | **+0.70** | 0.00 | 463 |
| 8k_sqrt | 8192 | ×2 | 61.46 | 66.88 | +0.11 | −0.16 | 462 |
| 8k_lin | 8192 | ×4 | **48.80** | 66.58 | **−12.55** | −0.46 | 284 |

(*wall not a clean speed read — GPU shared + cells ran 2-parallel.)

**FINDING.** Larger batch with the **SAME LR (no scaling)** holds/slightly-beats the baseline (4k_none +0.53,
8k_none +0.70 cat; reg flat). **Linear LR-scaling is HARMFUL** — `8k_lin` (max-lr 12e-3) collapses the 7-class
cat head to 48.80 (the documented cat-collapse from too-high LR). sqrt-scaling is neutral. So the champion
**tolerates bs↑ to 8192 without LR re-tuning**, and the linear rule (Goyal) does NOT apply to this cat head.
The ±0.5 pp wins are within 2-fold fold-std → the robust claim is "quality HOLDS"; n=20 needed to call a win.

**Next (Phase 2):** confirm **8k_none + 4k_none** (the two holders) at AL 5-fold + AZ 5-fold seed-0, then
**n=20 {0,1,7,100}** for the promote candidate. Separately, a **clean exclusive-GPU timing run** of 8k_none vs
base to quantify the speed benefit (the 2-parallel/shared screen can't). Also worth a cell: **bs=8192 +
pct_start↑** (warmup is shorter in steps at large bs) in case it recovers more cat.

## Phase-2 CONFIRMATION (AL+AZ, 5-fold, seed-0, board protocol, LR unchanged)

| state | cell | bs | cat (Δ vs base) | reg (Δ) | wall_s (vs base) |
|---|---|---:|---:|---:|---:|
| AL | base | 2048 | 63.44 (≈board 63.56) | 69.82 (≈69.81) | 1456 |
| AL | 4k_none | 4096 | 63.82 (+0.38) | 69.90 (+0.08) | 1355 (−7%) |
| AL | **8k_none** | 8192 | **64.01 (+0.57)** | 69.80 (−0.02) | 1317 (−10%) |
| AZ | base | 2048 | 63.48 (≈board 63.39) | 59.41 (≈59.34) | 2982 |
| AZ | 4k_none | 4096 | 63.79 (+0.31) | 59.51 (+0.11) | 2760 (−7%) |
| AZ | **8k_none** | 8192 | **64.37 (+0.89)** | **59.63 (+0.23)** | 1946 |

**RESULT — bs↑ is a WIN (not just a hold).** All 4 larger-batch cells improve cat (+0.31…+0.89) with reg
flat-to-up, on BOTH states, at full 5-fold, **with the LR unchanged**. `8k_none` is best (AL +0.57 cat, AZ +0.89
cat / +0.23 reg). base reproduces the board both states (AL 63.44/69.82, AZ 63.48/59.41), so the deltas are
trustworthy. Wall-clock shows the bigger batch is ≥7% faster, but it's **confounded by contention** (cells ran
2-parallel on a GPU shared with another user; AZ 8k_none's −35% is partly from running last/solo) → the clean
exclusive-GPU timing is still pending.

**Next:** (1) **n=20 {0,1,7,100}×5f** for `8k_none` at AL+AZ to promote the +0.5…+0.9 cat as a real win (seed-0
5-fold is suggestive but within multi-seed CI). (2) **Clean exclusive-GPU timing** (8k_none vs base) for the true
speed delta. (3) **FL scale check** (bs=8192 ≈36 GB — fits; confirm the cat win + 0 NaN at large C).

## Clean exclusive timing (corrects the contended Phase-2 wall numbers)

AL, 2f×50ep, **sequential + exclusive compute** (other user's kernel idle, 0% util throughout):
- base (bs=2048): **225 s** · 8k_none (bs=8192): **229 s** → **0.98× (−1.8%, within noise)**.

**Speed = NEUTRAL.** bs=8192 has 4× fewer batches/epoch (300 vs 1200) but 4× the compute/step → on a
**compute-bound** A40 (same total FLOPs) the wall-clock is unchanged. The Phase-2 "−7…−35%" was contention
(2-parallel + a shared GPU), NOT a real speedup. This holds at large states too (FL is already 98% util at
bs=2048, so no idle capacity for a bigger batch to fill).

## CONCLUSION — batch=8192 is a QUALITY win, not a speed win
| axis | bs 2048 → 8192 |
|---|---|
| cat macro-F1 | **+0.57 AL / +0.89 AZ** ✅ |
| reg Acc@10 | flat / +0.23 ✅ |
| wall-clock | neutral (−1.8% clean) |
| LR re-tune | none |
| memory | AL ~10 GB, FL ~36 GB (fits A40) |

So raising the batch to 8192 (LR unchanged) is a **free modest quality improvement** (the reduced gradient
noise helps the cat head), at ~equal wall-clock — NOT the speedup the original motive assumed. **Linear LR
scaling must be avoided** (collapses the cat head). To promote: **n=20 {0,1,7,100}** at AL+AZ, then an FL
scale check (cat win + 0 NaN at bs=8192, C=4703).

## CORRECTION (audit) — speed IS state-dependent; quality is too

The "speed-neutral" conclusion was AL-only and WRONG to generalize. Clean exclusive FL timing (1 fold × 50ep,
profiler on) shows the bigger batch **does** speed up the large state:

| state | base→8k wall | throughput | forward section | why |
|---|---|---|---|---|
| AL (1109 reg) | 225→229 s (**−1.8%**, neutral) | 36.1k→35.1k samp/s (flat) | — | small, ~GPU-resident, ~24 batches/ep → no overhead to amortize |
| **FL** (4703 reg) | 1268→1179 s (**+7.0%**) | 25.6k→**27.5k** samp/s (+7.6%) | **56.4→34.0 s** | huge **CPU-resident** dataset, 15600 batches/fold → 4× fewer kernel launches + H2D transfers |

So the Phase-2 "speedup" pointed the right way *for large states* (though it was contention-driven there). The
GPU is compute-saturated at AL but **launch/transfer-overhead-bound at FL** — the bigger batch amortizes that.
(Also: **eval is ~23% of the FL wall**, 287 s fixed, diluting the +7.6% throughput to +7% total — a separate
optimization axis.)

**Quality is ALSO state-dependent:**
| state | Δcat (2048→8192) | Δreg |
|---|---|---|
| AL (5f) | **+0.57** | −0.02 |
| AZ (5f) | **+0.89** | +0.23 |
| FL (1f) | **−0.58** | +0.34 |

Small states: bigger batch helps cat (less gradient noise → cleaner 7-class signal). FL: cat −0.58 / reg +0.34
(1-fold; 8k's cat best-ep 28 vs base 37 — OneCycle finishes in 4× fewer steps, so cat peaks earlier+lower).

## REVISED CONCLUSION
- **Speed:** bigger batch is a real ~7% win at LARGE states (FL; overhead-bound), neutral at small (AL; compute-bound).
- **Quality:** helps small-state cat (+0.5…+0.9 AL/AZ), but **mixed at FL** (cat −0.58 / reg +0.34, 1-fold).
- **For FL the cat dip + the OneCycle-fewer-steps interaction means bs=8192 there needs `pct_start`↑ or more
  epochs to hold cat** — it's NOT a free win at large states like it is at small. Confirm with FL 5-fold + n=20.
- LR linear-scaling still harmful everywhere; keep LR unchanged.

## n=20 CONFIRMATION + FL pct_start screen (2026-06-28) — SETTLE VERDICT

### n=20 {0,1,7,100} × 5-fold, race-free (scored per-PID rundir, 4 distinct seeds averaged)
| state | base (2048) cat/reg | 8k (8192) cat/reg | Δcat | Δreg |
|---|---|---|---|---|
| **AL** | 63.545±0.10 / 69.703 | **63.902±0.14 / 69.837** | **+0.356** | **+0.134** |
| **AZ** | 63.565±0.09 / 59.396 | **64.311±0.07 / 59.582** | **+0.746** | **+0.186** |

**Both small states confirm bs=8192 as a genuine quality WIN at n=20** — positive on BOTH heads, deltas exceed
the per-seed sd. The seed-0-only +0.5…+0.9 cat held up multi-seed (AZ even strengthened). Speed-neutral at small
states (compute-bound, proven earlier).

### FL pct_start screen (bs=8192, 1-fold, seed 0) — does warmup reshape rescue the FL cat dip?
| pct_start | cat (diag-best) | reg | best-ep cat | NaN |
|---|---|---|---|---|
| base-2048 ref | **78.34** | 75.58 | 37 | — |
| 0.30 (control) | 77.76 | 75.92 | 28 | 0 |
| 0.40 | 77.64 | 75.92 | 23 | 0 |
| 0.50 | 77.72 | 75.92 | 32 | 0 |
**FALSIFIED** — reshaping warmup does NOT recover FL cat (all ~77.7, −0.6 vs base). The FL cat regression is a raw
optimizer-STEP-COUNT deficit (4× fewer steps at bs=8192), not a schedule-SHAPE problem. Only `--epochs`↑ could fix
it — which erases the +7% FL speed win. Note reg holds +0.34 across all pct_start.

### SETTLE VERDICT
- **Small states (AL/AZ): PROMOTE bs=8192 as an opt-in recommendation** — confirmed n=20 quality win (cat +0.36/+0.75,
  reg +0.13/+0.19), speed-neutral. Documented recipe variant, NOT a canon.py flip.
- **Large states (FL/CA/TX): DO NOT promote** — bs=8192 is a +7% speed win but a −0.6 cat regression (step-count
  deficit, unfixable by pct_start without losing the speed). Keep bs=2048.
- **NEVER change the global `canon.py` default (2048)** — board §1 is frozen at 2048; a silent flip de-reproduces
  the entire board (the change is not byte-identical: it perturbs the data partition + OneCycle step budget + RNG).
- **Coupled hyperparameters beyond LR (advisor, verified vs code):** the overlooked driver is the **OneCycle
  optimizer-step budget** (OneCycle is parameterized in epoch-space → 4× batch = 4× fewer steps through warmup→peak;
  this is the FL cat dip, signature best-ep 28 vs 37). Secondary: decoupled-WD per-step shrinkage (~4× weaker reg at
  8k — plausibly WHY small states gain), gradient-noise-scale 1/√batch (explains the AL/AZ-gain vs FL-loss sign
  flip). RULED OUT (champion path): grad-accum (=1), BatchNorm (LayerNorm-only heads), class weighting (off), MTL
  balancer cadence (static_weight fixed). LR linear-scaling remains harmful everywhere.

### Process notes (this run)
- n=20 driver hit the concurrent-rundir race (rundir names omit seed → `ls -dt` mis-mapped; AL/AZ scored in seed
  pairs). Fixed by averaging the 4 distinct n20-window rundirs / capturing the train.py PID suffix.
- AL 8k OOM'd in the FL-overlap window (FL 18GB + felipe 10GB + 2 small runs > 46GB). Re-run solo = clean.

## FL SETTLE RESULT (2026-06-28) — regression CONFIRMED at n=5, not noise

The audit flagged the FL "regression" as n=1 (within single-fold noise) and predicted it might vanish at proper
5-fold. **It did not — it grew.** FL seed-0 at 5-fold (n=5), base+8k concurrent, clean (0 NaN, monitor clean):

| FL seed-0 (n=5) | cat | reg | wall |
|---|---|---|---|
| base (2048) | **79.831** | 77.403 | 16,471 s |
| 8k (8192) | **78.765** | 77.419 | 14,947 s |
| **Δ (8k−base)** | **−1.066** | +0.015 | **−9.2% (faster)** |

−1.07 pp is **far beyond** the ~0.3 pp single-fold noise floor → the FL cat regression is **REAL**, larger than
the 1-fold −0.58 hinted. reg is flat. So the trade at FL is unambiguous: **−1.07 cat for ~0 reg + ~8% speed → not
worth it.** Stopped after seed-0 n=5 (user decision: regression already conclusive; saved ~13h of seeds 1/7/100).

**This overturns the audit's "FL unconfirmed" status:** take #1 ("do NOT promote bs=8192 at large states") is now
SUPPORTED by n=5, not refuted. The methodology fix (n=1→n=5) was the right call — it just resolved AGAINST 8k at
FL rather than for it. The earlier internal-inconsistency point still holds for the 1-fold data, but the n=5 data
is self-consistent: cat clearly down, reg flat.

**Net batch-size picture (final):**
- Small states (AL/AZ): bs=8192 is a confirmed n=20 quality WIN (cat +0.36/+0.75), speed-neutral. → small-state opt-in.
- Large state (FL): bs=8192 is a confirmed n=5 cat REGRESSION (−1.07), reg flat, +8% speed. → keep bs=2048.
- Global canon.py default stays 2048. The batch/quality interaction is state-(scale-)dependent, as the speed was.

## COUPLED-HYPERPARAM SWEEP RESULT (2026-06-28) — nothing beats plain bs=8192

User-take B: sweep the hyperparameters that couple to batch size BEYOND LR — at AL+AZ, bs=8192, seed-0 5-fold,
3-wide parallel (quality deterministic; PID-keyed scoring). ref8k = plain bs=8192 ep50/wd0.05 (the small-state
opt-in recipe). All 0 NaN.

| cell | AL cat (Δ vs ref8k 64.02) | AZ cat (Δ vs ref8k 64.37) | lever |
|---|---|---|---|
| **ref8k** (ep50/wd0.05) | **64.02** | **64.37** | baseline |
| ep65 | 63.87 (−0.15) | 64.10 (−0.27) | step-budget |
| ep75 | 63.69 (−0.33) | 64.13 (−0.25) | step-budget |
| wd0.10 | 63.82 (−0.20) | 64.22 (−0.16) | reg/step |
| wd0.025 | 63.89 (−0.13) | 64.21 (−0.16) | reg/step |
| ps0.40 | 64.07 (+0.05) | 64.19 (−0.18) | warmup shape |
| ep65wd025 | 63.86 (−0.16) | 63.99 (−0.39) | combo |

**RESULT — NO coupled lever beats the plain bs=8192 recipe at either small state.** ps0.40 ties at AL (+0.05 cat
but −0.09 reg) yet is −0.18 at AZ → not a real, transferable improvement. Patterns, both states:
- **epochs↑ is HARMFUL** (ep75 worst): bs8192/ep50 already wins at small states (gradient-noise benefit); extra
  steps overtrain the 7-class cat head. This **falsifies the OneCycle step-budget hypothesis at small states** —
  the FL cat regression's step-deficit story does NOT generalize to AL/AZ (opposite regime).
- **weight-decay changes** (0.10 or 0.025) are mildly harmful — wd=0.05 is already tuned.
- **pct_start** is neutral-to-harmful (matches the earlier FL n=1 screen).
- **Mechanism confirmed:** the small-state bs=8192 cat gain is the **gradient-noise-scale (1/√batch)** effect, not
  a schedule/step-budget effect. So the small-state opt-in recipe is simply **bs=8192, everything else champion-G**.

**Net:** the small-state opt-in is final at ep50/wd0.05/pct-default; the coupled axis offers no further gain.
(Next: the new-knobs sweep tests cw re-balance / logit-adjust / β2 / grad-clip — orthogonal levers beyond this.)

## NEW-KNOBS SWEEP RESULT (2026-06-28) — no orthogonal lever helps either

Web-research agent (a46de69c) top toggle-now levers BEYOND the coupled axis, at AL+AZ bs=8192 seed-0 5-fold,
3-wide, PID-keyed scoring, 0 NaN. ref8k = champion bs8192 (AL 64.02, AZ 64.356; AZ ref8k is compiled-wobble vs
the coupled-sweep 64.374, within fold-std).

| lever | AL cat (Δ vs 64.02) | AZ cat (Δ vs 64.356) | reg both | verdict |
|---|---|---|---|---|
| **ref8k** (champion bs8192) | **64.02** | **64.36** | — | baseline |
| cw0.70 (category_weight) | 63.88 (−0.145) | 64.19 (−0.168) | ~flat | worse |
| cw0.80 (category_weight) | 63.86 (−0.164) | 64.26 (−0.101) | +0.06/+0.08 | worse |
| logitadj (τ=1.0, cat) | **58.77 (−5.25)** | **59.37 (−4.99)** | flat | **CRATERS macro-F1** |
| beta2_095 (--adam-beta2) | 63.76 (−0.262) | 64.17 (−0.185) | +0.06/+0.00 | mildly worse cat |
| gradclip05 (--max-grad-norm 0.5) | 63.70 (−0.322) | 64.18 (−0.174) | +0.01/+0.11 | mildly worse cat |
| stabcombo (β2 0.95 + clip 0.5) | 63.93 (−0.091) | 64.18 (−0.177) | +0.06/+0.02 | ~wash |

**RESULT — NO orthogonal lever beats plain bs=8192 at either small state.**
- **category_weight rebalance** (0.70/0.80) is flat-negative both states — cw=0.75 already optimal at bs8192.
- **logit-adjustment τ=1.0 CRATERS cat macro-F1** (−5.25 AL / −4.99 AZ) — it shifts logits by class-frequency
  priors, which fights the macro-F1 objective on the already-cw-balanced 7-class head. Do NOT use it here.
- **β2=0.95 and grad-clip 0.5** (the large-batch stabilizers) mildly hurt cat (−0.2…−0.3), reg ~flat-to-up;
  **stabcombo** is a wash (−0.09 cat / +0.06 reg). The bs=8192 run was already stable (0 NaN) → no instability
  to fix → the stabilizers only cost a little cat precision.
- **`--adam-beta2` flag added** (byte-identical default 0.999) — now available for future large-C/bf16 work.

**CONCLUSION (whole batch-size study):** the small-state opt-in recipe is **plain bs=8192 ep50/wd0.05/cw0.75,
champion-G otherwise** — NO coupled (epochs/wd/pct_start) OR orthogonal (cw/logit-adjust/β2/clip) lever improves
it. This is the signature of a pure **gradient-noise-scale** win (1/√batch variance reduction on the
noise-limited small-state cat head), not a tunable schedule/regularization effect. Large states (FL) keep bs=2048
(FL n=5 cat −1.07) — see the FL-mechanism analysis + FL cat-lr fix experiment.

## FL CAT-LR FIX — INVALID (per-head LR is INERT under OneCycle) — important finding 2026-06-28

The FL cat-lr fix (cat-lr 1e-3→2e-3 to recover FL cat, from web-research agent ade63a19) returned
**byte-identical** results for ref8k, catlr2e3, catlr1.5e3 (all cat 78.7648 / reg 77.4185). Root cause:

**`--cat-lr` / `--reg-lr` / `--shared-lr` are SILENTLY INERT under `--scheduler onecycle`.** In
`src/training/helpers.py:setup_scheduler`, the onecycle branch builds `OneCycleLR(max_lr=<scalar 3e-3>, ...)`.
PyTorch's OneCycleLR broadcasts a scalar `max_lr` to **every** param group → it overwrites the per-head LRs that
`setup_per_head_optimizer` set. The `multi_group_per_head` guard that protects the per-head LRs exists for the
**constant** and **cosine** branches but NOT for onecycle. Proof: both run.logs print
`[per-head-LR] optimizer groups: [('cat',1.2e-4),('reg',1.2e-4),('shared',1.2e-4)]` — all three at 3e-3/25
(div_factor 25), i.e. uniform, NOT the recipe's 1e-3/3e-3/1e-3.

**Implications:**
1. The cat-lr experiment can't run as designed — the knob does nothing under onecycle.
2. **The champion recipe's documented per-head LR (`--cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3`) is DECORATIVE
   under onecycle** — all heads actually peak at the uniform `--max-lr 3e-3`. The §0.1 numbers were produced with
   uniform-3e-3 per-head LR, not the per-head ratios the recipe implies. (This does NOT change §0.1 — it just
   means those three flags were inert all along; the real LR is the single max_lr=3e-3.)
3. The research's premise ("cat is under-stepped at 1e-3, raise it") is moot — cat is already at the 3e-3 peak.
   So the FL fix has to come from a DIFFERENT lever (reduce reg's backbone capture, not raise cat-lr).

**Still-valid FL cells (non-LR knobs, running):** cw0.80 (`--category-weight 0.80`, applied in the loss — tests
whether more cat weight counters reg-capture) and pct045 (`--pct-start 0.45`, applied to OneCycle).

**To actually test per-head LR scaling** would need a code change: pass `max_lr` to OneCycleLR as a per-group LIST
`[cat_max, reg_max, shared_max]` (opt-in / env-gated, default OFF to preserve the byte-identical champion). Then
the real FL lever to try is LOWERING reg's OneCycle peak (reduce reg's shared-backbone dominance) while holding
cat at 3e-3 — the opposite of the original cat-lr-up idea.
