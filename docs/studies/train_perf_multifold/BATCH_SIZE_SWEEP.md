# Batch-size × LR sweep — plan

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
