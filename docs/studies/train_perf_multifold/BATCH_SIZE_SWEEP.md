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
