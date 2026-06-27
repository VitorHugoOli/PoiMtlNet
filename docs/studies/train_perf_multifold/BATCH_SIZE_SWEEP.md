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
