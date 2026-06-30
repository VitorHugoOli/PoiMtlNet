# Per-head cat-LR 1e-3 + bs=8192 — n=20 confirmation

> **n=20 {0,1,7,100} × 5-fold** confirmation of the per-head-LR recipe from the `train_perf_multifold` study.
> Engine `check2hgi_dk_ovl`, champion-G recipe + **per-head cat-lr 1e-3** via `MTL_ONECYCLE_PER_HEAD_LR=1`
> (cat 1e-3 / reg 3e-3 / shared 3e-3). fp32, geom_simple selector. Driver: `run_n20_perhead.sh` (2026-06-29/30).
> cat = macro-F1 (diag-best), reg = next-region Acc@10. Full study: [`../train_perf_multifold/BATCH_SIZE_SWEEP.md`](../train_perf_multifold/BATCH_SIZE_SWEEP.md),
> [`../train_perf_multifold/RESULTS_SUMMARY.md`](../train_perf_multifold/RESULTS_SUMMARY.md).

## Results (n=20)

| State | recipe | bs | cat ± sd | reg ± sd |
|---|---|---:|---|---|
| **AL** | **bs8192 + cat-lr 1e-3** | 8192 | **64.540 ± 0.098** | 69.801 ± 0.052 |
| **AZ** | **bs8192 + cat-lr 1e-3** | 8192 | **65.835 ± 0.019** | 59.563 ± 0.053 |
| **FL** | bs2048 champion (baseline) | 2048 | 79.680 ± 0.264 | 77.224 ± 0.146 |
| **FL** | **bs8192 + cat-lr 1e-3** | 8192 | **79.848 ± 0.028** | 77.421 ± 0.025 |

## Deltas vs prior baselines (n=20)

| State | new recipe cat | vs bs2048-champion | vs bs8192-uniform (per-head lever) |
|---|---:|---:|---:|
| AL | 64.540 | **+0.995** (63.545) | **+0.638** (63.902) |
| AZ | 65.835 | **+2.270** (63.565) | **+1.524** (64.311) |
| FL | 79.848 | **+0.168 cat / +0.197 reg** (vs FL base 79.680 / 77.224) + **~7% faster** | — |

## Verdicts

1. **Per-head cat-lr 1e-3 is a confirmed n=20 quality WIN, board-wide.** The lever (activating the recipe's
   intended cat-lr 1e-3, which had been inert under OneCycle) adds **+0.64 (AL) / +1.52 (AZ)** cat over the
   bs8192-uniform run — the seed-0 finding (+0.59 AL) holds and AZ is even bigger. Full new recipe is
   **+1.0 (AL) / +2.3 (AZ)** over the frozen bs2048 champion, reg flat-to-up.
2. **bs=8192 + cat-lr 1e-3 is VIABLE — actually slightly better — at the large state (FL).** FL new beats the
   bs2048 champion by **+0.17 cat / +0.20 reg** and is **~7% faster**. This **flips the earlier "keep bs=2048 at
   FL"** conclusion (the bs8192 cat regression was pure cat-LR overshoot, fixed by cat-lr 1e-3).
3. **Mechanism:** the FL bs8192 cat regression was cat-LR overshoot (cat head overdriven at the uniform 3e-3),
   not reg-capture — proven by the isolation decomposition (only lowering cat-LR recovers FL cat).

## Recommendation
Promote **bs=8192 + per-head cat-lr 1e-3** (`MTL_ONECYCLE_PER_HEAD_LR=1`) as the recipe for all states: a cat win
at small states and equal-or-better + faster at large states. Requires the `MTL_ONECYCLE_PER_HEAD_LR` fix
(per-group OneCycle max_lr; default-OFF byte-identical) shipped in this branch. CA/TX still to be checked.
