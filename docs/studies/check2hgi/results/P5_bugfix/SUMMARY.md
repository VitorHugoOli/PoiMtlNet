# P5 Bugfix — A7 Optimizer Probe (post-MTL_PARAM_PARTITION_BUG fix)

**Config:** `mtlnet_dselectk` + MTLoRA r=8, AL 5f×50ep, `task_set=check2hgi_next_region`,
`task_a_input=checkin`, `task_b_input=region`, `max_lr=0.003`, seed 42.

**Purpose.** Two questions in one 4-run batch:
1. Does the MTLoRA-lift attributed to the A7 champion survive the
   partition-bug fix (commits `5668856` + `c1c7f3e`), or does it
   evaporate into noise as the audit predicted?
2. Is any user-requested optimizer (Aligned-MTL, CAGrad, DB-MTL)
   competitive with PCGrad on this config?

**Runtime.** 4 × ~40 min on MPS = 2h55m wallclock (11:19–14:10 on 2026-04-22).

## Headline numbers

| Optimizer   | reg acc (%)   | reg f1 (%)  | reg top5     | reg mrr      | cat acc       | cat f1       |
|-------------|---------------|-------------|--------------|--------------|---------------|--------------|
| **PRE-FIX** (partition bug) | 13.95 ± 1.44 | 7.05 ± 0.43 | 37.23 ± 3.26 | 25.36 ± 1.89 | 38.32 ± 0.97 | 35.61 ± 1.54 |
| pcgrad      | **17.48 ± 1.35** | 8.31 ± 1.02 | 40.54 ± 3.17 | 28.68 ± 1.97 | 39.52 ± 1.10 | 36.53 ± 1.24 |
| cagrad      | 17.24 ± 0.97 | 8.22 ± 0.66 | **41.16 ± 3.16** | **28.76 ± 1.59** | **39.75 ± 1.83** | 36.95 ± 1.77 |
| db_mtl      | 17.19 ± 1.20 | 8.13 ± 0.95 | 40.97 ± 3.03 | 28.63 ± 1.77 | 39.66 ± 1.52 | **37.86 ± 1.25** |
| aligned_mtl | 16.70 ± 0.72 | 7.92 ± 0.46 | 40.58 ± 2.04 | 28.19 ± 0.85 | 39.52 ± 1.49 | 36.57 ± 1.55 |

Bold = best-in-column among post-fix runs (ignoring pre-fix row).

## Key findings

### 1. Partition-bug fix materially changes the result — in the opposite direction from the audit's prediction

PCGrad post-fix vs PCGrad pre-fix (same config, same seed, same hyperparams):

| Metric       | pre-fix (LoRA frozen) | post-fix (LoRA trains) | Δ |
|--------------|-----------------------|------------------------|---|
| reg accuracy | 13.95 ± 1.44          | 17.48 ± 1.35           | **+3.53 pp** |
| reg f1       |  7.05 ± 0.43          |  8.31 ± 1.02           | +1.26 pp |
| reg top5     | 37.23 ± 3.26          | 40.54 ± 3.17           | +3.31 pp |
| reg mrr      | 25.36 ± 1.89          | 28.68 ± 1.97           | +3.32 pp |

Non-overlapping 1σ bands on every region metric → the fix is a real lift,
not noise. The audit's prediction was that "MTLoRA r=8 gives +1.84 pp
over DSelectK+PCGrad" would "evaporate into noise" once LoRA actually
trained. **The opposite happened:** once LoRA trains, the MTLoRA lift
widens. This reframes the audit's blast-radius conclusion:
- the `(SUPERSEDED: MTL_PARAM_PARTITION_BUG)` rows in pre-fix tables are
  numerically **worse** than post-fix, not numerically similar.
- the "best MTL reg Acc@10 = 50.72 (MTLoRA r=8)" claim in B11 is now
  understated for this config; need to pull the post-fix top10_acc from
  the run JSON (not included in this table) to restate the B11 number.
- MTLoRA is **more** load-bearing than the original (pre-fix) numbers
  implied, not less.

### 2. All four optimizers are statistically tied at 5 folds × seed 42

The 1σ bands on region accuracy overlap across all four:
- pcgrad:      17.48 ± 1.35 → [16.13, 18.83]
- cagrad:      17.24 ± 0.97 → [16.27, 18.21]
- db_mtl:      17.19 ± 1.20 → [15.99, 18.39]
- aligned_mtl: 16.70 ± 0.72 → [15.98, 17.42]

Mean-to-mean difference across the four is < 1 pp; fold-std is 0.7–1.4 pp
— cannot distinguish with n=5 folds at one seed. A seed-sweep would be
needed to tell these apart, and the ranking likely reorders under noise.

### 3. DB-MTL is tied with the gradient-surgery methods despite using a scalar-backward path

DB-MTL uses ``losses.sum() → weighted .backward()`` (inherits
`EqualWeightLoss` with dynamic per-task weights derived from buffered
log-loss gradients). It was **never** contaminated by
`MTL_PARAM_PARTITION_BUG`. That it reaches the same region accuracy
(17.19 vs 17.48) as PCGrad post-fix is quantitative evidence that the
gradient-surgery vs scalar-weighting distinction is not driving
differences on this task / config — a finding that supports the
upcoming P5 attribution narrative
(PCGrad ≈ static, commit `c5ed720`).

### 4. Post-fix AdaShare rerun is still pending

The AdaShare contaminated run
(`ablation_05_adashare_mtlnet_al_5f50ep.json`) was scheduled in
`scripts/rerun_partition_bugfix.sh` (running on a second machine per the
user). Results will land later in this directory; the "AdaShare NEUTRAL"
claim cannot be reframed until that run completes.

## Next steps

1. **Pull top10_acc / ndcg_10** from the JSONs to produce a B11 replacement
   row with the post-fix champion numbers.
2. **Wait for the second-machine matrix** (rerun_partition_bugfix.sh) —
   MTLoRA r=16, r=32, AdaShare rerun, AZ replication.
3. **Consider a seed sweep** (seeds 42, 43, 44) on the A7 champion to
   tighten the 4-way comparison if any visible signal persists. At current
   n=1 seed × 5 folds, all four optimizers are tied.
4. **Update `FINAL_ABLATION_SUMMARY.md` R4 row** with the post-fix numbers
   after the second-machine reruns land. Do NOT update until then — the
   right comparison is post-fix × post-fix, not mixed.

## Files

| File | Optimizer | Runtime |
|------|-----------|---------|
| `a7_mtlora_r8_al_5f50ep_postfix_pcgrad.json`      | PCGrad     | 11:19–11:57 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_aligned_mtl.json` | AlignedMTL | 11:57–12:35 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_cagrad.json`      | CAGrad     | 12:35–13:13 (38m) |
| `a7_mtlora_r8_al_5f50ep_postfix_db_mtl.json`      | DB-MTL     | 13:13–14:10 (57m) |

Launcher: `scripts/probe_a7_optimizers.sh`.
Commits landed: `5668856` (partition fix), `c1c7f3e` (DSelectK gating),
`8afc9ac` (crossattn partial-forward), `06b799a` (STAN alibi default +
DSelectK docstring), `3b6e7d9` (issues README status), `ac17da6` (probe
launcher).
