# B5 — Hard Index vs Soft Probe: Inference-Time Ablation

**Date:** 2026-04-22. Script: `scripts/eval_hard_vs_soft_region_idx.py`.
**Input checkpoints:** α-inspection 2f × 50-ep MTL-GETNext runs. Initial pass used
epoch 8-9 state dicts (way undertrained); corrected pass below uses epoch 46-47
state dicts, which are the near-final (champion-equivalent) trained weights.

## Question

Our GETNext adaptation uses a *soft probe* over the last-step shared-backbone
embedding to derive the region-identity prior:

```
prior_soft = softmax(probe(last_emb)) @ log_T        # trained jointly
```

The faithful GETNext (Yang et al., SIGIR 2022) uses a *hard index* at the
observed last POI's region:

```
prior_hard = log_T[last_region_idx]                  # no free parameters
```

**Question:** does the soft probe capture the same signal as the hard index,
or is it learning a different (noisy / frequency-biased) proxy?

## Method

For each saved MTL-GETNext checkpoint (AL epoch 9, AZ epoch 8) — both from
the 2-fold α-inspection runs — we run the full model forward on the val
fold and substitute the prior at inference time. The STAN backbone and α
remain those learned with the *soft* probe; only the prior-computation
arm swaps.

`last_region_idx` is derived from `sequences_next.parquet`'s `poi_{0..8}`
via `placeid_to_idx` → `poi_to_region`, picking the last non-pad POI per
row. This matches GETNext's original faithful index. ~12% of AL rows have
all-pad trajectories; for those we set the hard prior to zero (fallback
to pure STAN).

All metrics are restricted to in-distribution regions (labels that appear
in the training fold).

## Results — fully-trained (epoch 46-47) state dicts — AUTHORITATIVE

These are the near-final weights of the 2-fold × 50-ep α-inspection runs. The
MTL-GETNext 5f × 50-ep *champion* runs were launched with `--no-checkpoints` so
no state dicts exist for them directly; the α-inspection runs share exactly the
same config (`mtlnet_crossattn + pcgrad + GETNext d=256, 8h`) and seed=42, just
with 2 folds instead of 5. Hyperparameters and learned head weights should be
representative of the champion.

### Alabama

**Fold 0 (val = 6355, α = 0.543, epoch 46):**

| Prior source | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| soft (trained) | 14.13 | 39.64 | 52.43 | 25.02 |
| **hard (faithful)** | **18.27** | **47.68** | **61.54** | **30.73** |
| none (pure STAN) | 2.72 | 12.86 | 22.01 | 7.28 |
| **Δ (hard − soft)** | **+4.14** | **+8.04** | **+9.11** | **+5.71** |

Soft-probe argmax agrees with hard last_region_idx: **52.38%** (up from 7% at epoch 9).

**Fold 1 (val = 6354, α = 0.543, epoch 46):**

| Prior source | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| soft (trained) | 40.56 | 70.73 | 80.80 | 53.31 |
| **hard (faithful)** | **43.44** | **75.65** | **85.76** | **57.15** |
| none (pure STAN) | 25.26 | 44.85 | 57.19 | 33.91 |
| **Δ (hard − soft)** | **+2.88** | **+4.92** | **+4.96** | **+3.84** |

Soft/hard argmax agreement: **53.71%**.

Note: fold-1 numbers are much higher than fold-0 because the 2-fold split gives
this user subgroup a much more predictable trajectory distribution. Not directly
comparable to the 5-fold champion's per-fold numbers.

### Arizona

**Fold 0 (val = 13198, α = 0.715, epoch 47):**

| Prior source | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| soft (trained) | 12.48 | 33.04 | 42.88 | 21.23 |
| **hard (faithful)** | **16.03** | **40.19** | **52.24** | **26.39** |
| none (pure STAN) | 2.61 | 8.77 | 14.84 | 5.42 |
| **Δ (hard − soft)** | **+3.55** | **+7.15** | **+9.36** | **+5.16** |

Soft/hard argmax agreement: **55.44%**.

## Why α grows 4–5× over training

At the last epoch `α` is 0.54 (AL) / 0.72 (AZ) — an order of magnitude higher
than epoch-9 values (0.13 / 0.15). The model *learns to trust the graph prior
more* as the probe sharpens. This means early-epoch ablations overestimate the
B5 lift: at epoch 9 the prior contribution is small so swapping it matters a
lot; at epoch 46 the prior is already load-bearing and swapping it matters less
in relative terms.

## Correction log

A first pass of this ablation (committed in `6517255`) used epoch-9 state dicts
and reported Δ Acc@10 of +20 to +24 pp. **That number is misleading.** It
represents the cost of a poorly trained probe at early training — not the gap
between a fully trained soft probe and a hard index. The corrected epoch-46/47
numbers above (Δ Acc@10 between +5.0 and +9.4 pp) are the authoritative ones.

## Early-training numbers (epoch 9) — kept for reference

## Results — Alabama epoch 9 (2-fold α-inspection, α=0.131)

### Fold 0 (val = 6355)

| Prior source | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| soft (trained) | 6.15 | 20.61 | 30.04 | 12.40 |
| **hard (faithful)** | **16.10** | **38.62** | **50.65** | **25.60** |
| none (pure STAN) | 4.89 | 18.52 | 27.79 | 10.93 |
| **Δ (hard − soft)** | **+9.95** | **+18.01** | **+20.61** | **+13.20** |

Soft-probe argmax agrees with hard `last_region_idx` on only **7.03%** of
val samples.

### Fold 1 (val = 6354)

| Prior source | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| soft (trained) | 7.18 | 22.99 | 32.34 | 13.92 |
| **hard (faithful)** | **18.37** | **42.51** | **56.37** | **28.93** |
| none (pure STAN) | 6.42 | 21.07 | 30.48 | 12.72 |
| **Δ (hard − soft)** | **+11.19** | **+19.52** | **+24.03** | **+15.01** |

Soft/hard argmax agreement: **8.84%**.

## Results — Arizona epoch 8 (2-fold α-inspection, α=0.148)

### Fold 0 (val = 13198)

| Prior source | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---:|---:|---:|---:|
| soft (trained) | 7.66 | 15.79 | 22.16 | 11.26 |
| **hard (faithful)** | **16.12** | **34.59** | **44.64** | **24.01** |
| none (pure STAN) | 4.75 | 11.21 | 16.65 | 7.65 |
| **Δ (hard − soft)** | **+8.46** | **+18.80** | **+22.48** | **+12.75** |

Soft/hard argmax agreement: **11.92%**.

## Observations

1. **Hard index crushes soft probe.** On both states, hard gives Acc@10
   ~20 pp above soft and MRR ~13 pp above. The margin is many times larger
   than the typical σ across folds (≈ 3–4 pp Acc@10).

2. **The soft probe is almost random** w.r.t. the true last region —
   7–12% agreement between soft-probe argmax and hard last_region_idx.
   This is consistent with the B5 feasibility finding
   (`B5_PROBE_ENTROPY_FINDINGS.md`) that the probe is diffuse and argmax
   collapses to a few popular regions.

3. **Hard at epoch 9 ≈ soft at epoch 50.** The champion 5f × 50-ep
   PCGrad + GETNext AL Acc@10 = 56.38 ± 4.11. Hard index at epoch 9
   already hits 50.65 (fold 0) / 56.37 (fold 1) — **matching the full
   50-epoch soft-trained champion on fold 1**, with 41 epochs left to
   train. Strong reason to expect a retrained 50-ep hard-index model
   to land at 60+ Acc@10.

4. **STAN contribution is small** — pure-STAN (α·prior = 0) is only a
   few pp below soft-probe, confirming that most of the GETNext lift
   comes from the graph prior, not the backbone.

5. **α ≈ 0.13 (AL) / 0.15 (AZ)** — the model gives the prior modest
   weight during training. If the prior were more accurate (hard), the
   model might learn to increase α further.

## Implication for the paper — REVISED with epoch-46 data

The current MTL-GETNext numbers (56.49 AL, 46.66 AZ, 60.62 FL-1f) on
Acc@10 are suboptimal, but by less than the epoch-9 analysis suggested.
At near-convergence the soft probe agrees with the hard index on
~52% of samples and α grows to ~0.54 (AL) / ~0.72 (AZ), so the prior
is already load-bearing and the probe is at least partially faithful.

Retraining with `last_region_idx` as a hard input should lift all three
states by an amount in the range of **+3 to +9 pp Acc@10** (point
estimate from the epoch-46 inference-time gap, with some further claw-
back expected once the STAN backbone re-adapts to a sharper prior).

The "soft probe vs hard index" ablation is now a **paper-worthy
finding**, not just a nuisance variable. Claim:

> Our adaptation of GETNext from POI-granularity to check2HGI-region
> granularity initially used a soft probe because no per-row
> `last_region_idx` existed in the input parquet. An inference-time
> ablation (CPU-only, no retraining) shows the probe is acting as a
> diffuse noise floor — it agrees with the observed last-region index
> on only 7–12% of samples. Substituting the true index at inference
> lifts Acc@10 by +20 pp on both AL and AZ, indicating that the
> faithful GETNext formulation substantially outperforms our
> adaptation and should be adopted for the final headline numbers.

## Next step — full B5 retraining

With the hypothesis confirmed, B5 proper (retraining with
`last_region_idx` wired through the data pipeline) is now high-ROI.
Implementation plan in `B5_IMPLEMENTATION_PLAN.md` (to follow).

Estimated lift budget for the paper (revised with epoch-46 evidence):

| State | Current MTL-GETNext Acc@10 | Conservative B5 | Optimistic B5 |
|---|---:|---:|---:|
| AL | 56.38 ± 4.11 | 59 ± 3 (+3 pp) | 63 ± 3 (+7 pp) |
| AZ | 47.34 ± 2.93 | 50 ± 2 (+3 pp) | 54 ± 2 (+7 pp) |
| FL (1f) | 60.62 | 63 (+3 pp) | 67 (+7 pp) |

## Caveats

- Checkpoints are from 2f × 50ep runs at epoch 8-9, **not** 5f × 50ep.
  The champion 5-fold runs were launched with `--no-checkpoints`, so
  no state dicts exist for them. A 50-ep snapshot would be stronger
  evidence but isn't available.
- Inference-time substitution is not retraining. The STAN backbone was
  optimised against a *soft* prior; with a hard prior at inference,
  the backbone's residual path may be miscalibrated. Retraining
  end-to-end with hard prior could change magnitudes (up or down).
- AL has 12% all-pad trajectories (no valid `poi_0..8`). Those rows
  fall back to pure STAN — bounds the maximum B5 lift for that subset.

## Files

- Script: `scripts/eval_hard_vs_soft_region_idx.py`
- Checkpoints used (2f × 50ep α-inspection):
  - AL: `results/check2hgi/alabama/checkpoints/mtl__check2hgi_next_region_20260421_191156_29497/checkpoint_epoch_9.pt`
  - AZ: `results/check2hgi/arizona/checkpoints/mtl__check2hgi_next_region_20260421_191712_29627/checkpoint_epoch_8.pt`
