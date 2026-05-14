# B5 Feasibility — GETNext Soft-Probe Entropy Analysis

**Date:** 2026-04-22. Script: `scripts/inspect_probe_entropy.py`.

## Purpose

The GETNext head mixes a learnable soft probe `p = softmax(W · h_last)` with the
region-transition matrix `log_T`:

```
trans_prior  = p @ log_T
final_logits = stan_logits + α · trans_prior
```

If the probe is already near-one-hot (top-1 ≈ 1.0, entropy → 0), then replacing
it with a hard `last_region_idx` lookup changes the prior negligibly and B5
(hard-index path) is not worth a 4–6 h pipeline change. If the probe is diffuse,
hard-indexing would meaningfully alter behaviour.

## Method

1. Load MTL-GETNext state dict (AL 1f, AZ 1f, from the α-inspection runs of 2026-04-21).
2. Register a forward hook on `model.next_poi.region_probe`.
3. Run the full model forward on 5 validation batches (fold 0) and capture probe logits.
4. Report mean top-1, top-5, Shannon entropy, and argmax concentration.

Data size: 6355 AL val samples, 10240 AZ val samples (fold 0).

## Results

| State | N regions | Top-1 mean | Top-5 mean | Entropy mean | Uniform H | % of uniform | Unique argmax |
|-------|-----------|-----------:|-----------:|-------------:|----------:|-------------:|--------------:|
| AL    | 1109      |     0.053  |     0.160  |        5.76  |     7.01  |         82%  |  88 / 1109 (8%) |
| AZ    | 1540      |     0.123  |     0.313  |        4.93  |     7.34  |         67%  | 103 / 1540 (7%) |

### Top-10 argmax collapse

**AL** — 5 regions take 60% of mass:
```
r=41 (16.7%) · r=264 (10.6%) · r=6 (10.3%) · r=80 (10.1%) · r=362 (7.3%)
r=46 (4.1%)  · r=152 (3.4%)  · r=1 (3.1%)  · r=884 (3.1%) · r=109 (2.5%)
```

**AZ** — 5 regions take 59% of mass:
```
r=21 (24.7%) · r=20 (10.6%) · r=45 (9.6%) · r=89 (7.7%) · r=57 (6.4%)
r=423 (4.2%) · r=16 (3.4%)  · r=23 (2.9%) · r=619 (2.7%) · r=441 (2.1%)
```

## Interpretation

**The soft probe is DIFFUSE, not a hard index.** Mean top-1 probability is 5%
(AL) / 12% (AZ); Shannon entropy is 67–82% of uniform. In both states the
probe's output distribution is far from one-hot.

**BUT the argmax is highly collapsed** — in each state, only 7–8% of regions
are ever picked as argmax across thousands of val samples, and the top-5 argmax
regions together capture ≥ 59% of all argmaxes. The probe has learned to
concentrate *where it's most confident* on a handful of popular regions, but it
spreads the remaining mass widely.

**Functional consequence:** `p @ log_T` under a diffuse `p` degenerates toward
`mean(log_T, axis=0)` — a marginalised (frequency) prior over next regions,
*not* a transition-conditioned prior. The MTL-GETNext gains we observed on
Acc@10 (+11 pp AL, +5.6 pp AZ, +3 pp FL) are likely coming primarily from the
**frequency bias** of popular regions, not from per-trajectory conditional
transitions.

This explains the Acc@10 vs Acc@1/Acc@5/MRR divergence on FL: the head lifts
*coverage* of popular regions but does not improve *ranking precision* at the
top.

## AL vs AZ: why AZ probe is more confident

- AL top-1 = 0.053 (near-uniform), AZ top-1 = 0.123 (more concentrated).
- AZ has more data (10K samples, 1540 regions) and a higher learned α (per
  the α-inspection run: AZ α > AL α systematically).
- The probe gets more signal on AZ because per-POI transitions are visited
  more often, which matches the pattern of larger scale → more repeatable
  regional patterns.

## Decision on B5

**B5 is load-bearing, but its expected direction is non-obvious.**

Two competing hypotheses:

1. **Hard-index > soft-probe.** If the probe is genuinely noisy, replacing it
   with the ground-truth last region would sharpen the conditional
   transition prior and recover Acc@1/5 precision. Expected win: +2–5 pp on
   Acc@1 and MRR, possibly at a small cost in Acc@10.
2. **Hard-index ≈ soft-probe.** The probe is diffuse *because* the model has
   learned that a diffuse (frequency-like) prior works better than a sharp
   transition-conditional one at our scale. Hard-indexing would then lose
   the implicit smoothing and degrade all metrics.

**Recommendation:** run B5 after the static_weight + GETNext attribution test
(#148) resolves. If static_weight matches PCGrad + GETNext, the GETNext lift is
already confirmed to come from the prior; then B5 is worth 4–6 h to sharpen
the paper claim. If static_weight underperforms, the prior+PCGrad interaction
dominates and B5 should be deferred.

**Effort estimate:** 4–6 h (extend `next_region.parquet` schema with
`last_region_idx`, thread through dataset/dataloader, modify head to accept aux
column, re-run AL + AZ MTL GETNext at ~30 min each).

## Files

- Analysis script: `scripts/inspect_probe_entropy.py`
- State dicts used:
  - AL: `results/check2hgi/alabama/checkpoints/mtl__check2hgi_next_region_20260421_191156_29497/checkpoint_epoch_9.pt`
  - AZ: `results/check2hgi/arizona/checkpoints/mtl__check2hgi_next_region_20260421_191712_29627/checkpoint_epoch_8.pt`

## Caveat

The checkpoints analysed are **epoch 8–9 of 50**, from the 2-fold α-inspection
runs (not the full 5-fold MTL-GETNext runs). A 50-epoch probe may become more
(or less) concentrated. Re-running on a full-50-epoch checkpoint is cheap if
the analysis here turns load-bearing.
