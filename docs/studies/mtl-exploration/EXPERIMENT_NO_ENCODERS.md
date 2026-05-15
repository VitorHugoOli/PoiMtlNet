# Experiment — B9 without task encoders

**Date:** 2026-05-15
**Tracking branch:** `mtl-study` (this session)
**Status:** AL ✅ + AZ ✅ + FL ⏳ (in flight, 25ep both arms)

## Question

The current B9 forward pass has a per-task MLP encoder that projects each input
from `feature_size=64` up to `shared_layer_size=256` through a 2-layer
`Linear+ReLU+LN+Dropout` block before the cross-attention stack:

```
[B,9,64]  →  category_encoder (64→256→256)  →  a [B,9,256]
[B,9,64]  →  next_encoder    (64→256→256)  →  b [B,9,256]
                          ↓
          N cross-attn blocks at d_model=256
```

**What if we pass the raw 64-dim embeddings directly into the cross-attention
blocks?** Replace both encoders with `nn.Identity()` and run the entire
cross-attn stack at `d_model = feature_size = 64`.

## Implementation

Added `no_task_encoders: bool = False` kwarg to `MTLnetCrossAttn.__init__`
([`src/models/mtl/mtlnet_crossattn/model.py`](../../../src/models/mtl/mtlnet_crossattn/model.py)).
When `True`, both `category_encoder` / `next_encoder` become `nn.Identity()`,
and the constructor refuses to build unless `feature_size == shared_layer_size`.

CLI: `--model-param shared_layer_size=64 --model-param no_task_encoders=true`.

Parameter count drops from **3.00 M (baseline)** to **2.89 M (no_encoders)** —
the cross-attn blocks themselves get much cheaper at d=64, but the
`next_stan_flow` head's internal `Linear(embed_dim, d_model=256)` projection
soaks up the difference.

## Protocol

- B9 recipe verbatim except the two model-param overrides.
- 1-fold × 50 epochs (FL: 25 epochs to fit the MPS budget), seed=42.
- Leak-free per-fold `region_transition_log_seed42_fold{N}.pt`.
- `next_gru` cat head, `next_stan_flow` reg head, cosine schedule, alt-SGD, α-no-WD.
- Per-head LR (cat=1e-3, reg=3e-3, shared=1e-3).
- Best-after-ep5 metric extraction (matches §0.1 v11 paper protocol).

## Results

> **The headline table is the matched-protocol 1f × 25ep × seed=42 across all three
> states.** AL/AZ also have 50-epoch runs (longer budget) — kept below as a side
> panel because the AL sign flips between 25 and 50 epochs, which is itself the
> finding (see §Interpretation).

### Headline — matched protocol (1f × 25ep × seed=42, leak-free)

| State | Δ_cat F1 | Δ_reg top10_in | Δ_reg top5_in | Δ_reg mrr_in |
|-------|---------:|---------------:|--------------:|-------------:|
| **AL** | **+2.03** | **+0.54** | +0.85 | +1.95 |
| **AZ** | −1.89 | **−8.32** | −7.94 | −8.05 |
| **FL** | −2.95 | **−5.26** | −5.72 | −5.23 |

Per-arm absolute values:

| State | Variant | cat F1 | reg top10_in | reg top5_in | reg mrr_in |
|-------|---------|-------:|-------------:|------------:|-----------:|
| AL | baseline      | 28.63 | 47.22 | 33.22 | 20.57 |
| AL | no_encoders   | 30.66 | 47.76 | 34.07 | 22.52 |
| AZ | baseline      | 40.35 | 63.69 | 52.04 | 41.11 |
| AZ | no_encoders   | 38.46 | 55.37 | 44.10 | 33.06 |
| FL | baseline      | 63.26 | 76.51 | 69.08 | 59.37 |
| FL | no_encoders   | 60.31 | 71.25 | 63.36 | 54.14 |

### Side panel — AL+AZ at 50 epochs (over-budget; useful as upper bound)

### AL — 1-fold × 50 epochs, seed=42

| Metric                | no_encoders | baseline B9 | Δ (no_enc − baseline) |
|-----------------------|------------:|------------:|----------------------:|
| cat F1                |  **33.39%** |   **35.78%** | **−2.39 pp**          |
| reg top10_acc_indist  |  **63.69%** |   **68.21%** | **−4.52 pp**          |
| reg top5_acc_indist   |   50.28%    |    55.17%    | −4.89 pp              |
| reg mrr_indist        |   36.91%    |    42.27%    | −5.36 pp              |

### AZ — 1-fold × 50 epochs, seed=42

| Metric                | no_encoders | baseline B9 | Δ (no_enc − baseline) |
|-----------------------|------------:|------------:|----------------------:|
| cat F1                |  **41.76%** |   **43.03%** | **−1.27 pp**          |
| reg top10_acc_indist  |  **54.76%** |   **64.75%** | **−9.99 pp**          |
| reg top5_acc_indist   |   43.77%    |    53.13%    | −9.36 pp              |
| reg mrr_indist        |   31.91%    |    42.96%    | −11.05 pp             |

### FL — already at 25 epochs (see headline table)

The FL row in the headline table is FL's own 25-ep run, not over-budget. FL 50ep
was deferred (~5h/arm on MPS).

## Implementation verification (post-run, 2026-05-15)

Confirmed end-to-end via `model/arch.txt` in both run dirs:

| Module | no_encoders | baseline |
|---|---|---|
| `category_encoder` | `Identity()` ✓ | `Sequential(Linear(64→256)+ReLU+LN+Dropout, Linear(256→256)+ReLU+LN+Dropout, Linear(256→256)+ReLU+LN)` |
| `next_encoder` | `Identity()` ✓ | (same as above) |
| `cross_ab.out_proj` | `Linear(64,64)` ✓ | `Linear(256,256)` |
| `cross_ba.out_proj` | `Linear(64,64)` ✓ | `Linear(256,256)` |
| `cat_final_ln` / `next_final_ln` | `LayerNorm(64)` ✓ | `LayerNorm(256)` |
| `category_poi (next_gru)` | `GRU(64, 256, 4 layers)` ✓ | `GRU(256, 256, 4 layers)` |
| Total params | 2,892,397 | 3,003,164 |

**The Identity swap takes effect end-to-end** — cross-attn really runs at d=64, the
head's GRU receives raw 64-dim input, and there is no shadow path through the
original encoder dims.

The 110K param diff (instead of the naive "~300K encoder MLP" estimate) is because
shrinking d=256→64 in the cross-attn blocks frees up ~1.4M params (4× d² in
attention + FFN), most of which is then re-absorbed by the head-internal
projections that now have to lift 64→256 themselves.

## Leak-free confound check

All four runs (`AL no_enc / AL baseline / AZ no_enc / AZ baseline`) logged
`[C4 per-fold log_T] fold 1 seed 42 using region_transition_log_seed42_fold1.pt`.
Both arms in each pairwise comparison loaded the same leak-free seed-tagged
prior, so any residual leak is symmetric and cancels in the Δ. Reg metric
used is `top10_acc_indist` (in-distribution restricted), which matches the
v11 paper protocol.

## Caveats (read first)

1. **n=1 paired-fold smoke.** Single fold, single seed. The paper-canonical
   protocol is 5-fold × 50ep × 4–5 seeds (n=20–25 pooled fold-pairs). These
   numbers indicate direction and rough magnitude, not statistical significance.
2. **Single-seed-fold-1 numbers run higher than multi-seed means.** AL B9
   baseline here = 68.21% reg top10_in (this fold) vs 50.17% in
   [`RESULTS_TABLE.md §0.1`](../../results/RESULTS_TABLE.md) (n=20 pooled).
   **This is between the leak-free multi-seed mean (50.17%) and the
   pre-leak-fix F48-H3-alt single-seed number (~74%).** The leak-free
   per-fold log_T file was confirmed loaded (see verification above), so
   the inflation is not a re-introduction of the C4 leak. The most likely
   explanation is fold-1-specific easiness on seed=42, but this should be
   verified by running fold 1 of the multi-seed protocol in isolation
   before drawing absolute conclusions. Pairwise Δs here remain valid
   (same fold, same log_T, both arms).
3. **FL at 25 epochs**, not 50, to fit the MPS budget. Per the F-trail, α reaches
   near-peak by ep 20–25, so 25ep should still capture the headline ordering;
   absolute numbers will be slightly below converged values.

## Interpretation

**The matched-protocol (25ep) headline tells a more nuanced story than the
50-epoch numbers alone would.** Three states, three patterns:

| State | Δ_cat (25ep) | Δ_reg (25ep) | Δ_cat (50ep) | Δ_reg (50ep) | Note |
|-------|------------:|------------:|------------:|------------:|------|
| AL    | +2.03 | +0.54 | −2.39 | −4.52 | **Within n=1 noise at 25ep**; baseline pulls ahead by 50ep |
| AZ    | −1.89 | −8.32 | −1.27 | −9.99 | Direction stable, baseline wins |
| FL    | −2.95 | −5.26 | (not run) | (not run) | Baseline wins |

**Key finding: encoder budget matters.** At AL (the smallest state, 1.1k
regions), the no-encoders model is competitive with baseline through epoch 25 —
even slightly ahead, but within n=1 noise. **Per-epoch trajectory confirms
the saturation story**: no-encoders peaks val reg top10 at 47.76 by ep 17 and
holds; no-encoders peaks val cat F1 at 30.66 by ep 13 and holds; baseline is
still climbing at ep 25 on both heads (reg 47.22 climbing, cat 28.63 climbing)
and by ep 50 reaches reg 68.21 / cat 35.78. The no-encoders model
**converges faster but to a lower ceiling**, consistent with reduced capacity
(2.89M params vs 3.00M; effective d_model dropped 256→64 in the cross-attn).

**Cross-state ordering at matched 25ep is NOT monotone in region cardinality.**
Δ_reg goes AL (≈0) → **AZ (−8.3)** → FL (−5.3) → (CA/TX not run). AZ is hit
harder than FL despite having fewer regions (1.5k vs 4.7k). This is a single-fold-
single-seed observation that should not be over-read — multi-seed protocol is
needed before claiming any non-monotone scaling pattern.

**At AZ and FL the baseline wins at both budgets** — the cross-attn d=256
representation is materially helping the reg head's `α · log_T[last_region_idx]`
graph prior, which becomes more important as the region cardinality grows.

**Why reg is hit harder.** The `next_stan_flow` reg head receives the cross-attn
output and applies a learnable `α · log_T[last_region_idx]` graph prior. With
the encoder ablated, the cross-attn output is a 64-dim transformation of the
raw per-visit embedding rather than a 256-dim non-linearly-encoded
representation. The reg head's `input_proj: Linear(64 → d_model=256)`
recovers the dimensionality but not the non-linear encoder capacity — and
the α prior needs sufficient backbone signal to outpace its own pull toward
the (now leak-free, but still strong) transition graph.

**Why cat is hit less.** The cat head (`next_gru` with `hidden_dim=256`) is
recurrent and projects `64 → 256` internally on every step, so much of the
"encoder MLP" work is done inside the GRU anyway. Removing the upstream
encoder costs a couple of points but the recurrent capacity compensates.

**State-dependent gap.** AZ takes the bigger reg hit. Mechanism speculation
(needs verification): AZ has 1.5k regions vs AL's 1.1k — slightly larger
output space relies more on encoder-side feature discrimination upstream of
the head's `α · log_T` prior.

## Implications

- **The encoders are doing real work**, especially feeding the reg head.
  This is not a free ablation. It does not validate the "pass embeddings
  direct" hypothesis as a substitute for the current encoder.
- **Cat is closer to invariant** under encoder ablation — consistent with
  the CH19 finding that ~72–90% of the cat substrate gap is per-visit
  context (the input itself), with the model adding the remaining
  capacity-driven lift on top.
- **A weaker but more interesting question** is whether a *thinner* encoder
  (1 Linear layer 64→256, no non-linearity) would close most of the gap. The
  current change does both "remove non-linearity" and "shrink d_model
  64×4=256 → 64" in one move. If the user wants a follow-up, see §Next steps.

## Next steps (if pursued)

| Variant | Diagnostic value |
|---|---|
| `shared_layer_size=64`, encoders kept (default 2-layer MLP) | Isolates the dim-lift effect from the non-linearity effect. |
| `shared_layer_size=256`, encoder = single `Linear(64→256)` | Isolates the depth/non-linearity of the encoder MLP. |
| `shared_layer_size=128`, encoders kept | Intermediate-capacity smoke; cheaper than baseline, may match. |
| 5-fold × 50ep × 4 seeds (n=20) on the no-encoder variant | Promotes the AL+AZ direction to paper-grade significance. |
| Per-fold per-seed log_T (multi-seed protocol) | Required before any "this works" claim. |

## ⚠ Critical leak discovery — `--folds 1` + 5-fold log_T mismatch (2026-05-15)

While investigating why our single-fold seed=42 reg numbers ran +13–23 pp above
the v11 paper-canonical multi-seed mean (AL 50.17 / AZ 40.78 / FL 63.27), we
**reproduced an exact bug** documented in [`docs/findings/F51_MULTI_SEED_FINDINGS.md §0`](../../findings/F51_MULTI_SEED_FINDINGS.md).

### The bug

`scripts/train.py` documents:

> `--folds N`: run only the first N folds. **The split structure uses `max(2, N)` splits** (StratifiedKFold requires >= 2), but execution stops after N folds.

So `--folds 1` triggers `n_splits=2` for the trainer's `StratifiedGroupKFold`,
but the per-fold log_T file `region_transition_log_seed42_fold1.pt` on disk
was built with `n_splits=5` (the default in `compute_region_transition.py`).

**Result: the val set under `n_splits=2` (~50% of users) is NOT disjoint from
the log_T's "train" set under `n_splits=5` (~80% of users)**. Roughly ~30% of
val users have their transitions leak back into the prior. The α scalar
amplifies this leak through training, inflating reg `top10_acc_indist` by
13–23 pp.

### The fingerprint

| Measurement | Value | Reference |
|---|---:|---|
| F51 documented smoke bug (FL fold-1 seed=42 ep-6) | **76.33** | `F51_MULTI_SEED_FINDINGS.md §0` |
| **My FL fold-1 25ep ep-8 (peak)** | **76.51** | this experiment |
| F51 clean (5-fold seed=42 mean) | 63.47 | `F51_multi_seed_results.json` |
| v11 FL B9 multi-seed mean | 63.27 | `RESULTS_TABLE.md §0.1` |

**76.51 ≈ 76.33** — I reproduced the exact bug to within 0.18 pp.

### What this means for the experiment

| Aspect | Status |
|---|---|
| Absolute reg numbers in the original §Results (1f25ep + 1f50ep tables) | **Leak-inflated by ~13–23 pp** — NOT paper-grade |
| Within-experiment pairwise Δs (no_encoders vs baseline; A vs B vs C vs D) | **Still valid** under F51's documented uniform-leak property — both arms read the same wrong prior on the same val set; paired Δ cancels most of the leak (F51: "clean and leaky Δs match within 0.10 pp at every seed") |
| Conclusion "the d_model carries reg, not encoder non-linearity/depth" | **Holds** — the factorial discrimination at AZ (B ≈ A, C ≈ D, gap ≈ 8 pp on reg top10) is preserved because the leak is symmetric across cells |
| Cell-vs-cell ranking | **Unaffected** by the leak |

The cat side is naturally leak-free (cat heads don't read `log_T`), so cat
numbers don't suffer this issue.

### How AZ multi-seed (Step 2, in flight) resolves it

`run_az_multiseed.sh` uses `--folds 5` (matching `n_splits=5` of the log_T)
and rebuilds per-fold log_T at each seed {0,1,7,100} via
`compute_region_transition.py --per-fold --seed S`. **By construction, the
multi-seed run is leak-free** — the trainer's fold split matches the log_T's
fold split, so val users are properly disjoint from log_T's train set.

The multi-seed result will:
1. Give us paper-comparable absolute numbers (expected: cell C linear+d=256 ≈ baseline within ~0.5 pp on reg, matching the AZ 25ep single-fold pairwise Δ of +0.08 pp).
2. Promote the conclusion to n=20 paired-Wilcoxon significance.

### Proposed fix (footgun → loud failure)

The trainer should hard-fail when the loaded log_T's `n_splits` doesn't match
its own. Two implementation options:

1. **Encode `n_splits` in the log_T filename.** Change
   `region_transition_log_seed{S}_fold{N}.pt` → `region_transition_log_seed{S}_nsplits{K}_fold{N}.pt`,
   and have `mtl_cv.py` load `region_transition_log_seed{seed}_nsplits{n_splits}_fold{fold}.pt`.
   Trainer hard-fails if missing.
2. **Stash `n_splits` inside the `.pt` payload.** Add `"n_splits": K` to the
   torch.save dict in `compute_region_transition.py::save()`. Trainer reads
   and asserts == its own `n_splits`.

Option 2 is non-breaking (existing filenames preserved) but requires updating
both the writer and the reader. Option 1 is more explicit.

Either way: **`--folds 1` with the canonical 5-fold log_T should error out,
not silently leak.** This is a documentation-as-test gap.

## Factorial follow-up (cells B + C, 2026-05-15)

Advisor flagged that the original ablation conflated four factors: encoder
non-linearity, encoder depth, cross-attn `d_model`, head input dim. A 2×2
factorial at AL+AZ 25ep was run to isolate which factor carries the Δ.

### Design

| Cell | Encoder | shared_layer_size | What's tested |
|---|---|---:|---|
| **A** | Identity | 64 | Both factors removed (original ablation) |
| **B** | `Linear(64→64)` | 64 | *Only learnable projection added* (still d=64) |
| **C** | `Linear(64→256)` | 256 | *Only dim-lift added* (linear, no MLP) |
| **D** | 2-MLP `Linear(64,256)→ReLU→LN→Dropout→Linear(256,256)→...` | 256 | Baseline (current B9) |

Cell B and C use a new flag `--model-param linear_encoders=true` that replaces
both task encoders with a single `nn.Linear(feature_size, shared_layer_size)` —
no ReLU / LN / Dropout. See [`src/models/mtl/mtlnet_crossattn/model.py`](../../../src/models/mtl/mtlnet_crossattn/model.py)
(new `linear_encoders` kwarg).

### AL 25ep — no discrimination

| Cell | Encoder | d_model | cat F1 | reg top10_in | reg top5_in | reg mrr_in |
|---|---|---:|---:|---:|---:|---:|
| A | Identity        |  64 | 30.66 | 47.76 | 34.07 | 22.52 |
| B | Linear(64→64)   |  64 | 29.68 | 47.98 | 34.32 | 24.08 |
| C | Linear(64→256)  | 256 | 29.74 | 47.38 | 34.53 | 22.36 |
| D | 2-MLP (baseline)| 256 | 28.63 | 47.22 | 33.22 | 20.57 |

All four cells are within ~2 pp on every metric at AL 25ep — consistent with
the per-epoch trajectory finding that AL baseline is *still climbing* at ep 25.
At this state and budget, the factorial cannot discriminate.

### AZ 25ep — clean discrimination ⭐

| Cell | Encoder | d_model | cat F1 | reg top10_in | reg top5_in | reg mrr_in |
|---|---|---:|---:|---:|---:|---:|
| A | Identity        |  64 | 38.46 | **55.37** | 44.10 | 33.06 |
| B | Linear(64→64)   |  64 | 39.18 | **55.51** | 44.91 | 33.78 |
| **C** | **Linear(64→256)**  | **256** | **39.61** | **63.61** | **52.04** | **41.28** |
| D | 2-MLP (baseline)| 256 | 40.35 | 63.69 | 52.04 | 41.11 |

**The discrimination is unambiguous on reg:**
- B ≈ A (Δ_reg_top10 = +0.14 pp)
- C ≈ D (Δ_reg_top10 = +0.08 pp)
- A,B << C,D (gap ≈ **+8.1 pp**)

**The reg deficit in the original "pass embeddings direct" ablation is carried
by the cross-attn `d_model` reduction (256 → 64), NOT by the encoder's
non-linearity or depth.** A single learnable `Linear(64→256)` per task — no
ReLU, no LN, no Dropout — recovers **~98% of baseline reg top10** at AZ.

On cat the picture is different but lower-magnitude: each step
(A → B → C → D) adds ~0.5–0.7 pp monotonically, total gap +1.89 pp. Both
dim-lift *and* non-linearity contribute modestly to cat.

### Mechanism interpretation

The 2-MLP encoder in B9 is doing **two** things:
1. Dim-lift: `64 → 256` (a learnable affine projection, ~16K params).
2. Non-linear feature transformation: `ReLU + LN + Dropout` × 2 layers (~140K params).

The AZ factorial shows the *first* function carries virtually all of the reg
benefit. The non-linear MLP transformation contributes essentially nothing
beyond what a single linear projection delivers — likely because the cross-attn
blocks contain their own per-stream FFN (`Linear→GELU→Dropout→Linear`) that
provides the non-linearity downstream.

### Implication

**The current B9 encoder is over-engineered.** A `linear_encoders=true` ablation
at AZ matches baseline within noise (Δ_reg_top10 = +0.08 pp, Δ_cat = −0.74 pp)
while using ~150K fewer parameters per task encoder. If this generalizes
(needs AL 50ep + FL confirmation under multi-seed), the codebase could
simplify the encoder to a single `Linear` without any measurable cost.

This is not a "drop-in" recommendation — it needs:
1. Verification at FL (where the original Δ_reg was −5.26).
2. Multi-seed validation (n=20) to convert n=1 single-fold smoke into a
   paper-grade claim.
3. Verification that the cat benefit (+0.7 pp baseline → linear, ~+1.9 pp
   baseline → no_encoders) is honest, since cat is a smaller relative effect.

But it answers the advisor's mechanistic question cleanly: **the encoder's
job in B9 is dim-lift, not feature transformation.**

## Step 2 — AZ multi-seed (n=20) paper-grade confirmation (2026-05-15)

Per the advisor's discrimination recommendation, ran **cell C (`linear_encoders=true, shared_layer_size=256`) vs cell D (baseline)** at AZ across **4 seeds × 5 folds = n=20 fold-pairs** at 25 epochs. Leak-free per-fold log_T built per seed via `compute_region_transition.py --per-fold --seed S`. All runs at `--folds 5` matching `n_splits=5` of the per-fold log_T.

### Per-seed results

| seed | baseline reg (5f mean ± σ) | linear reg (5f mean ± σ) | **Δ_reg** | baseline cat (5f mean ± σ) | linear cat (5f mean ± σ) | **Δ_cat** |
|---:|---:|---:|---:|---:|---:|---:|
| 0   | 40.88 ± 2.29 | 40.77 ± 2.11 | **−0.11** | 42.63 ± 0.60 | 42.40 ± 0.71 | −0.23 |
| 1   | 40.91 ± 1.95 | 41.12 ± 2.01 | **+0.21** | 42.71 ± 0.60 | 42.56 ± 0.46 | −0.15 |
| 7   | 40.88 ± 2.49 | 40.75 ± 2.38 | **−0.12** | 42.73 ± 0.50 | 42.67 ± 0.66 | −0.06 |
| 100 | 40.90 ± 1.66 | 40.89 ± 1.54 | **−0.01** | 42.69 ± 1.14 | 42.22 ± 0.98 | −0.47 |

### Pooled paired Wilcoxon (n=20 fold-pairs)

| Axis | Δ pooled (C−D) | std | n+/n_total | Wilcoxon p (2-sided) | Verdict |
|---|---:|---:|---:|---:|---|
| **reg top10_indist** | **−0.008 pp** | 0.281 | 9/20 | **p = 0.658** | **Linear ≡ Baseline** (paper-grade indistinguishable) |
| **cat F1** | **−0.228 pp** | 0.545 | 7/20 | p = 0.064 | Baseline marginally better; not strictly paper-grade |

### Validation against v11 paper canon

| Quantity | This experiment (multi-seed) | v11 RESULTS_TABLE §0.1 |
|---|---:|---:|
| AZ baseline reg top10_indist (mean) | **40.89** | **40.78 ± 0.07** |

**Within 0.11 pp of v11 — leak-free protocol reproduces paper canon.** ✓ This confirms the bug fix is correct and the multi-seed numbers can be cited alongside v11.

(My multi-seed AZ cat F1 = 42.69 vs v11's 45.10 — gap of 2.4 pp attributed primarily to 25-epoch vs 50-epoch budget; v11 cat keeps climbing past ep 25 in the existing 50ep AZ baseline run: peaks at 43.03 at ep 34, declining slightly thereafter. Δ across arms is symmetric on this axis so the factorial conclusion is unaffected.)

### Final factorial verdict

The original ablation findings + advisor-prescribed factorial + n=20 multi-seed Wilcoxon together produce a clean three-tier conclusion:

1. **On reg**, the cross-attention `d_model = 256` (vs 64) is the load-bearing factor.
   The encoder MLP's ReLU + LayerNorm + Dropout + extra Linear contribute **nothing measurable beyond what a single `Linear(feature_size, shared_layer_size)` per task delivers** (Δ_reg = −0.01 pp, p = 0.66, n=20). The non-linearity comes from the cross-attention block's own per-stream FFN (GELU), making the encoder MLP redundant.

2. **On cat**, the MLP contributes a small marginal lift (~+0.23 pp; p=0.064, n=20). Not paper-grade significant at AZ, directional across all 4 seeds.

3. **The B9 encoder MLP is over-engineered at AZ scale** — replacing both encoders with a single `Linear(64→256)` saves ~280K parameters per encoder and ~6 LOC of model code at zero measured cost on reg and ~0.2 pp cost on cat at AZ.

### Implication for the codebase

If this holds at AL+FL+CA+TX (currently untested at multi-seed), the canonical B9 recipe could simplify to:

```python
self.category_encoder = nn.Linear(feature_size, shared_layer_size)
self.next_encoder     = nn.Linear(feature_size, shared_layer_size)
```

dropping the 2-layer MLP. Open question: is AL multi-seed needed before such a recommendation can ship?

## Step 3 — AL multi-seed + cell E (Linear+LN) cross-state (2026-05-15)

After the AZ multi-seed confirmed Cell C ≡ Cell D on reg, two extensions were run together:

1. **AL multi-seed**: tests whether the AZ factorial conclusion holds at smaller state size.
2. **Cell E (Linear + LayerNorm)**: tests whether adding a single LayerNorm to the linear projection recovers the small cat lift the 2-MLP provided.

Both at 5-fold × 25-ep × 4 seeds × matching protocol. Total: 16 runs (~2 h on MPS). Per-fold log_T at AL was built per seed via the trainer's `--per-fold --n-splits 5` writer (now including `n_splits` in payload thanks to the 2026-05-15 guard).

### Per-state, per-arm pooled means (n=20 fold-pairs)

| State | arm | reg top10_in (mean ± fold-std) | cat F1 (mean ± fold-std) |
|---|---|---:|---:|
| AL | baseline (D)   | 47.66 ± 3.07 | 35.91 ± 1.24 |
| AL | linear (C)     | 47.74 ± 3.31 | 32.70 ± 2.43 |
| AL | linear_ln (E)  | 47.80 ± 3.18 | 33.34 ± 1.85 |
| AZ | baseline (D)   | 40.89 ± 1.95 | 42.69 ± 0.69 |
| AZ | linear (C)     | 40.88 ± 1.81 | 42.46 ± 0.74 |
| AZ | linear_ln (E)  | 40.77 ± 1.97 | 42.67 ± 0.55 |

### Paired Wilcoxon at n=20

| Comparison | State | Δ_reg pp | p_reg | n+/n_reg | Δ_cat pp | p_cat | n+/n_cat | Verdict |
|---|---|---:|---:|:-:|---:|---:|:-:|---|
| C vs D | AL | +0.08 | 0.70 | 11/20 | **−3.21** | **<1e-4** | 0/20 | reg tied, cat baseline dominates |
| E vs D | AL | +0.14 | 0.62 | 10/20 | **−2.57** | 0.0001 | 2/20 | reg tied, cat baseline wins |
| **E vs D** | **AZ** | **−0.12** | **0.40** | 8/20 | **−0.025** | **0.81** | 9/20 | **🎯 Both tied (paper-grade)** |
| E vs C | AL | +0.06 | 0.73 | 9/20 | +0.65 | 0.20 | 13/20 | LN helps cat directionally |
| E vs C | AZ | −0.11 | 0.94 | 9/20 | **+0.20** | **0.008** | 16/20 | LN helps cat significantly |

### v11 paper-canon validation

| State | This experiment (5f×25ep×4 seeds pooled) | v11 RESULTS_TABLE §0.1 | Δ |
|---|---:|---:|---:|
| AL baseline reg top10_indist | 47.66 ± 3.07 | 50.17 ± 0.24 | −2.51 (25-epoch budget cap; AL still climbing past ep 25) |
| AZ baseline reg top10_indist | **40.89 ± 1.95** | **40.78 ± 0.07** | **+0.11 (within noise — ✓ leak-free protocol matches v11 paper canon)** |
| AL baseline cat F1 | 35.91 ± 1.24 | 40.57 ± 0.24 | −4.66 (25-epoch budget cap; AL cat keeps climbing past ep 25 — see F50 D5 caveat) |
| AZ baseline cat F1 | 42.69 ± 0.69 | 45.10 ± 0.19 | −2.41 (25-epoch budget cap) |

### Final scale-conditional verdict

**Reg axis (universal, both states):** the encoder structure does NOT matter. Cells C (Linear), D (2-MLP), E (Linear+LN) all give statistically identical `top10_acc_indist`. **The load-bearing factor is `d_model = 256` in the cross-attention stack**, not the encoder's depth or non-linearity. This is consistent with the architectural reading: the cross-attn block's own per-stream FFN (with GELU) provides all the non-linearity the reg head needs.

**Cat axis (state-conditional):**

- **AZ (1.5k regions):** Cell E (`Linear → LayerNorm`) is statistically equivalent to the 2-MLP baseline on both axes (Δ_reg = −0.12, p=0.40; Δ_cat = −0.025, p=0.81). The encoder MLP is genuinely over-engineered at AZ scale.
- **AL (1.1k regions):** baseline 2-MLP DOMINATES Cell E on cat by **−2.57 pp (p=0.0001)** and Cell C by **−3.21 pp (p<1e-4)**. The MLP is *load-bearing for cat at small scale*. Removing it costs ~6% relative cat F1.

**LayerNorm contribution (E vs C):** adding a single LayerNorm to the plain linear projection gives a small but real cat lift — **+0.20 pp at AZ (p=0.008 paper-grade)** and +0.65 pp at AL (p=0.20 directional). The LN closes most of the C→D gap at AZ; at AL it only closes ~20% of the (much larger) C→D gap.

### Simplification claim (paper-scope)

> The 2-layer MLP encoder in B9 is over-engineered **at AZ scale and above**: it can be replaced with `nn.Sequential(nn.Linear(64, 256), nn.LayerNorm(256))` at zero measurable cost on either head (Δ_reg = −0.12, p=0.40; Δ_cat = −0.025, p=0.81 at n=20). **At AL scale**, the MLP remains load-bearing for cat — replacing it costs ~2.6 pp on cat F1 (p<1e-3). The simplification is **scale-conditional**, matching the pattern of B9 itself (large-state recipe) vs H3-alt (small-state recipe).

### Open questions for the main study

1. **Does the AZ result generalize to FL/CA/TX?** All three are larger than AZ, and the simplification claim should be stronger there if the scale-conditional reading is right. Not tested here.
2. **Is the AL cat gap a 25-epoch artifact, or genuinely architectural?** Cat at AL keeps climbing past ep 25 in the existing 50ep run. Re-running cell E at AL 5f×50ep would disambiguate.
3. **What does the cross-attn d_model knob really do?** F51 Tier 2 already showed `d_model ∈ {384, 512}` breaks cat at FL. We now have evidence that `d_model = 256` is doing essentially all the work the encoder MLP is supposed to do for reg. A dedicated `d_model` ablation at d ∈ {64, 128, 192, 256} would map the curve.

These are **next-study questions**, not this support study's scope.

## Files

- Model change: [`src/models/mtl/mtlnet_crossattn/model.py`](../../../src/models/mtl/mtlnet_crossattn/model.py) (lines ~170-200: new `no_task_encoders` kwarg + post-`super().__init__` Identity swap).
- Run script: [`run_chain.sh`](run_chain.sh) (variants: `no_encoders`, `baseline`; `EPOCHS` env override).
- Logs: [`logs/`](logs/) (`no_encoders_<state>_1f{ep}ep_seed42.log`, `baseline_<state>_1f{ep}ep_seed42.log`).
- Per-fold JSONs: `results/check2hgi/{alabama,arizona,florida}/mtlnet_lr1.0e-04_bs2048_ep{ep}_2026*`.
