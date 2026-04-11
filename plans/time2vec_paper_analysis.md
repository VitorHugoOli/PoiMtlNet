# Time2Vec — paper vs our implementation, deep dive

**Status:** active reference document — read this before touching anything in
`research/embeddings/time2vec/`.
**Last updated:** 2026-04-11
**Scope:** critical comparison between the original paper, the canonical
community PyTorch implementation, and our migrated implementation. Identifies
deviations, judges whether each is justified, and points at the next change
worth trying.

---

## TL;DR

- The **mathematical core** (`t2v` function) in `research/embeddings/time2vec/model/activations.py` is **byte-for-byte identical** to the canonical reference implementation [`ojus1/Time2Vec-PyTorch`](https://github.com/ojus1/Time2Vec-PyTorch), which is itself a faithful PyTorch port of the paper's Eq. (1). This part is **solid and pinned by 44 equivalence tests** in `tests/test_embeddings/test_time2vec.py`.

- We deviate from the paper in **6 places**, ranked by severity:
  1. 🔴 **Self-supervised contrastive training instead of end-to-end supervised** — biggest departure, has structural issues, but the embedding still works downstream.
  2. 🟡 Explicit `Linear(64, 64)` projector on top of t2v (not in paper).
  3. 🟡 `F.normalize` (L2 unit norm) on the output (not in paper).
  4. 🟡 `out_features=64` instead of paper's `k+1=65` (off-by-one, cosmetic).
  5. 🟢 `in_features=2` instead of scalar τ — sound generalization for `(hour/24, dow/7)`.
  6. 🟢 Output ordering: linear term last instead of first (matches `ojus1` reference).

- **The downstream MTLnet next-task F1 on Alabama is 15.34**, above the original notebook's 14.54 — so the deviations don't break the embedding in practice.

- **The highest-leverage improvement** is to fix the contrastive training scheme to sample pairs by **feature-space distance** in `(hour, dow)` space instead of by absolute-time gap. This eliminates a structural conflict in the current sampler. Estimated cost: ~20-line change to `dataset.py`. Should be A/B tested on Alabama.

---

## What the paper actually says

Paper: **"Time2Vec: Learning a Vector Representation of Time"**, Kazemi et al., arXiv:1907.05321, July 2019.

### Eq. (1) — the formal definition

```
                ┌ ω_i · τ + φ_i,         if i = 0          (linear, 1 dim)
  t2v(τ)[i] = ─┤
                └ F(ω_i · τ + φ_i),      if 1 ≤ i ≤ k     (periodic, k dims)
```

- **τ is a scalar** (Section 3, Background & Notation: *"We use τ to represent a scalar notion of time (e.g., absolute time, time from start, time from the last event, etc.) and τ to represent a vector of time features."*).
- **`ω_i, φ_i` are scalars** — k+1 frequencies and k+1 phase shifts. They are **learnable** (the paper shows in Section 5.4 that learning > fixing).
- **Output is k+1 dim**, with the **linear term at index 0**.
- **F = sin** in the main experiments. Footnote 2: *"Using cosine (and some other similar activations) results in an equivalent representation."*
- **k = 64** in the main experiments (Section 5.3: *"We fixed the length of the Time2Vec to 64+1, i.e., 64 units with a non-linear transformation and 1 unit with a linear transformation"*).
- The **linear term `i=0`** captures non-periodic time progression (helps **extrapolation** to future times). Section 5.6 ablates this and shows the linear term matters for some datasets.
- **Invariance to time rescaling** (Proposition 1): if you scale τ by α, there exists another model in the same class that behaves on `α·τ` the same way the original behaves on `τ`. The proof relies on absorbing α into the learnable `ω_i`.

### Training methodology (paper)

The paper trains Time2Vec **end-to-end with a downstream supervised model**. There is no self-supervised pretraining. The downstream task supplies the loss; the t2v parameters get gradients from the downstream loss; the frequencies the model learns are the ones that are useful for that downstream task.

Examples in the paper:
- **LSTM+Time2Vec**: replace τ in the LSTM input with `t2v(τ)`, train end-to-end on classification.
- **TLSTM1+Time2Vec, TLSTM3+Time2Vec**: same idea, replace the time-aware components.
- All experiments use end-to-end supervised training with the downstream loss.

The paper **never** does contrastive learning, never has a "Time2Vec embedding" produced separately and consumed downstream as a precomputed parquet file.

### Other key facts from the paper

| | Paper |
|---|---|
| Periodic activation | sin (cos equivalent) |
| Number of components k | 64 (main experiments) |
| Linear term position | i = 0 (first) |
| Learning vs fixing frequencies | learning beats fixing (Section 5.4, Q5) |
| Importance of linear term | dataset-dependent, helps non-periodic patterns + extrapolation (Section 5.6) |
| Invariance to time rescaling | proven (Proposition 1, Appendix D) |
| L2 normalization of output | **never mentioned** |
| Projection layer on top | **none** — t2v feeds directly into the downstream model whose first layer plays the role of projection (`a = Θ · t2v(τ)`, Section 5.3) |

---

## What the canonical reference (ojus1) does

`ojus1/Time2Vec-PyTorch` is the most-cited community PyTorch implementation. Our notebook (`temp/tarik-new/Time_Encoder.ipynb`) copied its `periodic_activations.py` verbatim.

```python
# https://raw.githubusercontent.com/ojus1/Time2Vec-PyTorch/master/periodic_activations.py
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w  = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b  = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f  = torch.sin
    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

# Example from the same file:
sineact = SineActivation(1, 64)
print(sineact(torch.Tensor([[7]])).shape)   # ← scalar τ via in_features=1
```

ojus1's `experiment.py` then trains the model with **`CrossEntropyLoss`** on a toy 0/1 classification task (input integers, label = "is multiple of 7"), `bs=2048, lr=1e-3, 100 epochs`. This is faithful to the paper.

Two non-paper conventions ojus1 introduced (and we inherited):
- The linear term is **concatenated last** instead of placed at index 0 (`cat([v1, v2])`). Doesn't matter mathematically.
- `tau @ w` is a matrix multiply, which **silently allows** `in_features > 1`. The example uses `in_features=1`, but the formula generalizes.

---

## What our implementation does

### Files

```
research/embeddings/time2vec/
├── time2vec.py             ← create_embedding(), train_epoch(), CLI
├── model/
│   ├── activations.py      ← SineActivation, CosineActivation, t2v
│   ├── Time2VecModule.py   ← Time2VecContrastiveModel (encode, forward, loss)
│   └── dataset.py          ← TemporalContrastiveDataset (pair generation)
```

### Wrapper architecture

```
(hour/24, dow/7)              ← in_features = 2 (NOT scalar)
       │
       ▼
  SineActivation(2, 64)        ← cat([sin(τ@W+b)|63d, τ@w₀+b₀|1d]) = 64d
       │
       ▼
  Linear(64, 64)               ← projector (NOT in paper)
       │
       ▼
  F.normalize(dim=-1)          ← L2 unit norm (NOT in paper)
       │
       ▼
  64-d temporal embedding
```

### Training

`TemporalContrastiveDataset` generates ~1.79M `(t_i, t_j, label)` pairs from the check-in stream:
- **positive** (label=1) if `|t_i - t_j| ≤ r_pos_hours = 1.0`
- **negative** (label=0) if `|t_i - t_j| ≥ r_neg_hours = 24.0`
- Pair multiplicities are capped: `k_neg_per_i=5`, `max_pos_per_i=20`, `max_pairs=2_000_000`

The training loss is `BCEWithLogitsLoss(cosine_sim(z_i, z_j) / τ, label)` with `τ=0.3`. **This is self-supervised contrastive — not in the paper.**

After training the model is **frozen** and used to embed every check-in's `(hour/24, dow/7)`. The result is saved as `output/time2vec/{state}/embeddings.parquet` and consumed by MTLnet's next-task pipeline as **precomputed features** (not a learned layer).

---

## Six deviations, ranked

### 🟢 D1 — Output ordering: periodic before linear

| | Paper | ojus1 | ours |
|---|---|---|---|
| Linear term index | 0 (first) | last (`cat([v1, v2])`) | last |

Cosmetic. Downstream layers are init-permutation invariant, so no gradient or expressiveness difference. Matches the canonical community convention. **No fix.**

### 🟢 D2 — `in_features = 2` (cyclic features instead of scalar τ)

| | Paper | ours |
|---|---|---|
| Input | scalar τ | 2-vector `(hour/24, dow/7)` |
| Periodic weight shape | scalar `ω_i` per unit | matrix `W ∈ ℝ^(2 × 63)` |

Each periodic unit becomes `sin(α_i·hour + β_i·dow + b_i)`. This lets the model represent **diagonal periodicities** ("Friday evenings", "Sunday mornings") that a single scalar τ can't.

This is a **strict generalization** of the paper's formula (the paper is the special case `in_features=1`). It's also exactly what people typically do when applying Time2Vec to multiple time features. **Sound and justified.**

⚠️ Caveat: with cyclic inputs, the **paper's "linear term enables extrapolation" property is lost** because hour/24 and dow/7 are bounded. See D4.

### 🟡 D3 — Explicit `Linear(64, 64)` projector

The paper has nothing on top of t2v — the downstream model's first layer plays that role implicitly:
> *"the output of the first layer, after the Time2Vec transformation and before applying a non-linear activation, has the form `a = Θ · t2v(τ)`, where Θ is the first layer weights matrix"* — Section 5.3

In our setup the embedding is **standalone** (precomputed parquet → MTLnet), so we need an explicit projection space. The projector is essentially the would-be downstream first-layer baked into the embedding step. **Justified given the goal.**

If we ever switch to end-to-end joint training with MTLnet (see "Recommendations"), the projector becomes redundant and should be removed.

### 🟡 D4 — `F.normalize` (L2 unit norm)

This one needs care.

The paper's Section 5.6 argues that the **linear term is important** because it captures non-periodic time progression and **enables extrapolation to future times**. After L2 normalization, the linear term's monotonic growth is renormalized away — so for **scalar absolute-time** inputs, L2 normalization would destroy a key paper claim.

**However**: our input is `(hour/24, dow/7)` ∈ [0, 1)². These features are **cyclic and bounded**. The linear term `τ @ w₀ + b₀` is bounded too, and it's a learned affine map of cyclic inputs — it does **not** grow with absolute time and we have **no extrapolation expectation** anyway (a check-in next year still has `hour/24 ∈ [0, 1)`).

So L2 normalization is **harmless in our setting** because we already gave up on extrapolation by feeding cyclic features. **Justified, but contingent.**

⚠️ **If a future change ever switches the input to raw timestamps or "hours since epoch", the L2 normalize must be removed**, otherwise extrapolation breaks.

### 🟡 D5 — `out_features = 64` (paper used 65)

| | k_periodic | k_linear | total |
|---|---|---|---|
| Paper main | 64 | 1 | **65** |
| ojus1 example | 63 | 1 | 64 |
| Our notebook source | 127 | 1 | 128 |
| Our current default | 63 | 1 | **64** |

Tiny capacity difference. The user explicitly chose `out_features=64` for the current experiment (was 128 in the notebook). To exactly match the paper main experiments, pass `--out_features 65`. **No fix needed**, but worth a one-line note in the README/`PAPER_COMPARISON`.

### 🔴 D6 — Self-supervised contrastive training instead of end-to-end supervised

**This is the largest and most important deviation.** The paper trains Time2Vec **end-to-end with a downstream supervised model**. We train it in **isolation** with a self-supervised contrastive loss over time-pair similarity. There are **three concrete structural issues** with the current sampler.

#### Issue A — irreducible loss floor

The pair generator uses **absolute-time gap**:
```python
left  = np.searchsorted(times_sorted, t_i - r_pos, side="left")
right = np.searchsorted(times_sorted, t_i + r_pos, side="right")
```

Two check-ins at "Mon 2 PM, Jan 1" and "Mon 2 PM, Jan 8" have:
- **identical input features** (same hour, same dow) → identical t2v output → identical embedding → cosine sim = 1
- **labelled as negatives** in the dataset (168 hours apart > `r_neg = 24`)

The model is asked to make identical inputs have *low* similarity. This is physically impossible — the only way to comply is to make ALL embeddings less peaky, raising loss everywhere. The Alabama loss plateau at ~0.215 (instead of approaching 0) is partly explained by this conflict.

#### Issue B — partly redundant with the inductive bias

Two check-ins at "Mon 2:00 PM" and "Mon 2:30 PM" are labelled positive AND have very similar inputs. A randomly initialized Time2Vec already gives them similar embeddings (because `sin` is continuous and the inputs are close). Most of the gradient signal is teaching the network something it would do anyway.

#### Issue C — no task signal → no incentive to learn task-relevant frequencies

The paper's central message (Section 5.4): **learned frequencies > fixed frequencies** because the downstream loss tells the model which frequencies matter. Our loss only knows "close vs far in absolute time", which biases the model toward learning **low frequencies that match the dataset's typical inter-checkin gap**, not (e.g.) weekly periodicity that might matter for POI category prediction.

#### Why does it still work?

Downstream MTLnet next-task macro F1 on Alabama: **15.34** (best of all variants tested), above:
- the original notebook embedding (14.54)
- the bs=256 baseline subset (13.77)

The reason: **the inputs themselves carry the useful signal**. `(hour/24, dow/7)` already encodes everything MTLnet needs about time-of-day and day-of-week. Time2Vec is acting as a *learned smoothing* over those two features, and even a noisy training signal is enough to learn a reasonable smooth manifold.

**Hypothesis worth testing**: a **randomly initialized** Time2Vec (zero training, just frozen random weights) might give downstream F1 within 1pp of the trained version. If true, that would confirm the embedding works *in spite of* the training scheme, not *because of* it.

---

## Recommendations (in priority order)

### Recommendation 1 — Implement **feature-space contrastive sampling** (Option B)

**Change**: in `dataset.py`, replace the absolute-time pair generation with one that samples positive/negative pairs by **Cartesian distance in `(hour/24, dow/7)` space**.

**Why**: eliminates Issue A (no more "identical inputs labelled as negatives") and Issue C (the sampler now defines "similar" in terms of the actual feature space the model sees). Keeps the standalone-pretraining paradigm.

**Concrete sketch**:
```python
# Instead of: time_hours, r_pos_hours, r_neg_hours
# Use: feat_space_dist, r_pos_feat, r_neg_feat
# Where feat_space_dist(i, j) = ||(hour_i/24, dow_i/7) - (hour_j/24, dow_j/7)||_2
# (or use circular distance if you want to model wraparound)
```

**Cost**: ~20–30 line change to `dataset.py`. The training loop and model are unchanged.

**Verification**:
1. Run all 44 tests in `tests/test_embeddings/test_time2vec.py` (most should still pass; a few `TestTemporalContrastiveDataset` tests may need adjustment).
2. Train Alabama with new sampler, compare loss trajectory to current (expect lower final loss because Issue A is gone).
3. Run downstream MTLnet next-task on Alabama, compare F1 to current 15.34.

**Acceptance criterion**: downstream F1 on Alabama not worse than 15.0 (i.e. no quality regression).

### Recommendation 2 — Add a **random-init ablation** as a regression baseline

Train MTLnet next-task on Alabama using a Time2Vec embedding produced from a **frozen randomly initialized** model (no training at all). Compare downstream F1 to the trained-contrastive variant.

**Why**: tells us how much of our F1 actually comes from training vs. how much comes from the inductive bias of `sin(α·hour + β·dow + b)` with random `(α, β)`. If random ≈ trained, the contrastive training is approximately useless and we should drop it (or fix it).

**Cost**: trivial — disable training and call `model.eval()` immediately after init.

### Recommendation 3 — Investigate **end-to-end joint training** with MTLnet (Option A)

Make Time2Vec a learned layer **inside** MTLnet's input pipeline instead of a precomputed parquet. The frequencies get gradients from the MTLnet loss directly. This is the paper-faithful approach.

**Why**: highest expected upside if you want the embedding to be optimal for the actual downstream task.

**Cost**: medium-to-high. Requires changes to:
- `src/data/inputs/builders.py` (the time features need to flow through to training, not be pre-embedded)
- `src/data/folds.py` (same)
- `src/models/mtlnet.py` or wherever the input is consumed (add the t2v layer)
- The MTLnet input contract everywhere

**Risk**: harder to compose with other engines (HGI, Space2Vec, etc.) that expect pre-embedded features. Probably needs a separate "live time2vec" code path that coexists with the precomputed one.

**Defer until**: Recommendation 1 has been tried and we have clean numbers comparing standalone vs. joint training.

### Recommendation 4 — Document `out_features` more clearly

Tiny housekeeping. Add a one-line note in the README and `--help` text:

> The paper's main experiments use `k=64` periodic + 1 linear = 65-dim output. We default to 64-dim total (63 periodic + 1 linear) to match the experimental setup; pass `--out_features 65` to exactly match the paper's main configuration.

---

## What's already in the repo (don't redo this)

### Pinned equivalence with the original notebook

`tests/test_embeddings/test_time2vec.py` contains **44 tests**, all passing, that pin the migration to the original `Time_Encoder.ipynb`. The strongest are:

- `TestWeightTransferEquivalence`: inlines `_OriginalModel` (a verbatim copy of the notebook's `Time2VecPeriodicContrastiveModel`) and verifies that copying weights between models gives **bit-identical** outputs from `encode()`, `forward()`, `contrastive_loss()`.
- `TestTrainingStepEquivalence`: starts both models from the same weights, runs 10 Adam steps with the same batches, asserts every parameter still matches afterward.

**Implication**: the migration is provably correct as a port. Any future change to the model architecture or training loop should **first verify these tests still pass**, or explicitly mark them as superseded (e.g., if Recommendation 1 changes the dataset format, the `TestTemporalContrastiveDataset` group will need updates while the `TestWeightTransferEquivalence` group remains intact).

### Already-applied speed optimizations (4 commits on main)

| commit | what | speedup | quality |
|---|---|---|---|
| `a803d09` | manual slicing + `bs=2048` (no DataLoader) | 2.92× | bit-identical loss at 2 ep |
| `16c982f` | optional `--compile` flag | +10% | bit-identical loss |
| `5f293ff` | default device CPU → MPS auto-detect | +2.4× | bit-identical |
| `7fca270` | README documenting all of the above | docs only | n/a |

Stacked Alabama 100-epoch run: **18:33 → 2:20 (7.95×)** with `+2.5%` final-loss drift but **same downstream F1**.

The bottleneck pre-optimization was Python overhead (DataLoader iteration, op dispatch, tqdm) which dominated for this 4,352-parameter model. The MPS-vs-CPU story flipped between the old and new code paths because the old path launched 6,999 small kernels per epoch (bad on MPS) while the new path launches 219 larger kernels (good on MPS).

See `research/embeddings/time2vec/README.md` for the user-facing documentation and `scripts/profile_time2vec.py` + `scripts/bench_time2vec_configs.py` for the benchmark tools.

### Downstream MTLnet F1 reference

| Variant | Macro F1 (Alabama next-task, 20 ep × 2 folds) |
|---|---|
| Earlier `ours_sub` (bs=256 baseline subset, 93,402 rows) | 13.77 |
| Earlier `ref` (original notebook embedding, 93,402 rows) | 14.54 |
| Optimized run (bs=2048 + compile, full 113,753 rows) | **15.34** |

The new fast pipeline produces the **best downstream F1 of any Time2Vec variant tested** on Alabama. Use this as the regression baseline for any future change.

---

## Open questions / things future Claude should investigate

1. **Does a random-init Time2Vec match a trained one downstream?** (Recommendation 2 — easy to test, would tell us how much of F1 actually comes from training.)
2. **Does feature-space contrastive sampling improve loss trajectory and downstream F1?** (Recommendation 1.)
3. **Does end-to-end joint training with MTLnet beat both?** (Recommendation 3.)
4. **What does the model actually learn?** It would be useful to plot the learned frequencies (`W` rows) and phases (`b`) and see if they cluster around weekly (`2π/7`) or daily (`2π`) periods. Section 5.2 of the paper shows the toy dataset learns a clear `2π/7` peak. If our learned frequencies look random, that's evidence the contrastive loss isn't shaping them toward useful periods.
5. **What happens at `r_pos_hours = 0.5` or `2.0`?** The current default of 1h is arbitrary. Ablating this would tell us how sensitive the embedding is to the threshold.
6. **What if we use circular distance instead of Euclidean for `(hour/24, dow/7)`?** Hour 23 and hour 0 are 1 hour apart in real time but `|23-0|/24 = 23/24` apart in our linear feature. A circular embedding (`sin(2πhour/24), cos(2πhour/24)`) might be a better preprocessing step before t2v.

---

## File map for future work

| Path | Purpose | Touch when... |
|---|---|---|
| `research/embeddings/time2vec/model/activations.py` | `t2v` function, SineActivation, CosineActivation | changing the core formula (don't, it's pinned by tests) |
| `research/embeddings/time2vec/model/Time2VecModule.py` | `Time2VecContrastiveModel` (encode, forward, loss) | changing the projector or normalization |
| `research/embeddings/time2vec/model/dataset.py` | `TemporalContrastiveDataset`, pair generator | implementing Recommendation 1 (feature-space sampler) |
| `research/embeddings/time2vec/time2vec.py` | `create_embedding`, `train_epoch`, CLI | changing training loop, hyperparameters, device handling |
| `research/embeddings/time2vec/README.md` | User-facing run guide + decision log | adding new flags, changing defaults |
| `pipelines/embedding/time2vec.pipe.py` | Multi-state pipeline wrapper | changing the production config |
| `tests/test_embeddings/test_time2vec.py` | 44 equivalence + behavior tests | making any non-equivalent change to model or dataset (mark old tests as obsolete, add new ones) |
| `scripts/profile_time2vec.py` | Single-epoch breakdown | diagnosing a slowdown |
| `scripts/bench_time2vec_configs.py` | A/B over batch size, workers, manual-vs-DataLoader | trying a new optimization |
| `scripts/compare_time2vec_alabama.py` | A/B MTLnet next-task on multiple embedding variants | comparing the downstream impact of an embedding change |

---

## Sources

### Paper

- **Time2Vec: Learning a Vector Representation of Time** — Seyed Mehran Kazemi, Rishab Goel, Sepehr Eghbali, Janahan Ramanan, Jaspreet Sahota, Sanjay Thakur, Stella Wu, Cathal Smyth, Pascal Poupart, Marcus Brubaker (Borealis AI). arXiv:1907.05321, July 2019. <https://arxiv.org/abs/1907.05321>
- PDF used during this analysis: <https://arxiv.org/pdf/1907.05321> (downloaded into `webfetch-1775881535942-r2l30y.pdf` during the session — the relevant pages are 1–9).
- ICLR 2020 OpenReview thread (rejected from ICLR but became widely used as a reference): <https://openreview.net/forum?id=rklklCVYvB>

Specific claims used in this document, with paper citations:
- Eq. (1) — Section 4 ("Time2Vec"), page 3.
- "τ to represent a scalar notion of time" — Section 3 ("Background & Notation"), page 2.
- "We chose F to be the sine function in our experiments but we do experiments with other periodic activations as well" — Section 4, page 3.
- "Using cosine (and some other similar activations) results in an equivalent representation" — Footnote 2, page 3.
- "We fixed the length of the Time2Vec to 64+1, i.e., 64 units with a non-linear transformation and 1 unit with a linear transformation" — Section 5.3 ("Other activation functions"), page 6.
- "the output of the first layer, after the Time2Vec transformation and before applying a non-linear activation, has the form `a = Θ · t2v(τ)`" — Section 5.3, page 6.
- "learning the sine frequencies and phase-shifts of Time2Vec from the data offer any advantage compared to fixing them" — Section 5.4 ("Fixed frequencies and phase-shifts"), page 7.
- "Time2Vec is invariant to time rescaling" — Proposition 1, Section 4, page 3 (proof in Appendix D).
- Linear term ablation — Section 5.6 ("Why capture non-periodic patterns?"), page 8 + Figure 5(d).

### Reference implementations

- **`ojus1/Time2Vec-PyTorch`** — the canonical community PyTorch port; what our notebook copied: <https://github.com/ojus1/Time2Vec-PyTorch>
  - `periodic_activations.py` (the SineActivation/CosineActivation our code mirrors): <https://raw.githubusercontent.com/ojus1/Time2Vec-PyTorch/master/periodic_activations.py>
  - `experiment.py` (showing the supervised CrossEntropyLoss training): <https://raw.githubusercontent.com/ojus1/Time2Vec-PyTorch/master/experiment.py>
- **`gdetor/pytorch_time2vec`** — alternative PyTorch port: <https://github.com/gdetor/pytorch_time2vec>
- **`andrewrgarcia/time2vec`** — Keras + PyTorch layers: <https://github.com/andrewrgarcia/time2vec>
- **Author blog post by Surya Kant Sahu (ojus1) walking through the formula**: <https://ojus1.github.io/posts/time2vec/>
- **Time2Vec for Time Series features encoding** (Towards Data Science overview): <https://towardsdatascience.com/time2vec-for-time-series-features-encoding-a03a4f3f937e/>

### Internal references

- Original notebook: `temp/tarik-new/Time_Encoder.ipynb` (outside the repo).
- Migrated implementation: `research/embeddings/time2vec/`.
- Equivalence tests: `tests/test_embeddings/test_time2vec.py` (44 tests, all passing).
- User-facing run guide: `research/embeddings/time2vec/README.md`.
- Speedup history: commits `a803d09`, `16c982f`, `5f293ff`, `7fca270` on `main`.
- Earlier MTLnet A/B between our embedding and the reference's time_embedding.csv: `scripts/compare_time2vec_alabama.py` and `results/time2vec/alabama_ours_sub/`, `results/time2vec/alabama_ref/`.
