# Sphere2Vec — paper vs notebook discrepancy analysis

**Author:** Claude Opus 4.6 (analysis from 2026-04-11 session)
**Audience:** Future Claude Code session (or human) considering changes to `research/embeddings/sphere2vec/`
**Scope:** Architectural correctness. Documents that the notebook our migration is based on does NOT implement the paper's `sphereM` variant, despite being labeled as such. Recommends concrete actions and provides everything you need to make an informed decision before touching the code.
**Repo root:** `/Users/vitor/Desktop/mestrado/ingred`
**Target venv:** `.venv_new` (Python 3.12, PyTorch 2.9.1, MPS on Apple Silicon)
**Related plans:** `mtlnet_speed_optimization.md`, `todo.md`

---

## 0. TL;DR — read this first

The notebook at
`temp/tarik-new/Location Encoders/A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions (Sphere2Vec-sphereM).ipynb`
**is mislabeled**. It calls itself "Sphere2Vec-sphereM" but the architecture inside has **no relationship** to what Mai et al. (2023) define as sphereM in their paper.

- **Paper sphereM (Equation 8)**: closed-form sinusoidal multi-frequency encoder with output dim `5·S` (e.g. 160 for S=32). Deterministic, no learned anchors.
- **Notebook "sphereM"**: random unit-3D-sphere centroids + multi-scale RBF kernel with output dim `num_centroids · num_scales` (e.g. 256·32 = 8192). Frozen random buffers.

These are **fundamentally different architectures**, not minor implementation variations. The notebook also uses a **different training objective** (self-supervised contrastive on spatial pairs) than the paper (end-to-end with a downstream classifier + weighted cross-entropy).

Our migration faithfully ports the notebook (bit-equivalent under fixed seed, validated by `tests/test_embeddings/test_sphere2vec.py::test_per_poi_embeddings_match_notebook`). So the migration is correct *with respect to the source it was given* — the source itself is the problem.

**Recommended action:** at minimum, update package documentation to make the discrepancy explicit (Action A below). Consider implementing the actual paper variant as a parallel option (Action B). Do NOT silently replace the existing implementation (Action C — explicitly rejected).

---

## 1. The architectural mismatch — mathematical proof

### 1.1 Paper sphereM (Equation 8 in Mai et al. 2023)

```
PE^sphereM_S(x) = ⋃_{s=0}^{S-1} [
    sin φ^(s),
    cos φ^(s) · cos λ,
    cos φ      · cos λ^(s),
    cos φ^(s) · sin λ,
    cos φ      · sin λ^(s)
]
```

Where:
- `φ` = latitude (radians), `λ` = longitude (radians)
- `(s)` notates the per-scale frequency multiplier
- `S` = number of scales
- The 5-term inner bracket is concatenated across all `S` scales

**Output dimensionality: 5·S** (typically `5·32 = 160` or `5·64 = 320`).

The encoder is **fully deterministic** — no randomness, no learned buffers in the position encoding step. The motivation is that these 5 trigonometric products preserve great-circle distance between points on the sphere. The paper proves this in §3.

### 1.2 Official code (`SphereMixScaleSpatialRelationEncoder`)

The official repository (`gengchenmai/sphere2vec`) maps the paper variants to code class names in its README:

| Paper variant | Code class |
|---|---|
| Sphere2Vec-sphereC | `SphereSpatialRelationEncoder` (`sphere`) |
| Sphere2Vec-sphereC+ | `SphereGirdSpatialRelationEncoder` (`spheregrid`) |
| **Sphere2Vec-sphereM** | **`SphereMixScaleSpatialRelationEncoder` (`spheremixscale`)** |
| Sphere2Vec-sphereM+ | `SphereGridMixScaleSpatialRelationEncoder` (`spheregridmixscale`) |
| Sphere2Vec-dfs | `DFTSpatialRelationEncoder` (`dft`) |

The relevant class implements exactly Equation 8:

```python
# from main/SpatialRelationEncoder.py — verbatim from the official repo
spr_embeds_ = np.concatenate([
    lat_sin,                       # sin(lat * freq)        ← sin φ^(s)
    lat_cos * lon_single_cos,      # cos(lat*freq)*cos(lon) ← cos φ^(s) cos λ
    lat_single_cos * lon_cos,      # cos(lat)*cos(lon*freq) ← cos φ cos λ^(s)
    lat_cos * lon_single_sin,      # cos(lat*freq)*sin(lon) ← cos φ^(s) sin λ
    lat_single_cos * lon_sin       # cos(lat)*sin(lon*freq) ← cos φ sin λ^(s)
], axis = -1)
```

This is the **canonical sphereM implementation**, written by the paper's first author. It is purely sinusoidal, deterministic, and outputs `5 × frequency_num` features per point.

### 1.3 What the notebook actually implements (= our migration)

`research/embeddings/sphere2vec/model/Sphere2VecModule.py::SpherePositionEncoder`:

```python
def __init__(self, min_scale=1, max_scale=1000, num_scales=16, num_centroids=128, ...):
    # 256 RANDOM unit-3D-sphere centroids — frozen buffer
    centroids = torch.randn(num_centroids, 3)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)
    self.register_buffer('centroids', centroids)
    # 32 log-spaced RBF scales — frozen buffer
    scales = torch.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    self.register_buffer('scales', scales)

def forward(self, coords):
    # 1. lat/lon → 3D Cartesian on unit sphere
    lat_rad = deg2rad(coords[..., 0]); lon_rad = deg2rad(coords[..., 1])
    x = cos(lat_rad)*cos(lon_rad); y = cos(lat_rad)*sin(lon_rad); z = sin(lat_rad)
    input_vec = torch.stack([x, y, z], dim=-1)
    # 2. Cosine similarity vs random centroids (= dot product on unit sphere)
    dot_product = input_vec @ centroids.T          # [B, 256]
    # 3. Multi-scale RBF kernel
    distance = 1.0 - dot_product
    weighted_dist = distance.unsqueeze(-1) * scales.view(1,1,-1)
    rbf_feat = torch.exp(-weighted_dist)           # [B, 256, 32]
    # 4. Flatten
    return rbf_feat.flatten(1)                     # [B, 256*32 = 8192]
```

**Output dimensionality: `num_centroids × num_scales` = 256 × 32 = 8192.** 51× larger than the paper's sphereM at S=32.

This is a **multi-scale Random Basis Network on the unit 3-sphere**. It is closer in spirit to:
- Random Fourier Features (Rahimi & Recht 2007)
- Random Kitchen Sinks
- A frozen Gaussian Process kernel approximation
- An RBF Network with random sphere-distributed centers

…than to anything in the Sphere2Vec paper.

### 1.4 Side-by-side

| Aspect | Paper sphereM (Eq 8) | Notebook ("sphereM") | Match? |
|---|---|---|---|
| Output dim | `5·S` (e.g. 160 at S=32) | `num_centroids·num_scales` = 8192 | ❌ |
| Basis | Sinusoidal `sin/cos(angle·freq)` × 5 specific terms | Random unit centroids + RBF kernel `exp(−dist·scale)` | ❌ |
| Input transform | (lat, lon) → radians directly | (lat, lon) → 3D Cartesian on unit sphere | ❌ |
| Random / learned | None — deterministic | 256 random unit centroids (frozen buffer) | ❌ |
| Frequency role | Multiplies the angle inside sin/cos | Multiplies the cosine distance inside exp() | ❌ |
| Source in repo | `SphereMixScaleSpatialRelationEncoder` | **does not appear in the official repo** | ❌ |

I checked **all 17 classes** in `main/SpatialRelationEncoder.py` (`Sphere*`, `Grid*`, `Theory*`, `Naive*`, `XYZ*`, `NERF*`, `RBF*`, `DFT*`, `RFF*`, `AodhaFFT*`). **None** combines lat/lon → 3D Cartesian → cosine similarity vs random unit centroids → multi-scale `exp(−distance·scale)`. The notebook's encoder is a custom architecture invented somewhere outside the Sphere2Vec paper.

### 1.5 Note on `RBFSpatialRelationEncoder` (separate, also different)

The official repo does have an `RBFSpatialRelationEncoder` class. It is **not** what the notebook implements either:
- It uses 2D Euclidean RBF (not sphere-based).
- Anchor points are sampled either from training data ("global" mode) or uniformly from the `max_radius` box ("relative" mode), not random unit-3D-sphere centroids.
- Single kernel size, not multi-scale.

So the notebook's encoder is novel — it's not a misimplementation of any specific class in the official repo, it's a custom design.

---

## 2. The training-objective mismatch (separate issue, compounds the first)

### 2.1 Paper training (§4 of the paper)

```
ℒ^image(𝕏) = Σ β·log(σ(Enc(𝐱)·𝐓_{:,y}))
            + Σ log(1 − σ(Enc(𝐱)·𝐓_{:,i}))
            + Σ log(1 − σ(Enc(𝐱⁻)·𝐓_{:,i}))
```

The position encoder `Enc()` is trained **end-to-end with a downstream classifier `T`** (image classifier for Flickr / iNaturalist / fMoW). The encoder never exists as a standalone "feature extractor" — it is always optimized for a specific downstream task. There is no self-supervised pretraining stage anywhere in the paper.

### 2.2 Notebook training

```python
# Self-supervised contrastive pretraining (cell 11)
def contrastive_bce(z_i, z_j, label, tau):
    sim = F.cosine_similarity(z_i, z_j)
    return F.binary_cross_entropy_with_logits(sim/tau, label)

# Pairs from ContrastiveSpatialDataset (cell 7):
#   positive (label=1): (coord_i, coord_i + N(0, 0.01°))
#   negative (label=0): (coord_i, random_other_coord)
```

This is a **generic self-supervised contrastive pretraining task** — it teaches "nearby coordinates have similar embeddings". It is mathematically valid as a SSL task, and it is what the dev's notebook does, but **it is not what the paper does**, and it does not appear in the paper.

### 2.3 Implication for our context

The MTLnet pipeline in this repo expects standalone embedding artifacts (a parquet of POI features), not jointly-trained encoders. So we cannot exactly replicate the paper's training procedure inside our pipeline. We have two valid options:

| Option | Description | Closest to the paper? |
|---|---|---|
| **Frozen position encoder + downstream classifier in MTLnet** | Use the paper's deterministic Eq 8 directly with no pretraining; let MTLnet fine-tune the classifier on top. The encoder never trains. | Closest — matches the paper's "position encoder + classifier" pattern, just with the classifier being MTLnet's category head instead of an image classifier. |
| **Self-supervised pretraining of the encoder** | What the notebook does. Pretrain the encoder with some SSL task, then freeze and use as features. | A reinterpretation, not in the paper. |

**The notebook chose option 2 with a custom encoder. Both decisions are off-paper.**

---

## 3. Why this matters (and why it doesn't, completely)

### 3.1 Where it matters

1. **Citing the paper is misleading.** Anyone referencing "Sphere2Vec-sphereM (Mai et al. 2023)" based on this code is wrong — the implementation is unrelated.
2. **Naming consistency.** The package, README, equivalence-test snapshot, and `EmbeddingEngine.SPHERE2VEC` enum all carry the paper's name for an architecture that isn't the paper's.
3. **Hyperparameter intuition transfers wrong.** Numbers like `frequency_num=64`, `min_radius=10`, `max_radius=10000` from the Sphere2Vec paper apply to a 5S-dimensional sinusoidal encoder. Reusing them as `num_scales=32, min_scale=10, max_scale=1e7` for a 256×32 RBF encoder is a category error.
4. **Likely contributes to weak downstream signal.** Empirically, Alabama MTL gets ~13.6% category macro F1 with this encoder (see §5). That's barely above random for 7 classes. The notebook's RBF + contrastive-SSL combination is a much weaker prior than the paper's Eq 8 + task-specific cross-entropy. **A correct paper-sphereM implementation would plausibly do better — but we have not measured this.**

### 3.2 Where it doesn't (necessarily) matter

1. **The notebook architecture is not "wrong" in absolute terms.** Multi-scale RBF on the unit sphere is a legitimate modeling choice (used in geostatistics and spatial ML). It's just not Sphere2Vec.
2. **Our migration is faithful to the source the user gave us.** The bit-equivalence test is meaningful — it validates the port, not the source. The user's original ask was to migrate the notebook, and we did.
3. **The embedding works** (within the limits of the weak signal). Alabama MTL F1 is statistically tied with the dev's own CSV output (see §5). We're not shipping a broken model — we're shipping the dev's model.
4. **Replacing it now would invalidate prior work.** The Alabama checkpoint (`output/sphere2vec/alabama/sphere2vec_model.pt`), the existing input parquets, the MTL results in `results/sphere2vec/alabama/`, and the bit-equivalence test all assume the current architecture.

---

## 4. Three actions, ranked by invasiveness

### Action A — Documentation correction (DEFINITELY worth doing, very low risk)

**What:**
- Update `research/embeddings/sphere2vec/README.md` and `CLAUDE.md` to add a prominent **"⚠️ Discrepancy with the original paper"** section. Reproduce the side-by-side from §1.4 above. Cite the paper (arXiv: 2306.17624) and the official repo. Make crystal clear that this package implements a custom random-RBF-on-sphere variant labeled "sphereM" by the source notebook, NOT Equation 8 from Mai et al.
- Update `research/embeddings/README.md` (the knowledge-transfer doc) to add a paragraph in the sphere2vec section pointing readers at this analysis file.
- Optionally rename internal class name `SpherePositionEncoder` → `SphereRBFPositionEncoder` (or similar) to defuse misleading reads of the code, while keeping `EmbeddingEngine.SPHERE2VEC` and the package directory name unchanged for backward compatibility.

**Effort:** ~15-30 minutes of writing.
**Risk:** Zero — only documentation. No tests change. No retraining.
**Benefit:** Stops the misnaming from propagating into theses, papers, slides, or downstream collaborators' assumptions. Defends against accidentally citing Mai et al. for a different architecture.

**Recommendation:** **Do this regardless of whether you do Action B.** It is the minimum honest correction.

### Action B — Add the real paper sphereM as a parallel encoder (worth doing as a follow-up, moderate effort)

**What:**
- Add `SphereMixScaleEncoder` (or similar — match the paper's name) as a new class in `research/embeddings/sphere2vec/model/Sphere2VecModule.py`. Implement exactly Equation 8: `5·S` sinusoidal basis terms per point. Match the official `SphereMixScaleSpatialRelationEncoder` from `main/SpatialRelationEncoder.py` line by line.
- Add a CLI / Namespace flag like `--encoder-variant {paper, rbf}`. **Default `rbf`** to preserve backward compatibility. Pipeline file (`pipelines/embedding/sphere2vec.pipe.py`) keeps its current default.
- Add unit tests verifying that the new encoder matches Eq 8 on a fixed input batch (no randomness involved — should be a simple closed-form check).
- Run a head-to-head: train both variants on Alabama (50 epochs each), generate `embeddings.parquet`, run `scripts/train.py --task mtl --state alabama --engine sphere2vec` for each, compare cat F1.
- Document the result in `research/embeddings/sphere2vec/README.md`.

**Effort:** ~60-90 min total (30 min implementation + 30 min testing + ~30 min training/MTL each variant).
**Risk:** Low — additive change. Existing tests still pass, existing behavior is the default unless explicitly opted out.
**Benefit:** Gives the user the actual paper architecture as an option. Lets us empirically measure which is better on the user's data. The result is informative regardless of which one wins (a clean negative result is itself a useful finding for a thesis).

**Open question to settle before doing Action B:** what training objective should the paper variant use? Two reasonable choices:
1. **Skip training entirely** — use the deterministic Eq 8 output as features directly, no pretraining. Closest to the paper's design (the paper's encoder is also non-learnable in its position-encoding step; only the FFN above it is learned, and that's done jointly with the downstream task). For our context, the FFN would effectively be MTLnet's category head, so this becomes "no pretraining, just feed Eq 8 features straight into MTLnet".
2. **Reuse the notebook's contrastive task** — pretrain the new encoder with the same SSL loss the notebook uses. Less faithful to the paper but lets us isolate the encoder change from the training-procedure change in the comparison.

I would do **option 1** as the primary comparison and **option 2** as a tie-breaker if option 1 is unclear. Talk to the user before starting.

### Action C — Replace the notebook variant entirely (NOT recommended)

**What:**
- Delete the RBF encoder, replace with the paper's Equation 8 only. Discard the bit-equivalence test (since the source it equates against is the wrong architecture). Update all tests, docs, and the migration story.

**Effort:** ~2 hours including tests + retraining + MTL validation.
**Risk:** **High.** Breaks the migration's bit-equivalence guarantee. Invalidates any artifact the user has from prior runs. Removes a working (if mislabeled) code path. Rewrites history of the migration.
**Benefit:** Makes the package match the paper exactly.

**Recommendation:** **Do not do this.** The user's original ask was a migration of the notebook, and we delivered that with a verified bit-equivalence test. Replacing the encoder retroactively undoes that work. If the user ever wants the paper variant, do Action B and let them opt in.

---

## 5. Empirical context — the F1 numbers we actually have

These are the results from validating the migration on Alabama (113,753 checkins, 11,706 unique POIs) over the 2026-04-10/11 sessions. They serve as a regression baseline and a benchmark for any future "fix":

| Source | cat acc | cat F1 (macro) | next acc | next F1 | total time |
|---|---|---|---|---|---|
| Notebook author's CSV (10,269 POIs) | 29.72% ± 2.31 | 13.80% ± 1.46 | 34.16% ± 0.67 | 7.27% ± 0.11 | n/a |
| **Our migration, notebook-mode bs=64 (canonical baseline)** | **30.64% ± 2.63** | **13.59% ± 0.51** | 34.16% ± 0.67 | 7.27% ± 0.11 | 47 min |
| Our migration, eval-mode bs=64 (deterministic inference) | 31.71% ± 1.13 | 12.35% ± 0.66 | 34.16% ± 0.67 | 7.27% ± 0.11 | 47 min |
| **Our migration, optimized bs=4096 + fast dataset (current default)** | **27.92% ± 1.04** | **13.88% ± 1.55** | 34.16% ± 0.67 | 7.27% ± 0.11 | **14 min** |

Three observations matter for any future change:

1. **Cat F1 is statistically tied across all 4 runs** (12.35% to 13.88%, all within ±1 std). The notebook's RBF encoder produces the same downstream-task signal regardless of inference mode, batch size, or which exact training config we use. The encoder is clearly the bottleneck — the training procedure variations explore noise around the same plateau.

2. **Next-task is byte-equal-dead** in **every fold of every run** (cat F1 = 7.27%, best_epoch = 0). The transformer head extracts zero useful sequence signal from these embeddings. **A correct paper-sphereM implementation might or might not fix this — we haven't measured, and the next-POI task is fundamentally hard with location-only features, so don't expect a miracle.**

3. **The optimized bs=4096 default is 3.34× faster end-to-end** with no F1 regression. See `mtlnet_speed_optimization.md` for context (different work, separate plan).

If Action B is done, the head-to-head should replicate this table with one new column: "paper sphereM, no pretraining, frozen Eq 8 + MTLnet head". Use the same Alabama data, same MTL config (`scripts/train.py --task mtl --state alabama --engine sphere2vec`), same fold seeds.

---

## 6. Other knowledge a future Claude Code session needs to work on this

### 6.1 Where things live

```
research/embeddings/sphere2vec/
├── __init__.py                       # public API exports
├── sphere2vec.py                     # create_embedding(state, args) entry point + CLI
├── README.md                         # package-level docs (perf section, port decisions)
├── CLAUDE.md                         # short agent pointer
└── model/
    ├── __init__.py
    ├── Sphere2VecModule.py           # SpherePositionEncoder (the RBF one!) + everything else
    └── dataset.py                    # ContrastiveSpatialDataset + FastContrastiveSpatialDataset

pipelines/embedding/sphere2vec.pipe.py # multi-state pipeline runner
src/configs/paths.py                  # EmbeddingEngine.SPHERE2VEC + _Sphere2VecIoPath
src/configs/embedding_fusion.py       # POI-level validation list

tests/test_embeddings/
├── _sphere2vec_reference.py          # frozen verbatim notebook snapshot (DO NOT MODIFY)
└── test_sphere2vec.py                # 17 tests (architecture, fast-dataset, e2e equivalence)
```

### 6.2 The frozen reference snapshot

`tests/test_embeddings/_sphere2vec_reference.py` is a **verbatim copy of the notebook's model code**, used as the equivalence oracle. **Do not modify it.** If you want to add the paper's real sphereM as a new variant, add a new class to `Sphere2VecModule.py` and write new tests against the official `SphereMixScaleSpatialRelationEncoder` (which you'd vendor from the official repo into a new fixture file like `_sphere2vec_paper_reference.py`).

### 6.3 Tests to NOT break

Run this before and after any change:
```bash
source .venv_new/bin/activate
python -m pytest tests/test_embeddings/test_sphere2vec.py tests/test_data/ tests/test_configs/ -q
# Expect: 119 passed (or 120+ if you add new tests)
```

The critical bit-equivalence test is
`tests/test_embeddings/test_sphere2vec.py::TestEndToEndPipelineEquivalence::test_per_poi_embeddings_match_notebook`.
It uses `legacy_dataset=True` to exercise the slow per-item path. **It must keep passing** to validate that we still match the notebook source — even if/when we add a new paper-faithful variant in parallel.

### 6.4 Latent infrastructure bugs we found while validating sphere2vec end-to-end

Both are **already fixed on `main`** but worth knowing about because they were dormant for a long time and could resurface in similar ways:

1. **`src/data/folds.py` MTL category split with str placeids** (commit `3380e48` precursor + my fix).
   - **Symptom**: `ValueError: num_samples should be a positive integer value, but got num_samples=0` at fold 1.
   - **Root cause**: `category.parquet` from sphere2vec/space2vec stores `placeid` as `str` (from `.astype(str)` in the encoder code). `build_poi_user_mapping` returns the in-memory mapping with `int` keys (raw checkins are `int64`). `np.isin(str_array, int_set)` returned all-False.
   - **Fix**: defensive str-coercion at the comparison site in `_create_mtl_folds` (~line 606). Both sides cast to str before `np.isin`.
   - **Why it stayed dormant**: nobody had run `--task mtl` against any engine that stored str placeids before sphere2vec.

2. **`src/losses/nash_mtl.py` float64-on-MPS crash** (commit `3380e48` had my initial fix; PR #7 hardened it further).
   - **Symptom**: `TypeError: unsupported operand type(s) for *: 'Tensor' and 'Tensor'` on `losses[i] * alpha[i]`.
   - **Root cause**: `cvxpy.solve()` returns numpy `float64`. `torch.from_numpy(float64_array)` produces a CPU `float64` tensor. MPS does **not support `float64`**. Multiplying against the model's `float32` losses errors out.
   - **Why it stayed dormant**: notebook-mode embeddings produce a "soft" Gram matrix that the cvxpy solver converges in 1 iteration → `_stop_criteria` fires before the float64 reassignment → `prvs_alpha` stays at the constructor's `float32`. Eval-mode (deterministic) embeddings produce a stiffer Gram matrix that needs multiple iterations → bug triggered.
   - **Current state**: PR #7 added solver fallback (ECOS → SCS), explicit `cp.error.SolverError` catching with structured logging, and NaN-detection for degenerate Gram matrices. Original upstream code had a bare `except:` that silently degraded NashMTL into fixed `[1, 1]` weights — that's now caught.

If you implement Action B, **rerun MTL training to verify both fixes are still in place** and that the new encoder doesn't trip a third latent issue.

### 6.5 The two dataset implementations

`research/embeddings/sphere2vec/model/dataset.py` ships **two** dataset classes:

1. **`ContrastiveSpatialDataset`** (legacy, per-item) — bit-equivalent to the notebook. Used by the bit-equality test (`legacy_dataset=True`). Slow.
2. **`FastContrastiveSpatialDataset`** (default, vectorized) — implements PyTorch ≥2.0's `__getitems__` for batched fetch. ~9× faster on MPS at bs=4096. Statistically equivalent to the per-item version.

When using the fast dataset, you **must** pass `collate_fn=_identity_collate` to the `DataLoader` (the `_identity_collate` helper is in `sphere2vec.py`). Otherwise the default collator tries to `torch.stack` the 3 batched tensors against each other and fails on mismatched shapes. This is already handled in `create_embedding`.

### 6.6 Inference modes

`create_embedding(args)` supports two inference modes via `args.eval_inference`:

| Mode | `eval_inference` | Behavior | norms |
|---|---|---|---|
| Notebook-faithful (default) | `False` | `model.train()` + autograd alive — dropout is ACTIVE during inference | per-POI norms 0.6-1.0 (varies) |
| Deterministic fix | `True` | `model.eval()` + `torch.no_grad()` — dropout disabled | all norms = 1.0 |

The notebook-faithful default is empirically slightly better on macro F1 — the dropout noise during inference acts as implicit regularization for the downstream classifier. Documented in `research/embeddings/sphere2vec/README.md` and the bit-equality test enforces the faithful path.

### 6.7 Pitfalls to avoid

1. **Don't "fix" the notebook bug of running inference in `train()` mode without consulting the user.** It's empirically *better* than the eval-mode "fix" on Alabama MTL F1. We tested both. The notebook is bug-but-better.
2. **Don't `git add -A` in this repo.** Multiple agents and processes touch this codebase concurrently. Always stage files explicitly (the `git add file1 file2 file3` pattern in earlier commits). Otherwise, you'll sweep someone else's HGI README or time2vec changes into your commit.
3. **Don't push without explicit user approval.** The user's safety protocol is to commit but not push.
4. **Don't run the full 50-epoch training default unless you need it.** It takes ~25 min at bs=64 (notebook-faithful) or ~88s at bs=4096 (current optimized default). For quick smoke testing, use `epoch=2`.
5. **Don't use `num_workers > 0` with the legacy `ContrastiveSpatialDataset` without a `worker_init_fn`.** The dataset uses the global numpy RNG, and child workers fork-inherit a copy. Without re-seeding, every worker draws the same sequence. `_worker_init_fn` in `sphere2vec.py` handles this.
6. **MPS does not support `float64`**. Any new encoder must produce `float32` outputs and accept `float32` inputs. If you import a class from elsewhere, check its dtype assumptions.
7. **The `ExperimentConfig` factory signature is `default_mtl(name, state, embedding_engine, **overrides)`.** Required positional args. Easy to miss if you copy from older code.
8. **`scripts/train.py --folds 1` triggers a pre-existing framework bug** (`statistics.stdev requires at least two data points` in `src/tracking/display.py:188`). Use `--folds 2` minimum for smoke runs.

### 6.8 The test suite at a glance

`tests/test_embeddings/test_sphere2vec.py` has **17 tests** in 7 classes:

| Class | Tests | What it checks |
|---|---|---|
| `TestSpherePositionEncoderEquivalence` | 2 | Buffers + forward bit-equal vs `_sphere2vec_reference.py` |
| `TestSphereLocationEncoderEquivalence` | 2 | Parameters + forward bit-equal vs reference |
| `TestContrastiveModelEquivalence` | 2 | Parameters + forward (incl. unit-norm assertion) bit-equal vs reference |
| `TestContrastiveBCEEquivalence` | 1 | Loss bit-equal vs reference |
| `TestContrastiveDatasetEquivalence` | 1 | Per-item dataset sample sequence bit-equal vs reference under shared seed |
| `TestFastContrastiveDataset` | 6 | Shape, dtype, anchor preservation, Bernoulli ratio, noise bounds, DataLoader integration |
| `TestCreateEmbeddingSmoke` | 2 | End-to-end on synthetic data: notebook mode + eval mode |
| `TestEndToEndPipelineEquivalence` | 1 | **The big one**: full pipeline bit-equality vs inline notebook port |

If you add a paper-faithful encoder via Action B, mirror this structure: write its own equivalence tests against the official `SphereMixScaleSpatialRelationEncoder` source (which would need to be vendored as a new `_sphere2vec_paper_reference.py` fixture), plus shape/dtype contracts and an end-to-end smoke test.

---

## 7. Recommended sequence if Action A + B are approved

```
1. Edit research/embeddings/sphere2vec/README.md
   → add ⚠️ "Discrepancy with the original paper" section (~50 lines)
   → cite paper (arXiv:2306.17624) and official repo
   → reproduce the 5-row side-by-side from §1.4

2. Edit research/embeddings/sphere2vec/CLAUDE.md
   → add a one-line warning at the top: "this is a custom RBF variant, NOT the paper's Eq 8 sphereM"

3. Edit research/embeddings/README.md
   → in the "What it is" section, add a paragraph noting the discrepancy
   → link to plans/sphere2vec_paper_vs_notebook_analysis.md

4. (optional) Rename SpherePositionEncoder → SphereRBFPositionEncoder inside Sphere2VecModule.py
   → keep package + enum name unchanged for backward compat
   → update tests + reference snapshot? NO — do not touch _sphere2vec_reference.py
   → instead, alias inside the snapshot if needed

5. Run pytest (must still be 119 passing)

6. Commit Action A as a single docs commit with explicit file list

--- pause for user review ---

7. (if approved) Vendor SphereMixScaleSpatialRelationEncoder from the official repo
   → save as tests/test_embeddings/_sphere2vec_paper_reference.py (frozen, do not modify)
   → add SphereMixScaleEncoder to research/embeddings/sphere2vec/model/Sphere2VecModule.py
   → write equivalence tests against the new fixture

8. Add --encoder-variant {paper, rbf} flag to create_embedding (default: rbf)

9. Add SphereMixScaleEncoder export from package __init__ files

10. Run the existing 17 sphere2vec tests + the new ones (should be ~22-25 passing)

11. End-to-end Alabama validation:
    a. Train rbf variant (already done — current state)
    b. Train paper variant (new): regenerate embeddings.parquet + inputs
    c. MTL training on each
    d. Tabulate cat F1, next F1, wall-clock
    e. Document results in README.md

12. Commit Action B as a single feat commit
```

**Estimated total time for Action A + B: ~2-3 hours including the Alabama runs.**

---

## 8. Sources

### Paper
- **arXiv abstract**: [Sphere2Vec: A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions (arXiv:2306.17624)](https://arxiv.org/abs/2306.17624)
- **arXiv HTML version** (used for retrieving Equation 8 + training loss): [arxiv.org/html/2306.17624](https://arxiv.org/html/2306.17624)
- **Published version**: [ISPRS Journal of Photogrammetry and Remote Sensing, vol. 202, pp. 439-462, 2023](https://www.sciencedirect.com/science/article/abs/pii/S0924271623001818)
- **Earlier 2022 version**: [arXiv:2201.10489 — Sphere2Vec: Multi-Scale Representation Learning over a Spherical Surface for Geospatial Predictions](https://arxiv.org/abs/2201.10489)
- **Project page**: [gengchenmai.github.io/sphere2vec-website](https://gengchenmai.github.io/sphere2vec-website/)
- **Authors**: Gengchen Mai, Yao Xuan, Wenyun Zuo, Yutong He, Jiaming Song, Stefano Ermon, Krzysztof Janowicz, Ni Lao

### Code
- **Official repository**: [github.com/gengchenmai/sphere2vec](https://github.com/gengchenmai/sphere2vec)
- **Position encoder file** (contains `SphereMixScaleSpatialRelationEncoder` = the real sphereM): `main/SpatialRelationEncoder.py` (raw URL: `raw.githubusercontent.com/gengchenmai/sphere2vec/main/main/SpatialRelationEncoder.py`)
- **Variant → code-name mapping** (from the repo README):
  - `Sphere2Vec-sphereC` ↔ `sphere`
  - `Sphere2Vec-sphereC+` ↔ `spheregrid`
  - **`Sphere2Vec-sphereM` ↔ `spheremixscale`**
  - `Sphere2Vec-sphereM+` ↔ `spheregridmixscale`
  - `Sphere2Vec-dfs` ↔ `dft`

### The notebook (the source we ported)
- Located at: `/Users/vitor/Desktop/mestrado/temp/tarik-new/Location Encoders/A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions (Sphere2Vec-sphereM).ipynb`
- Frozen snapshot in our repo: `tests/test_embeddings/_sphere2vec_reference.py`

### Related project (in case the paper-sphereM ever gets ported by someone else)
- **TorchSpatial**: [github.com/seai-lab/TorchSpatial](https://github.com/seai-lab/TorchSpatial) — a successor framework that bundles Sphere2Vec and other location encoders. Could be useful as a reference implementation if you implement Action B.

### In-repo references
- Migration history: `git log --grep=sphere2vec`
- Speed optimization plan: `plans/mtlnet_speed_optimization.md`
- Project guide: `CLAUDE.md` (root)
- Embedding root README (knowledge transfer): `research/embeddings/README.md`
- Sphere2vec package README: `research/embeddings/sphere2vec/README.md`
- Sphere2vec CLAUDE pointer: `research/embeddings/sphere2vec/CLAUDE.md`
