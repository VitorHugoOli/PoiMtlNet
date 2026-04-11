# HGI Migration vs. Original Paper — Critical Analysis & Alignment Plan

**Author:** Claude Opus 4.6 (analysis from 2026-04-11 session)
**Audience:** Future Claude Code session that will decide whether to align our HGI migration with the canonical paper, OR who needs to understand what the migration actually is and what it isn't.
**Scope:** Documentation + a remediation plan. **No code is modified by following this document.** The plan ranks fixes by severity and gives the rationale, the file/line, and the expected blast radius for each.
**Repo root:** `/Users/vitor/Desktop/mestrado/ingred`
**Target venv:** `.venv_new` (Python 3.12, PyTorch 2.9.1, MPS on Apple Silicon)

---

## 1. Read this first — what this document IS and what it ISN'T

### 1.1 What it is

A side-by-side comparison of **four** sources of HGI:
1. The **paper**: Huang, Zhang, Mai, Guo, Cui (2023). "Learning urban region representations with POIs and hierarchical graph infomax." *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 196, pp. 134–145.
2. The **canonical code**: `github.com/RightBank/HGI` — the authors' own MIT-licensed implementation.
3. The **third-party reproduction**: `region-embedding-benchmark/baselines/HGI` — a community reproduction we initially used as a reference. Already drifts from (1) and (2).
4. **Our migration**: `research/embeddings/hgi/` — currently faithful to (3), partially faithful to (1)/(2).

### 1.2 What it isn't

This is **not** a "the migration is broken" document. Our migration is algorithmically correct — every operation in the paper's Equations (1)–(13) has a corresponding operation in our code, doing the right thing on the right tensors. The deviations are **constants and hyperparameters**, not architecture. The paper's headline result on Xiamen/Shenzhen could plausibly still be reproduced by our code with the right hyperparameters.

### 1.3 What "we already match the reference" actually means

Sessions before 2026-04-11 produced this claim repeatedly:

> "All 6 reference equivalence tests pass — bit-for-bit identical loss vs the original source code."

That statement is true, but the "original source" being tested against in `tests/test_embeddings/test_hgi_reference_equivalence.py` is **`region-embedding-benchmark/baselines/HGI`**, not the paper authors' own `RightBank/HGI`. The third-party reproduction had already drifted from the paper in 4 hyperparameters (`lr`, warmup, PReLU, cross-region edge weight), and we inherited all 4.

So the migration is bit-for-bit equivalent to the **third-party reproduction**, NOT to the canonical authors' code or the paper. The reference equivalence guarantee is real, just to the wrong target.

---

## 2. The four sources, side by side

### 2.1 Paths on disk

| Source | Location | Notes |
|---|---|---|
| Paper PDF | `temp/tarik-new/region-embedding-benchmark-main/.../HGI/...` (the third-party repo doesn't ship the paper) | Download from `https://raw.githubusercontent.com/RightBank/HGI/main/Paper/paper.pdf` if needed. The paper's author copy is in `RightBank/HGI/Paper/paper.pdf`. |
| Canonical `RightBank/HGI` | NOT cloned locally — fetch via `gh api repos/RightBank/HGI/contents/...` | The repo is small (6 files, ~20 KB of Python). |
| Third-party reproduction | `temp/tarik-new/region-embedding-benchmark-main/region-embedding-benchmark-main/region-embedding/baselines/HGI/` | What our migration was originally based on. Used by `tests/test_embeddings/_hgi_reference_shim.py`. |
| Our migration | `research/embeddings/hgi/` | Active code. |

### 2.2 Cheat sheet for fetching the canonical repo

```bash
mkdir -p /tmp/hgi_canonical && \
gh api repos/RightBank/HGI/contents/train.py            --jq .content | base64 -d > /tmp/hgi_canonical/train.py && \
gh api repos/RightBank/HGI/contents/Module/hgi_module.py --jq .content | base64 -d > /tmp/hgi_canonical/hgi_module.py && \
gh api repos/RightBank/HGI/contents/Module/set_transformer.py --jq .content | base64 -d > /tmp/hgi_canonical/set_transformer.py && \
gh api repos/RightBank/HGI/contents/Module/city_data.py --jq .content | base64 -d > /tmp/hgi_canonical/city_data.py && \
gh api repos/RightBank/HGI/contents/README.md           --jq .content | base64 -d > /tmp/hgi_canonical/README.md && \
gh api repos/RightBank/HGI/contents/evaluation.py       --jq .content | base64 -d > /tmp/hgi_canonical/evaluation.py && \
curl -sL -o /tmp/hgi_canonical/paper.pdf "https://raw.githubusercontent.com/RightBank/HGI/main/Paper/paper.pdf"
```

The canonical repo only has 6 files: `train.py`, `evaluation.py`, `Module/hgi_module.py`, `Module/set_transformer.py`, `Module/city_data.py`, `README.md`. **There is no preprocess script in the canonical repo** — the paper assumes you load a precomputed `{city}_data.pkl` from `./Data/`. This means the canonical authors' code does NOT specify the cross-region edge weight — that constant only exists in the third-party reproduction's preprocess code (which we inherited from).

---

## 3. The three-way diff table (the meat of this document)

| # | Component | Paper / Eq. | Canonical (`RightBank/HGI`) | `region-embedding-benchmark` | **Our migration** | Verdict |
|---|---|---|---|---|---|---|
| **A** | **Hyperparameters (paper §4.2)** | | | | | |
| 1 | Learning rate | **`0.006`** | `0.006` (default) | `0.001` ❌ | `0.001` ❌ (currently undefined — was just removed from `HGI_CONFIG` and the pipeline would crash on `args.lr`) | **HIGH severity** — wrong by 6×, inherited from baseline |
| 2 | Linear LR warmup | **40 epochs** | 40 epochs (`pytorch_warmup.LinearWarmup`) | none ❌ | none ❌ | **HIGH severity** — paper says it's needed for stability with the high LR |
| 3 | Epochs | 2000 | 2000 | 2000 | 2000 | ✅ |
| 4 | Gradient clip max_norm | 0.9 | 0.9 | 0.9 | 0.9 | ✅ |
| 5 | Hidden dim `d` | 64 (best) | 64 | 64 | 64 | ✅ |
| 6 | Attention heads `h` | 4 (best) | 4 | 4 | 4 | ✅ |
| 7 | Loss balance α | 0.5 (best) | 0.5 | 0.5 | 0.5 (currently undefined) | ✅ when defined |
| 8 | Hard-neg probability | 0.25 | 0.25 | 0.25 | 0.25 | ✅ |
| 9 | Hard-neg similarity range | `[0.6, 0.8]` (closed in paper §3.7) | `(0.6, 0.8)` open | `(0.6, 0.8)` open | `(0.6, 0.8)` open | ⚠️ trivial — boundary samples are measure-zero |
| **B** | **Model architecture** | | | | | |
| 10 | POIEncoder GCNConv | one-layer GCN, PReLU (Eq. 3) | `GCNConv(cached=True, bias=True)` + `nn.PReLU(hidden_channels)` (channelwise) | `cached=False` ❌, `nn.PReLU()` ❌ | `cached=False` ❌, `nn.PReLU()` ❌ | **MEDIUM severity** — wrong PReLU param count + perf opt missing |
| 11 | POI2Region GCNConv | one-layer GCN, PReLU (Eq. 9) | `GCNConv(cached=True, bias=True)` + `nn.PReLU(hidden_channels)` | `cached=False` ❌, `nn.PReLU()` ❌ | `cached=False` ❌, `nn.PReLU()` ❌ | **MEDIUM severity** — same as above |
| 12 | PMA aggregation | `H + rFF(H)` where `H = s + Multihead(s, P, P)` (Eq. 7-8) | `Set Transformer` PMA (matches) | matches | matches (incl. fixed multi-head split bug from earlier session) | ✅ |
| 13 | Region2city | `σ(Σ r_i · aw_i)` (Eq. 10) | `lambda z, area: sigmoid((z.T * area).sum(1))` | matches | matches | ✅ |
| 14 | Discriminator bilinear | `D(p, r) = σ(pᵀ W_pr r)` (Eq. 12-13) | `torch.matmul(poi, torch.matmul(W, region))` | matches | matches (vectorized, mathematically equivalent) | ✅ |
| 15 | Corruption fn | row-wise shuffle of feature matrix (paper §3.7) | `x[torch.randperm(x.size(0))]` | matches | matches | ✅ |
| **C** | **Loss & negative sampling** | | | | | |
| 16 | Loss EPS in `log()` | not specified | **`EPS = 1e-15`** | `EPS = 1e-7` ❌ | `EPS = 1e-7` ❌ | **LOW severity** — defensive, harmless |
| 17 | Negative POI/region pair | `p̃_j` from another region (Eq. 12) | foreign POIs vs current region | foreign POIs vs current region | foreign POIs vs current region (after our fix in commit `604b4d7`) | ✅ — we now match |
| 18 | Region-city neg samples | corrupted region embeddings (Eq. 13) | `neg_region_emb` from corrupted POIs | matches | matches | ✅ |
| 19 | Loss combine | `α L_pr + (1-α) L_rc` (Eq. 11) | matches | matches | matches | ✅ |
| **D** | **POI graph & features** | | | | | |
| 20 | POI input features | second-level **category embeddings** via `φ_c` (Huang 2022, paper §3.2) | loaded from external `.pt` file (paper assumes precomputed) | `embeddings_poi_encoder.csv` from a different project (the same `φ_c` algorithm via `poi-encoder/POIEmbedding.py`) | **POI2Vec** in-pipeline (the same `φ_c` algorithm: random walks + skip-gram + Laplacian eigenmaps L2 hierarchy loss) | ✅ **algorithmically identical** to paper §3.2 + Eq. (1). The paper's `φ_c` IS POI2Vec under a different name. |
| 21 | Edge weight formula | Eq. 2: `log((1+L^1.5)/(1+l^1.5)) × w_r`, then min-max → [0,1] | (no preprocess script in canonical repo) | uses `0.5` ❌ | uses **`0.5`** ❌ (`research/embeddings/hgi/preprocess.py:138`) | **MEDIUM-HIGH severity** — paper explicitly says **`w_r = 0.4`** for cross-region in Eq. (2), repeated below it as "the choices of [...] `w_r = 0.4` (cross-region) are in view of previous practices in Calafiore et al. (2021) and Huang et al. (2022)" |
| 22 | Region adjacency | "no specific weight is assigned" (paper §3.5) | passed to GCN as `region_adjacency` only (no edge weight) | matches | matches | ✅ |

---

## 4. Severity legend & rationale

### 4.1 HIGH severity (could materially change quality)

**(1) `lr=0.001` instead of paper's `0.006`** — 6× too small. Paper §4.2 verbatim:

> "We train HGI in the two study areas separately for 2,000 epochs without a minibatch mode. During training, we set the learning rate to **0.006**, and use the gradient clipping technique (maximum norm of the gradients is 0.9) as well as **linear learning rate warmup for a period of 40 epochs**. We find that with this training strategy, the training is both stable (not prone to collapse despite its complex loss landscapes) and efficient (with generally a large learning rate)."

The high LR + warmup combo is a **deliberate co-design**. Our setup uses neither.

**(2) No linear LR warmup** — see above. Paper §4.2 explicit: "linear learning rate warmup for a period of 40 epochs". Without warmup, you can't safely use lr=0.006 (it would diverge in the first few epochs). With our lr=0.001 we don't need warmup, but we also don't reach the paper's working point.

### 4.2 MEDIUM-HIGH severity

**(21) Cross-region edge weight `0.5` instead of `0.4`** — Paper Eq. (2) is unambiguous:

> `a_p(p_i, p_j) = log((1+L^1.5) / (1+l^1.5)) × w_r`
> 
> where `w_r = 1` (intra-region) and **`w_r = 0.4`** (cross-region).

This affects the POI graph the GCN sees, and therefore every downstream embedding. The constant lives in `research/embeddings/hgi/preprocess.py:138`:

```python
w2 = 1.0 if self.pois.iloc[x]["GEOID"] == self.pois.iloc[y]["GEOID"] else 0.5
```

The third-party reproduction's preprocess script uses `0.5`, and we inherited it without checking the paper.

### 4.3 MEDIUM severity

**(10), (11) Single-param `nn.PReLU()` instead of channelwise `nn.PReLU(hidden_channels)`** — both `POIEncoder` (`research/embeddings/hgi/model/POIEncoder.py:32`) and `POI2Region` (`research/embeddings/hgi/model/RegionEncoder.py:38`). Channelwise PReLU has 64 learnable leak parameters per layer vs 1 for the single-param version. The canonical authors' code uses channelwise (`nn.PReLU(hidden_channels)`). Tiny capacity difference but it's a deliberate authors' choice — and the paper explicitly cites He et al. 2015 (the PReLU paper) where channelwise is the default.

### 4.4 LOW severity (defensible / cosmetic)

**(16) `EPS = 1e-7` vs canonical `1e-15`** — `research/embeddings/hgi/model/HGIModule.py:16`. Our value is more conservative against `log(0)` blow-up in fp32 and doesn't change behavior except in degenerate cases. Defensible — keep.

**`nan_to_num` in `RegionEncoder`** — `research/embeddings/hgi/model/RegionEncoder.py:193` adds `torch.nan_to_num(region_emb, nan=0.0)` after the GCN; canonical doesn't have this. We added it because isolated regions with no POIs produce NaN through the PMA. Defensible defensive guard — keep.

**`get_region_emb` returns tuple** — Our `HGIModule.get_region_emb()` returns `(region_embedding, poi_embedding)` whereas canonical returns only `region_embedding`. We need POI embeddings for downstream MTL — keep.

**`(0.6, 0.8)` open vs `[0.6, 0.8]` closed similarity range** — Boundary samples are measure-zero. Ignore.

### 4.5 Pure perf opportunity

**`GCNConv(cached=False)` vs canonical's `cached=True`** — `research/embeddings/hgi/model/POIEncoder.py:30` and `research/embeddings/hgi/model/RegionEncoder.py:35`. The canonical's `cached=True` would save the second normalization in our positive+negative double-pass. Zero correctness risk; pure speed-up. Should enable.

---

## 5. Concrete fix list, ranked by priority

| Priority | Change | File / line | Risk | Test impact |
|---|---|---|---|---|
| **0 (CRITICAL — pipeline is broken)** | Re-add `alpha=0.5` and `lr=...` (whatever value you pick) to `HGI_CONFIG` | `pipelines/embedding/hgi.pipe.py:54-65` | The pipeline currently crashes on `args.alpha` access in `train_hgi` | None |
| **1** | Set `lr = 0.006` and add `pytorch_warmup.LinearWarmup` over 40 epochs to `train_hgi` | `research/embeddings/hgi/hgi.py` (`train_hgi`), `pipelines/embedding/hgi.pipe.py` (`HGI_CONFIG`) | Medium — needs `pytorch-warmup` dep added to `requirements.txt`; should re-validate Alabama F1 | None directly; the perf regression test should still pass |
| **2** | Fix cross-region edge weight from `0.5` to `0.4` | `research/embeddings/hgi/preprocess.py:138` | Low — one constant; will produce different `embeddings.parquet` so all CSV-equivalence tests need re-baselining | `test_hgi_alabama_csv_equivalence.py` thresholds may need to be re-tuned (procrustes cosine, k-NN overlap) |
| **3** | Use `nn.PReLU(hidden_channels)` (channelwise) in `POIEncoder` and `POI2Region` | `research/embeddings/hgi/model/POIEncoder.py:32`, `research/embeddings/hgi/model/RegionEncoder.py:38` | Low — 64 extra params per layer; weight shape changes so existing checkpoints won't load | The bit-for-bit reference equivalence tests will fail (they're pinned to the third-party reproduction). Either rebase them against the canonical `RightBank/HGI` (preferred) or delete them. |
| **4** | Set `GCNConv(..., cached=True, ...)` in both encoders | Same files as #3 | Zero correctness; pure speed-up | None |
| — | Leave `EPS = 1e-7` (defensive) | — | — | — |
| — | Leave `nan_to_num` in RegionEncoder (defensive) | — | — | — |
| — | Leave `get_region_emb` returning tuple (downstream needs POI emb) | — | — | — |
| — | Leave `(0.6, 0.8)` open interval (measure-zero) | — | — | — |

### 5.1 Recommended sequencing for the future agent

1. **Apply fix #0 first, in isolation**, with whatever `lr` value you intend for the rest of the work. Verify the existing Alabama benchmark still runs end-to-end. Commit.
2. **Apply fix #4 (cached=True) in isolation**. This should not change embeddings, only speed. Re-run all 52 HGI tests to verify (the bit-for-bit reference equivalence tests should still pass — `cached=True` is still mathematically the same op). Commit.
3. **Then apply #1 + #2 + #3 together** as one "align HGI with paper §4.2" change, because they all change the embedding values and need a coordinated test rebase + benchmark regeneration. Re-run Alabama, regenerate `embeddings.parquet`, re-run MTL training, compare F1 against the saved `results_save/alabama_baseline_*` directory.

### 5.2 What to expect from the F1 comparison

The most important question after the fixes is: **does `lr=0.006 + warmup=40` actually produce better Cat F1 / Next F1 on Alabama than `lr=0.001 + no warmup`?**

If F1 is the same or better → keep all the changes, and you've proven the migration matches the paper.

If F1 regresses → the most likely cause is dataset size. The paper trains on Xiamen (45,033 POIs / 661 regions) and Shenzhen (303,428 POIs / 5,461 regions). Our Alabama has 11,706 POIs / 1,108 regions — much smaller. Smaller datasets often need smaller learning rates because the loss landscape has more random structure relative to the global signal. In that case the **third-party reproduction's `lr=0.001`** may have been an empirical adjustment for smaller datasets, even if it's not what the paper says. Document this and either keep `lr=0.001` with a comment explaining why, or try a 3rd value (e.g. `lr=0.003`).

### 5.3 What to do with the bit-for-bit equivalence tests

`tests/test_embeddings/test_hgi_reference_equivalence.py` and `tests/test_embeddings/_hgi_reference_shim.py` currently pin our migration to the **third-party reproduction**. After fixes #1-#3 the pin will break.

**Options:**

1. **Rebase the shim to import the canonical `RightBank/HGI` source instead.** This is the right thing to do scientifically — pin to the authors' code, not the community reproduction. The challenge is that `RightBank/HGI` uses `pytorch_warmup` which has to be installed, and the import path is `from Module.hgi_module import *`, so the shim needs to fake the `Module` package the way the existing one fakes `model`.
2. **Delete those tests entirely** if you decide it's not worth maintaining bit-for-bit equivalence to either reference. The other 36 unit tests, 8 CSV-structural tests, and 2 perf tests still cover the algorithmic correctness.

Recommendation: **option 1** if you're aligning with the paper anyway, **option 2** if you're keeping `lr=0.001` for empirical reasons (then the equivalence-to-anything claim is moot).

---

## 6. The "we already pin perfectly" claim — what's true and what isn't

For future agents who read the README or earlier session summaries and see claims like:

- "All 50 HGI tests pass, including 6 bit-for-bit reference equivalence tests."
- "Loss values match the reference to 0.00 (bit-equal). Gradient cosine similarity: 1.000000."

**These are true statements about a specific reference: `region-embedding-benchmark/baselines/HGI`.** They are NOT statements about the paper or about the canonical authors' code. The third-party reproduction had:

- `lr = 0.001` (paper: 0.006)
- no warmup (paper: 40 epochs)
- single-param PReLU (canonical authors: channelwise)
- `cross-region weight = 0.5` (paper: 0.4)
- `EPS = 1e-7` (canonical authors: 1e-15)
- `cached=False` (canonical authors: `cached=True`)

We inherited every one of these. The bit-for-bit equivalence is real, but it's equivalence to a reference that itself has 6 known deviations from the paper.

**TL;DR:** if a future agent says "the migration is faithful to the original", they should clarify which "original": the paper, the canonical authors' code, or the third-party reproduction. They are three different things.

---

## 7. The migration's algorithmic correctness — what IS faithful

To balance section 6, here's what is genuinely faithful to the paper, with no caveats:

| Paper component | Equation(s) | Where in our code |
|---|---|---|
| POI category encoder `φ_c` | §3.2 + Eq. (1) | `research/embeddings/hgi/poi2vec.py` (`POI2Vec`, `EmbeddingModel`, `POISet`) |
| POI encoder `φ_p` | Eq. (3) | `research/embeddings/hgi/model/POIEncoder.py` |
| Edge weight formula structure | Eq. (2) (only the `w_r` constant deviates) | `research/embeddings/hgi/preprocess.py:_create_graph` |
| POI aggregation `AGG_POI-region` | Eq. (7), (8) | `research/embeddings/hgi/model/RegionEncoder.py` (vectorized PMA — fixed multi-head split bug in earlier session) |
| Region encoder `φ_r` | Eq. (9) | `research/embeddings/hgi/model/RegionEncoder.py` (the `self.conv` stage) |
| Region aggregation `AGG_region-city` | Eq. (10) | `research/embeddings/hgi/hgi.py:train_hgi` (the `region2city` lambda) |
| Hard-negative selection (25%, sim ∈ [0.6, 0.8]) | §3.7 paragraph 3 | `research/embeddings/hgi/model/HGIModule.py:forward` |
| Negative city sample = corrupted region | Eq. (13) | `research/embeddings/hgi/model/HGIModule.py:forward` (via `corruption()` and a second pass) |
| Bilinear discriminator | Eq. (12), (13) (`D(·,·) = σ(·ᵀ W ·)`) | `research/embeddings/hgi/model/HGIModule.py:discriminate_*` |
| Loss combine | Eq. (11) | `research/embeddings/hgi/model/HGIModule.py:loss` |
| Best model selection (lowest loss) | §4.2 ("region embeddings obtained in the epoch with the lowest loss") | `research/embeddings/hgi/hgi.py:train_hgi` |

The architecture is right. The paper is implementable from our code. The deviations are surface-level constants and a missing optimizer schedule.

---

## 8. Things a future agent will probably want to know

### 8.1 The paper is in the repo

`temp/tarik-new/region-embedding-benchmark-main/...` does NOT have the paper. The paper is in the canonical `RightBank/HGI` repo at `Paper/paper.pdf`. Get it via:

```bash
curl -sL -o /tmp/hgi_canonical/paper.pdf "https://raw.githubusercontent.com/RightBank/HGI/main/Paper/paper.pdf"
```

The paper is 41 pages, ~800 KB. Pages 10-22 contain the methodology and the §4.2 "Implementation details of HGI" section that has all the canonical hyperparameters. Read those first.

### 8.2 The paper's `φ_c` IS POI2Vec under a different name

Section 3.2 ("POI category embedding") describes the encoder `φ_c` and gives its loss in Eq. (1):

```
L_φ_c = Σ_c Σ_q -(log σ(c·c'_q) + Σ_i log σ(-c·c'_n_i)) + (λ/2) Σ_{i,j ∈ pairs} w_ij ||c_i - c_j||²
```

This is **exactly** what `research/embeddings/hgi/poi2vec.py:EmbeddingModel.forward` implements:
- The first term is skip-gram with negative sampling on the random walks (`F.logsigmoid(pos_dot)` + `F.logsigmoid(neg_dot)`).
- The second term is the Laplacian eigenmaps regularizer that pulls together fclasses sharing the same parent category — which is our "hierarchy loss" using `(diff*diff).sum()`.

The paper cites Huang et al. 2022 (the same author's earlier work) for the encoder, but the equation is exhibited in this paper. We are faithful to it.

The terminology mapping:
- Paper "first-level category" = our `category` column (e.g. "Food")
- Paper "second-level category" = our `fclass` column (e.g. "Coffee Shop")
- Paper `c_i` = our fclass embedding
- Paper `w_ij` = 1 if same parent category else 0 = our hierarchy_pairs `[(category_id, fclass_id), ...]`

### 8.3 The third-party reproduction has its own preprocess; the canonical doesn't

`RightBank/HGI` ships only the model + train + eval. It assumes you bring your own preprocessed `Data/{city}_data.pkl`. The paper authors' actual preprocess is **NOT** in the repo — it lives somewhere in the Huang et al. 2022 codebase or was custom for the paper's Xiamen/Shenzhen experiments.

The third-party reproduction at `region-embedding-benchmark/baselines/HGI/preprocess/main.py` IS a preprocess implementation. It's the one we inherited. **It's also where the `0.5` cross-region constant comes from** — the canonical code can't be checked for this constant because it doesn't have a preprocess script.

So when fixing the cross-region weight, you're fixing a deviation that exists in the **third-party reproduction's preprocess**, propagated into our code, and is contradicted only by the paper itself (Eq. 2).

### 8.4 The canonical model file is tiny — read it

`/tmp/hgi_canonical/hgi_module.py` is **132 lines** total. It is THE source of truth for the model architecture and the loss. It takes 5 minutes to read end-to-end and is much shorter than our `HGIModule.py` (which has the vectorized neg-pair build, the cached lookup, comments, etc.).

If you ever need to verify "does the canonical do X?", just read `hgi_module.py` directly. It's almost entirely identical to the third-party reproduction at `region-embedding-benchmark/baselines/HGI/model/hgi.py` **except for**:

- `EPS = 1e-15` (canonical) vs `1e-7` (third-party)
- `cached=True, bias=True` (canonical) vs `cached=False, bias=True` (third-party)
- `nn.PReLU(hidden_channels)` (canonical) vs `nn.PReLU()` (third-party)
- `get_region_emb` returns just region (canonical) vs returns (region, poi) tuple (third-party)
- `print(self.weight_poi2region, self.weight_region2city)` debug line in the third-party (canonical doesn't have it)
- Hard-negative sampling uses `random.sample(...)` from a Python set in canonical, vs `nonzero().tolist()` in third-party (functionally equivalent)

That's it. The 4 things flagged as "wrong" in our migration came from the third-party not the canonical. **The third-party reproduction was faithful to the paper EXCEPT for those 4 hyperparameters/constants.** We faithfully reproduced the third-party reproduction.

### 8.5 The canonical training loop is also tiny

`/tmp/hgi_canonical/train.py` is **94 lines**. The relevant block:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, verbose=False)
warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_period)

def train():
    model.train()
    optimizer.zero_grad()
    pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb = model(data)
    loss = model.loss(...)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    optimizer.step()
    with warmup_scheduler.dampening():
        scheduler.step()
    return loss.item()
```

Note `with warmup_scheduler.dampening(): scheduler.step()` — that's the `pytorch_warmup` library idiom. The warmup wraps the StepLR scheduler. Our `train_hgi` has neither.

To add warmup to our code:

```python
import pytorch_warmup as warmup

# in train_hgi setup:
warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_period)

# in train_epoch:
optimizer.step()
with warmup_scheduler.dampening():
    scheduler.step()
```

Add `pytorch-warmup>=0.1.0` to `requirements.txt`. The library is small and well-maintained.

### 8.6 The currently-broken HGI_CONFIG

Right now `pipelines/embedding/hgi.pipe.py:54` has:

```python
HGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    attention_head=4,
    gamma=1.0,
    max_norm=0.9,
    epoch=2000,
    poi2vec_epochs=100,
    force_preprocess=True,
    device='cpu',
    shapefile=None
)
```

`alpha` and `lr` were removed. `train_hgi` in `research/embeddings/hgi/hgi.py` reads `args.alpha` and `args.lr` and **will crash** when called via the pipeline. The argparse defaults at the bottom of `hgi.py` (`--alpha 0.5`, `--lr 0.001`) only apply when running `python research/embeddings/hgi/hgi.py` directly, NOT when running the pipeline.

**Minimum viable fix** before doing anything else (fix #0 from the table):

```python
HGI_CONFIG = Namespace(
    dim=InputsConfig.EMBEDDING_DIM,
    attention_head=4,
    alpha=0.5,                 # ADD: paper §4.2, best of {0.1, 0.3, 0.5, 0.7, 0.9}
    lr=0.006,                  # ADD: paper §4.2 says 0.006 (we were using 0.001)
    warmup_period=40,          # ADD: paper §4.2 linear warmup over 40 epochs
    gamma=1.0,
    max_norm=0.9,
    epoch=2000,
    poi2vec_epochs=100,
    force_preprocess=True,
    device='cpu',
    shapefile=None
)
```

If the agent decides to keep `lr=0.001` for empirical reasons (smaller dataset), they should add a comment explaining why and link to this document.

### 8.7 The 4 commits that produced the current state

The git log on `main` between the post-perf state and now:

| Commit | What it did | What it did NOT do |
|---|---|---|
| `604b4d7 feat(time2vec)...` | Loss negative-pair fix in HGIModule, RegionEncoder PMA bug fix, all the unit + reference + CSV + perf regression tests | Did not fix any of the deviations from the paper |
| `be2cc62 perf(hgi)...` | Vectorized neg_pair build with cached lookup, threads=6, perf regression tests | Did not fix paper deviations |
| `38d19a4 perf(hgi/poi2vec)...` | Cached neg candidates in POISet, vectorized hierarchy loss in EmbeddingModel | Did not fix paper deviations |
| `078ec0a docs(hgi)...` | Fixed misleading device log line | Documentation only |

The "perf" commits are still correct as-is. The "feat" commit is correct but was tested against the wrong reference.

### 8.8 If you re-run Alabama after applying the fixes, here's what to compare against

The pre-fix baseline is saved at:

```
results_save/alabama_baseline_20260410_231419/
├── embeddings.parquet
├── region_embeddings.parquet
├── input/
│   ├── category.parquet
│   └── next.parquet
├── temp/
│   └── gowalla.pt
├── poi2vec_fclass_embeddings_Alabama.pt
├── poi2vec_poi_embeddings_Alabama.csv
└── hgi.csv                 (the user's reference output)
```

And the corresponding pre-fix MTL run is at:

```
results/hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260410_2316/summary/full_summary.json
```

with the metrics: Cat F1 = **78.25 ± 2.30**, Cat Acc = **81.88**, Next F1 = **27.92 ± 1.54**, Next Acc = **36.97**.

After applying the paper alignment fixes, run the full pipeline + MTL again, then compare. Report deltas in pp. If any task drops by more than ~2 pp, that's a meaningful regression and the fix should be reverted (or `lr` re-tuned for our smaller dataset).

---

## 9. Sources

### 9.1 Primary

1. **The paper** — Huang, W., Zhang, D., Mai, G., Guo, X., & Cui, L. (2023). Learning urban region representations with POIs and hierarchical graph infomax. *ISPRS Journal of Photogrammetry and Remote Sensing*, 196, 134–145. https://doi.org/10.1016/j.isprsjprs.2022.11.021
   - Author copy PDF: https://raw.githubusercontent.com/RightBank/HGI/main/Paper/paper.pdf
   - Key sections referenced in this document:
     - **§3.1** Overview — six-step recipe
     - **§3.2** POI category embedding — defines `φ_c` and Eq. (1) (the POI2Vec loss)
     - **§3.3** POI encoder — defines `φ_p`, Eq. (2) (edge weight formula with `w_r = 0.4`), Eq. (3) (one-layer GCN with PReLU)
     - **§3.4** POI aggregation — Eqs. (4)-(8), the multi-head attention and PMA structure
     - **§3.5** Region encoder — Eq. (9), one-layer GCN, "no specific weight is assigned" to region adjacency edges
     - **§3.6** Region aggregation — Eq. (10), area-weighted summarization
     - **§3.7** Negative sampling and training objective — Eqs. (11)-(13), the hierarchical loss and the hard-negative sampling description (cosine similarity in [0.6, 0.8])
     - **§4.2** Implementation details — `d=64, h=4, α=0.5, lr=0.006, max_norm=0.9, epochs=2000, warmup=40`. This is the section that contains all the canonical hyperparameters.

2. **The canonical code** — Huang et al. (2022). RightBank/HGI [Source code]. GitHub. https://github.com/RightBank/HGI
   - License: MIT
   - Last commit (as of 2026-04-11): 2023-12-05
   - Files used in this analysis:
     - `train.py` — training loop, hyperparameter defaults, warmup setup
     - `Module/hgi_module.py` — `HierarchicalGraphInfomax`, `POIEncoder`, `POI2Region`, `corruption`
     - `Module/set_transformer.py` — `MAB`, `SAB`, `PMA`
     - `Module/city_data.py` — data loader
     - `README.md` — POI features described as "POI category embeddings (can even be one-hot vectors) or POI embeddings learned by other methods"
     - `Paper/paper.pdf` — author copy of the ISPRS paper

### 9.2 Secondary

3. **The third-party reproduction** — `region-embedding-benchmark/baselines/HGI`. This is what our migration was originally based on. Located on disk at `temp/tarik-new/region-embedding-benchmark-main/region-embedding-benchmark-main/region-embedding/baselines/HGI/`. Key file: `model/hgi.py` (the model + loss the third-party uses) and `preprocess/main.py` (the preprocess that introduced `cross-region weight = 0.5`).

4. **Set Transformer** — Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). Set transformer: A framework for attention-based permutation-invariant neural networks. In *International conference on machine learning* (pp. 3744–3753). https://arxiv.org/abs/1810.00825
   - Source of the PMA / MAB / SAB blocks. Both canonical and our code use the implementation from `github.com/juho-lee/set_transformer` (modified, as noted in `set_transformer.py` header).

5. **Deep Graph Infomax** — Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y., & Hjelm, R. D. (2019). Deep graph infomax. In *International Conference on Learning Representations*. https://arxiv.org/abs/1809.10341
   - The infomax framework that HGI generalizes hierarchically.

6. **Huang et al. 2022 (POI category encoder)** — Huang, W., Cui, L., Chen, M., Zhang, D., & Yao, Y. (2022). Estimating urban functional distributions with semantics preserved POI embedding. *International Journal of Geographical Information Science*, 36(10), 1905–1930. https://doi.org/10.1080/13658816.2022.2040510
   - The paper our HGI cites for `φ_c` (the POI category encoder). Defines the random-walk + skip-gram + Laplacian eigenmaps loss that we implement as `research/embeddings/hgi/poi2vec.py:EmbeddingModel`.

### 9.3 Code references in this repo

7. Our migration's HGI module: `research/embeddings/hgi/`
   - `model/HGIModule.py` — `HierarchicalGraphInfomax`, the bilinear discriminator, the `corruption` function, the (now vectorized) hard-negative sampling
   - `model/POIEncoder.py` — single-layer GCN + PReLU
   - `model/RegionEncoder.py` — vectorized PMA (with the multi-head split bug fix from earlier session) + region-level GCN
   - `model/SetTransformer.py` — MAB / SAB / PMA blocks (faithful port)
   - `hgi.py` — `train_hgi`, `create_embedding`, `_hgi_thread_context`
   - `poi2vec.py` — `POI2Vec`, `EmbeddingModel`, `POISet` (our `φ_c` implementation)
   - `preprocess.py` — Delaunay graph + edge weights (where the `0.5` constant lives)

8. Test suite that pins the migration to the **third-party reproduction**:
   - `tests/test_embeddings/test_hgi.py` — 36 unit tests
   - `tests/test_embeddings/test_hgi_reference_equivalence.py` — 6 bit-for-bit tests against the third-party reproduction
   - `tests/test_embeddings/_hgi_reference_shim.py` — the live-import shim
   - `tests/test_embeddings/test_hgi_alabama_csv_equivalence.py` — 8 structural tests against the user's `output/hgi/alabama/hgi.csv`
   - `tests/test_embeddings/test_hgi_perf_regression.py` — 2 perf regression tests

9. Profiling scripts created during the perf optimization sessions:
   - `scripts/profile_hgi_alabama.py` — phase-by-phase wall-clock profiler
   - `scripts/profile_poi2vec_alabama.py` — POI2Vec model + dataloader breakdown

### 9.4 Related plan documents

10. `plans/mtlnet_speed_optimization.md` — The MTL speed optimization plan. Cross-references this document for the device choice rationale (CPU > MPS for HGI).

---

**End of analysis.** Last updated: 2026-04-11. If you (a future agent) act on this document, please update it with what you found, what you changed, and what the new Alabama F1 numbers were.