# HGI Data Leakage Audit

**Date:** 2026-04-15
**Scope:** `research/embeddings/hgi/` (HGI + POI2Vec pipeline)
**Purpose:** Followup to `DATA_LEAKAGE_ANALYSIS.md` — that doc was written pre-refactor and overstated severity for HGI. This doc is a code-anchored audit of the current pipeline to identify every path through which category labels (or any downstream-task labels) can influence the final POI embeddings.

**Companion doc:** `DATA_LEAKAGE_ANALYSIS.md` (the original inductive-split proposal) and `DATE_LEAKAGE_IMPLEMENTATION_GUIDE.md`.

---

## TL;DR

- **One explicit label-injection path**: POI2Vec's hierarchy loss (`poi2vec.py:347-349` feeding `EmbeddingModel.forward` at `poi2vec.py:162-174`) pulls fclass embeddings toward category embeddings in a shared table. Weight `le_lambda = 1e-8`. Magnitude unknown without ablation.
- **Two emergent/indirect channels**: (a) `coarse_region_similarity` (computed from per-region fclass distributions) drives hard-negative sampling in HGI training; (b) fclass-level embedding sharing broadcasts whatever category structure exists in the fclass space to every POI with that fclass.
- **No hidden red flags**: HGI's training objective itself is purely self-supervised (contrastive, POI↔Region↔City). No auxiliary classification head. No `data.y` reads during training. No validation-metric-based model selection. 15 candidate paths checked and cleared.
- **Cleanest ablation requires three variants**, not one — see §6.

---

## 1. Pipeline recap

```
Phase 3a: preprocess_hgi(poi_emb_path=None)   → edges.csv, pois.csv (Delaunay graph)
Phase 3b: POI2Vec.generate_walks()             → fclass walks + co-occurrence
Phase 3c: POI2Vec.train()                      → fclass embeddings (skip-gram + hierarchy L2)
Phase 3d: POI2Vec.reconstruct_poi_embeddings() → POI-level embeddings via fclass lookup
Phase 4:  preprocess_hgi(poi_emb_path=<path>)  → graph_data.pkl (complete data dict)
Phase 5:  train_hgi()                          → final POI + region embeddings.parquet
```

POI2Vec produces **fclass-level** embeddings that are shared across all POIs with the same fclass. HGI runs hierarchical mutual-information maximisation on the Delaunay POI graph using those fclass embeddings as input node features.

---

## 2. Explicit leak path — POI2Vec hierarchy loss

### Location

```python
# research/embeddings/hgi/poi2vec.py:347-349
hierarchy_pairs = list(set([
    tuple(row) for row in self.pois[['category', 'fclass']].values
]))
```

```python
# research/embeddings/hgi/poi2vec.py:162-174  (EmbeddingModel.forward)
if self.hierarchy_pairs.numel() > 0:
    cat_idx    = self.hierarchy_pairs[:, 0]
    fclass_idx = self.hierarchy_pairs[:, 1]
    cat_embs    = self.in_embed(cat_idx)       # category embedding in SAME table
    fclass_embs = self.in_embed(fclass_idx)    # fclass embedding in SAME table
    diff = cat_embs - fclass_embs
    loss_hierarchy = 0.5 * self.le_lambda * (diff * diff).sum()
```

### Mechanism

1. `self.pois` is the full state POI set — no train/val split is ever applied at preprocessing time. Every POI's `(category, fclass)` pair (including val POIs') feeds into `hierarchy_pairs`.
2. Category indices and fclass indices share `self.in_embed` (a single table of size `vocab_size = max(fclass)+1`). The L2 loss pulls each fclass's row toward its category's row in the same table.
3. Because every POI of a given fclass later reuses that fclass's row as its node feature, any category-aligned structure learnt at this step propagates to all downstream POI embeddings.

### Severity

- Loss weight `le_lambda = 1e-8` (`poi2vec.py:122`, `poi2vec.py:367`) — 6–8 orders of magnitude smaller than the skip-gram loss per batch.
- **Direction matters more than magnitude**: the skip-gram gradient is category-agnostic noise w.r.t. the evaluation axis; the hierarchy gradient is tiny but consistently pushes fclass embeddings into category-consistent clusters. A small task-aligned bias can still move a category-separation metric.
- **Verdict**: real explicit label leakage. Magnitude must be measured by ablation (§6), not assumed from `le_lambda` alone.

---

## 3. Emergent channel — region-similarity-driven hard negatives

### Location

```python
# research/embeddings/hgi/preprocess.py:198-206
# region_fclass_dist = crosstab of (region_id, fclass) → counts
coarse_region_similarity = cosine_similarity(region_fclass_dist)
```

```python
# research/embeddings/hgi/model/HGIModule.py:125-200  (corruption / hard-negative sampling)
# 25% "hard" negatives drawn where similarity ∈ [0.6, 0.8], 75% random
```

### Mechanism

- Region-region similarity is derived from **per-region fclass distributions**. In OSM, fclass is strongly correlated with the 7-class `category` (fclass is a finer-grained category-like tag). So regions with similar fclass mix tend to have similar category mix.
- Hard negatives are biased toward "regions with a similar fclass mix". During HGI's contrastive training, pushing apart a positive from such hard negatives effectively teaches the encoder to separate regions by category mix — without the category label ever being read directly.

### Severity

- Not explicit label injection — fclass is a public OSM tag, legitimately available for every POI including future ones.
- But it's a second channel through which category geography shapes embeddings. If the explicit hierarchy loss (§2) is cosmetic, this may become the dominant category-derived signal.
- **Verdict**: defensible in a paper as "we use public OSM tags", but needs to be reported. Should be held out in one of the ablation arms to bound its contribution.

---

## 4. Architectural amplifier — fclass-level embedding sharing

```python
# research/embeddings/hgi/poi2vec.py:438-444
# poi_embedding[i] = fclass_embedding[poi.fclass[i]]
```

All POIs with the same fclass get the **identical** POI2Vec embedding. This means any category-correlated structure in fclass-embedding space (whether from §2 or §3) is broadcast to every POI of that fclass. This is not a leak in itself — it's an architectural choice that amplifies whatever signal exists upstream.

Relevant only as interpretation: expect category prediction from these embeddings to be easy even with §2 and §3 disabled, because **fclass → category is near-deterministic in OSM** and fclass drives everything.

---

## 5. Cleared paths (15)

| # | Path | Location | Why safe |
|---|------|----------|----------|
| 1 | Delaunay edge construction | `preprocess.py:130-162` | Coordinates only |
| 2 | Edge weights (`cross_region_weight`) | `preprocess.py:130-162` | 1.0 intra-region / 0.7 cross-region — spatial only, not category-conditional |
| 3 | POI→Region spatial join | `preprocess.py:126` | `sjoin(..., predicate='intersects')` on geometry only |
| 4 | Region adjacency | `preprocess.py:182-196` | Geometry-only intersects |
| 5 | Node2Vec walk params (`p`, `q`) | `poi2vec.py:213, 257-266` | `p=q=0.5` hardcoded, uniform |
| 6 | POISet hard-negative sampling | `poi2vec.py:49-104` | "Never co-occurred in walks" — purely walk-derived |
| 7 | POI reconstruction step | `poi2vec.py:438-495` | Pure fclass→POI lookup; `add_category_label` only decorates the output DataFrame |
| 8 | HGIModule contrastive loss | `HGIModule.py:259-300` | POI↔Region + Region↔City, no classification head, no CE against `y` |
| 9 | HGI best-epoch selection | `hgi.py:168-177` | Lowest training loss only; no val metric |
| 10 | POI2Vec best-epoch selection | `poi2vec.py:424-433` | Lowest training loss only |
| 11 | PMA / MAB attention | `RegionEncoder.py`, `SetTransformer.py` | Seed vectors + POI embeddings, no label-conditional weights |
| 12 | `encodings.json` | `preprocess.py:207-230` | Saved for human inspection only; never reloaded in HGI |
| 13 | `data.y` field | `preprocess.py:316`, `hgi.py:140, 185-187` | Stored in pickle for downstream only; grep confirms zero reads during HGI training; used post-hoc to decode output parquet |
| 14 | One-hot fallback | `preprocess.py:240-242` | Only reached when `poi_emb_path=None`; default pipeline always trains POI2Vec first (`hgi.pipe.py:108-120`) |
| 15 | Optimizer / scheduler / early stopping in training loop | `hgi.py` | No label-derived terms; LR schedule is loss-based |

---

## 6. Ablation plan — the minimum clean experiment

To bound how much each leakage channel actually contributes, three paired runs are needed. All use **alabama** (HGI + fusion folds already frozen per `P0.8`), all reuse the frozen `alabama/hgi/mtl` fold indices for clean paired comparison.

| Arm | Changes vs baseline | Tests channel |
|-----|---------------------|---------------|
| **Baseline** | None (current code) | — |
| **A** | POI2Vec `le_lambda=0` | §2 explicit hierarchy loss |
| **B** | HGI hard-negative sampling → uniform random (disable `coarse_region_similarity`-driven selection) | §3 emergent region-similarity signal |
| **A+B** | Both changes together | Total category-derived influence on embeddings |

### Expected outcomes

- If A alone moves joint/category F1 by **<0.5 p.p.** → `le_lambda=1e-8` is cosmetic. Defend in paper as "verified negligible by ablation".
- If A moves it by **≥1 p.p.** → explicit leak is material; recommend setting `le_lambda=0` as the default and reporting the delta.
- If B moves it materially while A doesn't → emergent channel dominates. Paper must either (i) defend fclass-based region similarity as legitimate (fclass is public OSM data), or (ii) replace with coordinate-only region similarity.
- If A+B is larger than A + B (synergy) → channels interact; discuss explicitly.

### Protocol

- **State**: alabama first (embeddings already built, folds already frozen). If results are material, repeat on florida.
- **Seeds**: same seed as baseline (paired comparison). If time permits, 3 seeds for variance estimate.
- **Folds**: reuse frozen `alabama/hgi/mtl` fold indices — mandatory for paired tests.
- **Task config**: use the current champion config (DSelectK + aligned_mtl on HGI-only) so the ablation measures leak, not config interaction.
- **Metrics reported**: category F1, next F1, joint F1 (mean + std across folds), plus per-fold deltas vs baseline.

### Required code changes

1. Expose `le_lambda` as a CLI / pipeline parameter in `pipelines/embedding/hgi.pipe.py` (currently hardcoded at `poi2vec.py:367`).
2. Add a `hard_negative_mode` flag to `HGIModule` (`similarity_weighted` | `uniform`) so arm B doesn't require a code branch.
3. Namespace outputs so the three arms don't overwrite baseline: e.g. `output/hgi_ablation_A/alabama/`, etc.

### Time estimate

- POI2Vec retrain: ~5 min per arm.
- HGI retrain: ~30–60 min per arm.
- MTL input regeneration: ~5 min per arm.
- MTL 5-fold training: ~30 min per arm.
- **Total per arm**: ~1.5 hours. **Three arms + baseline comparison**: ~5 hours wall-clock.

### Result archival

- One JSON per arm under `docs/studies/fusion/results/P0/leakage_ablation/alabama/{baseline,A,B,AB}.json`.
- Summary table in `docs/studies/fusion/results/P0/leakage_ablation/README.md`.
- Claim status update: if results are material, add to `CLAIMS_AND_HYPOTHESES.md` as C29 ("HGI's category-derived training signals are negligible / material / dominant").

---

## 7. Out of scope for this audit

- **Check2HGI**: confirmed to concatenate category one-hot directly into node features (`research/embeddings/check2hgi/preprocess.py:217-219`). Not currently in any active fusion preset. **Must be fixed before Check2HGI is activated** per `CHECK2HGI_ENRICHMENT_PROPOSAL.md`.
- **MTL fold creation**: previously audited. `StratifiedGroupKFold` on userids gives user-disjoint next-task folds; category folds are POI-disjoint (`src/data/folds.py:585-626`). No leak at split time.
- **DGI, Time2Vec, Sphere2Vec**: not audited here. Time2Vec and Sphere2Vec are deterministic feature encoders (no label-dependent training), so they're low-risk. DGI uses similar self-supervised training and should be audited before any DGI-based conclusions are drawn.

---

## 7b. What is `fclass`, and how deterministic is it?

`fclass` is the **sub-category** of a POI as tagged in the raw Gowalla
data (column `spot`, renamed to `fclass` at
`research/embeddings/hgi/preprocess.py:55`). It is one step finer than
the 7-class `category`. Concrete examples from alabama:

| category (coarse, 7 classes) | fclass / spot (fine, sub-type) |
|---|---|
| Food | `American`, `Mexican`, `Burgers`, `BBQ`, `Pizza`, `Bakery`, `Doughnuts`, `Sandwich Shop`, `Asian` |
| Travel | `Modern Hotel`, `Motel`, `Resort`, `Subway`, `Bridge` |
| Nightlife | `Bar`, `Pub`, `Sports Bar`, `Dive Bar`, `Ultra-lounge` |
| Outdoors | `City Park`, `Historic Landmark`, `Campground`, `Lake & Pond`, `Pool / Waterpark` |
| Community | `Church`, `High School`, `Apartment`, `Craftsman`, `Corporate Office` |
| Entertainment | `Stadium`, `Golf Course`, `Cineplex`, `Soccer Field`, `Special Event` |
| Shopping | `Grocery`, `Bank & Financial`, `Salon & Barbershop`, `Gas & Automotive` |

By construction the sub-types cannot cross coarse categories: "Burgers"
is always Food, "Motel" is always Travel, "Church" is always Community.

### Purity check across all 6 state datasets

Measured as: for each fclass value, the fraction of POIs whose category
is the fclass's most-common category. 1.0 ⇒ every POI of a given fclass
has the same category (perfect deterministic mapping).

| state | POIs | fclasses | categories | unique pairs | macro purity | size-weighted purity |
|---|---:|---:|---:|---:|---:|---:|
| Alabama | 11 848 | 284 | 7 | 284 | **1.0000** | **1.0000** |
| Arizona | 20 666 | 305 | 7 | 305 | **1.0000** | **1.0000** |
| California | 169 145 | 333 | 7 | 333 | **1.0000** | **1.0000** |
| Florida | 76 544 | 324 | 7 | 324 | **1.0000** | **1.0000** |
| Georgia | 29 667 | 313 | 7 | 313 | **1.0000** | **1.0000** |
| Texas | 160 938 | 365 | 7 | 365 | **1.0000** | **1.0000** |

`unique pairs == fclasses` confirms the mapping is a strict function:
no fclass ever splits across multiple categories. The fclass-identity
shortcut found in arm C generalizes to every Gowalla state we can
evaluate on — this is a taxonomy property of the dataset, not a
per-state artefact. Raw data lives at `data/checkins/<State>.parquet`;
per-state details archived at
`docs/studies/fusion/results/P0/leakage_ablation/fclass_purity.json`.

## 8. Ablation results (2026-04-15, alabama, 1 fold, seed 42)

Ran on alabama, HGI-only, DSelect-k(e=4,k=2) + aligned_mtl, fold 0, seed 42.
Driver: `experiments/hgi_leakage_ablation.py`. Full write-up at
`docs/studies/fusion/results/P0/leakage_ablation/alabama/README.md`.

```
arm                  cat_f1   Δcat    cat_acc   next_f1   Δnxt    next_acc
baseline             0.7855  +0.00    0.8250    0.2383   +0.00    0.3029
A_no_hierarchy       0.7992  +1.36    0.8296    0.2390   +0.06    0.3177   (le_lambda=0)
B_uniform_negs       0.7723  −1.32    0.8165    0.2623   +2.40    0.3314   (hard_neg_prob=0)
AB_both              0.7690  −1.66    0.8162    0.2550   +1.66    0.3503
C_fclass_shuffle     0.1437  −64.19   0.2623    0.1988   −3.95    0.2587   (fclass permuted across POIs)
```

### Arm A — explicit hierarchy loss is cosmetic

Removing the `(category, fclass)` L2 path in POI2Vec (`poi2vec.py:162-174`)
**improves** Category F1 by 1.36 p.p. If this were a real leak, the
expected direction would be a *drop*. The `le_lambda=1e-8` weight is too
small to bias fclass embeddings in a task-useful direction, so the path
exists structurally but contributes nothing substantive. **Not a leak.**

### Arm B — hard-negative sampling is a design trade-off, not a leak

Replacing similarity-weighted hard negatives with uniform random negatives
produces *asymmetric* per-task effects: cat F1 −1.32 p.p. but next F1
+2.40 p.p. A genuine label-leakage channel would hurt the supervised task
uniformly; the observed trade-off is consistent with a modeling choice.
The underlying similarity is computed from per-region **fclass**
distributions (a public OSM tag), not the 7-class `category`. **Not a leak.**

### Arm A+B — weakly sub-additive

Tracks arm B closely (−1.66 / +1.66) with slightly reduced next-task gain
vs arm B alone. The two channels interact weakly; most movement is driven
by hard-negative sampling.

### Arm C — fclass shuffle: decisive, and replicated on florida

**Cross-state confirmation** (paired baseline + shuffle per state,
1 fold, seed 42):

| state | Δ Category F1 | Δ Next-POI F1 | baseline cat → shuffle cat | baseline next → shuffle next |
|---|---|---|---|---|
| Alabama | **−64.19 p.p.** | −3.95 p.p. | 0.7855 → 0.1437 | 0.2383 → 0.1988 |
| Florida | **−61.43 p.p.** | −6.46 p.p. | 0.7649 → 0.1506 | 0.3627 → 0.2982 |

On both states the Category metric lands at the 1/7 ≈ 0.143 random-chance
floor, and the drop is an order of magnitude larger than the Next-POI
drop. Florida was run on the same driver with `--state Florida` — full
run log at `docs/studies/fusion/results/P0/leakage_ablation/florida/run_log.json`.

Florida's absolute Next-POI F1 is higher (0.36 vs 0.24) simply because
Florida has ~6× more sequences. The Next-POI shuffle-drop is mildly
larger on Florida (−6.46 vs −3.95), indicating fclass identity carries a
slightly stronger contextual signal on the larger dataset — still an
order of magnitude smaller than the Category collapse and insufficient
to change C29's verdict.

### Arm C — original (alabama) findings

Permuting the encoded `fclass` column across POIs (keeping `category`
intact, applied identically in Phase 3a and Phase 4 via
`shuffle_fclass_seed=42`) **collapses Category F1 by 64.19 p.p.**
to 0.1437 — indistinguishable from the 1/7 ≈ 0.143 random-chance
ceiling for 7-class macro F1. Accuracy (0.262) sits near the Food-class
majority rate (32%), so the model essentially defaults to majority
guessing once fclass identity is scrambled.

This is the key empirical finding of the audit:

- **fclass → category is 100% deterministic in the alabama OSM data**
  (284 unique fclasses, each mapping to exactly 1 of 7 categories, macro +
  size-weighted purity both 1.0).
- **POI2Vec embeds at fclass level** — every POI's embedding is a
  deterministic function of its fclass (`poi2vec.py:438-444`).
- Therefore the Category task is a **near-trivial fclass-identity lookup**,
  not a test of learned spatial/semantic representation.
- Removing the fclass shortcut (arm C) leaves the model with only the
  signal HGI's graph training extracts from Delaunay topology + edge
  weights + region-area aggregation — which carries **essentially zero
  category-discriminative signal on its own** (Δmacro-F1 is at random
  chance).

**Next-POI F1 drops only −3.95 p.p.** under the same shuffle, so the
Next-POI task is *not* riding on fclass identity as a shortcut — it was
never the dominant signal there. Next-POI F1 is therefore the meaningful
representation-quality metric; Category F1 primarily reports
"how faithfully the embedding preserves fclass identity".

### What arm C does NOT mean

- **This is not classical label leakage.** No val-set `category` values
  flow into HGI or POI2Vec training. The code audit confirms this
  (preprocess.py lines 126, 316; grep shows `y` is never read during
  training). `le_lambda=1e-8` is cosmetic (arm A).
- **fclass is a public OSM attribute**, legitimately available at
  inference time for any new POI. A deployed system can use the
  fclass→category lookup directly, with or without an embedding pipeline.
- **The paper task definition is still well-formed** — the Category head
  can solve the task; it just happens to be easy.

### What arm C DOES mean for the paper

- **Category F1 cannot be reported as evidence of learned representations
  on this dataset.** Across-engine deltas (HGI vs DGI vs Fusion) on
  Category F1 primarily reflect how well each engine preserves fclass
  identity in a 64-dim space, not spatial/semantic structure learning.
- **Next-POI F1 is the only defensible representation-quality metric.**
  Baselines, deltas, and architecture comparisons should be evaluated
  against Next-POI F1, joint-F1 (if reported), or downstream tasks that
  don't admit an fclass shortcut.
- **A paper-level caveat is required** — a one-paragraph discussion in
  the evaluation section explicitly stating: (i) the fclass→category
  determinism observation, (ii) the arm C result, (iii) that Category F1
  is therefore a sanity check on embedding fidelity, not a representation
  benchmark.

### What this does NOT rule out

- **Transductive GNN training** (val POIs are nodes in the HGI training
  graph). Still present. Mild, standard, defensible as methodology caveat.
- **Statistical noise on 1 fold / seed.** Arm C's −64 p.p. Category
  delta is far outside any reasonable noise envelope (>30σ for any
  plausible per-fold variance); replication would move the number by
  <5 p.p., not flip the conclusion. Arms A and B deltas (±1–3 p.p.) are
  in the noise band and would benefit from 3-seed replication.

### Gotcha fixed during the ablation

First attempt used `--folds-path` pointing at the frozen
`output/hgi/alabama/folds/fold_indices_mtl.pt`. That file stores the
**feature tensors** from the time of freezing, not just indices — so every
arm trained on the pre-ablation embeddings and produced bit-identical
results. Fixed by switching to `--no-folds-cache` and regenerating folds
from the current input parquets per arm (indices still deterministic from
seed+userids). Worth remembering for any future paired HGI ablation.

## 9. Decision (revised after arm C)

**No classical label leakage in HGI.** The prior `DATA_LEAKAGE_ANALYSIS.md`
concerns about val-label flow through HGI training are **not supported** by
the audit:
- HGI's training objective is purely self-supervised contrastive
  (`HGIModule.py:259-300`).
- POI2Vec's explicit `(category, fclass)` L2 path is cosmetic at
  `le_lambda=1e-8` (arm A null result).
- Hard-negative sampling uses fclass distributions (public OSM), not
  category labels (arm B has asymmetric per-task effects, not
  one-sided degradation).
- `data.y` is never read during training (grep-verified).

**But the Category task is near-trivial due to fclass → category 1:1
determinism in OSM** (arm C: −64.19 p.p. collapse on shuffle). Category F1
primarily reports fclass-identity preservation, not representation
quality. This is not leakage, but it has equivalent paper-relevance
impact — it changes which metric the paper should anchor claims on.

### Required actions

1. **No code changes to the HGI pipeline.** Existing alabama/hgi
   embeddings are scientifically defensible.
2. **Paper framing must change.** Next-POI F1 becomes the primary
   representation-quality metric. Category F1 becomes a sanity check.
   Include a one-paragraph caveat in the evaluation section citing
   the arm C result.
3. **Across-engine Category F1 comparisons** (HGI vs DGI vs Fusion) must
   be re-framed as "fclass-identity preservation" benchmarks, not
   representation benchmarks.
4. **C29 (new claim):** add to `CLAIMS_AND_HYPOTHESES.md` — "Category F1
   on OSM-tagged data is upper-bounded by fclass-identity preservation
   because fclass → category is deterministic; spatial structure alone
   contributes negligible category-discriminative signal. Evidence:
   `docs/studies/fusion/results/P0/leakage_ablation/alabama/C_fclass_shuffle/`."
5. **Replicate arm C on florida** before BRACIS submission (~15 min)
   to confirm the 1:1 mapping holds there too. If purity < 1.0 on
   florida, the Category F1 ceiling shifts accordingly.
6. **Check2HGI** remains a separate landmine that must be fixed before
   it's ever activated (category one-hot in node features,
   `research/embeddings/check2hgi/preprocess.py:217-219`). Unrelated to
   this audit's scope.

### Summary of what the 5 arms proved

| Channel | Tested | Result |
|---|---|---|
| Explicit hierarchy L2 loss | arm A | cosmetic (null) |
| Similarity-weighted hard negatives | arm B | design trade-off, not leak |
| A + B interaction | arm A+B | weakly sub-additive |
| fclass-identity shortcut via POI2Vec embedding | arm C | **~100% of Category F1** |
| Transductive graph training | not tested | presumed mild; standard GNN practice |
| Walk-based emergent category signal | not tested | not material given arm C |
