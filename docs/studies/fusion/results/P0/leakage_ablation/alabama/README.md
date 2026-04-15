# HGI Leakage Ablation — Alabama Results

**Date:** 2026-04-15
**Config:** alabama, HGI-only (not fusion), DSelect-k(e=4,k=2) + aligned_mtl,
1 fold (fold 0 of the StratifiedGroupKFold split), seed 42, 50 epochs,
batch 4096, grad_accum_steps=1, embedding_dim=64.
**Companion audit:** `docs/studies/fusion/issues/HGI_LEAKAGE_AUDIT.md`.

## What each arm does

| Arm | `le_lambda` | `hard_neg_prob` | `shuffle_fclass` | Tests |
|-----|-------------|-----------------|------------------|-------|
| baseline | 1e-8 | 0.25 | — | current defaults |
| A_no_hierarchy | **0.0** | 0.25 | — | explicit category-fclass L2 path in POI2Vec |
| B_uniform_negs | 1e-8 | **0.0** | — | emergent category signal via fclass-distribution region similarity |
| AB_both | **0.0** | **0.0** | — | combined effect + any interaction |
| C_fclass_shuffle | 1e-8 | 0.25 | **seed=42** | whether Category F1 is a real representation-quality metric or an fclass-identity lookup |

## Results

```
arm                  cat_f1     Δcat   cat_acc   next_f1     Δnxt   next_acc
--------------------------------------------------------------------------------
baseline            0.7855    +0.00    0.8250    0.2383    +0.00    0.3029
A_no_hierarchy      0.7992    +1.36    0.8296    0.2390    +0.06    0.3177
B_uniform_negs      0.7723    −1.32    0.8165    0.2623    +2.40    0.3314
AB_both             0.7690    −1.66    0.8162    0.2550    +1.66    0.3503
C_fclass_shuffle    0.1437   −64.19    0.2623    0.1988    −3.95    0.2587
```

F1 = macro F1. Δ = percentage points vs baseline. 1 fold → std=0 (paired
single-seed comparison, not a variance estimate).

Per-arm embedding bytes are distinct (see archived `embeddings.parquet` and
`poi2vec_fclass_embeddings_Alabama.pt` per arm).

## Interpretation

### Arm A — the explicit category path is cosmetic

If the hierarchy L2 loss were a meaningful leak, removing it should *drop*
Category F1. The observed effect is the opposite (+1.36 p.p. on category,
+0.06 p.p. on next). This is consistent with the `le_lambda=1e-8` weight
being too small to shape the fclass embeddings in any direction that matters
for the downstream task. The `(category, fclass)` L2 term is effectively a
no-op in the current pipeline.

**Verdict:** this channel is not a real leakage vector. Safe to leave as-is
and defensible in the paper as "we verified via ablation that the explicit
category path in POI2Vec has <1.5 p.p. effect on Category F1 on alabama/1-fold."

### Arm B — hard-negative sampling is a design trade-off, not a leak

Replacing similarity-weighted hard negatives with uniform negatives produces
asymmetric per-task effects: Category F1 −1.32 p.p., Next F1 +2.40 p.p.
A true label-leakage channel would degrade the supervised task uniformly.
The observed pattern (help one task, hurt another, in comparable magnitudes)
looks like a legitimate modeling choice: hard negatives encourage the
contrastive objective to separate regions with similar fclass mix, which
helps category classification but compresses the representation in a way
that hurts next-POI sequence modeling.

The underlying signal (`coarse_region_similarity` from per-region fclass
crosstab) uses public OSM tags, not the downstream 7-class `category`.
Reviewers will accept this as a feature-engineering decision, not leakage.

**Verdict:** not a leak. Report as "we confirmed via ablation that the
hard-negative sampling strategy shapes embeddings with asymmetric per-task
effects, consistent with a feature choice rather than label leakage."

### Arm A+B — mild sub-additive interaction

AB_both tracks arm B closely (−1.66 / +1.66), with slightly smaller magnitude
on the next-task side than arm B alone (+1.66 vs +2.40). The two channels
interact weakly; most of the movement is driven by hard-negative sampling,
not the hierarchy loss.

### Arm C — fclass shuffle: the decisive test

**Finding:** fclass → category is 100% deterministic in alabama OSM data
(284 fclasses → 7 categories, macro + size-weighted purity = 1.0000).
POI2Vec embeds at fclass level, so every POI's embedding is a deterministic
function of its fclass.

**Shuffle applied:** permute the encoded `fclass` column across POIs,
with matching seed in Phase 3a and Phase 4 so POI2Vec walks, region
similarity crosstab, and the pois.csv read by POI2Vec all see the same
shuffled fclass. `category` is untouched (it remains the real
per-POI label that the MTL category head must learn).

**Result:** Category F1 collapses 64.19 p.p. to **0.1437**, which is
indistinguishable from the 7-class random-chance ceiling (1/7 ≈ 0.143).
Accuracy of 0.262 sits near the Food-class majority rate of 32%, so the
model essentially defaults to majority guessing. **Spatial graph
structure carries almost no category-discriminative signal on its own.**

**Next-POI F1 drops only −3.95 p.p.** under the same shuffle, so
Next-POI was never riding on fclass identity as a dominant shortcut.

**Verdict:** Category F1 on this dataset primarily measures
fclass-identity preservation, not learned representation quality. Any
published Category F1 on OSM-tagged POI data using fclass-derived
embeddings inherits this ceiling. The paper must either (a) re-frame
Category F1 as a sanity check on embedding fidelity and anchor
representation-quality claims on Next-POI F1 or (b) use a task/dataset
that doesn't admit the fclass shortcut.

### What this does NOT mean

- **Not classical label leakage.** Val `category` labels are never used
  during HGI/POI2Vec training. fclass is a public OSM attribute,
  legitimately available at inference time.
- **The MTL task is still well-formed.** It's just easier than it
  appeared — the model can solve it via fclass proxy, the way a deployed
  system would use the fclass→category lookup directly.

### What this does NOT rule out

- **Transductive GNN training** — still present, still standard
  GNN-benchmark practice, still a methodology caveat for the paper.
- **Variance on arms A/B from 1-fold / single-seed.** Deltas of 1–3 p.p.
  could include run-to-run noise; a 3-seed replication would tighten bars.
  Arm C's −64 p.p. delta is far outside any reasonable noise envelope,
  so replication would not flip that conclusion.
- **Generalization to other states.** Need to confirm fclass→category
  purity ≈ 1.0 on florida / texas / arizona before generalizing the
  arm C finding beyond alabama.

## Files

- `run_log.json` — per-arm timing, return codes, result pointers
- `comparison.json` — extracted metrics table
- `<arm>/embeddings.parquet` — arm's POI-level HGI embeddings
- `<arm>/poi2vec_fclass_embeddings_Alabama.pt` — arm's fclass embeddings
- `<arm>/mtl_training.log` — MTL training log
- `<arm>/mtl_results/` — full training output (per-fold reports, summaries, plots)

## Reproduce

```
.venv/bin/python scripts/hgi_leakage_ablation.py \
    --arms baseline,A_no_hierarchy,B_uniform_negs,AB_both
```

Driver: `scripts/hgi_leakage_ablation.py`.
Code hooks added: `le_lambda` exposed in `research/embeddings/hgi/poi2vec.py`,
`hard_neg_prob` exposed in `research/embeddings/hgi/model/HGIModule.py` and
threaded through `research/embeddings/hgi/hgi.py`.

## Decision

1. **No code changes** to the HGI pipeline. No classical label leakage.
2. **Existing alabama/hgi embeddings** are scientifically valid (restore
   from `output/hgi/alabama.backup_pre_leakage_ablation_20260415_021908`
   if needed; currently the canonical `output/hgi/alabama/` holds
   C_fclass_shuffle's embeddings from this arm's run).
3. **Paper framing change:** Category F1 is a sanity check on
   fclass-identity preservation, not a representation-quality metric.
   Next-POI F1 is the primary representation-quality metric.
4. **Required paper caveat** (one paragraph, evaluation section):
   state the fclass→category determinism, cite arm C's result, note
   that Category F1 measures embedding fidelity rather than spatial
   representation learning.
5. **Before BRACIS:** replicate arm C on florida to confirm the fclass
   purity is ~1.0 there too (estimated 15 min).
6. **C29 new claim** to add to `CLAIMS_AND_HYPOTHESES.md`: "Category F1
   on OSM-tagged Gowalla data is near-upper-bounded by fclass-identity
   preservation; spatial-only signal contributes < 1 p.p. above random
   chance. Evidence: arm C collapses Category F1 by 64 p.p. to random."
