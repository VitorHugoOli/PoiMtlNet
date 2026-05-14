# CH14 — Preprocessing shortcut audit

**Date:** 2026-04-15
**Verdict:** `confirmed_by_construction` — no fclass-identity shortcut exists in Check2HGI.

---

## Step 1: Code inspection

### POI2Vec dependency?

**NO.** Check2HGI does not import or use POI2Vec anywhere.

- `research/embeddings/check2hgi/check2hgi.py` → `create_embedding()` calls `preprocess_check2hgi()` then `train_check2hgi()`. No POI2Vec pre-training step.
- `research/embeddings/check2hgi/preprocess.py` → `Check2HGIPreprocess` class builds node features directly from raw check-in data (category one-hot + temporal sin/cos). No external embedding lookup.
- `research/embeddings/check2hgi/model/CheckinEncoder.py` → Multi-layer GCN on the built node features. No pre-trained initialisation.

The HGI pipeline's shortcut (fusion C29) was specifically that POI2Vec creates per-fclass embeddings which map 1:1 to categories, making category classification trivially solvable. **That mechanism does not exist in Check2HGI.**

### How category enters the model

`preprocess.py:210-237` (`_build_node_features`):

```python
# Category one-hot: 7 dimensions (Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel)
category_onehot = np.zeros((num_checkins, num_categories), dtype=np.float32)
category_onehot[np.arange(num_checkins), self.checkins['category_encoded'].values] = 1.0

# Temporal encoding: 4 dimensions (hour_sin, hour_cos, dow_sin, dow_cos)
temporal = np.zeros((num_checkins, 4), dtype=np.float32)
...

# Total: F = 7 + 4 = 11 features per check-in node
node_features = np.concatenate([category_onehot, temporal], axis=1)
```

Category is one of 11 input features — it tells the GCN "this check-in is at a Food place." This is a legitimate feature for next-POI prediction (users have category preferences), not a shortcut. The GCN learns to weight it against temporal and structural (user-sequence edge) features.

### fclass → category mapping

From fusion C29: Gowalla's fclass→category purity = 1.0 across all 6 states. But category→fclass is many-to-one (multiple fclasses map to the same category). Using a 7-dim category one-hot cannot distinguish individual POIs within a category — there's no identity channel.

### coarse_region_similarity

`preprocess.py:282-283` computes region similarity from per-region category distributions. This is a legitimate geographic signal (regions with similar restaurant/shop mixes are probably nearby or similar in function). Not a shortcut.

---

## Step 2: Unconditional fclass-shuffle ablation

**Rationale:** Per critical-review agent §2.3, even when code inspection shows no shortcut, the shuffle ablation is cheap (~30 min) and the result (expected: modest Acc@10 drop, not a collapse) is publishable evidence that the representation does real work beyond category identity.

**Status:** DEFERRED — the shuffle requires regenerating Check2HGI embeddings with a modified `_build_node_features` that randomises the category column. This is a preprocessing-level change, not a training-level one.

**Implementation plan for when compute is free:**
1. Add `shuffle_category_seed: Optional[int] = None` param to `Check2HGIPreprocess.__init__`.
2. In `_build_node_features`, if `shuffle_category_seed is not None`, apply `np.random.default_rng(seed).permutation(checkins['category_encoded'])` before building the one-hot.
3. Regenerate Check2HGI embeddings for Alabama with the shuffle.
4. Run single-task next-POI, 1 fold, 10 epochs on the shuffled embeddings.
5. Compare Acc@10 to the P0.4 smoke result.

**Expected outcome:** Acc@10 drops modestly (category is 7/11 of the feature vector by dimensionality, so removing it should cost some signal). If the drop is < 3pp → confirms the GCN is extracting mostly temporal + structural signal. If > 10pp → the model leans heavily on category features, which is interesting but not a shortcut.

---

## Conclusion

**CH14 = `confirmed_by_construction`.** The Check2HGI preprocessing pipeline has no POI2Vec dependency, no fclass-level embedding sharing, and no identity channel. Category enters as one of 11 input features (7-dim one-hot), which is legitimate signal for next-POI prediction.

The unconditional shuffle ablation is deferred to when compute is free; the expected result (modest, not catastrophic, drop) would strengthen the paper's representation-quality argument but is not a blocker for P1.
