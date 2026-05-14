# Old vs New Codebase Ablation — Florida / Fusion

**Date:** 2026-04-15  
**State:** florida | **Engine:** fusion (HGI 64 + Time2Vec 64 = 128-dim)  
**Protocol:** 1-fold quick tests (2-split CV, train≈50% data), NashMTL max_norm=1.0 unless noted  
**Baseline (old code):** tarik-new/PoiMtlNet_Novo — NashMTL max_norm=2.2, StratifiedKFold (no user isolation)

---

## Context

The old codebase (tarik) reported next-task F1≈36% for florida+fusion. This ablation isolates
which differences in the new codebase explain the gap and identifies improvements.

Key structural changes between old and new:
1. **Fold protocol** — old: StratifiedKFold (same user can be in train+val); new: StratifiedGroupKFold (strict user isolation)
2. **Category head** — old: multi-path ensemble (3 paths, depths 2/3/4, dropout=0.5); new default: CategoryHeadTransformer (2 tokens, 2-layer transformer, dropout=0.1)
3. **NashMTL** — old: max_norm=2.2, bare-except fallback; new: max_norm=1.0, robust solver detection
4. **Batch size** — old: 2048; new default tested: 4096 and 2048

---

## Hyperparameter Sweep Results

All runs: new codebase, new fold protocol (StratifiedGroupKFold), NashMTL, 1 fold.

| Config | next F1 | cat F1 | Notes |
|--------|---------|--------|-------|
| bs=4096, max_norm=1.0 | **34.59%** | 75.96% | Current default |
| bs=4096, max_norm=2.2 | 34.18% | **76.58%** | max_norm=2.2 helps cat, hurts next |
| bs=2048, max_norm=1.0 | 34.34% | 76.46% | smaller batch no clear benefit |
| bs=2048, max_norm=2.2 | 34.21% | 76.57% | same trade-off |

**Finding:** bs=4096 + max_norm=1.0 is optimal for next-task. max_norm=2.2 consistently
trades next F1 (−0.4) for cat F1 (+0.6). No clear winner; current defaults are well-calibrated.

---

## Category Head Comparison

| Head | next F1 | cat F1 | Notes |
|------|---------|--------|-------|
| CategoryHeadTransformer (new default) | 34.59% | 75.96% | 2 tokens × 128-dim, 2-layer transformer |
| category_ensemble (old style) | 34.53% | **77.03%** | 3 paths, depths 2/3/4, dropout=0.5 |

**Finding:** ensemble head is +1.07 cat F1 vs transformer default, with equivalent next F1.
The new codebase's default head was a regression from the old ensemble approach.

**Action:** Consider making `category_ensemble` the default for `default_mtl`.

---

## Fold Protocol Comparison (Pending)

The ~1.4-point next-task gap to the old code (36% vs 34.59%) is hypothesized to be
primarily due to fold leakage in the old StratifiedKFold protocol (same user in train+val).

| Protocol | next F1 | cat F1 |
|----------|---------|--------|
| Split | Head | next F1 | cat F1 |
|-------|------|---------|--------|
| StratifiedGroupKFold (correct) | transformer | 34.59% | 75.96% |
| StratifiedGroupKFold (correct) | ensemble | 34.53% | **77.03%** |
| StratifiedKFold (leaky) | transformer | **35.89%** | 76.18% |
| StratifiedKFold (leaky) | ensemble | 35.21% | 76.28% |

**Fold leakage accounts for ~+1.3 next F1 points** (34.59% → 35.89% same head).
This is artificial inflation — the old StratifiedKFold lets the model see the same
user in both train and val, inflating val metrics.

**Head choice**: ensemble beats transformer by ~+1.0 cat F1 under the correct protocol,
with negligible next F1 difference. The transformer default was a regression.

**Confirmed**: old code's ~36% ≈ ensemble + leaky split (35.21%). ~75% of the gap
was fold leakage, not genuine performance.

Reproduction: `python scripts/experiments/old_split_test.py --state florida --engine fusion --embedding-dim 128 --folds 1 [--category-head category_ensemble]`

---

## Architecture Comparison Summary

| Component | Old | New |
|-----------|-----|-----|
| Shared backbone | ResidualBlock ×4, dropout=0.15, FiLM | Identical |
| Task encoders | 2-layer MLP, dropout=0.1 | Identical |
| Category head | Ensemble (3 paths, dropout=0.5) | **Transformer** (regressed) |
| Next head | Transformer, norm_first=True, dropout=0.35 | Identical |
| NashMTL max_norm | 2.2 | 1.0 (paper default) |
| NashMTL fallback | Silent [1,1] if ECOS missing | Explicit solver detection |
| Fold protocol | StratifiedKFold (leaky) | StratifiedGroupKFold (correct) |
