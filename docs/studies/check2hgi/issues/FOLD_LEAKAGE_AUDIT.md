# Fold Leakage in STL Next-Task Folds — Audit

**Severity:** HIGH — invalidated STL baselines and biased MTL-vs-STL comparisons by 3 pp on AL.

**Detected:** 2026-04-17 during P2 critical review, after noticing MTL cat F1 (35–37%) consistently below STL cat F1 (39.16%) and investigating the STL baseline's fold protocol.

**Status:** FIXED + VERIFIED (commit `5217095`).

---

## TL;DR

`FoldCreator._create_single_task_folds()` used plain `StratifiedKFold` for the NEXT task, stratifying only on `next_category` label. The same user's check-ins could appear in both train and val, inflating F1 by allowing user-taste memorisation. MTL's `_create_check2hgi_mtl_folds()` has always used `StratifiedGroupKFold(groups=userid)`, so STL and MTL F1 numbers were measuring different tasks. Fix: force `StratifiedGroupKFold` for NEXT in single-task path as well.

After the fix:
- Check2HGI STL next-category F1 dropped from 39.16% to **38.58%** (−0.57 pp — robust to fair split).
- HGI STL next-category F1 dropped from 23.48% to **20.29%** (−3.20 pp — heavy leakage).
- CH16 delta (Check2HGI − HGI) grew from +15.67 pp to **+18.30 pp** — primary substrate claim is *stronger* with fair folds.

---

## Timeline

1. **2026-04-16 evening** — P1.5b runs Check2HGI vs HGI single-task next-category, 5f × 50ep seed 42. Numbers: Check2HGI F1 = 39.16 ± 0.83, HGI F1 = 23.48 ± 1.19. Declared CH16 confirmed at Δ = +15.68 pp.

2. **2026-04-17 00:00** — P2 screen runs MTL grid (5 arches × 5 losses × 1f × 15ep). Top configs sit at 35–37% cat F1, below STL's 39.16%. Preliminary conclusion in the leaderboard: *"category F1 saturates at 35–37% across the grid — AL MTL does not improve category."*

3. **2026-04-17 ~04:30** — During P2 critical review, I enumerated hypotheses for the MTL < STL gap. H2 (fold-protocol mismatch) was flagged as HIGH prior. Inspected `src/data/folds.py`:

   ```python
   # Line 537 — STL next-task:
   skf = StratifiedKFold(n_splits=..., shuffle=True, random_state=...)
   ...
   for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):

   # Line 833 — MTL check2hgi path:
   sgkf = StratifiedGroupKFold(...)
   ...
   sgkf.split(X, y_cat, groups=userids)
   ```

   Confirmed mismatch. STL userids were loaded (`load_next_data` returns them) but the variable was bound to `_userids` (underscore prefix = "ignore") and discarded.

4. **2026-04-17 05:10** — Fix applied (see "Fix" below). Both P1.5b arms rerun. Numbers in TL;DR.

---

## Why plain `StratifiedKFold` is wrong for sequence prediction

The NEXT task predicts the category of a user's next check-in given a 9-step history. Each row in `next.parquet` is a `(user, history, target_category)` tuple. With plain `StratifiedKFold`:

- Fold splits are made on `(X_row, y_row)` stratified by `y_row` (target category).
- Multiple rows from the **same user** — which share some history windows and user-specific taste patterns — get distributed across folds.
- The val set has users whose other sequences are in the train set.
- A model can memorise "user U tends to visit Food after Shopping" and score well on U's val rows without learning the generalisable pattern.

With `StratifiedGroupKFold(groups=userid)`:

- User assignments to folds are disjoint; no user appears in both train and val of a single fold.
- The val F1 measures generalisation to **unseen users** — the metric we actually want for a check-in prediction system.

HGI (POI-level embeddings) benefits disproportionately from the leak because its per-POI vectors are stable fingerprints of POI semantics that cluster by user taste. Check2HGI (check-in-level contextual vectors) varies across visits of the same POI, weakening the stable-fingerprint effect, so it generalises better — and thus was less affected by the leakage.

---

## Impact

### Invalidated numbers

- **Any "MTL ≈ STL" or "MTL < STL" category-F1 comparison made before the fix**, including the P2-screen leaderboard's "AL MTL does not improve category" finding. That conclusion was an artefact of comparing user-disjoint MTL to leaky STL. Retracted until MTL is re-measured at matched compute with fair folds.

- **Any earlier fusion-study STL next-task F1 numbers** (outside this study's scope but worth noting to the fusion track owners — the bug was identical in that path before we forked).

### Unaffected numbers

- **Region single-task baselines (P1)**: the `scripts/p1_region_head_ablation.py` script already used `StratifiedGroupKFold` (line ~375). P1 region numbers (56.94 ± 4.01 AL GRU 5f × 50ep, etc.) are already fair.

- **CH16 direction**: both P1.5b arms had the same leak; the relative comparison (Δ Check2HGI − HGI = +15.67 pp leaky) was valid. Only absolute F1s were inflated. Post-fix Δ grew to +18.30 pp — the substrate claim is *stronger*, not weaker.

- **P0 simple baselines (Markov, top-k, majority)**: these are computed with their own fold logic in `scripts/compute_simple_baselines.py`; they use grouped folds by user. Unaffected.

---

## Fix

`src/data/folds.py::_create_single_task_folds`:

```python
# Before:
X, y, _userids, embedding_dim = load_next_data(state, embedding_engine)
...
skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):

# After:
X, y, userids, embedding_dim = load_next_data(state, embedding_engine)
...
if task == TaskType.NEXT and userids is not None:
    skf = StratifiedGroupKFold(n_splits=..., shuffle=True, random_state=self.seed)
    split_iter = skf.split(X, y, groups=userids)
    logger.info("NEXT single-task: user-disjoint folds via StratifiedGroupKFold.")
else:
    skf = StratifiedKFold(...)
    split_iter = skf.split(X, y)
for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
```

Category single-task (flat POI-level) keeps `StratifiedKFold` — for a POI-property classifier, user grouping is not meaningful (categorise a POI, not predict a user's next action).

---

## Verification

P1.5b rerun on 2026-04-17 at 05:15 (bg id `b4p19zyhx`):

- Check2HGI: F1 = 38.58 ± 1.23 (was 39.16 ± 0.83). Drop: −0.57 pp.
- HGI: F1 = 20.29 ± 1.34 (was 23.48 ± 1.19). Drop: −3.20 pp.
- New Δ: **+18.30 pp** (was +15.67 pp).

Both arms dropped (confirms the leak existed and was closed). The differential drop (HGI much more than Check2HGI) is consistent with HGI's POI-level per-user memorisation being the stronger beneficiary of the leak.

Result files:
- `docs/studies/check2hgi/results/P1_5b/next_category_alabama_check2hgi_5f_50ep_fair.json`
- `docs/studies/check2hgi/results/P1_5b/next_category_alabama_hgi_5f_50ep_fair.json`

(The leaky JSONs are also kept — `*_5f_50ep.json` — for the leaky-vs-fair comparison table in `CLAIMS_AND_HYPOTHESES.md §CH16`.)

---

## Paper implications

1. **All STL next-task baselines in final paper table must be rerun with user-disjoint folds.** The fix is now in the code; any new run picks it up. Old numbers should be discarded or clearly labelled as "non-grouped (inflated)".

2. **Frame the fair-vs-leaky comparison as evidence of Check2HGI's design**. The robustness differential (Check2HGI loses 0.57 pp; HGI loses 3.20 pp under user-disjoint splits) is a paper-quality finding in its own right: it shows Check2HGI's check-in-level contextual variance forces user-agnostic learning. A reviewer asking "why does Check2HGI help?" gets an empirical, mechanistic answer.

3. **Revalidate any claim of the form "MTL ≈ STL" or "STL > MTL" from before the fix.** Current candidates: the P2-screen finding about AL MTL category saturation. Rerun with fair STL baseline before reporting.

---

## Lessons

- **Never trust a fold creator without reading its code.** The function name `_create_single_task_folds` did not tell me whether it used grouped folds.
- **Compare protocols first, numbers second.** The MTL < STL gap was initially mystifying; it became obvious once the protocol mismatch was found. Protocol diff should be the first check in any "method A underperforms method B" investigation.
- **Keep both leaky and fair numbers for publication.** The comparison itself is informative — but only if you audit both sides against the same standard.
