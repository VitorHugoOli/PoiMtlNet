# Plan: port single-task eval helpers from sklearn to torchmetrics

**Status**: not started — design doc only
**Author**: drafted during torch 2.11 upgrade audit (PR #9 follow-up)
**Estimated effort**: 1–2 hours
**Priority**: low (cosmetic / consistency win, not a performance hot path)

---

## 1. Why

PR #8 commit `2d73fe3 perf(mtl): replace sklearn classification_report with torchmetrics on hot paths` already migrated the **MTL training runner** off sklearn:

```python
# src/training/runners/mtl_cv.py:159-174 (post-PR-8)
f1_next = multiclass_f1_score(
    epoch_next_preds, epoch_next_targets,
    num_classes=num_classes, average='macro', zero_division=0,
).item()
next_acc = multiclass_accuracy(
    epoch_next_preds, epoch_next_targets,
    num_classes=num_classes, average='micro',
).item()
# ... same for category
```

The port stayed on **on-device tensors** through the per-epoch metric computation, eliminating the per-batch CPU transfer that the sklearn version required (and that the torch 2.11 audit later identified as the cause of the `.numpy()`-on-MPS bug).

The MTL **eval helpers** (`src/training/evaluate.py` and `src/training/shared_evaluate.py`) were NOT touched by PR #8 because:
1. They produce a `dict` of per-class precision/recall/f1/support that downstream display code consumes — not just a scalar F1.
2. They were not on the per-epoch hot path (called once at end-of-training per fold).
3. PR #8 was scoped to "batch 1" perf wins; the post-training reporting helpers were left for follow-up.

The torch 2.11 audit confirmed both helpers still call `sklearn.metrics.classification_report` and still need to materialize numpy arrays (which is why my one-line `.cpu()` defensive fix was required to unbreak single-task training on MPS — see commit `1746920`). Porting these helpers to torchmetrics would:

1. **Remove the numpy detour entirely** — no `.cpu().numpy()`, no MPS sync at fold-end. Tiny perf win (~tens of ms per fold), but architecturally cleaner.
2. **Match the MTL hot-path style** — single source-of-truth for metric computation across runners.
3. **Enable on-device metric aggregation** for any future eval-time enhancement (e.g. per-class confusion matrices, top-k accuracy) without re-introducing the sklearn dependency at boundary points.

**This is NOT a perf-critical change.** sklearn's `classification_report` on ~12k samples runs in single-digit milliseconds; saving it across 5 folds is ~50 ms total. Treat this as a refactor for consistency, not a perf win.

---

## 2. What to change

### 2.1 Files in scope

| File | LOC | Changes |
|---|---|---|
| `src/training/evaluate.py` | ~75 | Replace `sklearn.classification_report` with torchmetrics functional API. Build a matching dict shape so consumers don't break. |
| `src/training/shared_evaluate.py` | ~76 | Same. The `evaluate()` function (line 10) and `collect_predictions()` + `build_report()` helpers all share the same pattern. |
| `src/training/runners/category_cv.py` | small | Adjust `compute_class_weights` consumer + downstream `evaluate()` call signature if it changes. |
| `src/training/runners/next_cv.py` | small | Same. |
| `tests/test_training/*` (if any cover these helpers) | small | Update tolerances if sklearn vs torchmetrics produce slightly different f1-macro for edge classes. |

### 2.2 Functional mapping

| sklearn call | torchmetrics replacement |
|---|---|
| `precision_score(y, yhat, average=None)` | `multiclass_precision(preds, targets, num_classes=N, average=None)` |
| `recall_score(y, yhat, average=None)` | `multiclass_recall(preds, targets, num_classes=N, average=None)` |
| `f1_score(y, yhat, average=None)` | `multiclass_f1_score(preds, targets, num_classes=N, average=None)` |
| `accuracy_score(y, yhat)` | `multiclass_accuracy(preds, targets, num_classes=N, average='micro')` |
| support per-class | manual: `torch.bincount(targets, minlength=N)` |

All four torchmetrics functions live under `torchmetrics.functional.classification` and accept tensors on any device (CPU, CUDA, MPS). They return tensors of shape `(num_classes,)` for `average=None`, or scalar tensors for `average={'macro','micro','weighted'}`.

### 2.3 Output shape compatibility (the load-bearing constraint)

The current `classification_report(..., output_dict=True)` returns a dict like:

```python
{
    '0': {'precision': 0.42, 'recall': 0.31, 'f1-score': 0.36, 'support': 1757},
    '1': {'precision': ..., ...},
    ...
    '6': {...},
    'accuracy': 0.5346,
    'macro avg': {'precision': ..., 'recall': ..., 'f1-score': ..., 'support': 11706},
    'weighted avg': {'precision': ..., 'recall': ..., 'f1-score': ..., 'support': 11706},
}
```

Downstream consumers (`src/tracking/display.py`, `src/tracking/storage.py`, MLHistory's per-task report tracking) iterate this dict by class label and pull `precision`/`recall`/`f1-score`/`support`. **The replacement must produce a dict with byte-identical keys and structure**, otherwise display/storage will silently mis-aggregate or KeyError.

Reference adapter to drop into both files:

```python
import torch
from torchmetrics.functional.classification import (
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score,
    multiclass_accuracy,
)


def _torchmetrics_classification_report(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    label_names: list[str] | None = None,
) -> dict:
    """Drop-in replacement for sklearn.metrics.classification_report(output_dict=True).

    Both ``preds`` and ``targets`` may live on any device. Internally torchmetrics
    keeps the computation on-device; only the final scalar/per-class tensors are
    transferred via .tolist() (a single sync per metric, not per-batch).

    The returned dict matches sklearn's structure exactly so downstream display
    and storage code requires no changes.
    """
    # Per-class (vector) metrics
    precision_per = multiclass_precision(preds, targets, num_classes=num_classes, average=None, zero_division=0).tolist()
    recall_per    = multiclass_recall(preds, targets, num_classes=num_classes, average=None, zero_division=0).tolist()
    f1_per        = multiclass_f1_score(preds, targets, num_classes=num_classes, average=None, zero_division=0).tolist()

    # Macro / micro / weighted aggregates
    precision_macro = multiclass_precision(preds, targets, num_classes=num_classes, average='macro', zero_division=0).item()
    recall_macro    = multiclass_recall(preds, targets, num_classes=num_classes, average='macro', zero_division=0).item()
    f1_macro        = multiclass_f1_score(preds, targets, num_classes=num_classes, average='macro', zero_division=0).item()

    precision_weighted = multiclass_precision(preds, targets, num_classes=num_classes, average='weighted', zero_division=0).item()
    recall_weighted    = multiclass_recall(preds, targets, num_classes=num_classes, average='weighted', zero_division=0).item()
    f1_weighted        = multiclass_f1_score(preds, targets, num_classes=num_classes, average='weighted', zero_division=0).item()

    accuracy = multiclass_accuracy(preds, targets, num_classes=num_classes, average='micro').item()

    # Per-class support — count of true labels in each class
    support_per = torch.bincount(targets.long(), minlength=num_classes).tolist()
    total_support = int(sum(support_per))

    label_names = label_names or [str(i) for i in range(num_classes)]
    report: dict = {}
    for i, name in enumerate(label_names):
        report[name] = {
            'precision': precision_per[i],
            'recall':    recall_per[i],
            'f1-score':  f1_per[i],
            'support':   support_per[i],
        }
    report['accuracy'] = accuracy
    report['macro avg'] = {
        'precision': precision_macro,
        'recall':    recall_macro,
        'f1-score':  f1_macro,
        'support':   total_support,
    }
    report['weighted avg'] = {
        'precision': precision_weighted,
        'recall':    recall_weighted,
        'f1-score':  f1_weighted,
        'support':   total_support,
    }
    return report
```

### 2.4 Caller-side changes

Both helper signatures need a `num_classes: int` parameter (torchmetrics requires it explicitly). All three runners already have `num_classes` in scope (from `config.model_params['num_classes']`), so threading it through is mechanical:

```python
# Before
report = evaluate(model, val_loader, DEVICE, best_state=...)

# After
report = evaluate(model, val_loader, DEVICE, num_classes=num_classes, best_state=...)
```

Same pattern for `collect_predictions` + `build_report` in `shared_evaluate.py`.

---

## 3. Risks

### 3.1 Numerical drift between sklearn and torchmetrics

`sklearn.metrics.classification_report` and `torchmetrics.functional.classification.*` are known to **agree to within ~1e-6** for `average='macro' | 'micro' | 'weighted'`. Per-class F1 may differ in the 1e-4 range under specific edge cases:

- **Empty classes**: sklearn defaults to `zero_division=0` and emits a UserWarning; torchmetrics defaults to `zero_division=0.0` (returns 0.0) — same behavior, silent.
- **Macro avg with empty classes**: sklearn averages over PRESENT classes only (non-NaN); torchmetrics averages over ALL classes including the empty ones (treating them as 0). This can shift macro-F1 by a few percent on tiny eval sets where some classes never appear in val.
- **Weighted avg**: equivalent.

**Mitigation**: regression test must compare a known fold's report between the old sklearn path and the new torchmetrics path on the same fixed seed. Tolerance band: ±0.005 absolute on macro F1, ±0.01 absolute on per-class F1. If a class has 0 support in val, accept either definition.

### 3.2 Display/storage breakage

Downstream consumers may depend on quirks of sklearn's dict structure that I missed. The byte-for-byte structural match in §2.3 is the only safety gate. **Test by running a category-only smoke (`--task category --folds 2 --epochs 5`) and checking that summary CSVs and plots in `results/<engine>/<state>/<run>/` are byte-identical to a baseline run.**

### 3.3 The MTL hot path uses scalar metrics, not the report dict

PR #8's MTL port only computes scalar `f1_macro` + `accuracy` via torchmetrics — it does NOT build the per-class dict. The eval helpers DO need the per-class dict because the display layer renders per-class precision/recall/f1 tables. The adapter in §2.3 is the bridge; do not try to reuse PR #8's pattern verbatim.

### 3.4 Sklearn import removal

Once both files are migrated, `from sklearn.metrics import classification_report` can be deleted from both. **Do NOT remove sklearn from `pyproject.toml`** — `src/data/folds.py` still uses `StratifiedKFold` from `sklearn.model_selection`, and several tests rely on `sklearn.metrics.f1_score` for parity checks against torchmetrics. Sklearn stays as a transitive dep.

---

## 4. Test plan

| Step | Command | Expected |
|---|---|---|
| 1. Pre-port baseline | `pytest tests/test_training tests/test_models -q` on current main | Note count (e.g. 240 passed) |
| 2. Port `evaluate.py` and `shared_evaluate.py` to use `_torchmetrics_classification_report` | — | — |
| 3. Update runner call sites in `category_cv.py`, `next_cv.py`, `mtl_cv.py` if they call these helpers | — | — |
| 4. Re-run baseline tests | `pytest tests/test_training tests/test_models -q` | Same count, no failures |
| 5. Numerical parity check | `python scripts/train.py --task category --state alabama --engine dgi --folds 2 --epochs 10` then diff its `summary/full_summary.json` against a sklearn-baseline run with the same seed | All scalar metrics within ±0.005, per-class within ±0.01 |
| 6. Same for next | `python scripts/train.py --task next --state alabama --engine dgi --folds 2 --epochs 10` | Same |
| 7. MTL regression | `python scripts/train.py --task mtl --state alabama --engine dgi --folds 5 --epochs 50` | Bit-exact match against torch 2.11 baseline (Cat F1 0.4435, Next F1 0.2603) — these helpers are not on the MTL hot path, so MTL metrics should be unchanged |
| 8. Display smoke | Inspect `results/dgi/alabama/<run>/summary/` for the new run | All CSVs render, no missing columns, support sums match |

If steps 5–6 show drift > tolerance on macro F1 specifically, the most likely cause is the empty-class handling difference described in §3.1 — investigate before merging.

---

## 5. Out of scope for this port

- Removing sklearn from `pyproject.toml` (kept for `StratifiedKFold` and test parity).
- Migrating `src/training/runners/mtl_cv.py` further — already on torchmetrics for hot-path metrics; the post-training helper calls (which use `evaluate()`) will benefit transitively from this port.
- Adding new metrics (top-k accuracy, confusion matrix display, ROC) — separate PR.
- Changing the MLHistory report aggregation logic — strictly preserve current dict shape.

---

## 6. Why this is low priority

- The current `.cpu().numpy()` fix (commit `1746920`) already unblocks single-task training on MPS. Functionally complete.
- The perf win is bounded by sklearn's `classification_report` runtime on ~12k samples ≈ 5–10 ms per call, ≈ 50 ms across a 5-fold run. Negligible vs. ~12 minute Gate 5 wall time.
- The architectural consistency win is real but cosmetic. Two slightly different metric computation paths in the codebase is not a maintenance burden today.

**Recommendation**: do this port only if/when someone is already touching `evaluate.py` / `shared_evaluate.py` for another reason (e.g. adding a new eval metric). Don't carve out a dedicated session for it.
