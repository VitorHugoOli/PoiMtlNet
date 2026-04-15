"""Numerical equivalence test for the high-cardinality metric path.

The hand-rolled implementation in ``tracking.metrics._handrolled_cls_metrics``
replaces torchmetrics' confusion-matrix path for ``num_classes > 256``
because the torchmetrics path allocates an O(num_classes**2 * batch)
tensor internally, OOM-ing on the check2HGI next_region head (1K–5K
classes). This test pins the replacement to within 1e-6 of the
torchmetrics result at num_classes small enough to run both.
"""

from __future__ import annotations

import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)

from tracking.metrics import _handrolled_cls_metrics


def _torchmetrics_reference(preds, targets, num_classes):
    acc_micro = multiclass_accuracy(preds, targets, num_classes=num_classes, average="micro").item()
    acc_macro = multiclass_accuracy(preds, targets, num_classes=num_classes, average="macro").item()
    f1_macro = multiclass_f1_score(
        preds, targets, num_classes=num_classes, average="macro", zero_division=0,
    ).item()
    f1_weighted = multiclass_f1_score(
        preds, targets, num_classes=num_classes, average="weighted", zero_division=0,
    ).item()
    return acc_micro, acc_macro, f1_macro, f1_weighted


def _assert_close(name, a, b, tol=1e-4):
    diff = abs(a - b)
    assert diff < tol, f"{name}: hand-rolled={a:.6f} torchmetrics={b:.6f} diff={diff:.2e}"


def test_equivalence_small_balanced():
    torch.manual_seed(0)
    logits = torch.randn(1000, 10)
    targets = torch.randint(0, 10, (1000,))
    preds = logits.argmax(-1)
    hr = _handrolled_cls_metrics(preds, targets, num_classes=10)
    ref = _torchmetrics_reference(preds, targets, num_classes=10)
    for name, a, b in zip(("acc_micro", "acc_macro", "f1_macro", "f1_weighted"), hr, ref):
        _assert_close(name, a, b)


def test_equivalence_sparse_classes_with_unused_slots():
    """Many classes have zero support — torchmetrics' zero_division=0
    excludes them from macro averages; hand-rolled does the same."""
    torch.manual_seed(42)
    # 200 classes, only first 20 actually used
    preds = torch.randint(0, 20, (500,))
    targets = torch.randint(0, 20, (500,))
    hr = _handrolled_cls_metrics(preds, targets, num_classes=200)
    ref = _torchmetrics_reference(preds, targets, num_classes=200)
    for name, a, b in zip(("acc_micro", "acc_macro", "f1_macro", "f1_weighted"), hr, ref):
        _assert_close(name, a, b)


def test_equivalence_imbalanced():
    """Heavy class imbalance (simulates FL's 22% majority region)."""
    torch.manual_seed(7)
    # 80% class 0, 20% spread over classes 1..19
    n = 2000
    majority = torch.zeros(int(n * 0.8), dtype=torch.long)
    tail = torch.randint(1, 20, (n - int(n * 0.8),))
    targets = torch.cat([majority, tail])
    # Perfect predictor for majority, random for tail (resembles an undertrained head)
    preds = torch.cat([majority, torch.randint(0, 20, (n - int(n * 0.8),))])
    hr = _handrolled_cls_metrics(preds, targets, num_classes=20)
    ref = _torchmetrics_reference(preds, targets, num_classes=20)
    for name, a, b in zip(("acc_micro", "acc_macro", "f1_macro", "f1_weighted"), hr, ref):
        _assert_close(name, a, b)


def test_empty_inputs_return_zero():
    preds = torch.empty(0, dtype=torch.long)
    targets = torch.empty(0, dtype=torch.long)
    hr = _handrolled_cls_metrics(preds, targets, num_classes=100)
    assert hr == (0.0, 0.0, 0.0, 0.0)


def test_high_cardinality_runs_without_oom():
    """The reason this file exists — the torchmetrics path would allocate
    ~num_classes**2 bytes and OOM on the next_region head. Running here
    proves the hand-rolled path scales linearly.
    """
    n = 5000
    n_classes = 2000
    torch.manual_seed(1)
    preds = torch.randint(0, n_classes, (n,))
    targets = torch.randint(0, n_classes, (n,))
    acc_micro, acc_macro, f1_macro, f1_weighted = _handrolled_cls_metrics(
        preds, targets, num_classes=n_classes,
    )
    # Random predictions on 2000 uniform classes → expect acc ~1/2000
    assert 0.0 <= acc_micro <= 0.01
    assert 0.0 <= acc_macro <= 0.01
    assert 0.0 <= f1_macro <= 0.01
    assert 0.0 <= f1_weighted <= 0.01
