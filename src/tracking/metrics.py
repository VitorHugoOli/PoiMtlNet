"""Classification metric computation for POI tasks.

A single pure function converts a batch of logits + targets into a flat dict
of scalar Python floats. The dict keys are stable names (optionally prefixed)
that downstream ``MetricStore.log(**kwargs)`` calls can consume directly.

Metric choices are grounded in ``docs/LITERATURE_SURVEY_POI_METRICS.md`` and
``docs/POI_RELATED_WORK_METRICS.md``:

  * ``accuracy`` (micro) and ``f1`` (macro) — unchanged primary metrics.
  * ``accuracy_macro`` — balanced accuracy, complements micro under imbalance.
  * ``f1_weighted`` — robustness-to-imbalance secondary signal.
  * ``top{k}_acc`` for k in ``top_k`` — bridge to the next-POI recommendation
    literature (Acc@K / Recall@K in the single-label case).
  * ``mrr`` — mean reciprocal rank over the full class ranking.
  * ``ndcg_{k}`` — NDCG@K with binary single-item relevance (reduces to the
    well-defined single-label form used by ranking papers).

The ``'f1'`` key stays identical to the previous
``multiclass_f1_score(..., average='macro')`` value so existing
``BestModelTracker(monitor='f1')`` behaviour is byte-identical.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)


def _handrolled_cls_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[float, float, float, float]:
    """Memory-O(N + num_classes) Acc / macro-F1 for high-cardinality heads.

    torchmetrics' ``multiclass_accuracy`` / ``multiclass_f1_score``
    build an internal confusion matrix of size ``num_classes**2``
    via ``bincount(minlength=num_classes**2)``. That allocation is
    ~30GB for 1K classes × 6K batch and OOMs on any device.

    This implementation uses per-class TP / FP / FN counts via
    ``torch.bincount`` on the 1-D predicted and target class streams —
    no ``num_classes**2`` tensor anywhere. Matches torchmetrics output
    for micro-accuracy, macro-accuracy (balanced) and macro/weighted F1
    on single-label multi-class data.
    """
    n = preds.numel()
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    correct = (preds == targets).to(torch.float32)
    acc_micro = correct.mean().item()

    # Per-class TP / (TP + FN) = recall; we also need precision via FP.
    # ``bincount(targets, weights=correct)`` → per-class TP counts.
    # ``bincount(targets)``                 → per-class support (TP + FN).
    # ``bincount(preds)``                   → per-class predicted (TP + FP).
    tp = torch.bincount(targets, weights=correct, minlength=num_classes)
    support = torch.bincount(targets, minlength=num_classes).to(torch.float32)
    predicted = torch.bincount(preds, minlength=num_classes).to(torch.float32)

    # macro-accuracy = mean over per-class recall on classes with support.
    recall_per_class = torch.where(
        support > 0, tp / support.clamp_min(1.0), torch.zeros_like(tp),
    )
    n_present = int((support > 0).sum().item())
    acc_macro = float(recall_per_class.sum().item() / max(n_present, 1))

    precision_per_class = torch.where(
        predicted > 0, tp / predicted.clamp_min(1.0), torch.zeros_like(tp),
    )
    # F1 = 2PR/(P+R), 0 when P+R==0.
    denom = precision_per_class + recall_per_class
    f1_per_class = torch.where(
        denom > 0,
        2 * precision_per_class * recall_per_class / denom.clamp_min(1e-12),
        torch.zeros_like(denom),
    )
    # macro F1 over *present* classes (matches torchmetrics' zero_division=0
    # behaviour for absent classes — they're excluded, not averaged as 0).
    f1_macro = float(f1_per_class[support > 0].mean().item()) if n_present > 0 else 0.0
    # Weighted F1 — weights = support / total support.
    total_support = support.sum().clamp_min(1.0)
    f1_weighted = float((f1_per_class * (support / total_support)).sum().item())

    return acc_micro, acc_macro, f1_macro, f1_weighted


def _top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Fraction of samples whose true class is in the top-k predictions.

    Caller must pass ``k >= 2`` — the main API skips ``k == 1`` since that
    would duplicate ``accuracy``.
    """
    k_eff = min(k, logits.shape[-1])
    top_k = logits.topk(k_eff, dim=-1).indices  # (N, k_eff)
    hit = (top_k == targets.unsqueeze(-1)).any(dim=-1)
    return hit.float().mean().item()


def _rank_of_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-indexed "best-case" rank of the true class in each row.

    Computed as ``1 + #{classes with a strictly higher logit than the target}``.
    This is the canonical tie-handling for ranking metrics: if two classes
    share the top score and one of them is the target, the target ranks 1,
    not 2. An ``argsort``-based implementation would break ties by the
    platform's sort stability, which is an implementation detail the caller
    shouldn't depend on.

    Returns an int64 tensor of shape ``(N,)``.
    """
    # (N, 1) gather — score of the true class per row
    target_scores = logits.gather(dim=-1, index=targets.unsqueeze(-1))
    # Count strictly higher-scoring classes; add 1 for the target itself.
    higher = (logits > target_scores).sum(dim=-1)
    return higher + 1


def _mean_reciprocal_rank(logits: torch.Tensor, targets: torch.Tensor) -> float:
    rank = _rank_of_target(logits, targets).float()
    return (1.0 / rank).mean().item()


def _ndcg_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """NDCG@K with single-relevant-item binary relevance.

    IDCG is 1 (one relevant item, perfectly ranked). DCG is
    ``1 / log2(rank + 1)`` if the true class falls in the top-k, else 0.
    """
    rank = _rank_of_target(logits, targets)                  # (N,)
    hit = rank <= k
    # Use float math; add 1 so rank=1 -> log2(2) = 1 -> DCG=1.
    dcg = torch.where(
        hit,
        1.0 / torch.log2(rank.float() + 1.0),
        torch.zeros_like(rank, dtype=torch.float32),
    )
    return dcg.mean().item()


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    top_k: Iterable[int] = (3, 5),
    prefix: str = "",
) -> Dict[str, float]:
    """Compute every classification/ranking metric we track, in one pass.

    Args:
        logits: ``(N, C)`` float tensor. May live on CPU, CUDA or MPS.
        targets: ``(N,)`` int tensor of class indices.
        num_classes: Number of classes ``C``. Required by torchmetrics.
        top_k: K values to report Top-K accuracy and NDCG@K for. ``k=1`` is
            skipped because it would duplicate ``accuracy``.
        prefix: Optional key prefix (e.g. ``"val_"``). Applied to every
            returned key.

    Returns:
        ``dict[str, float]``: every value is a plain Python float, safe to
        pass straight into ``MetricStore.log(**kwargs)``.

    Empty batch (``N == 0``) is tolerated and returns zeros for every key
    the non-empty path would have produced — this keeps test harnesses with
    degenerate loaders happy without forcing a guard at every call site.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C); got shape {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1-D (N,); got shape {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"logits N={logits.shape[0]} != targets N={targets.shape[0]}"
        )

    def _key(name: str) -> str:
        return f"{prefix}{name}"

    # Degenerate batch — return zeros for every key we would have produced.
    if logits.shape[0] == 0:
        out: Dict[str, float] = {
            _key("accuracy"): 0.0,
            _key("accuracy_macro"): 0.0,
            _key("f1"): 0.0,
            _key("f1_weighted"): 0.0,
            _key("mrr"): 0.0,
        }
        for k in top_k:
            if k > 1:
                out[_key(f"top{k}_acc")] = 0.0
                out[_key(f"ndcg_{k}")] = 0.0
        return out

    # Targets and logits must share a device for torchmetrics / torch ops.
    if targets.device != logits.device:
        targets = targets.to(logits.device)

    # torchmetrics expects integer predictions for the top-1 path.
    preds = logits.argmax(dim=-1)

    # High-cardinality heads (e.g. next_region with ~10^3 classes) blow up
    # **any** device in torchmetrics' confusion-matrix path — bincount
    # allocates ``num_classes**2 * batch_size`` floats internally. For
    # 1K classes × 6K batch that's ~30GB regardless of CPU/MPS/CUDA.
    # Above the threshold, use hand-rolled per-class accumulation via
    # ``torch.bincount`` on 1-D inputs, which stays O(N + num_classes)
    # memory and is numerically equivalent to torchmetrics for the
    # micro/macro-averaged accuracy and F1 we report.
    _CARDINALITY_HAND_ROLLED_THRESHOLD = 256
    if num_classes > _CARDINALITY_HAND_ROLLED_THRESHOLD:
        acc_micro, acc_macro, f1_macro, f1_weighted = _handrolled_cls_metrics(
            preds, targets, num_classes,
        )
    else:
        acc_micro = multiclass_accuracy(
            preds, targets, num_classes=num_classes, average="micro"
        ).item()
        acc_macro = multiclass_accuracy(
            preds, targets, num_classes=num_classes, average="macro"
        ).item()
        f1_macro = multiclass_f1_score(
            preds, targets, num_classes=num_classes, average="macro", zero_division=0
        ).item()
        f1_weighted = multiclass_f1_score(
            preds, targets, num_classes=num_classes, average="weighted", zero_division=0
        ).item()

    metrics: Dict[str, float] = {
        _key("accuracy"): acc_micro,
        _key("accuracy_macro"): acc_macro,
        _key("f1"): f1_macro,
        _key("f1_weighted"): f1_weighted,
        _key("mrr"): _mean_reciprocal_rank(logits, targets),
    }

    for k in top_k:
        if k <= 1:
            continue
        metrics[_key(f"top{k}_acc")] = _top_k_accuracy(logits, targets, k)
        metrics[_key(f"ndcg_{k}")] = _ndcg_at_k(logits, targets, k)

    return metrics


__all__ = ["compute_classification_metrics"]
