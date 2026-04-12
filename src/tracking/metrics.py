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

import math
from typing import Dict, Iterable, Tuple

import torch
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_f1_score,
)


def _top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Fraction of samples whose true class is in the top-k predictions."""
    if k <= 1:
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()
    k_eff = min(k, logits.shape[-1])
    top_k = logits.topk(k_eff, dim=-1).indices  # (N, k_eff)
    hit = (top_k == targets.unsqueeze(-1)).any(dim=-1)
    return hit.float().mean().item()


def _rank_of_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """1-indexed rank of the true class in each row's descending sort.

    Uses ``argsort`` rather than ``multiclass_accuracy(top_k=...)`` because we
    want the actual rank, not a thresholded hit. Ties are broken by the
    underlying sort stability (same contract as ``torch.argsort``).
    """
    order = logits.argsort(dim=-1, descending=True)         # (N, C)
    matches = (order == targets.unsqueeze(-1)).int()         # (N, C)
    rank_zero_indexed = matches.argmax(dim=-1)               # (N,)
    return rank_zero_indexed + 1                              # 1-indexed


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

    Edge cases:
        * Empty batch (``N == 0``): returns all zeros. Callers should guard
          against this upstream, but we handle it defensively so tests with
          degenerate loaders don't crash.
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
