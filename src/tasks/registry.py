"""``TaskConfig`` — per-slot description of an MTL head.

Keeping the dataclass narrow on purpose: fields are what the model and
runner actually read. Wider knobs (loss weights, class-weight overrides)
belong on ``ExperimentConfig`` where the existing knobs already live.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PrimaryMetric(str, Enum):
    """Which scalar drives checkpoint monitoring for this head.

    Key names match ``tracking.metrics.compute_classification_metrics``
    (``f1`` = macro-F1, ``accuracy`` = Acc@1, ``mrr`` = mean reciprocal
    rank), so a ``PrimaryMetric`` value is a valid monitor key after the
    per-task ``val_<name>_`` prefix is added.
    """

    F1 = "f1"
    ACCURACY = "accuracy"
    MRR = "mrr"


@dataclass(frozen=True)
class TaskConfig:
    """Per-slot task description consumed by MTLnet and the runner.

    Attributes:
        name: stable task label used in metric keys, tracking, and logs.
            Legacy slots are ``"category"`` and ``"next"``.
        num_classes: label-space size for this head. Data-derived for
            ``next_region`` (run-time); fixed at 7 for category/next.
        head_factory: model-registry name for the head. Legacy uses
            ``"category_transformer"`` and ``"next_mtl"``.
        head_params: extra kwargs for the head factory. ``None`` triggers
            the legacy-default construction path in ``MTLnet`` (see the
            bit-exact contract in ``MTLnet._build_{category,next}_head``).
        is_sequential: ``True`` if the head expects ``[B, T, D]`` inputs
            with left-padding masking, ``False`` for ``[B, D]`` flat input.
            Category-slot defaults to flat; next-slot and region heads are
            sequential.
        primary_metric: which metric drives the per-head checkpoint monitor.
            Category/next_category default to F1; next_region defaults to
            Accuracy (Acc@1) per the HMT-GRN / MGCL ranking-metric lineage
            — see ``docs/plans/CHECK2HGI_MTL_OVERVIEW.md`` §2.
    """

    name: str
    num_classes: int
    head_factory: str
    head_params: Optional[dict] = None
    is_sequential: bool = True
    primary_metric: PrimaryMetric = PrimaryMetric.F1


__all__ = ["PrimaryMetric", "TaskConfig"]
