"""Canonical ``TaskSet`` presets: legacy (category+next) and check2HGI (next+region).

A ``TaskSet`` is a pair ``(task_a, task_b)`` — the MTLnet shared-backbone
topology is always two tasks. Adding a third task would require a
different architecture (``MTLnetCGC`` / ``MTLnetPLE`` and a rewritten
runner); for now we stay with the 2-slot contract to avoid that work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from tasks.registry import PrimaryMetric, TaskConfig


@dataclass(frozen=True)
class TaskSet:
    """Ordered pair of task configs and a preset name.

    ``task_a`` maps to the category-slot in the legacy MTLnet topology
    (flat input by default, first in the tuple passed to ``forward``),
    ``task_b`` maps to the next-slot (sequential input). When both new
    tasks are sequential (check2HGI case), ``task_a.is_sequential`` is
    simply set to True.
    """

    name: str
    task_a: TaskConfig
    task_b: TaskConfig

    def __iter__(self) -> Iterator[TaskConfig]:
        yield self.task_a
        yield self.task_b

    @property
    def names(self) -> tuple[str, str]:
        return self.task_a.name, self.task_b.name


LEGACY_CATEGORY_NEXT = TaskSet(
    name="legacy_category_next",
    task_a=TaskConfig(
        name="category",
        num_classes=7,
        head_factory="category_transformer",
        is_sequential=False,
        primary_metric=PrimaryMetric.F1,
    ),
    task_b=TaskConfig(
        name="next",
        num_classes=7,
        head_factory="next_mtl",
        is_sequential=True,
        primary_metric=PrimaryMetric.F1,
    ),
)
"""The historical MTLnet pair. Defaults reproduce the pre-parameterisation
behaviour bit-exactly; do not change these fields without recalibrating
``tests/test_regression``."""


CHECK2HGI_NEXT_REGION = TaskSet(
    name="check2hgi_next_region",
    task_a=TaskConfig(
        name="next_category",
        num_classes=7,
        head_factory="next_mtl",
        is_sequential=True,
        primary_metric=PrimaryMetric.F1,
    ),
    task_b=TaskConfig(
        # num_classes is a placeholder — the runner resolves it from the
        # check2HGI region map before constructing the model. Alabama
        # has 1109 regions, Florida is similar in scale; see P-1 in
        # docs/plans/CHECK2HGI_MTL_BRANCH_PLAN.md.
        name="next_region",
        num_classes=0,
        head_factory="next_mtl",
        is_sequential=True,
        primary_metric=PrimaryMetric.ACCURACY,
    ),
)
"""Check2HGI-track pair: both heads are sequential NextHead transformers.
Next-category keeps macro-F1 as primary (category space is 7 and
imbalanced); next-region primary is Acc@1 because region cardinality is
~10^3 (ranking-metric regime, per HMT-GRN / MGCL)."""


_PRESETS: Dict[str, TaskSet] = {
    LEGACY_CATEGORY_NEXT.name: LEGACY_CATEGORY_NEXT,
    CHECK2HGI_NEXT_REGION.name: CHECK2HGI_NEXT_REGION,
}


def get_preset(name: str) -> TaskSet:
    if name not in _PRESETS:
        raise KeyError(
            f"TaskSet preset '{name}' not found. Available: {sorted(_PRESETS)}"
        )
    return _PRESETS[name]


def list_presets() -> list[str]:
    return sorted(_PRESETS)


__all__ = [
    "TaskSet",
    "LEGACY_CATEGORY_NEXT",
    "CHECK2HGI_NEXT_REGION",
    "get_preset",
    "list_presets",
]
