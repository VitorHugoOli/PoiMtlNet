"""Shim — canonical location is tracking/ (Phase 5).

This package will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "common.ml_history is deprecated; use tracking instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from tracking.experiment import MLHistory, FlopsMetrics  # noqa: F401
from tracking.fold import FoldHistory, TaskHistory  # noqa: F401
from tracking.metric_store import MetricStore  # noqa: F401
from tracking.best_tracker import BestModelTracker  # noqa: F401
from tracking.parms.neural import HyperParams, NeuralParams  # noqa: F401
from tracking.utils.dataset import DatasetHistory  # noqa: F401

__all__ = [
    'MLHistory',
    'FlopsMetrics',
    'FoldHistory',
    'TaskHistory',
    'MetricStore',
    'BestModelTracker',
    'NeuralParams',
    'HyperParams',
    'DatasetHistory',
]
