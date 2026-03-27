"""Shim — canonical location is data.folds (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "etl.create_fold is deprecated; use data.folds instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from data.folds import (  # noqa: F401
    TaskType,
    FoldConfig,
    FoldResult,
    TaskFoldData,
    POIDataset,
    FoldCreator,
)
