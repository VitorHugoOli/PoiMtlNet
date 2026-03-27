"""Shim — canonical location is data.dataset (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "common.poi_dataset is deprecated; use data.dataset instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from data.dataset import POIDataset  # noqa: F401
