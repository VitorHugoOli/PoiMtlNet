"""Shim — canonical location is training.runners.category_cv (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.category.cross_validation is deprecated; use training.runners.category_cv instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.category_cv import run_cv  # noqa: F401
