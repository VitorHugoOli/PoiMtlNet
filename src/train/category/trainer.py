"""Shim — canonical location is training.runners.category_trainer (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.category.trainer is deprecated; use training.runners.category_trainer instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.category_trainer import train  # noqa: F401
