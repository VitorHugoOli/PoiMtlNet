"""Shim — canonical location is training.runners.next_trainer (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.next.trainer is deprecated; use training.runners.next_trainer instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.next_trainer import train  # noqa: F401
