"""Shim — canonical location is training.runners.next_eval (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.next.evaluation is deprecated; use training.runners.next_eval instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.next_eval import evaluate  # noqa: F401
