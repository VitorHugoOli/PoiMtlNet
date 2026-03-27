"""Shim — canonical location is training.shared_evaluate (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.shared.evaluate is deprecated; use training.shared_evaluate instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.shared_evaluate import evaluate  # noqa: F401
