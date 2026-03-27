"""Shim — canonical location is training.runners.mtl_validation (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.mtlnet.validation is deprecated; use training.runners.mtl_validation instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.mtl_validation import validation_best_model  # noqa: F401
