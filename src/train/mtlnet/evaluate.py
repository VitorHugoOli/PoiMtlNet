"""Shim — canonical location is training.runners.mtl_eval (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.mtlnet.evaluate is deprecated; use training.runners.mtl_eval instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.mtl_eval import evaluate_model  # noqa: F401
