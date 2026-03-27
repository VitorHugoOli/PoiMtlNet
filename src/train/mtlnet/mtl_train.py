"""Shim — canonical location is training.runners.mtl_cv (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "train.mtlnet.mtl_train is deprecated; use training.runners.mtl_cv instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from training.runners.mtl_cv import (  # noqa: F401
    train_model,
    train_with_cross_validation,
)
