"""Shim — canonical location is losses.focal (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "criterion.FocalLoss is deprecated; use losses.focal instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from losses.focal import FocalLoss  # noqa: F401
