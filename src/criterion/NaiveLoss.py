"""Shim — canonical location is losses.naive (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "criterion.NaiveLoss is deprecated; use losses.naive instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from losses.naive import NaiveLoss  # noqa: F401
