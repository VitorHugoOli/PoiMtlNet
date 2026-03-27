"""Shim — canonical location is losses.pcgrad (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "criterion.pcgrad is deprecated; use losses.pcgrad instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from losses.pcgrad import PCGrad  # noqa: F401
