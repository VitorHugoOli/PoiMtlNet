"""Shim — canonical location is losses.nash_mtl (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "criterion.nash_mtl is deprecated; use losses.nash_mtl instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from losses.nash_mtl import NashMTL  # noqa: F401
