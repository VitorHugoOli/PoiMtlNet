"""Shim — canonical location is losses.gradnorm (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "criterion.gradnorm is deprecated; use losses.gradnorm instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from losses.gradnorm import GradNormLoss  # noqa: F401
