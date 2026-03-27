"""Shim — canonical location is models.heads.next (Phase 4a/5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "model.mtlnet.next_head is deprecated; use models.heads.next instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from models.heads.next import NextHeadMTL  # noqa: F401
