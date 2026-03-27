"""Shim — canonical location is models.heads.next (Phase 4a/5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "model.next.next_head is deprecated; use models.heads.next instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from models.components.positional import PositionalEncoding  # noqa: F401
from models.heads.next import NextHeadSingle  # noqa: F401
