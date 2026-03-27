"""Shim — canonical location is models.heads.category (Phase 4a/5).

Phase 1 contract: CategoryHeadMTL is an alias for CategoryHeadEnsemble.
This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "model.mtlnet.category_head is deprecated; use models.heads.category instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from models.heads.category import CategoryHeadMTL  # noqa: F401
