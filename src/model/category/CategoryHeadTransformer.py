"""Shim — canonical location is models.heads.category (Phase 4a/5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "model.category.CategoryHeadTransformer is deprecated; use models.heads.category instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from models.heads.category import CategoryHeadTransformer  # noqa: F401
