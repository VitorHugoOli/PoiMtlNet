"""Shim — canonical location is models.heads.category (Phase 4a/5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "model.category.category_head_enhanced is deprecated; use models.heads.category instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from models.heads.category import (  # noqa: F401
    CategoryHeadResidual,
    CategoryHeadGated,
    CategoryHeadEnsemble,
    CategoryHeadAttentionPooling,
    CategoryHeadMTL,
)
# Backward-compat: old code imports ResidualBlock from here
from models.heads.category import _CategoryResidualBlock as ResidualBlock  # noqa: F401
# Backward-compat: old code imports GatedLayer from here
from models.heads.category import _GatedLayer as GatedLayer  # noqa: F401
