"""Shim — canonical location is models.heads.category (Phase 4a)."""
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
