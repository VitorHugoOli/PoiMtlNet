"""Shim — canonical location is models.mtlnet (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "model.mtlnet.mtl_poi is deprecated; use models.mtlnet instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from models.mtlnet import (  # noqa: F401
    ResidualBlock,
    FiLMLayer,
    MTLnet,
)
