"""Shim — canonical location is data.inputs.fusion (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "etl.mtl_input.fusion is deprecated; use data.inputs.fusion instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from data.inputs.fusion import *  # noqa: F401, F403
