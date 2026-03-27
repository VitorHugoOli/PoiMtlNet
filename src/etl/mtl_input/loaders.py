"""Shim — canonical location is data.inputs.loaders (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "etl.mtl_input.loaders is deprecated; use data.inputs.loaders instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from data.inputs.loaders import *  # noqa: F401, F403
