"""Shim — canonical location is data.inputs.builders (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "etl.mtl_input.builders is deprecated; use data.inputs.builders instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from data.inputs.builders import *  # noqa: F401, F403
