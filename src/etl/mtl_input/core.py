"""Shim — canonical location is data.inputs.core (Phase 5).

This module will be removed at the end of Phase 6.
"""
import warnings as _warnings

_warnings.warn(
    "etl.mtl_input.core is deprecated; use data.inputs.core instead. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)

from data.inputs.core import *  # noqa: F401, F403
