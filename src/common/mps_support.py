"""Shim — canonical location is utils.mps (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.mps_support is deprecated; use utils.mps instead.", DeprecationWarning, stacklevel=2)
from utils.mps import *  # noqa: F401, F403
