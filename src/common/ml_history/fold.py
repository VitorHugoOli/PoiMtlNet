"""Shim — canonical location is tracking.fold (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.fold is deprecated; use tracking.fold instead.", DeprecationWarning, stacklevel=2)
from tracking.fold import *  # noqa: F401, F403
