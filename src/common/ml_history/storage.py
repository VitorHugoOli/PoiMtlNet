"""Shim — canonical location is tracking.storage (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.storage is deprecated; use tracking.storage instead.", DeprecationWarning, stacklevel=2)
from tracking.storage import *  # noqa: F401, F403
