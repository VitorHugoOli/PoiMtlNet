"""Shim — canonical location is tracking.utils.time_history (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.utils.time_history is deprecated; use tracking.utils.time_history instead.", DeprecationWarning, stacklevel=2)
from tracking.utils.time_history import *  # noqa: F401, F403
