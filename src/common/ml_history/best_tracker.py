"""Shim — canonical location is tracking.best_tracker (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.best_tracker is deprecated; use tracking.best_tracker instead.", DeprecationWarning, stacklevel=2)
from tracking.best_tracker import *  # noqa: F401, F403
