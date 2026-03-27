"""Shim — canonical location is tracking.metric_store (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.metric_store is deprecated; use tracking.metric_store instead.", DeprecationWarning, stacklevel=2)
from tracking.metric_store import *  # noqa: F401, F403
