"""Shim — canonical location is tracking.display (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.display is deprecated; use tracking.display instead.", DeprecationWarning, stacklevel=2)
from tracking.display import *  # noqa: F401, F403
