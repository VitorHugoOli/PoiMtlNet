"""Shim — canonical location is tracking.experiment (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.ml_history.experiment is deprecated; use tracking.experiment instead.", DeprecationWarning, stacklevel=2)
from tracking.experiment import *  # noqa: F401, F403
