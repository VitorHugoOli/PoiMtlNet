"""Shim — canonical location is utils.progress (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.training_progress is deprecated; use utils.progress instead.", DeprecationWarning, stacklevel=2)
from utils.progress import *  # noqa: F401, F403
