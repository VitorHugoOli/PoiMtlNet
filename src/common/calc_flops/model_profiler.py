"""Shim — canonical location is utils.profiler (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.calc_flops.model_profiler is deprecated; use utils.profiler instead.", DeprecationWarning, stacklevel=2)
from utils.profiler import *  # noqa: F401, F403
