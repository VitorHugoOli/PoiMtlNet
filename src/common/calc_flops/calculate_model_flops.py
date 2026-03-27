"""Shim — canonical location is utils.flops (Phase 5)."""
import warnings as _warnings
_warnings.warn("common.calc_flops.calculate_model_flops is deprecated; use utils.flops instead.", DeprecationWarning, stacklevel=2)
from utils.flops import *  # noqa: F401, F403
