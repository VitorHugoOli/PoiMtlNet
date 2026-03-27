"""Shim — canonical location is models.heads.next (Phase 4a)."""
from models.heads.next import (  # noqa: F401
    NextHeadLSTM,
    NextHeadGRU,
    NextHeadTemporalCNN,
    NextHeadHybrid,
    NextHeadTransformerOptimized,
)
