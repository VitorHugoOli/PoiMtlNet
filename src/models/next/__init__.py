"""Next-head domain."""

from .head import (
    NextHeadGRU,
    NextHeadHybrid,
    NextHeadLSTM,
    NextHeadMTL,
    NextHeadSingle,
    NextHeadTemporalCNN,
    NextHeadTransformerOptimized,
)

__all__ = [
    "NextHeadSingle",
    "NextHeadMTL",
    "NextHeadLSTM",
    "NextHeadGRU",
    "NextHeadTemporalCNN",
    "NextHeadHybrid",
    "NextHeadTransformerOptimized",
]
