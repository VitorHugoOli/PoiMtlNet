"""Next-head domain."""

from .next_gru.head import NextHeadGRU
from .next_hybrid.head import NextHeadHybrid
from .next_lstm.head import NextHeadLSTM
from .next_mtl.head import NextHeadMTL
from .next_single.head import NextHeadSingle
from .next_temporal_cnn.head import NextHeadTemporalCNN
from .next_transformer_optimized.head import NextHeadTransformerOptimized

__all__ = [
    "NextHeadSingle",
    "NextHeadMTL",
    "NextHeadLSTM",
    "NextHeadGRU",
    "NextHeadTemporalCNN",
    "NextHeadHybrid",
    "NextHeadTransformerOptimized",
]
