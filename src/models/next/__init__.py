"""Next-head domain."""

from .next_conv_attn.head import NextHeadConvAttn
from .next_gru.head import NextHeadGRU
from .next_hybrid.head import NextHeadHybrid
from .next_lstm.head import NextHeadLSTM
from .next_mtl.head import NextHeadMTL
from .next_single.head import NextHeadSingle
from .next_stan.head import NextHeadSTAN
from .next_tcn_residual.head import NextHeadTCNResidual
from .next_temporal_cnn.head import NextHeadTemporalCNN
from .next_transformer_optimized.head import NextHeadTransformerOptimized
from .next_transformer_relpos.head import NextHeadTransformerRelPos

__all__ = [
    "NextHeadSingle",
    "NextHeadMTL",
    "NextHeadLSTM",
    "NextHeadGRU",
    "NextHeadSTAN",
    "NextHeadTemporalCNN",
    "NextHeadHybrid",
    "NextHeadTransformerOptimized",
    "NextHeadTCNResidual",
    "NextHeadConvAttn",
    "NextHeadTransformerRelPos",
]
