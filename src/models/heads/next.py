"""All next-POI head variants — consolidated in Phase 4a.

Registry names:
    next_single              — NextHeadSingle (Transformer + attention pooling)
    next_mtl                 — NextHeadMTL (sinusoidal PE + causal mask)
    next_lstm                — NextHeadLSTM
    next_gru                 — NextHeadGRU
    next_temporal_cnn        — NextHeadTemporalCNN
    next_hybrid              — NextHeadHybrid (GRU + self-attention)
    next_transformer_optimized — NextHeadTransformerOptimized
"""

import torch
import torch.nn as nn

from models.components.positional import PositionalEncoding
from models.registry import register_model


# ---------------------------------------------------------------------------
# NextHeadSingle (standalone Transformer)
# ---------------------------------------------------------------------------
@register_model("next_single")
class NextHeadSingle(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, seq_length, num_layers, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length

        # Learned positional embeddings (better than sinusoidal for short fixed-length sequences)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.sequence_reduction = nn.Linear(embed_dim, 1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Temporal decay bias for attention pooling (recent visits weighted more)
        self.temporal_bias = nn.Parameter(torch.linspace(-2.0, 0.0, seq_length))

    def forward(self, x, return_attention=False):
        """
        x: (batch_size, seq_length, embed_dim)
        return_attention: If True, return (logits, attn_weights)
        """
        batch_size, seq_length, _ = x.size()

        padding_mask = (x.abs().sum(dim=-1) == 0)

        mask = (~padding_mask).unsqueeze(-1).float()
        x = (x + self.pos_embedding[:, :seq_length, :]) * mask
        x = self.dropout(x)

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        attn_logits = self.sequence_reduction(x).squeeze(-1)
        attn_logits = attn_logits + self.temporal_bias[:seq_length].unsqueeze(0)
        attn_logits = attn_logits.masked_fill(padding_mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        out = self.classifier(pooled)

        if return_attention:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# NextHeadMTL (used inside MTLnet)
# ---------------------------------------------------------------------------
@register_model("next_mtl")
class NextHeadMTL(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, seq_length, num_layers, dropout=0.35):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length

        self.pe = PositionalEncoding(embed_dim, max_seq_length=seq_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.sequence_reduction = nn.Linear(embed_dim, 1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        device = x.device

        padding_mask = (x.abs().sum(dim=-1) == 0)
        x = self.pe(x, padding_mask)

        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=device),
            diagonal=1
        )

        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        attn_logits = self.sequence_reduction(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(padding_mask, float('-inf'))

        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0).unsqueeze(-1)

        pooled = torch.sum(x * attn_weights, dim=1)
        out = self.classifier(pooled)
        return out


# ---------------------------------------------------------------------------
# NextHeadLSTM
# ---------------------------------------------------------------------------
@register_model("next_lstm")
class NextHeadLSTM(nn.Module):
    """LSTM-based next-POI predictor."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = (x.abs().sum(dim=-1) == 0)
        output, (h_n, c_n) = self.lstm(x)
        seq_lengths = (~padding_mask).sum(dim=1)

        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(output[i, last_idx])

        last_output = torch.stack(last_outputs)
        return self.classifier(last_output)


# ---------------------------------------------------------------------------
# NextHeadGRU
# ---------------------------------------------------------------------------
@register_model("next_gru")
class NextHeadGRU(nn.Module):
    """GRU-based next-POI predictor."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = (x.abs().sum(dim=-1) == 0)
        seq_lengths = (~padding_mask).sum(dim=1)
        output, h_n = self.gru(x)

        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(output[i, last_idx])

        last_output = torch.stack(last_outputs)
        return self.classifier(last_output)


# ---------------------------------------------------------------------------
# NextHeadTemporalCNN
# ---------------------------------------------------------------------------
@register_model("next_temporal_cnn")
class NextHeadTemporalCNN(nn.Module):
    """Temporal CNN for next-POI prediction using causal convolutions."""

    def __init__(
        self,
        embed_dim: int,
        hidden_channels: int = 128,
        num_classes: int = 7,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels = embed_dim

        for i in range(num_layers):
            out_channels = hidden_channels
            padding = (kernel_size - 1)
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_channels = out_channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = x[:, :, :x.size(2) - (conv[0].padding[0])]
        return self.classifier(x)


# ---------------------------------------------------------------------------
# NextHeadHybrid (GRU + Self-Attention)
# ---------------------------------------------------------------------------
@register_model("next_hybrid")
class NextHeadHybrid(nn.Module):
    """Hybrid GRU + Self-Attention for next-POI prediction."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 7,
        num_heads: int = 4,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = (x.abs().sum(dim=-1) == 0)
        gru_out, _ = self.gru(x)
        gru_out = self.norm1(gru_out)

        attn_out, attn_weights = self.attention(
            gru_out, gru_out, gru_out,
            key_padding_mask=padding_mask,
            need_weights=True
        )
        out = self.norm2(gru_out + attn_out)

        seq_lengths = (~padding_mask).sum(dim=1)
        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(out[i, last_idx])

        last_output = torch.stack(last_outputs)
        return self.classifier(last_output)


# ---------------------------------------------------------------------------
# NextHeadTransformerOptimized
# ---------------------------------------------------------------------------
@register_model("next_transformer_optimized")
class NextHeadTransformerOptimized(nn.Module):
    """Optimized Transformer for short sequences with temporal decay pooling."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 2,
        seq_length: int = 9,
        dropout: float = 0.3,
        use_temporal_decay: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.use_temporal_decay = use_temporal_decay

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        if use_temporal_decay:
            decay = torch.exp(-torch.arange(seq_length - 1, -1, -1).float() * 0.5)
            self.register_buffer('temporal_decay', decay)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device

        padding_mask = (x.abs().sum(dim=-1) == 0)

        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        if self.use_temporal_decay:
            decay_weights = self.temporal_decay[:seq_len].unsqueeze(0).unsqueeze(-1)
            weighted_x = x * decay_weights
            mask = (~padding_mask).unsqueeze(-1).float()
            weighted_x = weighted_x * mask
            pooled = weighted_x.sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        else:
            seq_lengths = (~padding_mask).sum(dim=1)
            last_outputs = []
            for i in range(batch_size):
                last_idx = seq_lengths[i] - 1
                last_outputs.append(x[i, last_idx])
            pooled = torch.stack(last_outputs)

        return self.classifier(pooled)
