import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=5000, dropout=0.1):
        """
        Args:
            embed_dim: The dimensionality of the embedding space.
            max_seq_length: The maximum sequence length to handle.
            dropout: Dropout rate applied to positional encodings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        positional_encoding = torch.zeros(max_seq_length, embed_dim)
        positions = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

        # Add a small epsilon to avoid numerical issues
        self.eps = 1e-12

    def forward(self, x, padding_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim).
            padding_mask: Boolean mask indicating which positions are padding (True for padding)
        Returns:
            Tensor with positional encodings added to the input embeddings.
        """
        seq_length = x.size(1)

        # Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_length, :]

        # If padding mask is provided, zero out positions for padded tokens
        if padding_mask is not None:
            # Create mask tensor (1.0 for valid positions, 0.0 for padding)
            # We add a small epsilon to avoid exact zeros for numerical stability
            mask_tensor = (~padding_mask).unsqueeze(-1).float() + self.eps

            # Apply the mask to both input and positional encoding
            # This ensures padded positions retain their zero value
            x = x * mask_tensor
            pos_enc = pos_enc * mask_tensor

        # Add positional encoding to input
        x = x + pos_enc

        return self.dropout(x)


class NextHeadSingle(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, seq_length, num_layers, dropout=0.35):
        super(NextHeadSingle, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_length = seq_length

        self.pe = PositionalEncoding(embed_dim, max_seq_length=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Simpler attention pooling to avoid NaN issues
        self.sequence_reduction = nn.Linear(embed_dim, 1)

        self.linear_layers = nn.Linear(embed_dim, num_classes)

        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # Shape: [batch_size, 9, 64]
        batch_size, seq_length, _ = x.size()

        padding_mask = (x.abs().sum(dim=-1) == 0)  # (batch_size, seq_len)
        x = self.pe(x, padding_mask)
        key_padding_mask = padding_mask
        causal_mask = torch.zeros(seq_length, seq_length, device=x.device)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_length, seq_length, device=x.device), diagonal=1).bool(),
            float('-inf')
        )
        x = self.transformer_encoder(
            x,
            mask=causal_mask,
        )
        x = self.layer_norm(x)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0)
        attn_logits = self.sequence_reduction(x)  # [batch_size, seq_length, 1]

        attn_logits = attn_logits.masked_fill(padding_mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        mask_all = (attn_weights != attn_weights)  # Find NaN positions
        if mask_all.any():
            uniform_weights = torch.ones_like(attn_weights) / seq_length
            attn_weights = torch.where(mask_all, uniform_weights, attn_weights)

        # Weighted sum to get final embedding
        x = torch.sum(x * attn_weights, dim=1)  # [batch_size, embed_dim]

        # Final classification
        x = self.linear_layers(x)

        return x