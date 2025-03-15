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

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim).
        Returns:
            Tensor with positional encodings added to the input embeddings.
        """
        seq_length = x.size(1)
        x = x + self.positional_encoding[:, :seq_length, :]
        return self.dropout(x)


class NextPoiNet(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, seq_length, num_layers, dropout=0.1):
        super(NextPoiNet, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.pe = PositionalEncoding(embed_dim, 9)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.linear_layers = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.pe(x)
        batch_size, seq_length, _ = x.size()

        attn_mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1).to(x.device)

        x = self.transformer_encoder(x, mask=attn_mask)

        x = self.linear_layers(x)

        return x

    @staticmethod
    def reshape_output(out_next, y_next, num_classes):
        B, S, _ = out_next.shape

        out_next = out_next.view(B * S, -1)
        y_next = y_next.view(B * S, -1)

        idx_valid = (y_next < num_classes).view(-1)
        y_next = y_next[idx_valid].view(-1)
        out_next = out_next[idx_valid]

        out_next = out_next.view(-1, num_classes)
        return out_next, y_next
