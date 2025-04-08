import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value):
        """
        Cross-attention from query to key_value
        Args:
            query: tensor of shape (batch_size, seq_length_q, embed_dim)
            key_value: tensor of shape (batch_size, seq_length_kv, embed_dim)
        """
        batch_size, query_len, _ = query.size()
        kv_batch_size, kv_len, _ = key_value.size()

        # Handle batch size mismatch by repeating key_value if needed
        if kv_batch_size != batch_size:
            # You can either subsample or repeat based on your needs
            if kv_batch_size > batch_size:
                key_value = key_value[:batch_size]
            else:
                repeat_factor = (batch_size + kv_batch_size - 1) // kv_batch_size  # ceiling division
                key_value = key_value.repeat(repeat_factor, 1, 1)[:batch_size]

        # Project inputs
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)

        return self.out_proj(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length=512, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)

        return x
