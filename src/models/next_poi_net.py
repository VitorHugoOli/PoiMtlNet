import math
import torch
import torch.nn as nn

from models.support.utils import PositionalEncoding, TransformerBlock, MultiHeadCrossAttention


class NextPoiNet(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads, seq_length, num_layers, dropout=0.1):
        super(NextPoiNet, self).__init__()

        self.positional_encoding = PositionalEncoding(embed_dim, seq_length, dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])

        # Cross-attention for task interaction
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

        # Output projection
        self.output_projection = nn.Linear(embed_dim, num_classes)

    def forward(self, x, context=None, mask=None):
        # Add positional encoding
        x = self.positional_encoding(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # Apply cross-attention if context is provided
        if context is not None:
            x = x + self.cross_attention(x, context)

        # Project to output classes
        x = self.output_projection(x)

        return x

    @staticmethod
    def reshape_output(pred, truth):
        """
        Reshape the output of the model to match the expected shape for loss calculation.

        Args:
            pred (torch.Tensor): Model predictions.
            truth (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Reshaped predictions.
        """
        B, S, _ = pred.shape
        next_poi_output = pred.view(B * S, -1)
        next_poi_labels = truth.view(-1)

        valid_indices = (next_poi_labels != -100)
        pred_next = next_poi_output[valid_indices]
        truth_next = next_poi_labels[valid_indices]
        return pred_next, truth_next
