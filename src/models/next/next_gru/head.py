"""GRU next-task head."""

import torch
import torch.nn as nn

from models.registry import register_model


@register_model("next_gru")
class NextHeadGRU(nn.Module):
    """GRU-based next-category predictor."""

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
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = (x.abs().sum(dim=-1) == 0)
        seq_lengths = (~padding_mask).sum(dim=1)
        output, _ = self.gru(x)

        batch_size = x.size(0)
        last_outputs = []
        for i in range(batch_size):
            last_idx = seq_lengths[i] - 1
            last_outputs.append(output[i, last_idx])

        last_output = torch.stack(last_outputs)
        return self.classifier(last_output)


__all__ = ["NextHeadGRU"]
