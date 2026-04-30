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

        # Vectorised last-valid-timestep extraction. The previous version
        # had a Python for loop over batch_size (~2048 iterations/forward);
        # on FL (127k rows) this added ~6M Python iterations per 50-epoch
        # run and slowed MPS training from ~35min (Transformer head) to
        # ~2h. This advanced-index fetch runs entirely on-device.
        # Clamp handles the pathological all-padded row (seq_lengths=0 →
        # idx=-1 wraps; clamp to 0 for safety — those rows get zeroed in
        # the MTL forward pre-head anyway, so the 0-index read is a no-op
        # downstream).
        batch_size = x.size(0)
        last_idx = (seq_lengths - 1).clamp(min=0)
        batch_idx = torch.arange(batch_size, device=output.device)
        last_output = output[batch_idx, last_idx]
        return self.classifier(last_output)


__all__ = ["NextHeadGRU"]
