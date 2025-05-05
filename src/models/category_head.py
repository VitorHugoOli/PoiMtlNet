import torch
from torch import nn


class CategoryHead(nn.Module):
    """
     Multi-layer perceptron for category classification.
     """

    def __init__(
            self,
            input_dim: int = 256,
            hidden_dims: tuple[int, ...] = (512, 256, 128, 64, 32),
            num_classes: int = 7,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            prev_dim = h
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)
