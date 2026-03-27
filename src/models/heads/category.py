"""All category head variants — consolidated in Phase 4a.

Registry names:
    category_single       — CategoryHeadSingle (MLP)
    category_residual     — CategoryHeadResidual (MLP + residual)
    category_gated        — CategoryHeadGated (MLP + gating)
    category_ensemble     — CategoryHeadEnsemble (multi-path, used by MTL)
    category_attention     — CategoryHeadAttentionPooling
    category_transformer  — CategoryHeadTransformer
    category_dcn          — DCNHead (Deep & Cross Network)
    category_se           — SEHead (Squeeze-and-Excitation)
"""

import torch
from torch import nn

from models.registry import register_model


# ---------------------------------------------------------------------------
# CategoryHeadSingle
# ---------------------------------------------------------------------------
@register_model("category_single")
class CategoryHeadSingle(nn.Module):
    """Multi-layer perceptron for category classification."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float,
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


# ---------------------------------------------------------------------------
# CategoryHeadResidual (helper: _CategoryResidualBlock)
# ---------------------------------------------------------------------------
class _CategoryResidualBlock(nn.Module):
    """Single residual block with expansion."""

    def __init__(self, dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.needs_projection = (dim != output_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

        if self.needs_projection:
            self.shortcut = nn.Linear(dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.shortcut(x)


@register_model("category_residual")
class CategoryHeadResidual(nn.Module):
    """MLP with residual connections for deeper, more stable training."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            current_dim = hidden_dims[i]
            next_dim = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else current_dim
            self.blocks.append(
                _CategoryResidualBlock(
                    dim=current_dim,
                    hidden_dim=current_dim * 2,
                    output_dim=next_dim,
                    dropout=dropout
                )
            )

        final_dim = hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# CategoryHeadGated (helper: _GatedLayer)
# ---------------------------------------------------------------------------
class _GatedLayer(nn.Module):
    """Single gated layer with GLU-style gating."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transform(x)
        g = self.gate(x)
        return self.dropout(self.activation(h * g))


@register_model("category_gated")
class CategoryHeadGated(nn.Module):
    """MLP with gating mechanism to dynamically emphasize important features."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.input_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(_GatedLayer(prev_dim, h, dropout))
            prev_dim = h

        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.input_gate(x)
        x = x * gate
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# CategoryHeadEnsemble (also aliased as CategoryHeadMTL)
# ---------------------------------------------------------------------------
@register_model("category_ensemble")
class CategoryHeadEnsemble(nn.Module):
    """Multi-path architecture that processes the embedding through different
    pathways and combines them."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_paths: int = 3,
            num_classes: int = 7,
            dropout: float = 0.5,
    ):
        super().__init__()
        self.num_paths = num_paths

        self.paths = nn.ModuleList([
            self._make_path(input_dim, hidden_dim, depth=i + 2, dropout=dropout)
            for i in range(num_paths)
        ])

        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * num_paths, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _make_path(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Module:
        layers = []
        prev_dim = input_dim
        for i in range(depth):
            current_hidden = hidden_dim if i == depth - 1 else hidden_dim // (2 ** (depth - i - 1))
            current_hidden = max(current_hidden, hidden_dim)
            layers.extend([
                nn.Linear(prev_dim, current_hidden),
                nn.LayerNorm(current_hidden),
                nn.GELU(),
                nn.Dropout(dropout) if i < depth - 1 else nn.Identity(),
            ])
            prev_dim = current_hidden
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Support [B, 1, input_dim] (MTL) and [B, input_dim] (standalone) inputs
        x = x.squeeze(1)
        path_outputs = [path(x) for path in self.paths]
        combined = torch.cat(path_outputs, dim=-1)
        return self.combiner(combined)


# Phase 1 contract: CategoryHeadMTL is an alias for CategoryHeadEnsemble
CategoryHeadMTL = CategoryHeadEnsemble


# ---------------------------------------------------------------------------
# CategoryHeadAttentionPooling
# ---------------------------------------------------------------------------
@register_model("category_attention")
class CategoryHeadAttentionPooling(nn.Module):
    """MLP with attention-based pooling over feature dimensions."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_classes: int = 7,
            dropout: float = 0.5,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        att_scores = self.attention_weights(h)
        att_weights = torch.sigmoid(att_scores)
        weighted_h = h * att_weights
        return self.classifier(weighted_h)


# ---------------------------------------------------------------------------
# CategoryHeadTransformer
# ---------------------------------------------------------------------------
@register_model("category_transformer")
class CategoryHeadTransformer(nn.Module):
    """Split embedding into L tokens, run through a small Transformer encoder,
    pool and project to num_classes."""

    def __init__(
            self,
            input_dim: int,
            num_tokens: int = 4,
            token_dim: int = 16,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.2,
            num_classes: int = 7,
    ):
        super().__init__()
        assert num_tokens * token_dim == input_dim, \
            f"need num_tokens*token_dim==input_dim ({num_tokens}x{token_dim}!={input_dim})"

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.token_proj = nn.Linear(input_dim, num_tokens * token_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(token_dim)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        tokens = self.token_proj(x)
        tokens = tokens.view(B, self.num_tokens, self.token_dim)
        tokens = tokens + self.pos_emb
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# DCNHead (Deep & Cross Network)
# ---------------------------------------------------------------------------
@register_model("category_dcn")
class DCNHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, cross_layers=2, dropout=0.):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=True)
            for _ in range(cross_layers)
        ])
        prev = input_dim
        deep = []
        for h in hidden_dims:
            deep += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*deep)
        self.classifier = nn.Linear(prev + input_dim, num_classes)

    def forward(self, x):
        x0 = x
        x_cross = x
        for layer in self.cross_layers:
            scalar = layer(x_cross)
            x_cross = x0 * scalar + x_cross
        x_deep = self.deep(x)
        return self.classifier(torch.cat([x_cross, x_deep], dim=-1))


# ---------------------------------------------------------------------------
# SEHead (Squeeze-and-Excitation)
# ---------------------------------------------------------------------------
@register_model("category_se")
class SEHead(nn.Module):
    def __init__(self, input_dim, reduction=8, hidden_dims=(128, 64), num_classes=7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction)
        self.fc2 = nn.Linear(input_dim // reduction, input_dim)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.deep = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, num_classes)

    def forward(self, x):
        s = torch.relu(self.fc1(x))
        g = torch.sigmoid(self.fc2(s))
        x = x * g
        x = self.deep(x)
        return self.classifier(x)
