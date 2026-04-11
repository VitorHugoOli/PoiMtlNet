"""Baseline MTLnet architecture."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
from torch import nn

from configs.model import InputsConfig
from models.category import CategoryHeadTransformer
from models.mtl._components import FiLMLayer, ResidualBlock
from models.next import NextHeadMTL
from models.registry import register_model


@register_model("mtlnet")
class MTLnet(nn.Module):
    """Baseline shared-backbone multitask architecture."""

    def __init__(
        self,
        feature_size: int,
        shared_layer_size: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        seq_length: int,
        num_shared_layers: int,
        encoder_layer_size: int = 256,
        num_encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
        shared_dropout: float = 0.15,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.category_encoder = self._build_encoder(
            in_size=feature_size,
            hidden_size=encoder_layer_size,
            num_layers=num_encoder_layers,
            out_size=shared_layer_size,
            dropout=encoder_dropout,
        )
        self.next_encoder = self._build_encoder(
            in_size=feature_size,
            hidden_size=encoder_layer_size,
            num_layers=num_encoder_layers,
            out_size=shared_layer_size,
            dropout=encoder_dropout,
        )

        self.task_embedding = nn.Embedding(2, shared_layer_size)
        self.film = FiLMLayer(emb_dim=shared_layer_size, layer_size=shared_layer_size)

        self.shared_layers = self._build_shared_layers(
            layer_size=shared_layer_size,
            num_blocks=num_shared_layers,
            dropout=shared_dropout,
        )

        self.category_poi = CategoryHeadTransformer(
            input_dim=shared_layer_size,
            num_tokens=2,
            token_dim=shared_layer_size // 2,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=0.1,
            num_classes=num_classes,
        )
        self.next_poi = NextHeadMTL(
            shared_layer_size,
            num_classes,
            num_heads,
            seq_length,
            num_layers,
            dropout=0.1,
        )

    def _build_encoder(
        self,
        in_size: int,
        hidden_size: int,
        num_layers: int,
        out_size: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            ]
        layers += [
            nn.Linear(hidden_size, out_size),
            nn.ReLU(),
            nn.LayerNorm(out_size),
        ]
        return nn.Sequential(*layers)

    def _build_shared_layers(
        self,
        layer_size: int,
        num_blocks: int,
        dropout: float,
    ) -> nn.Sequential:
        layers = [
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
            nn.LayerNorm(layer_size),
            nn.Dropout(dropout),
        ]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(layer_size, dropout))
        return nn.Sequential(*layers)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        category_input, next_input = inputs

        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)

        enc_cat = self.category_encoder(category_input)
        enc_next = self.next_encoder(next_input)

        b_cat = enc_cat.size(0)
        b_next = enc_next.size(0)
        id_cat = torch.zeros(b_cat, dtype=torch.long, device=enc_cat.device)
        id_next = torch.ones(b_next, dtype=torch.long, device=enc_next.device)

        emb_cat = self.task_embedding(id_cat)
        emb_next = self.task_embedding(id_next)

        mod_cat = self.film(enc_cat, emb_cat)
        mod_next = self.film(enc_next, emb_next)

        shared_cat = self.shared_layers(mod_cat)
        shared_next = self.shared_layers(mod_next)

        out_cat = self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)
        out_next = self.next_poi(shared_next)
        return out_cat, out_next

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if "shared_layers" in name
            or "task_embedding" in name
            or "film" in name
        )

    def task_specific_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if any(
                key in name
                for key in (
                    "category_encoder",
                    "next_encoder",
                    "category_poi",
                    "next_poi",
                )
            )
        )


__all__ = ["MTLnet"]
