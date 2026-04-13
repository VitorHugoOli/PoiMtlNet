"""PLE (Progressive Layered Extraction) MTLnet variant.

Reference:
    Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task
    Learning Model for Personalized Recommendations", RecSys 2020.
    https://dl.acm.org/doi/10.1145/3383313.3412236
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import torch
from torch import nn

from configs.model import InputsConfig
from models.mtl._components import PLELiteLayer
from models.mtl.mtlnet.model import MTLnet
from models.registry import register_model


@register_model("mtlnet_ple")
class MTLnetPLE(MTLnet):
    """MTLnet variant with stacked CGC layers (Progressive Layered Extraction)."""

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
        num_levels: int = 2,
        num_shared_experts: int = 2,
        num_task_experts: int = 2,
        category_head: Optional[str] = None,
        next_head: Optional[str] = None,
        category_head_params: Optional[dict[str, Any]] = None,
        next_head_params: Optional[dict[str, Any]] = None,
    ):
        self._num_levels = int(num_levels)
        self._num_shared_experts = int(num_shared_experts)
        self._num_task_experts = int(num_task_experts)
        super().__init__(
            feature_size=feature_size,
            shared_layer_size=shared_layer_size,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_length=seq_length,
            num_shared_layers=num_shared_layers,
            encoder_layer_size=encoder_layer_size,
            num_encoder_layers=num_encoder_layers,
            encoder_dropout=encoder_dropout,
            shared_dropout=shared_dropout,
            category_head=category_head,
            next_head=next_head,
            category_head_params=category_head_params,
            next_head_params=next_head_params,
        )

    def _build_shared_backbone(
        self,
        shared_layer_size: int,
        num_shared_layers: int,
        shared_dropout: float,
    ) -> None:
        self.ple = PLELiteLayer(
            layer_size=shared_layer_size,
            num_shared_layers=num_shared_layers,
            num_levels=self._num_levels,
            num_shared_experts=self._num_shared_experts,
            num_task_experts=self._num_task_experts,
            dropout=shared_dropout,
        )

    @property
    def last_gate_stats(self) -> dict[str, torch.Tensor]:
        return self.ple.last_gate_stats

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

        shared_cat, shared_next = self.ple(enc_cat, enc_next)

        out_cat = self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)
        out_next = self.next_poi(shared_next)
        return out_cat, out_next

    def cat_forward(self, category_input: torch.Tensor) -> torch.Tensor:
        enc_cat = self.category_encoder(category_input)
        dummy_next = torch.zeros_like(enc_cat)
        shared_cat, _ = self.ple(enc_cat, dummy_next)
        return self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)

    def next_forward(self, next_input: torch.Tensor) -> torch.Tensor:
        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)
        enc_next = self.next_encoder(next_input)
        dummy_cat = torch.zeros(enc_next.size(0), enc_next.size(-1), device=enc_next.device)
        _, shared_next = self.ple(dummy_cat, enc_next)
        return self.next_poi(shared_next)

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if "ple.levels" in name and "shared_experts" in name
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
                    "category_experts",
                    "next_experts",
                    "category_gate",
                    "next_gate",
                )
            )
        )


__all__ = ["MTLnetPLE"]
