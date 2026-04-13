"""Baseline MTLnet architecture."""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import torch
from torch import nn

from configs.model import InputsConfig
from models.category import CategoryHeadTransformer
from models.mtl._components import FiLMLayer, ResidualBlock
from models.next import NextHeadMTL
from models.registry import create_model, register_model


@register_model("mtlnet")
class MTLnet(nn.Module):
    """Baseline shared-backbone multitask architecture.

    Head selection can be overridden via ``category_head`` / ``next_head``
    (registry names) plus ``category_head_params`` / ``next_head_params``
    (kwargs forwarded to the head constructor). The defaults reproduce the
    historical `CategoryHeadTransformer` + `NextHeadMTL` configuration
    bit-exactly so existing regression floors stay valid.
    """

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
        category_head: Optional[str] = None,
        next_head: Optional[str] = None,
        category_head_params: Optional[dict[str, Any]] = None,
        next_head_params: Optional[dict[str, Any]] = None,
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

        self._build_shared_backbone(
            shared_layer_size=shared_layer_size,
            num_shared_layers=num_shared_layers,
            shared_dropout=shared_dropout,
        )

        self.category_poi = self._build_category_head(
            name=category_head,
            shared_layer_size=shared_layer_size,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            overrides=category_head_params,
        )
        self.next_poi = self._build_next_head(
            name=next_head,
            shared_layer_size=shared_layer_size,
            num_classes=num_classes,
            num_heads=num_heads,
            num_layers=num_layers,
            seq_length=seq_length,
            overrides=next_head_params,
        )

    def _build_shared_backbone(
        self,
        shared_layer_size: int,
        num_shared_layers: int,
        shared_dropout: float,
    ) -> None:
        """Register the shared mixing block on ``self``.

        Hook for subclasses. The baseline uses FiLM + a stack of
        residual blocks; MMoE/CGC/DSelect-k variants override this to
        register a mixture-of-experts block instead — without paying
        for (and then `del`-ing) the baseline FiLM/shared_layers
        allocation at init time.
        """
        self.task_embedding = nn.Embedding(2, shared_layer_size)
        self.film = FiLMLayer(
            emb_dim=shared_layer_size, layer_size=shared_layer_size
        )
        self.shared_layers = self._build_shared_layers(
            layer_size=shared_layer_size,
            num_blocks=num_shared_layers,
            dropout=shared_dropout,
        )

    @staticmethod
    def _build_category_head(
        name: Optional[str],
        shared_layer_size: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        overrides: Optional[dict[str, Any]],
    ) -> nn.Module:
        """Instantiate the category head.

        When ``name`` is None, build the historical default
        (``CategoryHeadTransformer`` with 2 tokens of half-width) directly
        — this path is deliberately not routed through the registry so
        parameter ordering and RNG consumption stay bit-exact with the
        pre-parameterization model.
        """
        if name is None:
            return CategoryHeadTransformer(
                input_dim=shared_layer_size,
                num_tokens=2,
                token_dim=shared_layer_size // 2,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=0.1,
                num_classes=num_classes,
            )
        params: dict[str, Any] = {
            "input_dim": shared_layer_size,
            "num_classes": num_classes,
        }
        params.update(overrides or {})
        return create_model(name, **params)

    @staticmethod
    def _build_next_head(
        name: Optional[str],
        shared_layer_size: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        seq_length: int,
        overrides: Optional[dict[str, Any]],
    ) -> nn.Module:
        """Instantiate the next-POI head (see ``_build_category_head``).

        Only the universal ``embed_dim`` + ``num_classes`` pair is
        injected for non-default heads. The caller must supply the
        remaining kwargs (``num_heads``, ``seq_length``, ``hidden_dim``…)
        in ``overrides`` — heads in this codebase have divergent
        constructor signatures so there is no one-shot default that
        works across GRU, LSTM, transformer and CNN heads.
        """
        if name is None:
            return NextHeadMTL(
                shared_layer_size,
                num_classes,
                num_heads,
                seq_length,
                num_layers,
                dropout=0.1,
            )
        params: dict[str, Any] = {
            "embed_dim": shared_layer_size,
            "num_classes": num_classes,
        }
        params.update(overrides or {})
        return create_model(name, **params)

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
        # This method interleaves the category and next paths intentionally:
        # under train-mode dropout, the RNG state advances across both paths
        # in a specific order, and the regression floors in
        # tests/test_regression were calibrated against that order. Do not
        # refactor into sequential `_forward_category` + `_forward_next`
        # calls without recalibrating — the bit-exact contract only holds
        # for cat_forward / next_forward in eval mode (no dropout).
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

    def cat_forward(self, category_input: torch.Tensor) -> torch.Tensor:
        """Run only the category subgraph.

        Exists so inference and ablation code can evaluate one head
        without feeding a dummy zero-tensor on the unused side.
        In eval mode (no dropout) the output is bit-exactly equal to
        ``forward((x, anything))[0]`` — see tests/test_models/test_mtlnet.py
        for the pinned contract. In train mode it will differ because
        Dropout samples a different RNG subsequence; use only for eval.
        """
        enc = self.category_encoder(category_input)
        task_id = torch.zeros(enc.size(0), dtype=torch.long, device=enc.device)
        task_emb = self.task_embedding(task_id)
        modulated = self.film(enc, task_emb)
        shared = self.shared_layers(modulated)
        return self.category_poi(shared.squeeze(1)).view(-1, self.num_classes)

    def next_forward(self, next_input: torch.Tensor) -> torch.Tensor:
        """Run only the next-POI subgraph (see ``cat_forward``)."""
        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)

        enc = self.next_encoder(next_input)
        task_id = torch.ones(enc.size(0), dtype=torch.long, device=enc.device)
        task_emb = self.task_embedding(task_id)
        modulated = self.film(enc, task_emb)
        shared = self.shared_layers(modulated)
        return self.next_poi(shared)

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
