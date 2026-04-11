from typing import Iterator, Tuple
import torch
from torch import nn

from configs.model import InputsConfig
from models.heads.category import CategoryHeadTransformer
from models.heads.next import NextHeadMTL
from models.registry import register_model


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and Dropout for deep shared layers."""

    def __init__(self, hidden_size: int, dropout_rate: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual sublayer
        residual = x
        out = self.norm1(x)
        out = self.activation(self.layer1(out))
        out = self.dropout1(out) + residual

        # Second residual sublayer
        residual = out
        out = self.norm2(out)
        out = self.activation(self.layer2(out))
        out = self.dropout2(out) + residual

        return out


class FiLMLayer(nn.Module):
    """
    Feature‐Wise Linear Modulation: gamma * x + beta,
    where gamma/beta come from a small task embedding.
    """

    def __init__(self, emb_dim: int, layer_size: int):
        super().__init__()
        self.gamma = nn.Linear(emb_dim, layer_size)
        self.beta = nn.Linear(emb_dim, layer_size)

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        # x: [batch, ... , layer_size]
        # task_emb: [batch, emb_dim]
        gamma = self.gamma(task_emb)  # [batch, layer_size]
        beta = self.beta(task_emb)  # [batch, layer_size]

        # unsqueeze so gamma/beta can broadcast along any extra dims of x
        # e.g. if x is [batch, seq_len, layer_size] we need [batch, 1, layer_size]
        for _ in range(x.dim() - gamma.dim()):
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return gamma * x + beta


@register_model("mtlnet")
class MTLnet(nn.Module):
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

        # Task‐specific encoders
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

        # Task‐ID embedding + FiLM block for shared layers
        self.task_embedding = nn.Embedding(2, shared_layer_size)
        self.film = FiLMLayer(emb_dim=shared_layer_size, layer_size=shared_layer_size)

        # Shared processing layers (post‐FiLM)
        self.shared_layers = self._build_shared_layers(
            layer_size=shared_layer_size,
            num_blocks=num_shared_layers,
            dropout=shared_dropout,
        )

        # Task heads
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
            shared_layer_size, num_classes, num_heads, seq_length, num_layers,
            dropout=0.1
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

    def cat_forward(self, category_input: torch.Tensor) -> torch.Tensor:
        """Run only the category-prediction subgraph.

        Use this when evaluating or doing inference on the category head
        alone — avoids wasting compute on the next-head subgraph and
        avoids the dummy-zero indirection that would otherwise be needed
        to feed a 3D placeholder into MTLnet.forward().
        """
        enc_cat = self.category_encoder(category_input)  # [B, shared_size]
        emb_cat = self.task_embedding.weight[0].expand(enc_cat.size(0), -1)
        mod_cat = self.film(enc_cat, emb_cat)
        shared_cat = self.shared_layers(mod_cat)
        return self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)

    def next_forward(self, next_input: torch.Tensor) -> torch.Tensor:
        """Run only the next-POI prediction subgraph.

        Use this when evaluating or doing inference on the next head
        alone. Recomputes the padding mask from the raw sequence positions
        the same way ``forward()`` does, so behaviour is identical to
        the next-head output of a full MTLnet.forward() call.
        """
        pad_value = InputsConfig.PAD_VALUE
        next_padding_mask = (next_input.abs().sum(dim=-1) == pad_value)  # (B, seq_len)
        next_input = next_input.masked_fill(next_padding_mask.unsqueeze(-1), 0)
        enc_next = self.next_encoder(next_input)  # [B, seq_len, shared_size]
        emb_next = self.task_embedding.weight[1].expand(enc_next.size(0), -1)
        mod_next = self.film(enc_next, emb_next)
        shared_next = self.shared_layers(mod_next)
        return self.next_poi(shared_next, padding_mask=next_padding_mask)

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        category_input, next_input = inputs  # ([B, 1, feature_size], [B, seq_len, feature_size])

        pad_value = InputsConfig.PAD_VALUE
        next_padding_mask = (next_input.abs().sum(dim=-1) == pad_value)  # (batch_size, seq_len)
        next_input = next_input.masked_fill(next_padding_mask.unsqueeze(-1), 0)  # zero out all-pad tokens

        # Task‐specific encoding
        enc_cat = self.category_encoder(category_input)  # [batch, shared_size]
        enc_next = self.next_encoder(next_input)  # [batch, shared_size]

        # Look up the two task embeddings directly from the weight matrix
        # and broadcast to the batch dimension. Equivalent to building a
        # long-int id vector and calling self.task_embedding(ids), but
        # avoids allocating an int tensor and an Embedding gather kernel
        # on every forward pass — the gradient still flows because the
        # weight slice itself is a view into the Embedding parameter.
        emb_cat = self.task_embedding.weight[0].expand(enc_cat.size(0), -1)
        emb_next = self.task_embedding.weight[1].expand(enc_next.size(0), -1)

        # FiLM modulation
        mod_cat = self.film(enc_cat, emb_cat)
        mod_next = self.film(enc_next, emb_next)

        # Shared processing
        shared_cat = self.shared_layers(mod_cat)
        shared_next = self.shared_layers(mod_next)

        # Heads
        # Cat in: [batch, 1, shared_size] → squeeze → [batch, shared_size]
        out_cat = self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)
        # Next in: [batch, seq_len, shared_size] Next out: [batch, seq_len, num_classes]
        # Reuse the padding mask we already computed at input time — the
        # padding pattern is determined by raw sequence positions, not by
        # post-shared-layers activations, so this avoids a duplicate
        # abs().sum(-1) reduction inside NextHeadMTL.
        out_next = self.next_poi(shared_next, padding_mask=next_padding_mask)

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


class CGCLiteLayer(nn.Module):
    """Single-level CGC block with shared and task-specific experts."""

    def __init__(
        self,
        layer_size: int,
        num_shared_layers: int,
        num_shared_experts: int = 2,
        num_task_experts: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()
        if num_shared_experts < 1:
            raise ValueError("num_shared_experts must be >= 1")
        if num_task_experts < 1:
            raise ValueError("num_task_experts must be >= 1")

        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts

        self.shared_experts = nn.ModuleList(
            [
                self._build_expert(layer_size, num_shared_layers, dropout)
                for _ in range(num_shared_experts)
            ]
        )
        self.category_experts = nn.ModuleList(
            [
                self._build_expert(layer_size, num_shared_layers, dropout)
                for _ in range(num_task_experts)
            ]
        )
        self.next_experts = nn.ModuleList(
            [
                self._build_expert(layer_size, num_shared_layers, dropout)
                for _ in range(num_task_experts)
            ]
        )

        gate_outputs = num_shared_experts + num_task_experts
        self.category_gate = nn.Linear(layer_size, gate_outputs)
        self.next_gate = nn.Linear(layer_size, gate_outputs)
        self.last_gate_stats: dict[str, torch.Tensor] = {}

    @staticmethod
    def _build_expert(layer_size: int, num_blocks: int, dropout: float) -> nn.Sequential:
        layers = [
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
            nn.LayerNorm(layer_size),
            nn.Dropout(dropout),
        ]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(layer_size, dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def _gate_entropy(weights: torch.Tensor) -> torch.Tensor:
        log_weights = torch.log(weights.clamp_min(1e-8))
        return -(weights * log_weights).sum(dim=-1).mean()

    def _mix(
        self,
        x: torch.Tensor,
        shared_outputs: list[torch.Tensor],
        task_experts: nn.ModuleList,
        gate: nn.Linear,
        task_name: str,
    ) -> torch.Tensor:
        task_outputs = [expert(x) for expert in task_experts]
        expert_outputs = shared_outputs + task_outputs
        stacked = torch.stack(expert_outputs, dim=-2)
        weights = torch.softmax(gate(x), dim=-1)
        mixed = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)

        with torch.no_grad():
            flat_weights = weights.reshape(-1, weights.size(-1))
            self.last_gate_stats[f"{task_name}_entropy"] = self._gate_entropy(weights).detach()
            self.last_gate_stats[f"{task_name}_mean_weights"] = flat_weights.mean(dim=0).detach()

        return mixed

    def forward(
        self,
        category_x: torch.Tensor,
        next_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_cat = [expert(category_x) for expert in self.shared_experts]
        shared_next = [expert(next_x) for expert in self.shared_experts]

        category_out = self._mix(
            category_x, shared_cat, self.category_experts, self.category_gate, "category"
        )
        next_out = self._mix(
            next_x, shared_next, self.next_experts, self.next_gate, "next"
        )
        return category_out, next_out


class MMoELiteLayer(nn.Module):
    """Multi-gate mixture-of-experts block for category and next tasks."""

    def __init__(
        self,
        layer_size: int,
        num_shared_layers: int,
        num_experts: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")

        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                CGCLiteLayer._build_expert(layer_size, num_shared_layers, dropout)
                for _ in range(num_experts)
            ]
        )
        self.category_gate = nn.Linear(layer_size, num_experts)
        self.next_gate = nn.Linear(layer_size, num_experts)
        self.last_gate_stats: dict[str, torch.Tensor] = {}

    def _mix(
        self,
        x: torch.Tensor,
        expert_outputs: list[torch.Tensor],
        gate: nn.Linear,
        task_name: str,
    ) -> torch.Tensor:
        stacked = torch.stack(expert_outputs, dim=-2)
        weights = torch.softmax(gate(x), dim=-1)
        mixed = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)

        with torch.no_grad():
            flat_weights = weights.reshape(-1, weights.size(-1))
            self.last_gate_stats[f"{task_name}_entropy"] = (
                CGCLiteLayer._gate_entropy(weights).detach()
            )
            self.last_gate_stats[f"{task_name}_mean_weights"] = (
                flat_weights.mean(dim=0).detach()
            )

        return mixed

    def forward(
        self,
        category_x: torch.Tensor,
        next_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_cat = [expert(category_x) for expert in self.experts]
        shared_next = [expert(next_x) for expert in self.experts]

        category_out = self._mix(
            category_x, shared_cat, self.category_gate, "category"
        )
        next_out = self._mix(next_x, shared_next, self.next_gate, "next")
        return category_out, next_out


class DSelectKLiteLayer(nn.Module):
    """DSelect-k style sparse soft expert selection for category and next tasks."""

    def __init__(
        self,
        layer_size: int,
        num_shared_layers: int,
        num_experts: int = 4,
        num_selectors: int = 2,
        dropout: float = 0.15,
        temperature: float = 0.5,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if num_selectors < 1:
            raise ValueError("num_selectors must be >= 1")
        if num_selectors > num_experts:
            raise ValueError("num_selectors must be <= num_experts")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.num_experts = num_experts
        self.num_selectors = num_selectors
        self.temperature = float(temperature)

        self.experts = nn.ModuleList(
            [
                CGCLiteLayer._build_expert(layer_size, num_shared_layers, dropout)
                for _ in range(num_experts)
            ]
        )

        selector_dim = num_selectors * num_experts
        self.category_selector = nn.Linear(layer_size, selector_dim)
        self.next_selector = nn.Linear(layer_size, selector_dim)

        # Global selector mixture per task (k selectors)
        self.category_selector_weights = nn.Parameter(torch.zeros(num_selectors))
        self.next_selector_weights = nn.Parameter(torch.zeros(num_selectors))

        self.last_gate_stats: dict[str, torch.Tensor] = {}

    @staticmethod
    def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log(probs.clamp_min(1e-8))
        return -(probs * log_probs).sum(dim=-1)

    def _mix(
        self,
        x: torch.Tensor,
        expert_outputs: list[torch.Tensor],
        selector: nn.Linear,
        selector_weights: torch.nn.Parameter,
        task_name: str,
    ) -> torch.Tensor:
        stacked = torch.stack(expert_outputs, dim=-2)

        selector_logits = selector(x).view(
            *x.shape[:-1],
            self.num_selectors,
            self.num_experts,
        )
        selector_expert_probs = torch.softmax(
            selector_logits / self.temperature,
            dim=-1,
        )

        selector_mix = torch.softmax(selector_weights, dim=-1).to(
            dtype=x.dtype,
            device=x.device,
        )
        selector_view_shape = (1,) * (selector_expert_probs.dim() - 2) + (
            self.num_selectors,
            1,
        )
        selector_mix = selector_mix.view(selector_view_shape)

        weights = torch.sum(selector_expert_probs * selector_mix, dim=-2)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mixed = torch.sum(stacked * weights.unsqueeze(-1), dim=-2)

        with torch.no_grad():
            flat_weights = weights.reshape(-1, weights.size(-1))
            self.last_gate_stats[f"{task_name}_entropy"] = (
                CGCLiteLayer._gate_entropy(weights).detach()
            )
            self.last_gate_stats[f"{task_name}_mean_weights"] = (
                flat_weights.mean(dim=0).detach()
            )
            selector_probs = torch.softmax(selector_weights, dim=-1)
            self.last_gate_stats[f"{task_name}_selector_entropy"] = (
                self._entropy_from_probs(selector_probs).detach()
            )
            self.last_gate_stats[f"{task_name}_selector_weights"] = (
                selector_probs.detach()
            )

        return mixed

    def forward(
        self,
        category_x: torch.Tensor,
        next_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_cat = [expert(category_x) for expert in self.experts]
        shared_next = [expert(next_x) for expert in self.experts]

        category_out = self._mix(
            category_x,
            shared_cat,
            self.category_selector,
            self.category_selector_weights,
            "category",
        )
        next_out = self._mix(
            next_x,
            shared_next,
            self.next_selector,
            self.next_selector_weights,
            "next",
        )
        return category_out, next_out


@register_model("mtlnet_cgc")
class MTLnetCGC(MTLnet):
    """MTLnet variant that replaces FiLM hard sharing with CGC-lite experts."""

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
        num_shared_experts: int = 2,
        num_task_experts: int = 1,
    ):
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
        )
        del self.task_embedding
        del self.film
        del self.shared_layers
        self.cgc = CGCLiteLayer(
            layer_size=shared_layer_size,
            num_shared_layers=num_shared_layers,
            num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts,
            dropout=shared_dropout,
        )

    @property
    def last_gate_stats(self) -> dict[str, torch.Tensor]:
        return self.cgc.last_gate_stats

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        category_input, next_input = inputs

        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)

        enc_cat = self.category_encoder(category_input)
        enc_next = self.next_encoder(next_input)

        shared_cat, shared_next = self.cgc(enc_cat, enc_next)

        out_cat = self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)
        out_next = self.next_poi(shared_next)

        return out_cat, out_next

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if "cgc.shared_experts" in name
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
                    "cgc.category_experts",
                    "cgc.next_experts",
                    "cgc.category_gate",
                    "cgc.next_gate",
                )
            )
        )


@register_model("mtlnet_mmoe")
class MTLnetMMoE(MTLnet):
    """MTLnet variant that replaces FiLM hard sharing with MMoE experts."""

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
        num_experts: int = 4,
    ):
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
        )
        del self.task_embedding
        del self.film
        del self.shared_layers
        self.mmoe = MMoELiteLayer(
            layer_size=shared_layer_size,
            num_shared_layers=num_shared_layers,
            num_experts=num_experts,
            dropout=shared_dropout,
        )

    @property
    def last_gate_stats(self) -> dict[str, torch.Tensor]:
        return self.mmoe.last_gate_stats

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        category_input, next_input = inputs

        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)

        enc_cat = self.category_encoder(category_input)
        enc_next = self.next_encoder(next_input)

        shared_cat, shared_next = self.mmoe(enc_cat, enc_next)

        out_cat = self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)
        out_next = self.next_poi(shared_next)

        return out_cat, out_next

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if "mmoe.experts" in name
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
                    "mmoe.category_gate",
                    "mmoe.next_gate",
                )
            )
        )


@register_model("mtlnet_dselectk")
class MTLnetDSelectK(MTLnet):
    """MTLnet variant with DSelect-k style sparse expert selection."""

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
        num_experts: int = 4,
        num_selectors: int = 2,
        temperature: float = 0.5,
    ):
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
        )
        del self.task_embedding
        del self.film
        del self.shared_layers
        self.dselect = DSelectKLiteLayer(
            layer_size=shared_layer_size,
            num_shared_layers=num_shared_layers,
            num_experts=num_experts,
            num_selectors=num_selectors,
            dropout=shared_dropout,
            temperature=temperature,
        )

    @property
    def last_gate_stats(self) -> dict[str, torch.Tensor]:
        return self.dselect.last_gate_stats

    def forward(
            self,
            inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        category_input, next_input = inputs

        pad_value = InputsConfig.PAD_VALUE
        mask = (next_input.abs().sum(dim=-1) == pad_value)
        next_input = next_input.masked_fill(mask.unsqueeze(-1), 0)

        enc_cat = self.category_encoder(category_input)
        enc_next = self.next_encoder(next_input)

        shared_cat, shared_next = self.dselect(enc_cat, enc_next)

        out_cat = self.category_poi(shared_cat.squeeze(1)).view(-1, self.num_classes)
        out_next = self.next_poi(shared_next)

        return out_cat, out_next

    def shared_parameters(self) -> Iterator[nn.Parameter]:
        return (
            p
            for name, p in self.named_parameters()
            if "dselect.experts" in name
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
                    "dselect.category_selector",
                    "dselect.next_selector",
                    "dselect.category_selector_weights",
                    "dselect.next_selector_weights",
                )
            )
        )
