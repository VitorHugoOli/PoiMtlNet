"""STAN-inspired next-location head for region prediction.

Reference:
    Luo, Liu, Liu. "STAN: Spatio-Temporal Attention Network for Next
    Location Recommendation." The Web Conference (WWW) 2021.
    https://arxiv.org/abs/2102.04095

Architecture (as adapted here):

    Bi-layer self-attention over the 9-step check-in trajectory with an
    explicit pairwise spatio-temporal bias added to every attention
    logit. Layer 1 aggregates trajectory context; Layer 2 plays the role
    of STAN's candidate-matching layer (last-position queries all
    positions). Both layers attend bidirectionally across the trajectory
    (STAN has no causal mask — all 9 inputs are past observations of a
    target that comes later).

Adaptation note — ST bias source:

    STAN computes the pairwise bias ``B[i,j]`` from raw time-interval
    ``Δt_ij`` and great-circle distance ``Δd_ij`` between check-in i and
    j, looked up through a learned interval-embedding table with linear
    interpolation. Our ``next_region.parquet`` carries the 9-step
    check2HGI embedding vectors but no raw timestamps or coordinates, so
    the pairwise bias here is a **learnable matrix indexed by relative
    position** ``(i, j) ∈ [0, 9) × [0, 9)`` — one scalar per head per
    pair. This is a faithful architectural adaptation because check2HGI
    embeddings already internalize per-check-in spatio-temporal context
    at encode time (see ``docs/studies/check2hgi/`` and the ``check2hgi``
    engine); the role of STAN's explicit ΔT / ΔD features is subsumed
    by check2HGI's contextual encoding, and the pairwise bias here adds
    the position-aware prior that's orthogonal to that encoded signal.

Key differences from the existing registry heads:

    - ``next_gru`` / ``next_lstm``: recurrent, causal, no cross-position
      attention.
    - ``next_transformer_relpos``: single-stack causal Transformer with
      ALiBi-style slopes (non-learnable). STAN is bidirectional,
      bi-layer with distinct roles, and uses a fully-learnable pairwise
      bias per (head, i, j).
    - ``next_mtl``: causal Transformer with sinusoidal PE.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import register_model


class _STANAttention(nn.Module):
    """Multi-head self-attention with a fully-learnable pairwise bias.

    The pairwise bias has shape ``[num_heads, seq_length, seq_length]``.
    Every (head, i, j) entry is a free parameter — this mirrors STAN's
    design of embedding the ST interval between every pair of positions
    directly, rather than deriving it from a 1-D relative-position slope
    (ALiBi-style).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_length: int,
        dropout: float,
        bias_init: str = "gaussian",
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide num_heads"
        assert bias_init in ("gaussian", "alibi"), "bias_init must be gaussian|alibi"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.pair_bias = nn.Parameter(torch.zeros(num_heads, seq_length, seq_length))
        if bias_init == "alibi":
            # ALiBi-style recency-decay prior: head h gets slope
            # 2^(-8(h+1)/num_heads). Bias b_ij = -slope * |i - j|.
            # Newer tokens attended to more by default; lowers effective
            # DOF vs free Gaussian init.
            with torch.no_grad():
                positions = torch.arange(seq_length).float()
                rel_dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
                for h in range(num_heads):
                    slope = 1.0 / (2.0 ** ((h + 1) * 8.0 / num_heads))
                    self.pair_bias.data[h] = -slope * rel_dist
        else:
            nn.init.normal_(self.pair_bias, std=0.02)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        last_query_only: bool = False,
    ) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn + self.pair_bias[:, :S, :S].unsqueeze(0)

        if padding_mask is not None:
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(1), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if last_query_only:
            if padding_mask is None:
                return out[:, -1]
            seq_lengths = (~padding_mask).sum(dim=1)
            last_idx = (seq_lengths - 1).clamp(min=0)
            batch_idx = torch.arange(B, device=out.device)
            return out[batch_idx, last_idx]
        return out


class _STANBlock(nn.Module):
    """Self-attention + FFN with pre-norm residuals."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        seq_length: int,
        dropout: float,
        bias_init: str = "gaussian",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _STANAttention(d_model, num_heads, seq_length, dropout, bias_init=bias_init)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), padding_mask=padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


@register_model("next_stan")
class NextHeadSTAN(nn.Module):
    """STAN-inspired bi-layer self-attention region predictor.

    Input: ``[B, S, embed_dim]`` where S = sliding-window length (default 9).
    Output: ``[B, num_classes]`` logits over regions.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        seq_length: int = 9,
        d_model: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
        bias_init: str = "alibi",
    ):
        # Default changed from "gaussian" to "alibi" on 2026-04-22 per
        # docs/studies/check2hgi/issues/MODEL_DESIGN_REVIEW_2026-04-22.md
        # §4: the unregularised Gaussian pair_bias (324 params/block) was
        # overfit-prone; AZ ALiBi runs (commit f1ea416) showed scale-
        # dependent σ reduction. Best STAN+GETNext paper numbers (B-M6d)
        # already passed bias_init="alibi" explicitly, so the default
        # flip aligns new runs with the champion configuration.
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.d_model = d_model

        self.input_proj = nn.Linear(embed_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.trajectory_block = _STANBlock(d_model, num_heads, seq_length, dropout, bias_init=bias_init)
        self.matching_norm = nn.LayerNorm(d_model)
        self.matching_attn = _STANAttention(d_model, num_heads, seq_length, dropout, bias_init=bias_init)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run STAN's encoder up to the matching-attention output (pre-classifier).

        Returns the pooled `[B, d_model]` representation that the classifier
        consumes — exposed for downstream heads (e.g. ``next_getnext_hard_hsm``)
        that need the same pooled features but a different output structure.
        """
        padding_mask = (x.abs().sum(dim=-1) == 0)
        all_padded = padding_mask.all(dim=1)
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded, -1] = False

        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.input_dropout(h)

        h = self.trajectory_block(h, padding_mask=padding_mask)

        h_norm = self.matching_norm(h)
        last = self.matching_attn(h_norm, padding_mask=padding_mask, last_query_only=True)
        return last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = self.forward_features(x)
        return self.classifier(last)


__all__ = ["NextHeadSTAN"]
