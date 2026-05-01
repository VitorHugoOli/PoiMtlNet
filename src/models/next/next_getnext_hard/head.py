"""Faithful GETNext next-region head (B5) using a hard ``last_region_idx``.

This is the GETNext formulation the original SIGIR 2022 paper uses:

    final_logits = stan_logits + α · log_T[last_region_idx]

where ``last_region_idx`` is the region of the **observed** last-step POI,
not a learned probe output. Compare to ``NextHeadGETNext`` (soft probe)
which approximates this via ``softmax(probe(last_emb)) @ log_T``.

Our inference-time ablation
(``docs/studies/check2hgi/research/B5_HARD_VS_SOFT_INFERENCE.md``) showed
hard-index recovery of +3 to +9 pp Acc@10 at near-convergence over the
soft-probe adaptation we currently ship. This head tests whether that
gap widens, stays, or narrows when the STAN backbone is allowed to
co-adapt against the sharper prior.

Aux-channel plumbing
--------------------
The head reads ``last_region_idx`` via the thread-local
``src/data/aux_side_channel.py``, populated by ``AuxPublishingLoader``
before each ``forward`` call. This keeps the MTL training loop
(``mtl_cv.py``) unchanged — a shared hot path we do NOT want to touch
while partition-bugfix reruns are live.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from data.aux_side_channel import get_current_aux
from models.next.next_stan.head import NextHeadSTAN
from models.registry import register_model


@register_model("next_getnext_hard")
# Paper-facing alias (2026-05-01): "STAN-Flow" — STAN attention backbone +
# GETNext-style α·log_T trajectory-flow prior on regions. NOT a faithful
# reproduction of GETNext (Yang et al. 2022, which is a next-POI model with
# friendship + check-in graph priors). The new alias is what we cite in the
# paper; the legacy `next_getnext_hard` name is preserved for back-compat with
# existing scripts and result-file paths. See PAPER_BASELINES_STRATEGY.md and
# next_region/comparison.md "Substrate-head matched STL — leak-free" §.
@register_model("next_stan_flow")
class NextHeadGETNextHard(nn.Module):
    """STAN backbone + GETNext trajectory-flow prior using a **hard**
    ``last_region_idx`` gather in place of the soft probe.

    Parameters
    ----------
    embed_dim, num_classes, seq_length, d_model, num_heads, dropout, bias_init:
        Forwarded to the underlying ``NextHeadSTAN``.
    transition_path:
        Path to a ``.pt`` file with key ``log_transition`` — the
        pre-computed region-transition log-probability matrix. Same
        format as ``NextHeadGETNext``.
    alpha_init:
        Initial value for the learnable ``α`` scalar. Default ``0.1``,
        same as the soft variant, so that training dynamics start from
        the same point before diverging.
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
        transition_path: Optional[str] = None,
        alpha_init: float = 0.1,
        freeze_alpha: bool = False,
    ):
        super().__init__()
        self.stan = NextHeadSTAN(
            embed_dim=embed_dim,
            num_classes=num_classes,
            seq_length=seq_length,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            bias_init=bias_init,
        )
        # F50 D1 — when ``freeze_alpha=True``, register alpha as a buffer
        # (not a Parameter) so gradient descent cannot move it from its
        # initial value. Combined with ``alpha_init=0.0``, this disables
        # the α·log_T graph prior entirely — output = stan_logits alone.
        # Tests how much of the 82.44 STL ceiling comes from the prior vs
        # the encoder.
        if bool(freeze_alpha):
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))
        else:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self._num_classes = int(num_classes)

        if transition_path is not None:
            payload = torch.load(transition_path, map_location="cpu", weights_only=False)
            log_T = payload["log_transition"] if isinstance(payload, dict) else payload
            log_T = log_T.float()
            if log_T.shape[0] < num_classes or log_T.shape[1] < num_classes:
                raise ValueError(
                    f"Transition matrix shape {tuple(log_T.shape)} is smaller "
                    f"than num_classes={num_classes}. Rebuild for this state."
                )
            log_T = log_T[:num_classes, :num_classes].contiguous()
            self.register_buffer("log_T", log_T)
        else:
            self.register_buffer("log_T", torch.zeros(num_classes, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stan_logits = self.stan(x)

        aux = get_current_aux()  # [B] int64 last_region_idx, or None
        if aux is None:
            # Defensive: eval outside AuxPublishingLoader (e.g. isolated
            # unit tests, FLOPs probe, or a misconfigured training run
            # where the aux loader wasn't wired up). Fall back to pure
            # STAN, but keep ``alpha`` in the autograd graph (via a
            # zero-coefficient multiply) so gradient-surgery losses like
            # PCGrad that enumerate ``task_specific_parameters`` don't
            # crash with "unused in graph" when a batch reaches this
            # branch.
            return stan_logits + self.alpha * 0.0

        if aux.device != stan_logits.device:
            aux = aux.to(stan_logits.device)
        # Zero the prior for pad / out-of-bounds rows; clamp index so the
        # gather never crashes even on bad data.
        pad_mask = (aux < 0) | (aux >= self._num_classes)
        safe_idx = aux.clamp(min=0, max=self._num_classes - 1)
        transition_prior = self.log_T[safe_idx]  # [B, num_classes]
        if pad_mask.any():
            transition_prior = transition_prior.masked_fill(
                pad_mask.unsqueeze(-1), 0.0
            )

        return stan_logits + self.alpha * transition_prior


__all__ = ["NextHeadGETNextHard"]
