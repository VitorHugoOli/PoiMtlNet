"""GradNorm: Gradient Normalization for Adaptive Loss Balancing.

Reference: Chen et al., ICML 2018 (https://arxiv.org/abs/1711.02257)
Implementation follows LibMTL (https://github.com/median-research-group/LibMTL)
and lucidrains/gradnorm-pytorch.

Algorithm:
  1. Normalise task weights: w = n_tasks * softmax(loss_scale)
  2. Compute per-task gradient norms G_i = ||∇_W (w_i * L_i)||
     with create_graph=True so that L_grad can differentiate through G_i → loss_scale.
  3. Compute GradNorm auxiliary loss: L_grad = Σ |G_i - target_i|
     where target_i = G_avg * r̃_i^alpha  (r̃_i = relative inverse training rate).
  4. L_grad.backward() → populates loss_scale.grad (+ contaminates shared param grads).
  5. Zero shared param grads (remove contamination).
  6. (w.detach() * losses).sum().backward() → clean model gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class GradNormLoss:
    def __init__(self, n_tasks: int = 2, alpha: float = 1.5, **kwargs):
        self.n_tasks = n_tasks
        self.alpha = alpha
        # Learnable log-scale weights; optimizer will step these via parameters()
        self.loss_scale = nn.Parameter(torch.zeros(n_tasks))
        self._L0: torch.Tensor | None = None  # initial losses, recorded on first call

    # ------------------------------------------------------------------
    # Interface expected by mtl_cv.py: backward() returns (loss, extras)
    # and is treated as already_backpropagated=True by the runner.
    # ------------------------------------------------------------------

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: List[torch.nn.Parameter],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        device = losses.device

        # Normalised task weights: positive, sum = n_tasks
        w = self.n_tasks * F.softmax(self.loss_scale.to(device), dim=-1)

        # --- Record initial losses on first call ----------------------
        if self._L0 is None:
            self._L0 = losses.detach().clone().clamp(min=1e-8)
            # First step: plain weighted sum, skip GradNorm update
            total_loss = (w.detach() * losses).sum()
            total_loss.backward()
            return total_loss.detach(), {"weights": w.detach()}

        # --- GradNorm auxiliary loss (updates loss_scale) -------------
        if shared_parameters:
            G_norms = []
            for i in range(self.n_tasks):
                grads = torch.autograd.grad(
                    w[i] * losses[i],
                    shared_parameters,
                    retain_graph=True,
                    create_graph=True,   # needed so L_grad → loss_scale has a path
                    allow_unused=True,
                )
                valid = [g for g in grads if g is not None]
                norm = (
                    torch.norm(torch.stack([g.norm() for g in valid]))
                    if valid
                    else torch.zeros(1, device=device)
                )
                G_norms.append(norm)

            G_norms = torch.stack(G_norms)
            G_avg = G_norms.mean().detach()

            # Relative inverse training rate r̃_i (mean-normalised loss ratio)
            L_ratio = losses.detach() / self._L0.to(device)
            r_i = (L_ratio / L_ratio.mean().clamp(min=1e-8)).clamp(min=1e-8)
            targets = (G_avg * r_i ** self.alpha).detach()

            L_grad = torch.abs(G_norms - targets).sum()
            # Backward through G_norms → loss_scale (via create_graph path).
            # Side-effect: also contaminates shared_parameters.grad — zeroed below.
            L_grad.backward(retain_graph=True)

            # Remove contamination so model grads are clean
            for p in shared_parameters:
                if p.grad is not None:
                    p.grad = None

        # --- Model backward (detach w so loss_scale gets no extra grad) ---
        total_loss = (w.detach() * losses).sum()
        total_loss.backward()

        return total_loss, {"weights": w.detach()}

    def get_weighted_loss(self, losses, shared_parameters, **kwargs):
        # Force runner to use backward() path (already_backpropagated=True)
        raise NotImplementedError

    def parameters(self) -> List[torch.nn.Parameter]:
        """Exposed so the runner adds loss_scale to the optimizer."""
        return [self.loss_scale]
