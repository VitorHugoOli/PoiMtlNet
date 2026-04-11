"""Lightweight multi-task loss weighting baselines.

These losses use the same ``backward(losses, ...)`` contract as NashMTL:
``losses[0]`` is next-category loss and ``losses[1]`` is category loss.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F


def _as_parameter_list(
    parameters: Union[List[torch.nn.Parameter], Tuple[torch.nn.Parameter, ...], torch.Tensor, None],
) -> list[torch.nn.Parameter]:
    if parameters is None:
        return []
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return [param for param in parameters if param is not None]


def _flatten_task_grads(
    grads: tuple[torch.Tensor | None, ...],
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, parameters):
        if grad is None:
            chunks.append(torch.zeros_like(param).reshape(-1))
        else:
            chunks.append(grad.reshape(-1))
    if not chunks:
        if parameters:
            return torch.empty(0, device=parameters[0].device)
        return torch.empty(0)
    return torch.cat(chunks, dim=0)


def _compute_task_gradients(
    losses: torch.Tensor,
    shared_parameters: Union[List[torch.nn.Parameter], Tuple[torch.nn.Parameter, ...], torch.Tensor, None],
) -> torch.Tensor:
    parameters = _as_parameter_list(shared_parameters)
    if not parameters:
        return torch.empty(losses.shape[0], 0, device=losses.device, dtype=losses.dtype)

    gradients = []
    for loss in losses:
        task_grads = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients.append(_flatten_task_grads(task_grads, parameters))
    return torch.stack(gradients, dim=0)


class EqualWeightLoss:
    """Equal loss scalarization: L = sum_i L_i."""

    def __init__(self, n_tasks: int, device: torch.device):
        self.n_tasks = n_tasks
        self.device = device

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        weights = torch.ones(self.n_tasks, dtype=losses.dtype, device=losses.device)
        return torch.sum(losses * weights), {"weights": weights.detach()}

    def backward(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        loss, extra_outputs = self.get_weighted_loss(losses, **kwargs)
        loss.backward()
        return loss, extra_outputs

    def __call__(self, losses: torch.Tensor, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self) -> List[torch.Tensor]:
        return []


class StaticWeightLoss(EqualWeightLoss):
    """Static two-task scalarization.

    ``category_weight`` applies to losses[1]. The next-task weight is
    ``1 - category_weight``. This keeps the scale fixed across the grid.
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        category_weight: float = 0.5,
    ):
        if n_tasks != 2:
            raise ValueError("StaticWeightLoss currently expects exactly 2 tasks")
        if not 0.0 <= category_weight <= 1.0:
            raise ValueError(
                f"category_weight must be in [0, 1], got {category_weight}"
            )
        super().__init__(n_tasks=n_tasks, device=device)
        self.category_weight = float(category_weight)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        weights = torch.tensor(
            [1.0 - self.category_weight, self.category_weight],
            dtype=losses.dtype,
            device=losses.device,
        )
        return torch.sum(losses * weights), {"weights": weights.detach()}


class UncertaintyWeightingLoss(EqualWeightLoss):
    """Homoscedastic uncertainty weighting for task losses."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        initial_log_var: float = 0.0,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        self.log_vars = torch.nn.Parameter(
            torch.full((n_tasks,), float(initial_log_var), device=device)
        )

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        log_vars = self.log_vars.to(device=losses.device, dtype=losses.dtype)
        precision = torch.exp(-log_vars)
        weighted = torch.sum(precision * losses + log_vars)
        return weighted, {
            "weights": precision.detach(),
            "log_vars": log_vars.detach(),
        }

    def parameters(self) -> List[torch.Tensor]:
        return [self.log_vars]


class SoftOptimalUncertaintyWeightingLoss(EqualWeightLoss):
    """Soft Optimal Uncertainty weighting (UW-SO style).

    Based on recent uncertainty-weighting analysis, task weights are derived from
    inverse losses and normalized through a temperature-controlled softmax:
        w_i = softmax(-log(L_i + eps) / temperature).
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        safe_losses = torch.clamp(losses.detach(), min=self.eps)
        logits = -torch.log(safe_losses) / self.temperature
        weights = torch.softmax(logits, dim=-1).to(dtype=losses.dtype, device=losses.device)
        weighted = torch.sum(losses * weights.detach())
        return weighted, {"weights": weights.detach()}


class RandomWeightLoss(EqualWeightLoss):
    """Random Loss Weighting using Dirichlet-sampled task weights."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        alpha: Union[float, list[float], tuple[float, ...]] = 1.0,
        scale: float = 1.0,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if isinstance(alpha, (float, int)):
            alpha_tensor = torch.full((n_tasks,), float(alpha), device=device)
        else:
            if len(alpha) != n_tasks:
                raise ValueError(
                    f"alpha length must match n_tasks={n_tasks}, got {len(alpha)}"
                )
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=device)
        if torch.any(alpha_tensor <= 0):
            raise ValueError("Dirichlet alpha values must be > 0")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.alpha = alpha_tensor
        self.scale = float(scale)

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        # MPS does not implement Dirichlet sampling; sample weights on CPU and
        # move the small vector back to the active loss device.
        alpha = self.alpha.to(device=torch.device("cpu"), dtype=torch.float32)
        weights = torch.distributions.Dirichlet(alpha).sample().to(
            device=losses.device,
            dtype=losses.dtype,
        ) * self.scale
        return torch.sum(losses * weights), {"weights": weights.detach()}


class FAMOLoss(EqualWeightLoss):
    """Fast Adaptive Multitask Optimization-style dynamic weighting.

    This implementation follows the O(1) loss-history weighting idea from
    FAMO while fitting the project's existing loss interface. It updates task
    logits from consecutive observed task losses and uses the FAMO log-loss
    scalarization for the model backward pass.
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        weight_lr: float = 0.025,
        gamma: float = 0.001,
        eps: float = 1e-8,
        min_losses: Union[float, list[float], tuple[float, ...], None] = None,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        self.eps = float(eps)
        if min_losses is None:
            min_losses_tensor = torch.zeros(n_tasks, device=device)
        elif isinstance(min_losses, (float, int)):
            min_losses_tensor = torch.full((n_tasks,), float(min_losses), device=device)
        else:
            if len(min_losses) != n_tasks:
                raise ValueError(
                    f"min_losses length must match n_tasks={n_tasks}, got {len(min_losses)}"
                )
            min_losses_tensor = torch.tensor(min_losses, dtype=torch.float32, device=device)

        self.min_losses = min_losses_tensor
        self.logits = torch.nn.Parameter(torch.zeros(n_tasks, device=device))
        self._optimizer = torch.optim.Adam([self.logits], lr=weight_lr, weight_decay=gamma)
        self._previous_losses: torch.Tensor | None = None

    def _positive_losses(self, losses: torch.Tensor, detach: bool = True) -> torch.Tensor:
        min_losses = self.min_losses.to(device=losses.device, dtype=losses.dtype)
        source = losses.detach() if detach else losses
        return torch.clamp(source - min_losses, min=self.eps)

    def _update_logits(self, current_losses: torch.Tensor) -> None:
        current = self._positive_losses(current_losses, detach=True)
        if self._previous_losses is None:
            self._previous_losses = current
            return

        previous = self._previous_losses.to(device=current.device, dtype=current.dtype)
        delta = torch.log(previous) - torch.log(current)

        with torch.enable_grad():
            weights = F.softmax(self.logits, dim=-1)
            grad = torch.autograd.grad(
                weights,
                self.logits,
                grad_outputs=delta.detach(),
                retain_graph=False,
            )[0]

        self._optimizer.zero_grad(set_to_none=True)
        self.logits.grad = grad
        self._optimizer.step()
        self._previous_losses = current

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        self._update_logits(losses)

        weights = F.softmax(self.logits, dim=-1).to(device=losses.device, dtype=losses.dtype)
        positive_losses = self._positive_losses(losses, detach=False).to(dtype=losses.dtype)
        normalizer = torch.sum(weights.detach() / positive_losses).detach()
        weighted = torch.sum(torch.log(positive_losses) * weights.detach() / normalizer)

        return weighted, {
            "weights": weights.detach(),
        }

    def parameters(self) -> List[torch.Tensor]:
        # FAMO owns a private optimizer for task logits.
        return []


class FairGradLoss(EqualWeightLoss):
    """FairGrad-style weighting using gradient Gram matrix matching."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        alpha: float = 1.0,
        solver_steps: int = 25,
        step_size: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if solver_steps <= 0:
            raise ValueError(f"solver_steps must be > 0, got {solver_steps}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        self.alpha = float(alpha)
        self.solver_steps = int(solver_steps)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self._weights = torch.full((n_tasks,), 1.0 / n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        grads = _compute_task_gradients(losses, shared_parameters)
        if grads.numel() == 0:
            weights = torch.full((self.n_tasks,), 1.0 / self.n_tasks, device=losses.device, dtype=losses.dtype)
            return torch.sum(losses * weights), {"weights": weights.detach()}

        gtg = torch.mm(grads, grads.t()).detach().to(device=losses.device, dtype=losses.dtype)
        gtg = gtg + self.eps * torch.eye(self.n_tasks, device=gtg.device, dtype=gtg.dtype)

        weights = self._weights.to(device=losses.device, dtype=losses.dtype)
        residual_norm = torch.tensor(float("nan"), device=losses.device, dtype=losses.dtype)
        for _ in range(self.solver_steps):
            rhs = torch.pow(torch.clamp(weights, min=self.eps), -1.0 / self.alpha)
            residual = torch.mv(gtg, weights) - rhs
            residual_norm = torch.norm(residual)
            weights = torch.clamp(weights - self.step_size * residual, min=self.eps)
            weights = weights / torch.clamp(torch.sum(weights), min=self.eps)

        self._weights = weights.detach().to(device=self.device, dtype=torch.float32)
        weighted = torch.sum(losses * weights.detach())
        return weighted, {
            "weights": weights.detach(),
            "solver_residual": residual_norm.detach(),
        }


class ExcessMTLLoss(EqualWeightLoss):
    """ExcessMTL-style robust weighting from gradient excess risk."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        robust_step_size: float = 1e-2,
        eps: float = 1e-7,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if robust_step_size <= 0:
            raise ValueError(f"robust_step_size must be > 0, got {robust_step_size}")
        self.robust_step_size = float(robust_step_size)
        self.eps = float(eps)
        self.grad_sum: torch.Tensor | None = None
        self.initial_w: torch.Tensor | None = None
        self.loss_weight = torch.ones(n_tasks, device=device)

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        grads = _compute_task_gradients(losses, shared_parameters).detach()
        if grads.numel() == 0:
            weights = torch.ones(self.n_tasks, dtype=losses.dtype, device=losses.device)
            return torch.sum(losses * weights), {"weights": weights.detach()}

        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(grads)
        self.grad_sum = self.grad_sum.to(device=grads.device, dtype=grads.dtype)

        self.grad_sum = self.grad_sum + grads.pow(2)
        h = torch.sqrt(self.grad_sum + self.eps)
        w = torch.sum((grads * grads) / h, dim=1)

        if self.initial_w is None:
            self.initial_w = torch.clamp(w.detach(), min=self.eps)
        else:
            normalized = w / torch.clamp(
                self.initial_w.to(device=w.device, dtype=w.dtype),
                min=self.eps,
            )
            step_size = float(kwargs.get("robust_step_size", self.robust_step_size))
            updated = self.loss_weight.to(device=w.device, dtype=w.dtype) * torch.exp(normalized * step_size)
            updated = updated / torch.clamp(updated.sum(), min=self.eps) * self.n_tasks
            self.loss_weight = updated.detach()

        weights = self.loss_weight.to(device=losses.device, dtype=losses.dtype)
        weighted = torch.sum(losses * weights.detach())
        return weighted, {"weights": weights.detach()}


class STCHLoss(EqualWeightLoss):
    """Smooth Tchebycheff scalarization with warmup and nadir estimate."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        mu: float = 0.5,
        warmup_epochs: int = 1,
        eps: float = 1e-20,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if mu <= 0:
            raise ValueError(f"mu must be > 0, got {mu}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
        self.mu = float(mu)
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)
        self.nadir_vector: torch.Tensor | None = None
        self.average_loss: torch.Tensor | None = None
        self.average_loss_count = 0

    def get_weighted_loss(self, losses: torch.Tensor, epoch: int | None = None, **kwargs) -> Tuple[torch.Tensor, Dict]:
        epoch_idx = int(epoch) if epoch is not None else 0
        ones = torch.ones(self.n_tasks, dtype=losses.dtype, device=losses.device)
        log_losses = torch.log(losses + self.eps)

        if epoch_idx < self.warmup_epochs:
            return torch.sum(log_losses), {"weights": ones.detach()}

        if epoch_idx == self.warmup_epochs and self.nadir_vector is None:
            if self.average_loss is None:
                self.average_loss = torch.zeros_like(losses.detach())
            self.average_loss = self.average_loss.to(device=losses.device, dtype=losses.dtype)
            self.average_loss = self.average_loss + losses.detach()
            self.average_loss_count += 1
            return torch.sum(log_losses), {"weights": ones.detach()}

        if self.nadir_vector is None:
            if self.average_loss is not None and self.average_loss_count > 0:
                nadir = self.average_loss / float(self.average_loss_count)
            else:
                nadir = losses.detach()
            self.nadir_vector = torch.clamp(nadir, min=self.eps).detach()

        nadir_vector = self.nadir_vector.to(device=losses.device, dtype=losses.dtype)
        stch_losses = torch.log((losses / nadir_vector) + self.eps)
        reg_losses = stch_losses - torch.max(stch_losses.detach())
        weighted = self.mu * torch.logsumexp(reg_losses / self.mu, dim=0) * self.n_tasks
        weights = torch.softmax(reg_losses / self.mu, dim=-1)
        return weighted, {
            "weights": weights.detach(),
            "nadir_vector": nadir_vector.detach(),
        }


class DBMTLLoss(EqualWeightLoss):
    """Dual-Balancing MTL style weighting from buffered log-loss gradients."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        beta: float = 0.9,
        beta_sigma: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if beta_sigma < 0:
            raise ValueError(f"beta_sigma must be >= 0, got {beta_sigma}")
        self.beta = float(beta)
        self.beta_sigma = float(beta_sigma)
        self.eps = float(eps)
        self.step = 0
        self.grad_buffer: torch.Tensor | None = None

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        self.step += 1
        log_losses = torch.log(losses + self.eps)
        batch_grads = _compute_task_gradients(log_losses, shared_parameters).detach()
        if batch_grads.numel() == 0:
            weights = torch.ones(self.n_tasks, dtype=losses.dtype, device=losses.device)
            return torch.sum(losses * weights), {"weights": weights.detach()}

        if self.grad_buffer is None:
            self.grad_buffer = torch.zeros_like(batch_grads)
        self.grad_buffer = self.grad_buffer.to(device=batch_grads.device, dtype=batch_grads.dtype)

        beta_t = self.beta / (float(self.step) ** self.beta_sigma)
        self.grad_buffer = batch_grads + beta_t * (self.grad_buffer - batch_grads)

        grad_norms = torch.norm(self.grad_buffer, dim=-1)
        alpha = grad_norms.max() / torch.clamp(grad_norms, min=self.eps)
        alpha = alpha / torch.clamp(alpha.sum(), min=self.eps) * self.n_tasks

        weighted = torch.sum(losses * alpha.detach())
        return weighted, {"weights": alpha.detach()}


class BayesAggMTLLoss(EqualWeightLoss):
    """Bayesian gradient-uncertainty aggregation (diagonal approximation)."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        ema_beta: float = 0.9,
        uncertainty_power: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        if uncertainty_power <= 0:
            raise ValueError(f"uncertainty_power must be > 0, got {uncertainty_power}")
        self.ema_beta = float(ema_beta)
        self.uncertainty_power = float(uncertainty_power)
        self.eps = float(eps)
        self.running_var = torch.ones(n_tasks, device=device)
        self._initialized = False

    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        grads = _compute_task_gradients(losses, shared_parameters).detach()
        if grads.numel() == 0:
            weights = torch.full((self.n_tasks,), 1.0 / self.n_tasks, dtype=losses.dtype, device=losses.device)
            return torch.sum(losses * weights), {"weights": weights.detach()}

        grad_var = torch.mean(grads.pow(2), dim=1).to(device=losses.device, dtype=losses.dtype)
        if not self._initialized:
            self.running_var = torch.clamp(grad_var, min=self.eps).detach()
            self._initialized = True
        else:
            running = self.running_var.to(device=grad_var.device, dtype=grad_var.dtype)
            self.running_var = (
                self.ema_beta * running + (1.0 - self.ema_beta) * grad_var
            ).detach()

        running_var = self.running_var.to(device=losses.device, dtype=losses.dtype)
        precision = 1.0 / torch.pow(torch.clamp(running_var, min=self.eps), self.uncertainty_power)
        weights = precision / torch.clamp(precision.sum(), min=self.eps)

        weighted = torch.sum(losses * weights.detach())
        return weighted, {
            "weights": weights.detach(),
            "grad_var": running_var.detach(),
        }


class GO4AlignLoss(EqualWeightLoss):
    """GO4Align-style risk-guided weighting with dynamic task interaction signals."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        temperature: float = 1.0,
        ema_beta: float = 0.9,
        window_size: int = 12,
        corr_floor: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__(n_tasks=n_tasks, device=device)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        self.temperature = float(temperature)
        self.ema_beta = float(ema_beta)
        self.window_size = int(window_size)
        self.corr_floor = float(corr_floor)
        self.eps = float(eps)
        self.ema_losses: torch.Tensor | None = None
        self.risk_history: list[torch.Tensor] = []

    def _history_correlation(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if len(self.risk_history) < 2:
            return torch.eye(self.n_tasks, device=device, dtype=dtype)

        hist = torch.stack(
            [entry.to(device=device, dtype=dtype) for entry in self.risk_history],
            dim=0,
        )
        centered = hist - hist.mean(dim=0, keepdim=True)
        norm = torch.norm(centered, dim=0, keepdim=True).clamp(min=self.eps)
        corr = torch.mm(centered.t(), centered) / torch.mm(norm.t(), norm)
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = torch.clamp(corr, min=-1.0, max=1.0)
        diag_idx = torch.arange(self.n_tasks, device=device)
        corr[diag_idx, diag_idx] = 1.0
        return corr

    def get_weighted_loss(self, losses: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        observed = losses.detach().to(device=losses.device, dtype=losses.dtype)
        if self.ema_losses is None:
            self.ema_losses = observed
        else:
            ema = self.ema_losses.to(device=observed.device, dtype=observed.dtype)
            self.ema_losses = (self.ema_beta * ema + (1.0 - self.ema_beta) * observed).detach()

        ema_losses = self.ema_losses.to(device=losses.device, dtype=losses.dtype)
        risk = observed / torch.clamp(ema_losses, min=self.eps)

        self.risk_history.append(risk.detach())
        if len(self.risk_history) > self.window_size:
            self.risk_history.pop(0)

        corr = self._history_correlation(device=losses.device, dtype=losses.dtype)
        interaction = torch.clamp(corr, min=self.corr_floor).mean(dim=1)
        indicators = risk * (1.0 + interaction)
        weights = torch.softmax(indicators / self.temperature, dim=-1)
        weighted = torch.sum(losses * weights.detach())

        return weighted, {
            "weights": weights.detach(),
            "risk": risk.detach(),
            "interaction": interaction.detach(),
        }
