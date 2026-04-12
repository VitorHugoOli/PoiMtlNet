"""Tests for GradNorm gradient balancing."""

import pytest
import torch
import torch.nn as nn

from configs.globals import DEVICE
from losses.gradnorm import GradNormLoss


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_layers = nn.Sequential(nn.Linear(8, 8))
        self.head1 = nn.Linear(8, 2)
        self.head2 = nn.Linear(8, 2)

    def forward(self, x):
        shared = self.shared_layers(x)
        return self.head1(shared), self.head2(shared)


class TestGradNorm:
    """Test suite for GradNorm."""

    def test_initialization(self):
        """Test GradNorm initialization."""
        model = _SimpleModel().to(DEVICE)
        grad_norm = GradNormLoss(model, alpha=1.5)

        assert grad_norm.alpha == 1.5
        assert isinstance(grad_norm.task_weights, nn.Parameter)
        assert grad_norm.task_weights.shape == (2,)
        # Initial values should be ones
        assert torch.allclose(grad_norm.task_weights.data, torch.ones(2, device=DEVICE))

    def test_gradient_magnitude_balancing(self):
        """Test gradient magnitude balancing across tasks: compute_loss returns a Tensor."""
        torch.manual_seed(42)
        model = _SimpleModel().to(DEVICE)
        grad_norm = GradNormLoss(model, alpha=1.5)

        x = torch.randn(4, 8, device=DEVICE)
        out1, out2 = model(x)
        targets = torch.randint(0, 2, (4,), device=DEVICE)

        loss1 = nn.CrossEntropyLoss()(out1, targets)
        loss2 = nn.CrossEntropyLoss()(out2, targets)

        # L0: initial losses (detached scalars)
        L0 = [loss1.item(), loss2.item()]

        result = grad_norm.compute_loss(loss1, loss2, t=0, L0=L0)

        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result)

    def test_weight_updates(self):
        """Test task weight updates: after compute_loss, task_weights.grad is not None."""
        torch.manual_seed(0)
        model = _SimpleModel().to(DEVICE)
        grad_norm = GradNormLoss(model, alpha=1.5)

        x = torch.randn(4, 8, device=DEVICE)
        out1, out2 = model(x)
        targets = torch.randint(0, 2, (4,), device=DEVICE)

        loss1 = nn.CrossEntropyLoss()(out1, targets)
        loss2 = nn.CrossEntropyLoss()(out2, targets)

        L0 = [loss1.item(), loss2.item()]
        grad_norm.compute_loss(loss1, loss2, t=0, L0=L0)

        # GradNorm backward on task_weights should have populated .grad
        assert grad_norm.task_weights.grad is not None
