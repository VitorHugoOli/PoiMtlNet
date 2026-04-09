"""Shared training helpers — extracted in Phase 4a.

Deduplicates compute_class_weights / setup_optimizer / setup_fold patterns
that were copy-pasted across category, next, and MTL cross-validation files.
"""

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader


def compute_class_weights(
    targets: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute balanced class weights from training targets.

    Uses sklearn's 'balanced' mode: n_samples / (n_classes * bincount).

    Args:
        targets: 1D numpy array of integer class labels.
        num_classes: Total number of classes.
        device: Device to place the weight tensor on.

    Returns:
        Float32 tensor of shape (num_classes,) on the given device.
    """
    cls = np.arange(num_classes)
    weights = compute_class_weight('balanced', classes=cls, y=targets)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def setup_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    eps: float = 1e-8,
) -> AdamW:
    """Create an AdamW optimizer matching the project's conventions.

    Args:
        model: Model whose parameters to optimize.
        learning_rate: Base learning rate.
        weight_decay: L2 regularization weight.
        eps: Adam epsilon for numerical stability.

    Returns:
        Configured AdamW optimizer.
    """
    return AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=eps,
    )


def setup_scheduler(
    optimizer: AdamW,
    max_lr: float,
    epochs: int,
    steps_per_epoch: int,
) -> OneCycleLR:
    """Create a OneCycleLR scheduler matching the project's conventions.

    Args:
        optimizer: The optimizer to schedule.
        max_lr: Peak learning rate for OneCycleLR.
        epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.

    Returns:
        Configured OneCycleLR scheduler.
    """
    return OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
