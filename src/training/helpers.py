"""Shared training helpers — extracted in Phase 4a.

Deduplicates compute_class_weights / setup_optimizer / setup_fold patterns
that were copy-pasted across category, next, and MTL cross-validation files.
"""

from typing import Union

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from typing import Iterable


def compute_class_weights(
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute balanced class weights from training targets.

    Uses sklearn's 'balanced' mode: n_samples / (n_classes * bincount).

    Accepts either a numpy array or a ``torch.Tensor`` (any device).
    Tensors are routed through ``.cpu().numpy()`` exactly once at this
    boundary — sklearn requires CPU numpy and several call sites have
    targets pre-loaded onto MPS by PR #8's on-device-tensor optimization.
    Centralizing the conversion here means future call sites cannot
    silently re-introduce the ``can't convert mps:0 device type tensor
    to numpy`` bug.

    Args:
        targets: 1D class labels. Either ``np.ndarray`` or ``torch.Tensor``
            (CPU/MPS/CUDA — handled transparently).
        num_classes: Total number of classes.
        device: Device to place the resulting weight tensor on.

    Returns:
        Float32 tensor of shape (num_classes,) on the given device.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    # sklearn's ``compute_class_weight`` requires every class in ``classes``
    # to appear in ``y``. That's fine for 7-class category labels but fails
    # for high-cardinality heads (e.g. next_region with ~10^3 classes, many
    # of which are absent from any given train fold). Fill weights only
    # for observed classes and default to 1.0 for absent classes — the
    # head will never predict an absent class as positive, so its weight
    # is inert; keeping it at 1.0 preserves CE normalisation.
    present = np.unique(targets)
    weights_full = np.ones(num_classes, dtype=np.float32)
    if present.size > 0:
        weights_present = compute_class_weight('balanced', classes=present, y=targets)
        weights_full[present] = weights_present
    return torch.tensor(weights_full, dtype=torch.float32, device=device)


def setup_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    eps: float = 1e-8,
    extra_parameters: Iterable[torch.nn.Parameter] | None = None,
) -> AdamW:
    """Create an AdamW optimizer matching the project's conventions.

    Args:
        model: Model whose parameters to optimize.
        learning_rate: Base learning rate.
        weight_decay: L2 regularization weight.
        eps: Adam epsilon for numerical stability.
        extra_parameters: Optional non-model parameters, such as learnable
            MTL loss weights.

    Returns:
        Configured AdamW optimizer.
    """
    parameters = list(model.parameters())
    if extra_parameters is not None:
        parameters.extend(list(extra_parameters))

    return AdamW(
        parameters,
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
