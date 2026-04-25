"""Shared training helpers — extracted in Phase 4a.

Deduplicates compute_class_weights / setup_optimizer / setup_fold patterns
that were copy-pasted across category, next, and MTL cross-validation files.
"""

from typing import Optional, Union

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)
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


def setup_per_head_optimizer(
    model: torch.nn.Module,
    cat_lr: float,
    reg_lr: float,
    shared_lr: float,
    weight_decay: float,
    eps: float = 1e-8,
    extra_parameters: Iterable[torch.nn.Parameter] | None = None,
) -> AdamW:
    """Build an AdamW with three param groups (F48-H3 per-head LR).

    Requires the model to expose ``cat_specific_parameters``,
    ``reg_specific_parameters`` and ``shared_parameters``. Currently
    implemented by ``MTLnetCrossAttn`` only — F48-H3 is scoped to the
    cross-attention MTL backbone.

    Group layout:
      * ``cat``    — cat encoder + cat head  → ``cat_lr``
      * ``reg``    — next encoder + next head → ``reg_lr``
      * ``shared`` — cross-attn blocks + cat/next final_ln → ``shared_lr``

    ``extra_parameters`` (e.g. learnable MTL loss weights) ride along
    with the ``reg`` group since they are typically tied to reg-loss
    backward dynamics.
    """
    for required in ("cat_specific_parameters", "reg_specific_parameters",
                     "shared_parameters"):
        if not hasattr(model, required):
            raise ValueError(
                f"Per-head LR optimizer requires model.{required}(); "
                f"{type(model).__name__} does not expose it. "
                f"Currently supported: MTLnetCrossAttn."
            )
    cat_params = list(model.cat_specific_parameters())
    reg_params = list(model.reg_specific_parameters())
    shared_params = list(model.shared_parameters())
    if extra_parameters is not None:
        reg_params.extend(list(extra_parameters))
    return AdamW(
        [
            {"name": "cat",    "params": cat_params,    "lr": cat_lr},
            {"name": "reg",    "params": reg_params,    "lr": reg_lr},
            {"name": "shared", "params": shared_params, "lr": shared_lr},
        ],
        weight_decay=weight_decay,
        eps=eps,
    )


def setup_scheduler(
    optimizer: AdamW,
    max_lr: float,
    epochs: int,
    steps_per_epoch: int,
    scheduler_type: str = "onecycle",
    pct_start: Optional[float] = None,
):
    """Create an LR scheduler matching the project's conventions.

    Args:
        optimizer: The optimizer to schedule.
        max_lr: Peak learning rate. For ``onecycle`` this is the peak;
            for ``constant`` / ``cosine`` this is the initial/reference LR.
        epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps per epoch.
        scheduler_type: One of ``{"onecycle", "constant", "cosine"}``.
            Default ``onecycle`` preserves legacy behaviour bit-exactly.
        pct_start: For ``onecycle``, fraction of training spent on warmup.
            ``None`` → PyTorch default (0.3). Smaller values push peak LR
            earlier and leave more epochs in the annealing phase.

    Returns:
        Configured scheduler (OneCycleLR / ConstantLR / CosineAnnealingLR).
    """
    if scheduler_type == "onecycle":
        kwargs = dict(
            optimizer=optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
        )
        if pct_start is not None:
            kwargs["pct_start"] = float(pct_start)
        return OneCycleLR(**kwargs)
    # Per-head LR mode (F48-H3) builds an optimizer with multiple param
    # groups already at their target LRs. Detect and skip the single-LR
    # overwrite below — otherwise `setup_per_head_optimizer`'s per-group
    # LRs would be silently flattened to `max_lr` here.
    multi_group_per_head = len(optimizer.param_groups) > 1
    if scheduler_type == "constant":
        # Hold LR fixed at `max_lr` for the entire run (no warmup, no
        # annealing). Isolates "more epochs" from "stretched OneCycleLR
        # schedule" when paired with a higher --epochs value.
        # The optimizer's base `lr` must be overwritten to `max_lr`
        # before ConstantLR(factor=1.0) locks it — but only in single-
        # group (legacy) mode. Per-head mode preserves its own LRs.
        if not multi_group_per_head:
            for pg in optimizer.param_groups:
                pg["lr"] = max_lr
        return ConstantLR(optimizer=optimizer, factor=1.0, total_iters=1)
    if scheduler_type == "cosine":
        # Warmup-free cosine decay from `max_lr` → 0 over total steps.
        # First set the optimizer's base lr to max_lr so CosineAnnealingLR
        # starts at max_lr (not at AdamW's lr=1e-4 default).
        if not multi_group_per_head:
            for pg in optimizer.param_groups:
                pg["lr"] = max_lr
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=int(epochs * steps_per_epoch),
        )
    if scheduler_type == "warmup_constant":
        # F48-H2: linear warmup over `pct_start * total_steps` from a low
        # base, then hold constant at max_lr forever. Designed to give
        # cat a stable warmup phase (avoiding the F45 cat-collapse from
        # day-1 sustained 3e-3) and reg a long high-LR plateau (where
        # `α` in next_getnext_hard can grow, per F45 mechanism).
        # `pct_start` doubles as the warmup fraction (default 1/3 ≈
        # 50ep warmup of the 150ep design from FINDINGS doc).
        if not multi_group_per_head:
            for pg in optimizer.param_groups:
                pg["lr"] = max_lr
        warmup_frac = float(pct_start) if pct_start is not None else (1.0 / 3.0)
        if not (0 < warmup_frac < 1):
            raise ValueError(f"warmup_constant pct_start must be in (0,1), got {warmup_frac}")
        total_steps = int(epochs * steps_per_epoch)
        warmup_steps = max(1, int(warmup_frac * total_steps))
        # start at ~3% of target LR — gentle enough that 7-class cat head
        # doesn't diverge in epoch 1, while keeping a single knob.
        warmup = LinearLR(
            optimizer,
            start_factor=0.033,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        plateau = ConstantLR(optimizer, factor=1.0, total_iters=1)
        return SequentialLR(
            optimizer,
            schedulers=[warmup, plateau],
            milestones=[warmup_steps],
        )
    raise ValueError(
        f"Unknown scheduler_type '{scheduler_type}'; "
        f"expected one of {{'onecycle', 'constant', 'cosine', 'warmup_constant'}}."
    )
