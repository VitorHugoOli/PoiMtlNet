"""Shared evaluation helpers — extracted in Phase 4a.

Each task runner calls these. MTL calls them twice (once per task head).
Next adds attention extraction on top. No monolithic evaluate() is forced.
"""

from typing import Callable, Optional

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    forward_fn: Optional[Callable] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over loader, return (predictions, targets) as numpy.

    Args:
        model: The model to evaluate (must be in eval mode or will be set).
        loader: DataLoader yielding (X_batch, y_batch) tuples.
        device: Device to run on.
        forward_fn: Optional callable(model, X_batch) -> logits.
            If None, uses model(X_batch). Useful for MTL where the
            model returns a tuple and only one head is needed.

    Returns:
        Tuple of (predictions, targets) as numpy int arrays.
    """
    model.eval()
    preds_list, targets_list = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)

        if forward_fn is not None:
            logits = forward_fn(model, X_batch)
        else:
            logits = model(X_batch)

        preds_list.append(logits.argmax(dim=1))
        targets_list.append(y_batch)

    preds = torch.cat(preds_list).cpu().numpy()
    targets = torch.cat(targets_list).numpy()
    return preds, targets


def build_report(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dict: bool = True,
) -> dict:
    """Wrap sklearn classification_report with standard options.

    Args:
        preds: Predicted class indices.
        targets: True class indices.
        output_dict: If True (default), return dict. If False, return string.

    Returns:
        Classification report as dict (or string if output_dict=False).
    """
    return classification_report(
        targets,
        preds,
        output_dict=output_dict,
        zero_division=0,
    )
