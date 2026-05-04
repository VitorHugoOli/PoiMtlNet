from __future__ import annotations

from copy import deepcopy

import torch


class BestModelTracker:
    """Tracks the best model state based on a monitored metric.

    Separated from metric recording so concerns don't mix.

    Usage:
        tracker = BestModelTracker(monitor='f1', mode='max')
        improved = tracker.update(epoch=0, metric_value=0.5, model_state={...})
        improved = tracker.update(epoch=1, metric_value=0.7, model_state={...})
        tracker.best_state  # state dict from epoch 1
    """

    def __init__(
        self,
        monitor: str = 'f1',
        mode: str = 'max',
        min_epoch: int = 0,
    ):
        """
        Parameters
        ----------
        monitor: which metric key to track
        mode: 'max' or 'min'
        min_epoch:
            F50 B1 — earliest epoch (0-indexed) eligible to be selected
            as best. Epochs with ``epoch < min_epoch`` are skipped during
            ``update()``. Default 0 = legacy behaviour (any epoch eligible).
            Set >0 to skip an init-artifact window — e.g.
            ``min_epoch=2`` for the GETNext-prior-with-α=2.0 case where
            ep 1 is the un-trained prior and shouldn't be reported.
        """
        self.monitor = monitor
        self.mode = mode
        self.min_epoch = int(min_epoch)
        self.best_value: float = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch: int = -1
        self.best_time: float = 0.0
        self.best_state: dict = {}

    @staticmethod
    def _snapshot_state(model_state: dict) -> dict:
        """Create an isolated CPU copy of a model state dict.

        model.state_dict() already returns freshly allocated tensors (not views),
        so we only need to move them to CPU — no deepcopy required. For non-tensor
        values (e.g. plain dicts from tests), fall back to simple copy.
        """
        return {
            k: v.cpu().clone() if isinstance(v, torch.Tensor) else deepcopy(v)
            for k, v in model_state.items()
        }

    def update(
        self,
        epoch: int,
        metric_value: float,
        model_state: dict,
        elapsed_time: float = 0.0,
    ) -> bool:
        """Update best model if metric_value improves.

        Returns True if best was updated.
        """
        # B1 gate — refuse epochs before min_epoch. The metric value
        # may still be the true best later in training, so this is a
        # selector hint, not a destructive filter (the per-epoch CSV
        # records every value regardless).
        if epoch < self.min_epoch:
            return False

        if self.mode == 'max':
            improved = metric_value > self.best_value
        else:
            improved = metric_value < self.best_value

        if improved:
            self.best_value = metric_value
            self.best_epoch = epoch
            self.best_time = elapsed_time
            self.best_state = self._snapshot_state(model_state)
            return True
        return False
