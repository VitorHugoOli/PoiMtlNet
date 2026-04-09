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

    def __init__(self, monitor: str = 'f1', mode: str = 'max'):
        self.monitor = monitor
        self.mode = mode
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
