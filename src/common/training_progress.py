"""Training progress bar with multi-dataloader support."""
import time
from contextlib import contextmanager
from datetime import timedelta
from itertools import cycle
from typing import Iterator, List

from torch.utils.data import DataLoader
from tqdm import tqdm


def zip_longest_cycle(*dataloaders: DataLoader) -> Iterator:
    """Zip dataloaders, cycling shorter ones to match the longest.

    This is the 'max_size_cycle' strategy used in PyTorch Lightning.
    """
    if not dataloaders:
        return iter([])

    if len(dataloaders) == 1:
        return iter(dataloaders[0])

    max_len = max(len(dl) for dl in dataloaders)
    iterators = [
        cycle(dl) if len(dl) < max_len else iter(dl)
        for dl in dataloaders
    ]
    return zip(*iterators)


class TrainingProgressBar(tqdm):
    """Extended tqdm progress bar for training loops."""

    def __init__(self, num_epochs: int, dataloaders: List[DataLoader], **kwargs):
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.batches_per_epoch = max(len(dl) for dl in dataloaders)

        self.epoch_start_time = None
        self.current_epoch = 0
        self.metrics = {}

        super().__init__(
            total=self.batches_per_epoch * num_epochs,
            unit="batch",
            desc=f"Epoch 1/{num_epochs}",
            **kwargs
        )

    def __iter__(self) -> Iterator[int]:
        """Iterate over epochs."""
        for epoch_idx in range(self.num_epochs):
            self.epoch_start_time = time.time()
            self.current_epoch = epoch_idx
            self.set_description(f"Epoch {epoch_idx + 1}/{self.num_epochs}")
            yield epoch_idx
            self.write("")

    def iter_epoch(self) -> Iterator:
        """Iterate over batches in current epoch (max_size_cycle strategy)."""
        for data in zip_longest_cycle(*self.dataloaders):
            yield data
            self.update(1)

    @contextmanager
    def validation(self):
        """Context manager for validation phase."""
        self.set_description(f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Validating]")
        try:
            yield self
        finally:
            elapsed = timedelta(seconds=int(time.time() - self.epoch_start_time))
            self.set_description(f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{elapsed}]")

    def set_postfix(self, ordered_dict=None, **kwargs):
        """Update metrics and display."""
        self.metrics.update(ordered_dict or {})
        self.metrics.update(kwargs)
        super().set_postfix(**self.metrics)
        return self