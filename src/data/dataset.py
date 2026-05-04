import numpy as np
import torch
from torch.utils.data import Dataset


class POIDataset(Dataset):
    """
    PyTorch Dataset for POI category classification.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# NOTE: POIDatasetWithAux lives in ``src/data/folds.py`` next to the real
# ``POIDataset`` class used by ``_create_dataloader``. The ``POIDataset`` in
# this file is an orphan kept for backward reference but is not imported
# by any code path. See folds.py for the active classes.