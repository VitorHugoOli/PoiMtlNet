"""
Sphere2Vec contrastive dataset.

Verbatim port of `ContrastiveSpatialDataset` from the source notebook. Each
sample randomly emits either a positive pair (`coord_i + Gaussian noise`) or a
negative pair (`coord_i, random_other_coord`) with a binary label.

NOTE: The dataset uses the global numpy RNG (`np.random.*`), matching the
notebook. Seed it once via `np.random.seed(...)` (or `seed_everything`) before
iterating if you want reproducible runs. This is also why the equivalence test
in `tests/test_embeddings/test_sphere2vec.py` works — both the reference and
the migrated dataset draw from the same RNG stream.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ContrastiveSpatialDataset(Dataset):
    """
    Args:
        coords: ``np.ndarray`` of shape ``[N, 2]`` with ``(latitude, longitude)``
            in degrees.
        pos_radius: Standard deviation (in degrees) of the Gaussian noise applied
            to generate positive pairs. ``0.01`` ≈ 1.1 km.
    """

    def __init__(self, coords: np.ndarray, pos_radius: float = 0.01):
        self.coords = coords
        self.num_points = len(coords)
        self.pos_radius = pos_radius

    def __len__(self) -> int:
        return self.num_points

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord_i = self.coords[idx]

        if np.random.random() > 0.5:
            label = 1.0
            noise = np.random.normal(0, self.pos_radius, size=2)
            coord_j = coord_i + noise
        else:
            label = 0.0
            rand_idx = np.random.randint(0, self.num_points)
            coord_j = self.coords[rand_idx]

        return (
            torch.tensor(coord_i, dtype=torch.float32),
            torch.tensor(coord_j, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )
