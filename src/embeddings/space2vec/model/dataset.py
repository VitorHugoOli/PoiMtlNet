"""
Dataset classes and pair generation utilities for Space2Vec.

Contains functions for generating spatial proximity pairs using BallTree
and dataset classes for in-memory and memory-mapped pair storage.
"""

import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from sklearn.neighbors import BallTree
from torch.utils.data import Dataset
from tqdm import tqdm

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def latlon_to_radians(coords: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude from degrees to radians."""
    return np.radians(coords.astype(np.float32))


def to_xy_km(latlon_deg: np.ndarray) -> np.ndarray:
    """
    Convert lat/lon degrees to approximate XY coordinates in kilometers.

    Uses a simple cylindrical projection suitable for local areas.

    Args:
        latlon_deg: Array of shape (B, 2) in degrees [lat, lon]

    Returns:
        Array of shape (B, 2) in kilometers [x, y]
    """
    lat = np.radians(latlon_deg[:, 0])
    lon = np.radians(latlon_deg[:, 1])
    x = EARTH_RADIUS_KM * lon * np.cos(np.mean(lat))
    y = EARTH_RADIUS_KM * lat
    return np.stack([x, y], axis=1)


def spatial_proximity_pairs_kdtree_fast(
    coords: np.ndarray,
    r_pos_km: float = 1.0,
    r_neg_km: float = 5.0,
    max_pairs: int = 50_000,
    k_neg_per_i: int = 5,
    max_pos_per_i: int = 5,
    seed: int = 42,
    leaf_size: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate spatial proximity pairs using BallTree for efficient spatial queries.

    Positive pairs: locations within r_pos_km of each other
    Negative pairs: locations beyond r_neg_km from each other

    Args:
        coords: Lat/lon coordinates of shape (N, 2)
        r_pos_km: Positive pair radius in kilometers
        r_neg_km: Negative pair minimum distance in kilometers
        max_pairs: Maximum number of pairs to generate
        k_neg_per_i: Number of negative pairs per anchor
        max_pos_per_i: Maximum positive pairs per anchor
        seed: Random seed
        leaf_size: BallTree leaf size

    Returns:
        Tuple of (i_indices, j_indices, labels) as numpy arrays
    """
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.float32)
    N = coords.shape[0]
    coords_rad = latlon_to_radians(coords)

    r_pos_rad = r_pos_km / EARTH_RADIUS_KM
    r_neg_rad = r_neg_km / EARTH_RADIUS_KM

    tree = BallTree(coords_rad, metric="haversine", leaf_size=leaf_size)

    # Pre-allocate arrays
    i_arr = np.empty(max_pairs, dtype=np.int32)
    j_arr = np.empty(max_pairs, dtype=np.int32)
    y_arr = np.empty(max_pairs, dtype=np.uint8)

    pos_lists = tree.query_radius(coords_rad, r=r_pos_rad, return_distance=False)
    close_lists = tree.query_radius(coords_rad, r=r_neg_rad, return_distance=False)

    write = 0
    for i in range(N):
        if write >= max_pairs:
            break

        # Positive pairs
        pos_i = pos_lists[i]
        pos_i = pos_i[pos_i != i]
        if pos_i.size > 0:
            if pos_i.size > max_pos_per_i:
                pos_i = rng.choice(pos_i, size=max_pos_per_i, replace=False)
            m = min(pos_i.size, max_pairs - write)
            i_arr[write : write + m] = i
            j_arr[write : write + m] = pos_i[:m].astype(np.int32)
            y_arr[write : write + m] = 1
            write += m
            if write >= max_pairs:
                break

        # Negative pairs
        close_set = set(map(int, close_lists[i]))
        close_set.add(i)
        got = 0
        tries = 0
        max_trials = 50 * k_neg_per_i
        while got < k_neg_per_i and write < max_pairs and tries < max_trials:
            j = int(rng.integers(0, N))
            tries += 1
            if j in close_set:
                continue
            i_arr[write] = i
            j_arr[write] = j
            y_arr[write] = 0
            write += 1
            got += 1

    return i_arr[:write], j_arr[:write], y_arr[:write]


def build_pairs_memmap(
    coords_deg: np.ndarray,
    out_dir: Union[str, Path],
    max_pairs: int = 2_000_000,
    r_pos_km: float = 1.0,
    r_neg_km: float = 20.0,
    k_pos_per_i: int = 5,
    k_neg_per_i: int = 5,
    block: int = 5_000,
    seed: int = 42,
    leaf_size: int = 40,
    verbose: bool = True,
) -> int:
    """
    Build spatial proximity pairs and store in memory-mapped files.

    Uses block-based bulk BallTree queries for efficiency while keeping
    memory usage bounded.

    Args:
        coords_deg: Lat/lon coordinates in degrees, shape (N, 2)
        out_dir: Output directory for memmap files
        max_pairs: Maximum number of pairs to generate
        r_pos_km: Positive pair radius in kilometers
        r_neg_km: Negative pair minimum distance in kilometers
        k_pos_per_i: Maximum positive pairs per anchor
        k_neg_per_i: Maximum negative pairs per anchor
        block: Block size for bulk queries (default 5000 for memory safety)
        seed: Random seed
        leaf_size: BallTree leaf size
        verbose: Whether to print progress

    Returns:
        Number of pairs generated
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    coords_deg = np.asarray(coords_deg, dtype=np.float32)
    N = coords_deg.shape[0]
    coords_rad = np.radians(coords_deg)

    # Create memory-mapped files
    i_mm = np.memmap(
        out_dir / "pairs_i.int32", dtype=np.int32, mode="w+", shape=(max_pairs,)
    )
    j_mm = np.memmap(
        out_dir / "pairs_j.int32", dtype=np.int32, mode="w+", shape=(max_pairs,)
    )
    y_mm = np.memmap(
        out_dir / "pairs_y.uint8", dtype=np.uint8, mode="w+", shape=(max_pairs,)
    )
    wptr = 0

    r_pos = r_pos_km / EARTH_RADIUS_KM
    r_neg = r_neg_km / EARTH_RADIUS_KM

    if verbose:
        print("Building BallTree...")
    tree = BallTree(coords_rad, metric="haversine", leaf_size=leaf_size)

    # Progress bar for total points
    pbar = tqdm(total=N, desc="Generating pairs", disable=not verbose)

    for start in range(0, N, block):
        if wptr >= max_pairs:
            break

        end = min(start + block, N)
        block_coords = coords_rad[start:end]
        block_size = end - start

        # Bulk query for entire block (much faster than individual queries!)
        pos_lists = tree.query_radius(block_coords, r=r_pos, return_distance=False)
        close_lists = tree.query_radius(block_coords, r=r_neg, return_distance=False)

        # Process each point in the block
        for local_i in range(block_size):
            if wptr >= max_pairs:
                break

            global_i = start + local_i

            # Positive pairs
            pos_neighbors = pos_lists[local_i]
            pos_neighbors = pos_neighbors[pos_neighbors != global_i]

            if pos_neighbors.size > 0:
                if pos_neighbors.size > k_pos_per_i:
                    pos_neighbors = rng.choice(
                        pos_neighbors, size=k_pos_per_i, replace=False
                    )
                n = min(pos_neighbors.size, max_pairs - wptr)
                i_mm[wptr : wptr + n] = global_i
                j_mm[wptr : wptr + n] = pos_neighbors[:n]
                y_mm[wptr : wptr + n] = 1
                wptr += n

            if wptr >= max_pairs:
                break

            # Negative pairs
            close_neighbors = close_lists[local_i]
            close_set = set(map(int, close_neighbors))
            close_set.add(global_i)

            got = 0
            trials = 0
            max_trials = 50 * k_neg_per_i
            while got < k_neg_per_i and trials < max_trials and wptr < max_pairs:
                j = int(rng.integers(0, N))
                trials += 1
                if j in close_set:
                    continue
                i_mm[wptr] = global_i
                j_mm[wptr] = j
                y_mm[wptr] = 0
                got += 1
                wptr += 1

        # Update progress bar
        pbar.update(block_size)
        pbar.set_postfix(pairs=wptr)

    pbar.close()

    # Flush and save count
    i_mm.flush()
    j_mm.flush()
    y_mm.flush()
    np.save(out_dir / "pairs_count.npy", np.array([wptr], dtype=np.int64))

    if verbose:
        print(f"Generated {wptr} pairs, saved to {out_dir}")

    return wptr


class SpatialContrastiveDataset(Dataset):
    """
    In-memory dataset for spatial contrastive learning.

    Generates pairs on initialization and stores them in memory.
    Suitable for smaller datasets.
    """

    def __init__(
        self,
        coords: np.ndarray,
        r_pos_km: float = 1.0,
        r_neg_km: float = 5.0,
        max_pairs: int = 2_000_000,
        k_neg_per_i: int = 10,
        max_pos_per_i: int = 5,
        seed: int = 42,
    ):
        """
        Args:
            coords: Lat/lon coordinates of shape (N, 2)
            r_pos_km: Positive pair radius in kilometers
            r_neg_km: Negative pair minimum distance in kilometers
            max_pairs: Maximum number of pairs
            k_neg_per_i: Number of negative pairs per anchor
            max_pos_per_i: Maximum positive pairs per anchor
            seed: Random seed
        """
        self.coords = np.asarray(coords, dtype=np.float32)
        self.i_arr, self.j_arr, self.y_arr = spatial_proximity_pairs_kdtree_fast(
            self.coords,
            r_pos_km=r_pos_km,
            r_neg_km=r_neg_km,
            max_pairs=max_pairs,
            k_neg_per_i=k_neg_per_i,
            max_pos_per_i=max_pos_per_i,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.i_arr)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        i = int(self.i_arr[idx])
        j = int(self.j_arr[idx])
        y = int(self.y_arr[idx])
        return self.coords[i], self.coords[j], y


class PairsMemmapDataset(Dataset):
    """
    Memory-mapped dataset for spatial contrastive learning.

    Loads pre-computed pairs from memory-mapped files for efficient
    handling of large datasets.
    """

    def __init__(self, coords_deg: np.ndarray, out_dir: Union[str, Path]):
        """
        Args:
            coords_deg: Lat/lon coordinates in degrees, shape (N, 2)
            out_dir: Directory containing memmap pair files
        """
        out_dir = Path(out_dir)
        self.coords = np.asarray(coords_deg, dtype=np.float32)
        self.i = np.memmap(out_dir / "pairs_i.int32", dtype=np.int32, mode="r")
        self.j = np.memmap(out_dir / "pairs_j.int32", dtype=np.int32, mode="r")
        self.y = np.memmap(out_dir / "pairs_y.uint8", dtype=np.uint8, mode="r")
        self.count = int(np.load(out_dir / "pairs_count.npy")[0])

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        i = int(self.i[idx])
        j = int(self.j[idx])
        y = int(self.y[idx])
        return self.coords[i], self.coords[j], y
