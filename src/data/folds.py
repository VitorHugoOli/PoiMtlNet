"""
Unified Fold Creation Module for POI Classification Tasks.

Usage:
    # When using weighted CrossEntropyLoss for class imbalance (recommended)
    creator = FoldCreator(TaskType.MTL, use_weighted_sampling=False)
    folds = creator.create_folds("florida", EmbeddingEngine.DGI)

    # Save/Load
    path = creator.save(Path("./folds"))
    folds = FoldCreator.load(path)

Note:
    Choose ONE approach for class imbalance:
    - use_weighted_sampling=True  -> unweighted CrossEntropyLoss
    - use_weighted_sampling=False -> weighted CrossEntropyLoss (recommended)
    Using both causes over-correction.
"""

import gc
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.globals import CATEGORIES_MAP, DEVICE
from configs.model import InputsConfig
from configs.paths import EmbeddingEngine, IoPaths

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS & DATACLASSES
# ============================================================
class TaskType(Enum):
    CATEGORY = "category"
    NEXT = "next"
    MTL = "mtl"


@dataclass
class FoldConfig:
    n_splits: int
    batch_size: int
    seed: int
    use_weighted_sampling: bool
    embedding_dim: int
    slide_window: int


@dataclass
class FoldIndices:
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class TaskTensors:
    task_type: TaskType
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class SerializableFolds:
    task_tensors: Dict[TaskType, TaskTensors]
    fold_indices: Dict[TaskType, List[FoldIndices]]
    config: FoldConfig
    created_at: str


@dataclass
class FoldData:
    dataloader: DataLoader
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class TaskFoldData:
    train: FoldData
    val: FoldData


@dataclass
class FoldResult:
    next: Optional[TaskFoldData] = None
    category: Optional[TaskFoldData] = None


# ============================================================
# DATASET
# ============================================================
class POIDataset(Dataset):
    def __init__(
            self,
            features: torch.Tensor,
            targets: torch.Tensor,
            device: Optional[torch.device] = None,
    ):
        # When `device` is provided, pre-move the underlying tensors so that
        # __getitem__ returns slices that already live on the target device.
        # This eliminates the per-batch host->device copy in the training loop
        # — safe on MPS because the datasets fit comfortably in unified memory.
        # When `device` is None, tensors are kept on CPU so they can be shared
        # safely with forked DataLoader workers.
        if device is not None:
            self.features = features.to(device) if features.device != device else features
            self.targets = targets.to(device) if targets.device != device else targets
        else:
            self.features = features if features.device.type == 'cpu' else features.cpu()
            self.targets = targets if targets.device.type == 'cpu' else targets.cpu()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


# ============================================================
# UTILITIES
# ============================================================
def _get_num_workers() -> int:
    # MPS + in-memory tensor datasets: num_workers=0 is fastest.
    # Each worker is a forked Python process that pickles the tensor over
    # IPC per epoch — pure overhead when the dataset is already a torch
    # tensor in RAM. See PyTorch Lightning MPS docs.
    if DEVICE.type == 'mps':
        return 0
    return min(8, os.cpu_count() or 1)


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)


def _map_categories(y: pd.Series) -> pd.Series:
    inv_categories = {v: k for k, v in CATEGORIES_MAP.items()}
    return y.map(inv_categories)


def _get_class_distribution(y: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique.astype(int), counts.astype(int)))


def _convert_to_tensors(
        X: np.ndarray,
        y: np.ndarray,
        task_type: TaskType,
        embedding_dim: int,
        slide_window: int = InputsConfig.SLIDE_WINDOW,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_values = np.ascontiguousarray(X, dtype=np.float32)
    y_values = np.ascontiguousarray(y, dtype=np.int64)

    # from_numpy shares memory (zero-copy) since arrays are already contiguous
    x_tensor = torch.from_numpy(x_values)
    y_tensor = torch.from_numpy(y_values)

    if task_type == TaskType.NEXT:
        # Reshape to (num_samples, slide_window, embedding_dim) for sequence models
        x_tensor = x_tensor.view(-1, slide_window, embedding_dim)
    # CATEGORY: keep as (num_samples, embedding_dim) - no reshape needed

    return x_tensor, y_tensor


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _compute_overlap_seq_fraction(
    ambiguous_pois: List[int],
    val_next_idx: np.ndarray,
    seq_poi_mapping: Dict[int, set],
) -> float:
    """Compute fraction of val next-task sequences that touch ambiguous POIs.

    Uses the materialized sequence→POI mapping artifact to determine which
    val sequences contain ambiguous POIs (i.e., POIs that are in category
    training but appear in next-task validation through val-user sequences).
    """
    if not ambiguous_pois:
        return 0.0

    ambiguous_set = set(ambiguous_pois)
    affected = 0
    total_val = len(val_next_idx)

    for idx in val_next_idx:
        seq_pois = seq_poi_mapping.get(int(idx), set())
        if seq_pois & ambiguous_set:
            affected += 1

    return affected / max(total_val, 1)


def _create_weighted_sampler(y: np.ndarray, seed: int) -> WeightedRandomSampler:
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_per_class = dict(zip(classes, class_weights))
    sample_weights = torch.tensor(
        [weight_per_class[label] for label in y],
        dtype=torch.float32
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator
    )


def _create_dataloader(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        use_weighted_sampling: bool,
        seed: int,
) -> DataLoader:
    num_workers = _get_num_workers()

    # Pre-move tensors to DEVICE only when not using DataLoader workers.
    # Forked workers cannot share GPU/MPS memory with the parent process —
    # but with num_workers=0 (the MPS path), we can keep the dataset entirely
    # on-device and skip the per-batch host->device copy.
    dataset_device = DEVICE if num_workers == 0 else None

    sampler = None
    if use_weighted_sampling:
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        sampler = _create_weighted_sampler(y_np, seed)
        shuffle = False

    return DataLoader(
        POIDataset(x, y, device=dataset_device),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


# ============================================================
# DATA LOADING
# ============================================================
def load_category_data(state: str, embedding_engine: EmbeddingEngine) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Load category data. Returns (X, y, placeids, embedding_dim).

    placeids is None when the parquet predates Phase 2 (no 'placeid' column).
    Callers that require placeids (MTL user-level splits) must handle None by
    falling back to independent StratifiedKFold splits.
    """
    logger.info(f"Loading category data: {state}/{embedding_engine.value}")
    df = IoPaths.load_category(state, embedding_engine)

    df['label'] = _map_categories(df['category'])
    if 'placeid' in df.columns:
        placeids = df['placeid'].values.copy()
    else:
        logger.warning(
            "Category parquet for %s/%s has no 'placeid' column (pre-Phase-2 data). "
            "MTL user-level splits require regenerating input with the Phase 2 pipeline. "
            "Falling back to independent StratifiedKFold splits.",
            state, embedding_engine.value,
        )
        placeids = None
    df = df.drop(columns=['category'])

    # Infer embedding_dim from numeric columns in the artifact
    feature_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    embedding_dim = len(feature_cols)
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    logger.info(f"Category data: {X.shape} (dim={embedding_dim}), distribution: {_get_class_distribution(y)}")
    return X, y, placeids, embedding_dim


def load_next_data(
    state: str, embedding_engine: EmbeddingEngine
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load next-POI data. Returns (X, y, userids, embedding_dim)."""
    logger.info(f"Loading next-POI data: {state}/{embedding_engine.value}")
    df = IoPaths.load_next(state, embedding_engine)

    df['label'] = _map_categories(df['next_category'])
    # Ensure userid is int (may be stored as string in parquet)
    userids = df['userid'].astype(int).values.copy()
    df = df.drop(columns=['userid', 'next_category'])

    nan_count = df['label'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Dropping {nan_count} rows with NaN labels")
        valid_mask = ~df['label'].isna()
        df = df[valid_mask]
        userids = userids[valid_mask.values]

    # Infer dimensions from numeric columns in the artifact
    feature_cols = sorted([c for c in df.columns if c.isdigit()], key=int)
    num_features = len(feature_cols)
    slide_window = InputsConfig.SLIDE_WINDOW
    embedding_dim = num_features // slide_window

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    logger.info(f"Next-POI data: {X.shape} (dim={embedding_dim}, window={slide_window}), distribution: {_get_class_distribution(y)}")
    return X, y, userids, embedding_dim


# ============================================================
# SERIALIZATION
# ============================================================
def save_folds(serializable: SerializableFolds, save_dir: Path) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"folds_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    save_path = save_dir / filename

    save_dict = {
        'created_at': serializable.created_at,
        'config': serializable.config.__dict__,
        'task_tensors': {
            task.value: {'x': t.x, 'y': t.y}
            for task, t in serializable.task_tensors.items()
        },
        'fold_indices': {
            task.value: [
                {'fold_idx': idx.fold_idx, 'train': idx.train_indices, 'val': idx.val_indices}
                for idx in indices
            ]
            for task, indices in serializable.fold_indices.items()
        },
    }

    torch.save(save_dict, save_path)
    logger.info(f"Folds saved to {save_path}")
    return save_path


def load_folds(path: Path) -> SerializableFolds:
    save_dict = torch.load(path, weights_only=False)
    config = FoldConfig(**save_dict['config'])

    task_tensors = {
        TaskType(task): TaskTensors(TaskType(task), t['x'], t['y'])
        for task, t in save_dict['task_tensors'].items()
    }

    fold_indices = {
        TaskType(task): [
            FoldIndices(idx['fold_idx'], idx['train'], idx['val'])
            for idx in indices
        ]
        for task, indices in save_dict['fold_indices'].items()
    }

    logger.info(f"Folds loaded from {path}")
    return SerializableFolds(task_tensors, fold_indices, config, save_dict['created_at'])


def rebuild_dataloaders(
        serialized: SerializableFolds,
        batch_size: Optional[int] = None,
        use_weighted_sampling: Optional[bool] = None,
) -> Dict[int, FoldResult]:
    config = serialized.config
    batch_size = batch_size or config.batch_size
    use_weighted = use_weighted_sampling if use_weighted_sampling is not None else config.use_weighted_sampling

    fold_results: Dict[int, FoldResult] = {}
    tasks = list(serialized.task_tensors.keys())
    n_folds = len(serialized.fold_indices[tasks[0]])

    for fold_idx in range(n_folds):
        fold_result = FoldResult()

        for task in tasks:
            tensors = serialized.task_tensors[task]
            indices = serialized.fold_indices[task][fold_idx]

            train_x, train_y = tensors.x[indices.train_indices], tensors.y[indices.train_indices]
            val_x, val_y = tensors.x[indices.val_indices], tensors.y[indices.val_indices]

            task_data = TaskFoldData(
                train=FoldData(
                    _create_dataloader(train_x, train_y, batch_size, True, use_weighted, config.seed),
                    train_x, train_y
                ),
                val=FoldData(
                    _create_dataloader(val_x, val_y, batch_size, False, False, config.seed),
                    val_x, val_y
                ),
            )

            if task == TaskType.NEXT:
                fold_result.next = task_data
            else:
                fold_result.category = task_data

        fold_results[fold_idx] = fold_result
        gc.collect()

    return fold_results


# ============================================================
# FOLD CREATOR
# ============================================================
class FoldCreator:
    def __init__(
            self,
            task_type: TaskType,
            n_splits: int = 5,
            batch_size: int = 2048,
            seed: int = 42,
            use_weighted_sampling: bool = False,
    ):
        self.task_type = task_type
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.seed = seed
        self.use_weighted_sampling = use_weighted_sampling

        self._config = FoldConfig(
            n_splits=n_splits,
            batch_size=batch_size,
            seed=seed,
            use_weighted_sampling=use_weighted_sampling,
            embedding_dim=0,  # set from artifact in create_folds()
            slide_window=InputsConfig.SLIDE_WINDOW,
        )
        self._task_tensors: Dict[TaskType, TaskTensors] = {}
        self._fold_indices: Dict[TaskType, List[FoldIndices]] = {}

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def create_folds(
            self,
            state: str,
            embedding_engine: EmbeddingEngine,
    ) -> Dict[int, FoldResult]:
        logger.info(f"Creating {self.n_splits}-fold CV for {self.task_type.value}")

        if self.task_type == TaskType.MTL:
            return self._create_mtl_folds(state, embedding_engine)
        else:
            return self._create_single_task_folds(state, embedding_engine)

    def _create_single_task_folds(
            self,
            state: str,
            embedding_engine: EmbeddingEngine,
    ) -> Dict[int, FoldResult]:
        task = self.task_type

        if task == TaskType.CATEGORY:
            X, y, _placeids, embedding_dim = load_category_data(state, embedding_engine)
        else:
            X, y, _userids, embedding_dim = load_next_data(state, embedding_engine)

        x_tensor, y_tensor = _convert_to_tensors(X, y, task, embedding_dim=embedding_dim)
        self._task_tensors[task] = TaskTensors(task, x_tensor, y_tensor)
        self._fold_indices[task] = []

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        fold_results: Dict[int, FoldResult] = {}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self._fold_indices[task].append(FoldIndices(fold_idx, train_idx, val_idx))
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}: train={len(train_idx)}, val={len(val_idx)}")

            train_x, train_y = x_tensor[train_idx], y_tensor[train_idx]
            val_x, val_y = x_tensor[val_idx], y_tensor[val_idx]

            task_data = TaskFoldData(
                train=FoldData(
                    _create_dataloader(train_x, train_y, self.batch_size, True, self.use_weighted_sampling, self.seed),
                    train_x, train_y
                ),
                val=FoldData(
                    _create_dataloader(val_x, val_y, self.batch_size, False, False, self.seed),
                    val_x, val_y
                ),
            )

            fold_result = FoldResult()
            if task == TaskType.NEXT:
                fold_result.next = task_data
            else:
                fold_result.category = task_data

            fold_results[fold_idx] = fold_result
            gc.collect()

        return fold_results

    def _create_mtl_folds(
            self,
            state: str,
            embedding_engine: EmbeddingEngine,
    ) -> Dict[int, FoldResult]:
        """Create MTL folds using user-isolation split protocol (SPLIT_PROTOCOL.md).

        Algorithm:
        1. Split users via StratifiedGroupKFold(groups=userid, y=next_category)
        2. Build POI→users mapping to classify POIs per fold
        3. Next-task: train = train-user sequences, val = val-user sequences
        4. Category-task: train = train-exclusive + ambiguous POIs,
                          val = val-exclusive POIs
        """
        # Load data
        X_next, y_next, next_userids, next_dim = load_next_data(state, embedding_engine)
        X_cat, y_cat, cat_placeids, cat_dim = load_category_data(state, embedding_engine)

        x_next_tensor, y_next_tensor = _convert_to_tensors(X_next, y_next, TaskType.NEXT, embedding_dim=next_dim)
        x_cat_tensor, y_cat_tensor = _convert_to_tensors(X_cat, y_cat, TaskType.CATEGORY, embedding_dim=cat_dim)

        self._task_tensors[TaskType.NEXT] = TaskTensors(TaskType.NEXT, x_next_tensor, y_next_tensor)
        self._task_tensors[TaskType.CATEGORY] = TaskTensors(TaskType.CATEGORY, x_cat_tensor, y_cat_tensor)
        self._fold_indices[TaskType.NEXT] = []
        self._fold_indices[TaskType.CATEGORY] = []

        # Determine whether we can run the full user-isolation protocol.
        # cat_placeids is None when the category parquet predates Phase 2.
        use_poi_protocol = cat_placeids is not None

        if use_poi_protocol:
            # Build POI→users mapping from raw checkins
            from data.poi_user_mapping import build_poi_user_mapping
            poi_users = build_poi_user_mapping(state, embedding_engine)

            # Build sequence→POI mapping for overlap diagnostics
            from data.sequence_poi_mapping import build_sequence_poi_mapping
            seq_poi_mapping = build_sequence_poi_mapping(state, embedding_engine)
        else:
            logger.warning(
                "MTL split: no placeid data available; "
                "falling back to independent StratifiedKFold for category splits."
            )
            poi_users = {}
            seq_poi_mapping = {}
            # Pre-compute category splits so they align by fold index
            from sklearn.model_selection import StratifiedKFold
            _skf_cat = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            _cat_fold_splits = list(_skf_cat.split(X_cat, y_cat))

        # Step 1: Split users with StratifiedGroupKFold on next-task data
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        fold_results: Dict[int, FoldResult] = {}
        self._fold_manifests: List[dict] = []

        for fold_idx, (train_next_idx, val_next_idx) in enumerate(
                sgkf.split(X_next, y_next, groups=next_userids)
        ):
            train_users = set(next_userids[train_next_idx])
            val_users = set(next_userids[val_next_idx])

            logger.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"train_users={len(train_users)}, val_users={len(val_users)}"
            )

            if use_poi_protocol:
                # Step 2: Classify POIs
                train_exclusive = []
                val_exclusive = []
                ambiguous = []
                for poi, visitors in poi_users.items():
                    in_train = bool(visitors & train_users)
                    in_val = bool(visitors & val_users)
                    if in_train and in_val:
                        ambiguous.append(poi)
                    elif in_train:
                        train_exclusive.append(poi)
                    elif in_val:
                        val_exclusive.append(poi)

                logger.info(
                    f"  POIs: train_excl={len(train_exclusive)}, "
                    f"val_excl={len(val_exclusive)}, ambiguous={len(ambiguous)}"
                )

                # Step 3: Derive category fold indices
                # Category train = train-exclusive + ambiguous POIs
                # Category val = val-exclusive POIs only
                cat_train_pois = set(train_exclusive) | set(ambiguous)
                cat_val_pois = set(val_exclusive)

                # Coerce both sides to str before set membership.
                # `cat_placeids` dtype depends on the upstream embedding engine:
                #   - space2vec / sphere2vec / notebook-style engines store
                #     `placeid` as Python str (object dtype) in category.parquet
                #   - fusion (and any engine that preserves the raw int64) stores
                #     it as int64
                # `poi_users` (built from raw checkins via build_poi_user_mapping)
                # always has int keys, so without this coercion `np.isin` returns
                # all-False whenever the embedding engine cast placeid to str —
                # producing empty category folds and a downstream
                # `num_samples=0` crash in DataLoader.
                cat_placeids_str = cat_placeids.astype(str)
                cat_train_pois_str = {str(p) for p in cat_train_pois}
                cat_val_pois_str = {str(p) for p in cat_val_pois}

                train_cat_idx = np.where(np.isin(cat_placeids_str, list(cat_train_pois_str)))[0]
                val_cat_idx = np.where(np.isin(cat_placeids_str, list(cat_val_pois_str)))[0]
            else:
                # Fallback: use pre-computed StratifiedKFold splits for category
                train_exclusive, val_exclusive, ambiguous = [], [], []
                train_cat_idx, val_cat_idx = _cat_fold_splits[fold_idx]

            logger.info(
                f"  Category: train={len(train_cat_idx)}, val={len(val_cat_idx)}"
            )
            logger.info(
                f"  Next: train={len(train_next_idx)}, val={len(val_next_idx)}"
            )

            self._fold_indices[TaskType.NEXT].append(
                FoldIndices(fold_idx, train_next_idx, val_next_idx)
            )
            self._fold_indices[TaskType.CATEGORY].append(
                FoldIndices(fold_idx, train_cat_idx, val_cat_idx)
            )

            # Build fold manifest data for P2.9
            self._fold_manifests.append({
                'fold_idx': fold_idx,
                'train_users': sorted(train_users),
                'val_users': sorted(val_users),
                'train_exclusive_pois': sorted(train_exclusive),
                'val_exclusive_pois': sorted(val_exclusive),
                'ambiguous_pois': sorted(ambiguous),
                'split_mode': 'strict' if use_poi_protocol else 'legacy_stratified',
                'category_train_count': len(train_cat_idx),
                'category_val_count': len(val_cat_idx),
                'next_train_count': len(train_next_idx),
                'next_val_count': len(val_next_idx),
                'overlap': {
                    'ambiguous_poi_count': len(ambiguous),
                    'ambiguous_poi_fraction': len(ambiguous) / max(len(poi_users), 1),
                    'cat_train_next_val_poi_count': len(ambiguous),
                    'cat_train_next_val_seq_fraction': (
                        _compute_overlap_seq_fraction(ambiguous, val_next_idx, seq_poi_mapping)
                        if use_poi_protocol else 0.0
                    ),
                },
                'seed': self.seed,
            })

            fold_results[fold_idx] = FoldResult(
                next=TaskFoldData(
                    train=FoldData(
                        _create_dataloader(x_next_tensor[train_next_idx], y_next_tensor[train_next_idx],
                                           self.batch_size, True, self.use_weighted_sampling, self.seed),
                        x_next_tensor[train_next_idx], y_next_tensor[train_next_idx]
                    ),
                    val=FoldData(
                        _create_dataloader(x_next_tensor[val_next_idx], y_next_tensor[val_next_idx],
                                           self.batch_size, False, False, self.seed),
                        x_next_tensor[val_next_idx], y_next_tensor[val_next_idx]
                    ),
                ),
                category=TaskFoldData(
                    train=FoldData(
                        _create_dataloader(x_cat_tensor[train_cat_idx], y_cat_tensor[train_cat_idx],
                                           self.batch_size, True, self.use_weighted_sampling, self.seed),
                        x_cat_tensor[train_cat_idx], y_cat_tensor[train_cat_idx]
                    ),
                    val=FoldData(
                        _create_dataloader(x_cat_tensor[val_cat_idx], y_cat_tensor[val_cat_idx],
                                           self.batch_size, False, False, self.seed),
                        x_cat_tensor[val_cat_idx], y_cat_tensor[val_cat_idx]
                    ),
                ),
            )
            gc.collect()

        return fold_results

    def save(self, save_dir: Path) -> Path:
        if not self._task_tensors:
            raise ValueError("No folds created yet. Call create_folds() first.")

        serializable = SerializableFolds(
            task_tensors=self._task_tensors,
            fold_indices=self._fold_indices,
            config=self._config,
            created_at=datetime.now().isoformat(),
        )
        return save_folds(serializable, save_dir)

    def save_split_manifests(self, output_dir: Path) -> List[Path]:
        """Emit split_manifest_fold*.json for each fold.

        Only available after _create_mtl_folds() has been called.
        Returns list of paths written.
        """
        if not hasattr(self, '_fold_manifests') or not self._fold_manifests:
            logger.warning("No fold manifests to save (not an MTL split or create_folds not called)")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for manifest in self._fold_manifests:
            fold_idx = manifest['fold_idx']
            path = output_dir / f"split_manifest_fold{fold_idx}.json"
            with open(path, 'w') as f:
                json.dump(manifest, f, indent=2, default=_json_default)
            paths.append(path)
            logger.info(f"Split manifest written: {path}")

        return paths

    @classmethod
    def load(cls, path: Path) -> Dict[int, FoldResult]:
        serialized = load_folds(path)
        return rebuild_dataloaders(serialized)
