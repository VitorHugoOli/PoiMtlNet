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
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.globals import CATEGORIES_MAP, DEVICE
from configs.model import InputsConfig, MTLModelConfig
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
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features.cpu()
        self.targets = targets.cpu()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


# ============================================================
# UTILITIES
# ============================================================
def _get_num_workers() -> int:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_values = np.ascontiguousarray(X, dtype=np.float32)
    y_values = np.ascontiguousarray(y, dtype=np.int64)

    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.long)

    embedding_dim = InputsConfig.EMBEDDING_DIM
    slide_window = InputsConfig.SLIDE_WINDOW

    if task_type == TaskType.NEXT:
        # Reshape to (num_samples, slide_window, embedding_dim) for sequence models
        x_tensor = x_tensor.view(-1, slide_window, embedding_dim)
    # CATEGORY: keep as (num_samples, embedding_dim) - no reshape needed

    return x_tensor, y_tensor


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

    sampler = None
    if use_weighted_sampling:
        y_np = y.numpy() if isinstance(y, torch.Tensor) else y
        sampler = _create_weighted_sampler(y_np, seed)
        shuffle = False

    return DataLoader(
        POIDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
        persistent_workers=num_workers > 0,
        prefetch_factor=5 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


# ============================================================
# DATA LOADING
# ============================================================
def load_category_data(state: str, embedding_engine: EmbeddingEngine) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Loading category data: {state}/{embedding_engine.value}")
    df = IoPaths.load_category(state, embedding_engine)

    df['label'] = _map_categories(df['category'])
    df = df.drop(columns=['category'])

    feature_cols = list(map(str, range(InputsConfig.EMBEDDING_DIM)))
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    logger.info(f"Category data: {X.shape}, distribution: {_get_class_distribution(y)}")
    return X, y


def load_next_data(state: str, embedding_engine: EmbeddingEngine) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Loading next-POI data: {state}/{embedding_engine.value}")
    df = IoPaths.load_next(state, embedding_engine)

    df['label'] = _map_categories(df['next_category'])
    df = df.drop(columns=['userid', 'next_category'])

    nan_count = df['label'].isna().sum()
    if nan_count > 0:
        logger.warning(f"Dropping {nan_count} rows with NaN labels")
        df = df.dropna(subset=['label'])

    expected_features = InputsConfig.EMBEDDING_DIM * InputsConfig.SLIDE_WINDOW
    feature_cols = df.columns[:expected_features]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    logger.info(f"Next-POI data: {X.shape}, distribution: {_get_class_distribution(y)}")
    return X, y


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
            batch_size: int = MTLModelConfig.BATCH_SIZE,
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
            embedding_dim=InputsConfig.EMBEDDING_DIM,
            slide_window=InputsConfig.SLIDE_WINDOW,
        )
        self._task_tensors: Dict[TaskType, TaskTensors] = {}
        self._fold_indices: Dict[TaskType, List[FoldIndices]] = {}

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
            X, y = load_category_data(state, embedding_engine)
        else:
            X, y = load_next_data(state, embedding_engine)

        x_tensor, y_tensor = _convert_to_tensors(X, y, task)
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
        X_next, y_next = load_next_data(state, embedding_engine)
        X_cat, y_cat = load_category_data(state, embedding_engine)

        x_next_tensor, y_next_tensor = _convert_to_tensors(X_next, y_next, TaskType.NEXT)
        x_cat_tensor, y_cat_tensor = _convert_to_tensors(X_cat, y_cat, TaskType.CATEGORY)

        self._task_tensors[TaskType.NEXT] = TaskTensors(TaskType.NEXT, x_next_tensor, y_next_tensor)
        self._task_tensors[TaskType.CATEGORY] = TaskTensors(TaskType.CATEGORY, x_cat_tensor, y_cat_tensor)
        self._fold_indices[TaskType.NEXT] = []
        self._fold_indices[TaskType.CATEGORY] = []

        next_skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        cat_skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        fold_results: Dict[int, FoldResult] = {}

        for fold_idx, ((train_next, val_next), (train_cat, val_cat)) in enumerate(
                zip(next_skf.split(X_next, y_next), cat_skf.split(X_cat, y_cat))
        ):
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}")

            self._fold_indices[TaskType.NEXT].append(FoldIndices(fold_idx, train_next, val_next))
            self._fold_indices[TaskType.CATEGORY].append(FoldIndices(fold_idx, train_cat, val_cat))

            fold_results[fold_idx] = FoldResult(
                next=TaskFoldData(
                    train=FoldData(
                        _create_dataloader(x_next_tensor[train_next], y_next_tensor[train_next],
                                           self.batch_size, True, self.use_weighted_sampling, self.seed),
                        x_next_tensor[train_next], y_next_tensor[train_next]
                    ),
                    val=FoldData(
                        _create_dataloader(x_next_tensor[val_next], y_next_tensor[val_next],
                                           self.batch_size, False, False, self.seed),
                        x_next_tensor[val_next], y_next_tensor[val_next]
                    ),
                ),
                category=TaskFoldData(
                    train=FoldData(
                        _create_dataloader(x_cat_tensor[train_cat], y_cat_tensor[train_cat],
                                           self.batch_size, True, self.use_weighted_sampling, self.seed),
                        x_cat_tensor[train_cat], y_cat_tensor[train_cat]
                    ),
                    val=FoldData(
                        _create_dataloader(x_cat_tensor[val_cat], y_cat_tensor[val_cat],
                                           self.batch_size, False, False, self.seed),
                        x_cat_tensor[val_cat], y_cat_tensor[val_cat]
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

    @classmethod
    def load(cls, path: Path) -> Dict[int, FoldResult]:
        serialized = load_folds(path)
        return rebuild_dataloaders(serialized)
