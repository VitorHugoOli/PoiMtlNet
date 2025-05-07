import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from common.poi_dataset import POIDataset
from configs.globals import CATEGORIES_MAP
from configs.model import InputsConfig
from model.next.head.configs.next_config import CfgNextTraining, CfgNextModel


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    inv_map = {v: k for k, v in CATEGORIES_MAP.items()}
    df['label'] = df['next_category'].map(inv_map)
    df.drop(columns=['userid', 'next_category'], inplace=True)
    feature_cols = df.columns[0:CfgNextModel.INPUT_DIM * 9]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y


def create_folds(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        seed: int,
        batch_size: int = CfgNextTraining.BATCH_SIZE,
) -> list[tuple[DataLoader, DataLoader]]:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        X_train = X_train.reshape(X_train.shape[0], 9, CfgNextModel.INPUT_DIM)
        X_val = X_val.reshape(X_val.shape[0], 9, CfgNextModel.INPUT_DIM)

        train_ds = POIDataset(X_train, y_train)
        val_ds = POIDataset(X_val, y_val)

        num_workers = min(8, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        folds.append((train_loader, val_loader))
    return folds
