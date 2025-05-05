import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from configs.globals import DEVICE, CATEGORIES_MAP
from configs.model import MTLModelConfig, CategoryModelConfig
from configs.paths import OUTPUT_ROOT, RESULTS_ROOT
from utils.ml_history.metrics import MLHistory, FoldHistory
from utils.ml_history.parms.neural import NeuralParams
from utils.ml_history.utils.dataset import DatasetHistory


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


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


class CategoryClassifier(nn.Module):
    """
    Multi-layer perceptron for category classification.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            num_classes: int,
            dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(prev_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            prev_dim = h
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    inv_map = {v: k for k, v in CATEGORIES_MAP.items()}
    df['label'] = df['category'].map(inv_map)
    df.drop(columns=['placeid', 'category'], inplace=True)
    feature_cols = df.columns[:CategoryModelConfig.INPUT_DIM]
    X = df[feature_cols].values
    y = df['label'].values
    return X, y


def create_folds(
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int,
        seed: int,
) -> list[tuple[DataLoader, DataLoader]]:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = POIDataset(X_train, y_train)
        val_ds = POIDataset(X_val, y_val)

        num_workers = min(8, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_ds,
            batch_size=MTLModelConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=MTLModelConfig.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        folds.append((train_loader, val_loader))
    return folds


def train_one_fold(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: CategoryModelConfig,
        logger: logging.Logger,
        history: MLHistory,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
    )


    loop = tqdm(
        range(config.EPOCHS),
        unit="batch",
        desc="Training",
    )

    history.set_model_parms(
        NeuralParams(
            batch_size=config.BATCH_SIZE,
            num_epochs=config.EPOCHS,
            learning_rate=config.LR,
            optimizer="AdamW",
            optimizer_state={
                "lr": config.LR,
                "weight_decay": config.WEIGHT_DECAY,
            },
            scheduler="OneCycleLR",
            scheduler_state={
                "max_lr": config.MAX_LR,
                "epochs": config.EPOCHS,
                "steps_per_epoch": len(train_loader),
            },

        )
    )


    fold_history = history.get_curr_fold()

    for _ in loop:
        model.train()
        total_loss = total_correct = total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y_batch.size(0)
            total_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=1)
                val_loss += loss.item() * y_batch.size(0)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        fold_history.to('category').add(
            loss=train_loss,
            accuracy=train_acc,
            f1=0.0,
        )
        fold_history.to('category').add_val(
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_f1=0.0,
            model_state=model.state_dict(),
            best_metric='val_accuracy',
        )

        loop.set_postfix(
            {
                "tr_loss": f"{train_loss:.4f}",
                "tr_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
            }
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Union[str, dict]:
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            truths.append(y_batch.numpy())

    report = classification_report(
        np.concatenate(truths),
        np.concatenate(preds),
        output_dict=True,
        zero_division=0,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train POI Category Classifier"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        required=True,
        help="Path to the preprocessed CSV file",
    )
    return parser.parse_args()


def main():
    state = "florida_test"  # Replace with the desired state
    input_dir = f'{OUTPUT_ROOT}/{state}/pre-processing'
    output_dir = f'{RESULTS_ROOT}/{state}'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-csv",
        type=str,
        default=f"{input_dir}/category-input.csv",
        help="Path to the preprocessed CSV file",
    )

    logger = setup_logger()

    X, y = load_data(parser.parse_args().data_csv)
    folds = create_folds(
        X,
        y,
        n_splits=CategoryModelConfig.N_SPLITS,
        seed=CategoryModelConfig.SEED,
    )

    history: MLHistory = MLHistory(
        model_name="CategoryClassifier",
        model_type="Single-Task",
        tasks='category',
        num_folds=CategoryModelConfig.N_SPLITS,
        datasets={
            DatasetHistory(
                raw_data=f"{input_dir}/category-input.csv",
                description="POI Category Classification",
            )
        }
    )

    history.start()
    for idx, (train_loader, val_loader) in enumerate(folds):
        history.display.start_fold()
        model = CategoryClassifier(
            input_dim=CategoryModelConfig.INPUT_DIM,
            hidden_dims=tuple(CategoryModelConfig.HIDDEN_DIMS),
            num_classes=CategoryModelConfig.NUM_CLASSES,
            dropout=CategoryModelConfig.DROPOUT,
        ).to(DEVICE)

        train_one_fold(
            model,
            train_loader,
            val_loader,
            DEVICE,
            CategoryModelConfig(),
            logger,
            history=history,
        )

        report = evaluate(model, val_loader, DEVICE)

        history.get_curr_fold().to('category').add_report(report)
        history.step()
        history.display.end_fold()

    history.display.end_training()
    history.storage.save(path=output_dir)



if __name__ == "__main__":
    main()
