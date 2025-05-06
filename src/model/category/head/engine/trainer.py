import logging

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.category.head.configs.category_config import CfgCategoryHyperparams, CfgCategoryTraining
from utils.ml_history.metrics import MLHistory
from utils.ml_history.parms.neural import NeuralParams


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        history: MLHistory,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=CfgCategoryHyperparams.LR,
        weight_decay=CfgCategoryHyperparams.WEIGHT_DECAY,
    )
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=CfgCategoryHyperparams.MAX_LR,
        epochs=CfgCategoryTraining.EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    loop = tqdm(
        range(CfgCategoryTraining.EPOCHS),
        unit="batch",
        desc="Training",
    )

    history.set_model_parms(
        NeuralParams(
            batch_size=CfgCategoryTraining.BATCH_SIZE,
            num_epochs=CfgCategoryTraining.EPOCHS,
            learning_rate=CfgCategoryHyperparams.LR,
            optimizer=optimizer.__class__.__name__,
            optimizer_state=optimizer.state_dict(),
            scheduler=scheduler.__class__.__name__,
            scheduler_state=scheduler.state_dict(),
            criterion={
                'category': criterion.__class__.__name__},
            criterion_state={
                'category': criterion.state_dict()
            }

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), CfgCategoryHyperparams.MAX_GRAD_NORM)
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
