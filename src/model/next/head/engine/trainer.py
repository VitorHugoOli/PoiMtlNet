from typing import Optional

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.next.head.configs.next_config import CfgNextTraining, CfgNextHyperparams
from utils.ml_history.metrics import MLHistory, FoldHistory


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        history: FoldHistory,
        device: torch.device,
) -> None:
    loop = tqdm(
        range(CfgNextTraining.EPOCHS),
        unit="batch",
        desc="Training",
    )

    for _ in loop:
        model.train()
        total_loss = total_correct = total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)

            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CfgNextHyperparams.MAX_GRAD_NORM)
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
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=1)

                val_loss += loss.item() * y_batch.size(0)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(val_targets, val_preds, average='macro')

        history.to('next').add(
            loss=train_loss,
            accuracy=train_acc,
            f1=0.0,
        )
        history.to('next').add_val(
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_f1=val_f1,
            model_state=model.state_dict(),
        )

        loop.set_postfix(
            {
                "tr_loss": f"{train_loss:.4f}",
                "tr_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}({val_f1:.4f})",
            }
        )
