from typing import Optional
import time

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.globals import CATEGORIES_MAP
from tracking.fold import FoldHistory


def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        history: FoldHistory,
        device: torch.device,
        epochs: int = 100,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = -1,
        timeout: Optional[int] = None,
        target_cutoff: Optional[float] = None
) -> None:
    start_time = time.time()
    best_val_f1 = 0.0
    patience_counter = 0

    num_classes = len(CATEGORIES_MAP) - 1  # exclude 'None' class (key 7)
    class_labels = list(range(num_classes))
    class_names = [CATEGORIES_MAP[i] for i in class_labels]

    total_epochs = epochs
    loop = tqdm(
        range(total_epochs),
        unit="batch",
        desc="Training",
    )

    for epoch_idx in loop:
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, device=device, dtype=torch.long)
        total = 0
        train_preds_list, train_targets_list = [], []
        epoch_grad_norms = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)

            loss = criterion(logits, y_batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            epoch_grad_norms.append(grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.detach() * y_batch.size(0)
            running_correct += (preds == y_batch).sum()
            total += y_batch.size(0)
            train_preds_list.append(preds.detach())
            train_targets_list.append(y_batch)

        # Single MPS sync per epoch instead of per batch
        train_loss = running_loss.item() / total
        train_acc = running_correct.item() / total
        train_preds = torch.cat(train_preds_list).cpu().numpy()
        train_targets = torch.cat(train_targets_list).cpu().numpy()
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        avg_grad_norm = float(torch.stack(epoch_grad_norms).mean().item()) if epoch_grad_norms else 0.0

        # Validation
        model.eval()
        val_running_loss = torch.tensor(0.0, device=device)
        val_running_correct = torch.tensor(0, device=device, dtype=torch.long)
        val_total = 0
        val_preds_list, val_targets_list = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                preds = logits.argmax(dim=1)

                val_running_loss += loss.detach() * y_batch.size(0)
                val_running_correct += (preds == y_batch).sum()
                val_total += y_batch.size(0)

                val_preds_list.append(preds)
                val_targets_list.append(y_batch)

        # Single MPS→CPU transfer for all validation predictions
        val_loss = val_running_loss.item() / val_total
        val_acc = val_running_correct.item() / val_total
        val_preds = torch.cat(val_preds_list).cpu().numpy()
        val_targets = torch.cat(val_targets_list).cpu().numpy()
        val_f1 = f1_score(val_targets, val_preds, average='macro')

        # Per-class F1 and diagnostics
        per_class = f1_score(val_targets, val_preds, average=None, zero_division=0, labels=class_labels)
        history.log_diagnostic(
            grad_norm=avg_grad_norm,
            learning_rate=optimizer.param_groups[0]['lr'],
            **{f'per_class_f1_{name}': float(f1) for name, f1 in zip(class_names, per_class)},
        )

        history.log_train('next', loss=train_loss, accuracy=train_acc, f1=train_f1)
        # Only create state_dict when F1 improves (avoids MPS→CPU copy on every epoch)
        current_best = history.task('next').best.best_value
        is_improvement = val_f1 > current_best
        history.log_val(
            'next',
            loss=val_loss,
            accuracy=val_acc,
            f1=val_f1,
            model_state=model.state_dict() if is_improvement else None,
            elapsed_time=history.timer.timer(),
        )

        _, best_f1 = history.task('next').val.best('f1')
        loop.set_postfix(
            {
                "tr_loss": f"{train_loss:.4f}",
                "tr_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}({val_f1:.4f})",
                "best": f"{best_f1:.4f}",
            }
        )

        # Early stopping based on validation F1 patience
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch_idx + 1}. Best val F1: {best_val_f1:.4f}")
            break

        current_time = time.time()
        if (target_cutoff is not None and val_f1 * 100 >= target_cutoff) or (
                timeout is not None and (current_time - start_time) > timeout):
            print(f"\nStopping early at epoch {epoch_idx + 1} with validation F1 score: {val_f1:.4f}.")
            break
