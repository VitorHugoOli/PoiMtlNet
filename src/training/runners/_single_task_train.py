"""Unified single-task training loop for category and next-POI tasks."""

import contextlib
import logging
import time
from typing import Optional, List

import torch
from torch import nn
from torchmetrics.functional.classification import multiclass_f1_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tracking.fold import FoldHistory
from tracking.metrics import compute_classification_metrics
from training.callbacks import CallbackContext, CallbackList

logger = logging.getLogger(__name__)


def train_single_task(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        history: FoldHistory,
        task_name: str,
        num_classes: int,
        epochs: int = 50,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int = -1,
        timeout: Optional[int] = None,
        target_cutoff: Optional[float] = None,
        callbacks: Optional[list] = None,
        compute_train_f1: bool = False,
        diagnostic_class_names: Optional[List[str]] = None,
) -> None:
    """Train a single-task model with optional diagnostics and early stopping.

    Args:
        task_name: 'category' or 'next' — used for history logging.
        num_classes: number of output classes — required for torchmetrics
            multiclass F1 computation. Must match model output dim.
        compute_train_f1: If True, compute train F1 and track gradient norms.
        diagnostic_class_names: If provided, log per-class F1 diagnostics.
    """
    start_time = time.time()
    best_val_f1 = 0.0
    patience_counter = 0

    class_labels = list(range(num_classes)) if diagnostic_class_names else []

    cb = CallbackList(callbacks)
    for c in cb.callbacks:
        if hasattr(c, 'set_model'):
            c.set_model(model)

    cb.on_train_begin(CallbackContext(epoch=0, epochs_total=epochs))

    _autocast_ctx = (
        torch.autocast(device.type, dtype=torch.float16)
        if device.type == 'cuda'
        else contextlib.nullcontext()
    )

    loop = tqdm(range(epochs), unit="batch", desc="Training")
    epoch_idx = 0

    for epoch_idx in loop:
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        running_correct = torch.tensor(0, device=device, dtype=torch.long)
        total = 0
        train_logits_list, train_targets_list = [], []
        epoch_grad_norms = []

        for X_batch, y_batch in train_loader:
            if X_batch.device != device:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if compute_train_f1:
                epoch_grad_norms.append(grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Cheap top-1 accuracy accumulation (no logit retention).
            preds = logits.argmax(dim=1)
            running_loss += loss.detach() * y_batch.size(0)
            running_correct += (preds == y_batch).sum()
            total += y_batch.size(0)

            # Only keep logits when the caller opted into full train metrics.
            # Retaining all train logits would balloon memory for large
            # datasets, so the legacy ``compute_train_f1`` flag still gates it.
            if compute_train_f1:
                train_logits_list.append(logits.detach())
                train_targets_list.append(y_batch)

        if total == 0:
            continue
        train_loss = running_loss.item() / total
        train_acc = running_correct.item() / total

        if compute_train_f1 and train_logits_list:
            train_logits = torch.cat(train_logits_list)
            train_targets = torch.cat(train_targets_list)
            train_metrics = compute_classification_metrics(
                train_logits, train_targets, num_classes=num_classes,
            )
            train_f1 = train_metrics['f1']
            # Prefer the torchmetrics-derived accuracy — identical value,
            # avoids drift if the running_correct path is ever removed.
            train_acc = train_metrics['accuracy']
            avg_grad_norm = float(torch.stack(epoch_grad_norms).mean().item()) if epoch_grad_norms else 0.0
        else:
            # Fallback: only loss + cheap top-1 accuracy are computed. We
            # still emit an ``f1`` key (=0.0) so ``fold{i}_{task}_train.csv``
            # columns stay stable across runs with/without
            # ``compute_train_f1``; anything analysing CSV schemas would
            # otherwise see a column disappear from old-style runs.
            train_metrics = {'accuracy': train_acc, 'f1': 0.0}
            train_f1 = 0.0
            avg_grad_norm = 0.0

        # Validation
        model.eval()
        val_running_loss = torch.tensor(0.0, device=device)
        val_total = 0
        val_logits_list, val_targets_list = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if X_batch.device != device:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                with _autocast_ctx:
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)

                val_running_loss += loss.detach() * y_batch.size(0)
                val_total += y_batch.size(0)

                val_logits_list.append(logits)
                val_targets_list.append(y_batch)

        if val_total == 0:
            continue
        val_loss = val_running_loss.item() / val_total
        val_logits = torch.cat(val_logits_list)
        val_targets = torch.cat(val_targets_list)
        val_metrics = compute_classification_metrics(
            val_logits, val_targets, num_classes=num_classes,
        )
        val_f1 = val_metrics['f1']
        val_acc = val_metrics['accuracy']

        # Per-class F1 diagnostics (kept separate: goes to fold.diagnostics,
        # not to the task MetricStore — the docs call out per-class as a
        # detailed breakdown, not a headline metric).
        if diagnostic_class_names:
            val_preds = val_logits.argmax(dim=1)
            per_class = multiclass_f1_score(
                val_preds, val_targets,
                num_classes=num_classes, average=None, zero_division=0,
            ).tolist()
            history.log_diagnostic(
                grad_norm=avg_grad_norm,
                learning_rate=optimizer.param_groups[0]['lr'],
                **{f'per_class_f1_{name}': float(f1) for name, f1 in zip(diagnostic_class_names, per_class)},
            )

        history.log_train(task_name, loss=train_loss, **train_metrics)
        current_best = history.task(task_name).best.best_value
        is_improvement = val_f1 > current_best
        history.log_val(
            task_name,
            loss=val_loss,
            **val_metrics,
            model_state=model.state_dict() if is_improvement else None,
            elapsed_time=history.timer.timer(),
        )

        _, best_f1 = history.task(task_name).val.best('f1')
        loop.set_postfix({
            "tr_loss": f"{train_loss:.4f}",
            "tr_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}({val_f1:.4f})",
            "best": f"{best_f1:.4f}",
        })

        metrics = {
            "val_f1": val_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_loss": train_loss,
            "train_acc": train_acc,
        }
        if compute_train_f1:
            metrics["train_f1"] = train_f1
            metrics["grad_norm"] = avg_grad_norm

        cb.on_epoch_end(CallbackContext(
            epoch=epoch_idx,
            epochs_total=epochs,
            metrics=metrics,
        ))

        # Early stopping based on validation F1 patience
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            logger.info("Early stopping at epoch %d. Best val F1: %.4f", epoch_idx + 1, best_val_f1)
            break

        current_time = time.time()
        if (target_cutoff is not None and val_f1 * 100 >= target_cutoff) or (
                timeout is not None and (current_time - start_time) > timeout):
            logger.info("Stopping early at epoch %d with validation F1 score: %.4f.", epoch_idx + 1, val_f1)
            break

        if cb.stop_training:
            logger.info("Callback requested stop at epoch %d.", epoch_idx + 1)
            break

    cb.on_train_end(CallbackContext(epoch=epoch_idx, epochs_total=epochs))
