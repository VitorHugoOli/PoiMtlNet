import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from typing import Optional

from configs.globals import DEVICE, CATEGORIES_MAP
from configs.experiment import ExperimentConfig
from models.registry import create_model
from training.helpers import compute_class_weights, setup_optimizer, setup_scheduler
from training.runners.next_eval import evaluate
from training.runners.next_trainer import train
from utils.flops import calculate_model_flops
from tracking import MLHistory, FlopsMetrics, NeuralParams
from tracking.fold import FoldHistory
from utils.mps import clear_mps_cache

import torch
import torch.nn as nn

def _extract_diagnostics(
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        best_state: dict,
        fold_history: FoldHistory,
        num_classes: int = 7,
) -> None:
    """Extract confusion matrix and attention weights from the best model."""
    model.eval()
    model.load_state_dict(best_state)

    class_labels = list(range(num_classes))
    class_names = [CATEGORIES_MAP.get(i, f"Class-{i}") for i in class_labels]

    preds_list, targets_list = [], []
    attn_list = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits, attn = model(X_batch, return_attention=True)
            preds_list.append(logits.argmax(dim=1))
            targets_list.append(y_batch)
            attn_list.append(attn)

    # Single bulk GPU→CPU transfer
    all_preds = torch.cat(preds_list).cpu().numpy()
    all_targets = torch.cat(targets_list).numpy()
    all_attn = torch.cat(attn_list).cpu().numpy()  # (N, seq)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds, labels=class_labels)
    fold_history.add_artifact('confusion_matrix', {
        'matrix': cm.tolist(),
        'labels': class_names,
    })

    # Overall attention weights: mean/std per position
    fold_history.add_artifact('attention_weights', {
        'mean': all_attn.mean(axis=0).tolist(),
        'std': all_attn.std(axis=0).tolist(),
    })

    # Per-class attention: mean/std grouped by target class
    per_class_attn = {}
    for cls_idx, cls_name in zip(class_labels, class_names):
        mask = all_targets == cls_idx
        if mask.sum() > 0:
            cls_attn = all_attn[mask]
            per_class_attn[cls_name] = {
                'mean': cls_attn.mean(axis=0).tolist(),
                'std': cls_attn.std(axis=0).tolist(),
                'count': int(mask.sum()),
            }
    fold_history.add_artifact('attention_per_class', per_class_attn)


def run_cv(
        history: MLHistory,
        folds: list[tuple[DataLoader, DataLoader]],
        config: ExperimentConfig,
        results_path: Optional[Path] = None,
):
    """Run cross-validation for the model."""
    num_classes = config.model_params.get('num_classes', 7)

    for idx, (train_loader, val_loader) in enumerate(folds):
        model = create_model(config.model_name, **config.model_params).to(DEVICE)

        alpha = compute_class_weights(
            train_loader.dataset.targets.numpy(), num_classes, DEVICE
        )

        criterion = nn.CrossEntropyLoss(
            reduction='mean',
            weight=alpha if config.use_class_weights else None,
        )

        optimizer = setup_optimizer(model, config.learning_rate, config.weight_decay)
        scheduler = setup_scheduler(optimizer, config.max_lr, config.epochs, len(train_loader))

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=config.batch_size,
                num_epochs=config.epochs,
                learning_rate=config.learning_rate,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'next': criterion.__class__.__name__},
                criterion_state={
                    'next': criterion.state_dict()
                }
            )
        )

        # Calculate FLOPs only on first fold (architecture is identical across folds)
        if idx == 0:
            sample = next(iter(train_loader))[0].to(DEVICE)
            result = calculate_model_flops(model,
                                           sample_input=sample,
                                           print_report=True,
                                           units='K'
                                           )
            history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            history.get_curr_fold(),
            DEVICE,
            epochs=config.epochs,
            max_grad_norm=config.max_grad_norm,
            early_stopping_patience=config.early_stopping_patience,
            timeout=config.timeout,
            target_cutoff=config.target_cutoff,
        )

        report = evaluate(model, val_loader, DEVICE, best_state=history.fold.task('next').best.best_state)

        _extract_diagnostics(
            model, val_loader, DEVICE,
            best_state=history.fold.task('next').best.best_state,
            fold_history=history.fold,
            num_classes=num_classes,
        )

        history.fold.task('next').report = report
        history.step()

        # Free MPS memory between folds
        if DEVICE.type == 'mps':
            clear_mps_cache()

    # Write run manifest
    if results_path is not None:
        from configs.experiment import RunManifest
        from configs.paths import IoPaths, EmbeddingEngine
        engine = EmbeddingEngine(config.embedding_engine)
        manifest = RunManifest.from_current_env(
            config=config,
            dataset_paths={
                "next_input": IoPaths.get_next(config.state, engine),
            },
        )
        manifest.write(results_path)
