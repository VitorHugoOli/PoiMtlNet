import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from configs.globals import DEVICE, CATEGORIES_MAP
from configs.model import InputsConfig

from configs.next_config import CfgNextModel, CfgNextHyperparams, CfgNextTraining
from model.next.next_head_enhanced import NextHeadHybrid, NextHeadGRU, NextHeadTemporalCNN, NextHeadLSTM
from train.next.evaluation import evaluate
from train.next.trainer import train
from model.next.next_head import NextHeadSingle
from common.calc_flops.calculate_model_flops import calculate_model_flops
from common.ml_history import MLHistory, FlopsMetrics, NeuralParams
from common.ml_history.fold import FoldHistory
from common.mps_support import clear_mps_cache

import torch
import torch.nn as nn

def _extract_diagnostics(
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        best_state: dict,
        fold_history: FoldHistory,
) -> None:
    """Extract confusion matrix and attention weights from the best model."""
    model.eval()
    model.load_state_dict(best_state)

    num_classes = CfgNextModel.NUM_CLASSES
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
):
    """Run cross-validation for the model."""
    for idx, (train_loader, val_loader) in enumerate(folds):
        model = NextHeadSingle(
            embed_dim=CfgNextModel.INPUT_DIM,
            num_classes=CfgNextModel.NUM_CLASSES,
            num_heads=CfgNextModel.NUM_HEADS,
            seq_length=CfgNextModel.MAX_SEQ_LENGTH,
            num_layers=CfgNextModel.NUM_LAYERS,
            dropout=CfgNextModel.DROPOUT,
        ).to(DEVICE)

        y_all = train_loader.dataset.targets.numpy()
        cls = np.arange(CfgNextModel.NUM_CLASSES)
        weights = compute_class_weight('balanced', classes=cls, y=y_all)
        alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

        criterion = nn.CrossEntropyLoss(
            reduction='mean',
            weight=alpha,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=CfgNextHyperparams.LR,
            weight_decay=CfgNextHyperparams.WEIGHT_DECAY,
        )

        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=CfgNextHyperparams.MAX_LR,
            epochs=CfgNextTraining.EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=CfgNextTraining.BATCH_SIZE,
                num_epochs=CfgNextTraining.EPOCHS,
                learning_rate=CfgNextHyperparams.LR,
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
            timeout=InputsConfig.TIMEOUT_TEST,
            target_cutoff=InputsConfig.NEXT_TARGET,
        )

        report = evaluate(model, val_loader, DEVICE, best_state=history.fold.task('next').best.best_state)

        _extract_diagnostics(
            model, val_loader, DEVICE,
            best_state=history.fold.task('next').best.best_state,
            fold_history=history.fold,
        )

        history.fold.task('next').report = report
        history.step()
        history.display.end_fold()

        # Free MPS memory between folds
        if DEVICE.type == 'mps':
            clear_mps_cache()
