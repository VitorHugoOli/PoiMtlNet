import numpy as np
import torch
import torch.optim as optim
import time
from pathlib import Path
from typing import Optional

from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import OneCycleLR

from common.calc_flops.calculate_model_flops import calculate_model_flops
from common.mps_support import clear_mps_cache
from common.training_progress import TrainingProgressBar
from configs.globals import DEVICE
from configs.experiment import ExperimentConfig
from criterion.nash_mtl import NashMTL
from etl.create_fold import TaskFoldData, FoldResult
from model.mtlnet.mtl_poi import MTLnet
from common.ml_history import MLHistory, FlopsMetrics, NeuralParams
from common.ml_history.fold import FoldHistory, TaskHistory
from train.mtlnet.evaluate import evaluate_model
from train.mtlnet.validation import validation_best_model


# Training Function
def train_model(model: torch.nn.Module,
                optimizer,
                scheduler,
                dataloader_next: TaskFoldData,
                dataloader_category: TaskFoldData,
                next_criterion,
                category_criterion,
                mtl_criterion,
                num_epochs,
                num_classes,
                fold_history=FoldHistory.standalone({'next', 'category'}),
                gradient_accumulation_steps=1,
                max_grad_norm: float = 1.0,
                timeout: Optional[int] = None,
                next_target_cutoff: Optional[float] = None,
                category_target_cutoff: Optional[float] = None
                ):
    """
    Train the model with multi-task learning.
    """
    start_time = time.time()

    # Create progress bar that extends tqdm
    progress = TrainingProgressBar(
        num_epochs,
        [dataloader_next.train.dataloader,
         dataloader_category.train.dataloader],
    )

    cutoff_hits = {
        'next': False,
        'category': False
    }

    # Initialize model-level tracking
    if fold_history.model_task is None:
        fold_history.model_task = TaskHistory()

    # Main training loop
    for epoch_idx in progress:
        model.train()

        # Initialize on-device accumulators to avoid per-batch MPS syncs
        running_loss = torch.tensor(0.0, device=DEVICE)
        next_running_loss = torch.tensor(0.0, device=DEVICE)
        category_running_loss = torch.tensor(0.0, device=DEVICE)
        steps = 0

        # Collect predictions on-device, compute metrics once per epoch
        all_next_preds, all_next_targets = [], []
        all_cat_preds, all_cat_targets = [], []

        # Reset gradients at the beginning
        optimizer.zero_grad(set_to_none=True)

        # Iterate over batches with automatic progress tracking
        for data_next, data_category in progress.iter_epoch():
            # Move data to device
            x_next, y_next = data_next
            x_next = x_next.to(DEVICE, non_blocking=True)
            y_next = y_next.to(DEVICE, non_blocking=True)
            x_category, y_category = data_category
            x_category = x_category.to(DEVICE, non_blocking=True)
            y_category = y_category.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            category_output, next_poi_output = model((x_category, x_next))

            pred_next, truth_next = next_poi_output, y_next
            pred_category, truth_category = category_output, y_category

            # Calculate losses
            next_loss = next_criterion(pred_next, truth_next)
            category_loss = category_criterion(pred_category, truth_category)
            loss, _ = mtl_criterion.backward(
                torch.stack([next_loss, category_loss]),
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
            )

            optimizer.step()
            scheduler.step()

            # Accumulate on-device — no .item() sync per batch
            running_loss += loss.detach()
            next_running_loss += next_loss.detach()
            category_running_loss += category_loss.detach()

            # Collect predictions on-device for epoch-level metrics
            with torch.no_grad():
                all_next_preds.append(torch.argmax(pred_next, dim=1))
                all_next_targets.append(truth_next)
                all_cat_preds.append(torch.argmax(pred_category, dim=1))
                all_cat_targets.append(truth_category)

            steps += 1

        # Single MPS→CPU transfer per epoch for all predictions
        epoch_next_preds = torch.cat(all_next_preds).cpu().numpy()
        epoch_next_targets = torch.cat(all_next_targets).cpu().numpy()
        epoch_cat_preds = torch.cat(all_cat_preds).cpu().numpy()
        epoch_cat_targets = torch.cat(all_cat_targets).cpu().numpy()

        # Compute metrics once per epoch instead of per batch
        next_report = classification_report(
            epoch_next_targets, epoch_next_preds,
            output_dict=True, zero_division=0
        )
        category_report = classification_report(
            epoch_cat_targets, epoch_cat_preds,
            output_dict=True, zero_division=0
        )

        f1_next = next_report['macro avg']['f1-score']
        f1_category = category_report['macro avg']['f1-score']
        next_acc = next_report['accuracy']
        category_acc = category_report['accuracy']

        # Calculate epoch metrics (single sync for losses)
        epoch_loss = running_loss.item() / steps
        epoch_loss_next = next_running_loss.item() / steps
        epoch_loss_category = category_running_loss.item() / steps

        progress.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'next': f'{f1_next:.4f}({next_acc:.4f})',
            'cat': f'{f1_category:.4f}({category_acc:.4f})'
        })

        fold_history.model_task.log_train(loss=epoch_loss, accuracy=0)
        fold_history.log_train('next',
                               loss=epoch_loss_next,
                               f1=f1_next,
                               accuracy=next_acc)
        fold_history.log_train('category',
                               loss=epoch_loss_category,
                               f1=f1_category,
                               accuracy=category_acc)

        # Validation phase with progress tracking
        with progress.validation():
            acc_val_next, f1_val_next, acc_val_category, f1_val_category, loss_val = evaluate_model(
                model,
                [dataloader_next.val.dataloader, dataloader_category.val.dataloader],
                next_criterion,
                category_criterion,
                mtl_criterion,
                DEVICE,
            )

            # Only create state_dict when at least one task improves
            next_improved = f1_val_next > fold_history.task('next').best.best_value
            cat_improved = f1_val_category > fold_history.task('category').best.best_value
            state = model.state_dict() if (next_improved or cat_improved) else None

            fold_history.model_task.log_val(loss=loss_val, f1=0, accuracy=0)
            fold_history.log_val(
                'next',
                loss=0,
                accuracy=acc_val_next,
                f1=f1_val_next,
                model_state=state if next_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )
            fold_history.log_val(
                'category',
                loss=0,
                accuracy=acc_val_category,
                f1=f1_val_category,
                model_state=state if cat_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )

        # Update metrics on progress bar with validation results
        progress.set_postfix({
            'val_loss': f'{loss_val:.4f}',
            'next_val': f'{f1_val_next:.4f}({acc_val_next:.4f})',
            'cat_val': f'{f1_val_category:.4f}({acc_val_category:.4f})'
        })

        if next_target_cutoff is not None and f1_val_next * 100 >= next_target_cutoff:
            cutoff_hits['next'] = True

        if category_target_cutoff is not None and f1_val_category * 100 >= category_target_cutoff:
            cutoff_hits['category'] = True

        if cutoff_hits['next'] and cutoff_hits['category']:
            print(f"\nStopping early at epoch {epoch_idx + 1} with validation F1 scores: "
                  f"Next: {f1_val_next:.4f}, Category: {f1_val_category:.4f}.")
            break

        current_time = time.time()
        if timeout is not None and (current_time - start_time) > timeout:
            print(f"\nTraining timed out after {timeout:.2f} seconds during epoch {epoch_idx + 1}.")
            break

    return fold_history


# Cross-validation function
def train_with_cross_validation(dataloaders: dict[int, FoldResult],
                                history: MLHistory,
                                config: ExperimentConfig,
                                results_path: Optional[Path] = None):
    num_classes = config.model_params.get('num_classes', 7)

    for fold_idx, (i_fold, dataloader) in enumerate(dataloaders.items()):
        clear_mps_cache()

        # Initialize model
        model = MTLnet(**config.model_params)

        model = model.to(DEVICE)

        # Get dataloaders
        dataloader_next: TaskFoldData = dataloader.next
        dataloader_category: TaskFoldData = dataloader.category

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.optimizer_eps,
        )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=len(dataloader_next.train.dataloader),
        )

        cls = np.arange(num_classes)

        y_all_next = dataloader_next.train.y.numpy()
        weights_next = compute_class_weight('balanced', classes=cls, y=y_all_next)
        alpha_next = torch.tensor(weights_next, dtype=torch.float32, device=DEVICE)

        y_all_category = dataloader_category.train.y.numpy()
        weights_cat = compute_class_weight('balanced', classes=cls, y=y_all_category)
        alpha_cat = torch.tensor(weights_cat, dtype=torch.float32, device=DEVICE)

        next_criterion = CrossEntropyLoss(reduction='mean', weight=alpha_next)
        category_criterion = CrossEntropyLoss(reduction='mean', weight=alpha_cat)
        mtl_criterion = NashMTL(n_tasks=2, device=DEVICE, **config.mtl_loss_params)

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=dataloader_next.train.dataloader.batch_size,
                num_epochs=config.epochs,
                learning_rate=config.learning_rate,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'mtl': mtl_criterion.__class__.__name__,
                    'next': next_criterion.__class__.__name__,
                    'category': category_criterion.__class__.__name__
                },
                criterion_state={
                    'mtl': {},
                    'next': next_criterion.state_dict(),
                    'category': category_criterion.state_dict()
                }
            )
        )

        if history.flops is None:
            sample_category, _ = next(iter(dataloader_category.train.dataloader))
            sample_next, _ = next(iter(dataloader_next.train.dataloader))
            sample_category = sample_category.to(DEVICE)
            sample_next = sample_next.to(DEVICE)
            result = calculate_model_flops(model, [sample_category[1:], sample_next[1:]], print_report=True, units='K')
            history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))

        # Train the model
        train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            next_criterion, category_criterion, mtl_criterion,
            config.epochs, num_classes,
            fold_history=history.get_curr_fold(),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            timeout=config.timeout,
            next_target_cutoff=config.target_cutoff,
            category_target_cutoff=config.target_cutoff,
        )

        # Run final validation
        print("\nRunning final validation...")
        report_next, report_category = validation_best_model(
            dataloader_next.val.dataloader,
            dataloader_category.val.dataloader,
            history.fold.task('next').best.best_state,
            history.fold.task('category').best.best_state,
            model
        )

        history.fold.task('next').report = report_next
        history.fold.task('category').report = report_category

        history.step()
        history.display.end_fold()

    # Display summary metrics
    history.display.end_training()

    # Write run manifest
    if results_path is not None:
        from configs.experiment import RunManifest
        from configs.paths import IoPaths, EmbeddingEngine
        engine = EmbeddingEngine(config.embedding_engine)
        manifest = RunManifest.from_current_env(
            config=config,
            dataset_paths={
                "category_input": IoPaths.get_category(config.state, engine),
                "next_input": IoPaths.get_next(config.state, engine),
            },
        )
        manifest.write(results_path)
