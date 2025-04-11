import torch
import torch.optim as optim
import time
from datetime import timedelta

from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import OneCycleLR

from common.flops import calculate_flops
from common.mps_support import clear_mps_cache
from common.training_progress import TrainingProgressBar
from configs.globals import DEVICE
from configs.model import ModelParameters, ModelConfig
from criterion.FocalLoss import FocalLoss
from metrics.metrics import FoldResults, TrainingMetrics
from modeling.evaluate import evaluate_model
from modeling.validation import validation_best_model
from data.create_fold import SuperInputData
from models.mtl_poi import MTLnet
from criterion.nash_mtl import NashMTL, WeightMethod
from criterion.pcgrad import PCGrad


# Training Function
def train_model(model,
                optimizer,
                scheduler,
                dataloader_next: SuperInputData,
                dataloader_category: SuperInputData,
                next_criterion,
                category_criterion,
                mtl_criterion: WeightMethod,
                num_epochs,
                num_classes,
                gradient_accumulation_steps=1):
    """
    Train the model with multi-task learning.

    Args:
        model: The MTLPOI model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloader_next: SuperInputData for next POI prediction
        dataloader_category: SuperInputData for category prediction
        task_loss_fn: Task-specific loss function
        mtl_criterion: Multi-task loss function
        num_epochs: Number of training epochs
        num_classes: Number of POI classes
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Returns:
        FoldResults object with training metrics
    """
    # Initialize fold results and training components
    fold_results = FoldResults()
    pcgrad = PCGrad(n_tasks=2, device=DEVICE, reduction="sum")

    # Get FLOPS metrics
    fold_results.flops = calculate_flops(dataloader_category, dataloader_next, model)
    fold_results.flops.display()

    # Set gradient clipping norm
    max_grad_norm = 1.0

    # Create progress bar that extends tqdm
    progress = TrainingProgressBar(
        num_epochs,
        [dataloader_next.train.dataloader,
         dataloader_category.train.dataloader],
    )

    # Main training loop - iterate directly over progress bar for epochs
    for _ in progress:
        model.train()

        # Initialize metrics
        running_loss = 0.0
        next_running_loss = 0.0
        category_running_loss = 0.0
        next_running_acc = 0.0
        category_running_acc = 0.0
        steps = 0

        # Reset gradients at the beginning
        optimizer.zero_grad()
        batch_idx = 0

        # Iterate over batches with automatic progress tracking
        for data_next, data_category in progress.iter_epoch():
            # Move data to device
            x_next = data_next['x'].to(DEVICE, non_blocking=True)
            y_next = data_next['y'].to(DEVICE, non_blocking=True)
            x_category = data_category['x'].to(DEVICE, non_blocking=True)
            y_category = data_category['y'].to(DEVICE, non_blocking=True)

            # Convert to contiguous tensors for more efficient transfer
            x_next = x_next.contiguous()
            y_next = y_next.contiguous()
            x_category = x_category.contiguous()
            y_category = y_category.contiguous()

            optimizer.zero_grad()

            category_output, next_poi_output = model((x_category, x_next))

            # Process predictions
            # pred_next, truth_next = NextPoiNet.reshape_output(next_poi_output, y_next, num_classes)
            # pred_category, truth_category = CategoryPoiNet.reshape_output(category_output, y_category)

            pred_next, truth_next = next_poi_output, y_next
            pred_category, truth_category = category_output, y_category

            # Calculate losses (normalized by accumulation steps)
            next_loss = next_criterion(pred_next, truth_next)
            category_loss = category_criterion(pred_category, truth_category)
            loss, _ = mtl_criterion.backward(
                torch.stack([next_loss, category_loss]),
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
            )

            # Apply optimizer step and scaler update after accumulating gradients
            if DEVICE == 'mps':
                torch.mps.synchronize()  # Ensure all operations complete

            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scheduler.step()

            # Track metrics (use non-scaled loss for metrics)
            batch_loss = loss.item()
            batch_loss_next = next_loss.item()
            batch_loss_category = category_loss.item()

            running_loss += batch_loss
            next_running_loss += batch_loss_next
            category_running_loss += batch_loss_category

            # Calculate accuracy for monitoring
            with torch.no_grad():
                pred_next_cls = torch.argmax(pred_next, dim=1)
                pred_category_cls = torch.argmax(pred_category, dim=1)

                next_report = classification_report(
                    truth_next.cpu().numpy(),
                    pred_next_cls.cpu().numpy(),
                    output_dict=True,
                    zero_division=0
                )

                category_report = classification_report(
                    truth_category.cpu().numpy(),
                    pred_category_cls.cpu().numpy(),
                    output_dict=True,
                    zero_division=0
                )

                f1_next = next_report['macro avg']['f1-score']
                f1_category = category_report['macro avg']['f1-score']

                next_acc = next_report['accuracy']
                category_acc = category_report['accuracy']

                next_running_acc += next_acc
                category_running_acc += category_acc

            # Update metrics on progress bar (it automatically handles the progress update)
            progress.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'next': f'{f1_next:.4f}({next_acc:.4})',
                'cat': f'{f1_category:.4f}({category_acc:.4f})'
            })
            steps += 1

        # Calculate epoch metrics
        epoch_loss = running_loss / steps
        fold_results.mtl.add_loss(epoch_loss)
        fold_results.next.add_loss(0)
        fold_results.category.add_loss(0)
        fold_results.next.add_accuracy(next_running_acc / steps)
        fold_results.category.add_accuracy(category_running_acc / steps)

        # Validation phase with progress tracking
        with progress.validation():
            acc_val_next, f1_val_next, acc_val_category, f1_val_category, loss_val_next = evaluate_model(
                model,
                [dataloader_next.val.dataloader, dataloader_category.val.dataloader],
                next_criterion,
                category_criterion,
                mtl_criterion,
                DEVICE,
                num_classes=num_classes
            )

            # Store validation metrics
            fold_results.mtl.add_val_loss(loss_val_next)
            fold_results.next.add_val_loss(0)
            fold_results.category.add_val_loss(0)
            fold_results.next.add_val_accuracy(acc_val_next, model_state=model.state_dict())
            fold_results.category.add_val_accuracy(acc_val_category, model_state=model.state_dict())

        # Update metrics on progress bar with validation results
        progress.set_postfix({
            'val_loss': f'{loss_val_next:.4f}',
            'next_val': f'{f1_val_next:.4f}({acc_val_next:.4f})',
            'cat_val': f'{f1_val_category:.4f}({acc_val_category:.4f})'
        })

    return fold_results


# Cross-validation function
def train_with_cross_validation(dataloaders: dict[int, dict[str, SuperInputData]],
                                num_classes: int,
                                num_epochs: int,
                                learning_rate: float,
                                gradient_accumulation_steps: int = 2):
    """
    Train with cross-validation with efficient batching and gradient accumulation.

    Args:
        dataloaders: Dictionary of fold index to task dataloaders
        num_classes: Number of POI classes
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        gradient_accumulation_steps: Number of batches to accumulate gradients

    Returns:
        TrainingMetrics object with cross-validation results
    """
    training_metric = TrainingMetrics()

    # Track cross-validation time
    cv_start_time = time.time()
    total_folds = len(dataloaders)

    for fold_idx, (i_fold, dataloader) in enumerate(dataloaders.items()):
        # Track fold time
        fold_start_time = time.time()

        print(f"\n{'#' * 110}")
        print(f'FOLD {i_fold} [{fold_idx + 1}/{total_folds}]:')
        clear_mps_cache()

        # Initialize model with enhanced weight initialization
        model = MTLnet(
            feature_size=ModelParameters.INPUT_DIM,
            shared_layer_size=ModelParameters.SHARED_LAYER_SIZE,
            num_classes=ModelConfig.NUM_CLASSES,
            num_heads=ModelParameters.NUM_HEADS,
            num_layers=ModelParameters.NUM_LAYERS,
            seq_length=ModelParameters.SEQ_LENGTH,
            num_shared_layers=ModelParameters.NUM_SHARED_LAYERS
        )
        model = model.to(DEVICE)

        # Get dataloaders
        dataloader_next: SuperInputData = dataloader['next']
        dataloader_category = dataloader['category']

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate * 2,  # Start with higher base learning rate
            weight_decay=1e-5,  # Lower weight decay
            betas=(0.9, 0.95),  # More momentum
            eps=1e-8
        )

        # Learning rate scheduler with patience
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 20,  # Double the peak learning rate
            epochs=num_epochs,
            steps_per_epoch=len(dataloader_next.train.dataloader),
            pct_start=0.2,  # Reach peak LR earlier
            div_factor=25,
            final_div_factor=10000  # Stronger annealing at the end
        )

        # Initialize loss functions
        next_criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
        category_criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
        mtl_criterion = NashMTL(n_tasks=2, device=DEVICE, max_norm=1.0, update_weights_every=1)
        # mtl_criterion = NaiveLoss(alpha=0.5, beta=0.5)

        # Train the model with gradient accumulation
        results = train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            next_criterion,category_criterion, mtl_criterion, num_epochs, num_classes,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Run final validation
        print("\nRunning final validation...")
        report_next, report_category = validation_best_model(
            dataloader_next.val.dataloader,
            dataloader_category.val.dataloader,
            results.next.best_model,
            results.category.best_model,
            model
        )

        results.add_next_report(report_next)
        results.add_category_report(report_category)

        results.display_final_summary()

        # Add fold results to training metrics
        training_metric.add_fold_results(results)

        # Display fold time statistics
        fold_time = time.time() - fold_start_time
        elapsed_time = time.time() - cv_start_time
        remaining_folds = total_folds - (fold_idx + 1)
        estimated_time_remaining = (fold_time * remaining_folds) if remaining_folds > 0 else 0

        print(f"\nFold {i_fold} completed in {timedelta(seconds=int(fold_time))}")
        print(f"Cross-validation progress: {fold_idx + 1}/{total_folds} folds")
        print(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        print(f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}")

    # Show total cross-validation time
    total_cv_time = time.time() - cv_start_time
    print(f"\n{'#' * 110}")
    print(f"Cross-validation completed in {timedelta(seconds=int(total_cv_time))}")

    # Display summary metrics
    training_metric.display_summary()
    training_metric.export_to_csv()

    return training_metric
