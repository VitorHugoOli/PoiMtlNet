import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from datetime import timedelta
from tqdm.auto import tqdm
from calflops import calculate_flops
from sklearn.metrics import classification_report, accuracy_score
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from itertools import cycle

from common.training_progress import TrainingProgressBar
from configs.globals import DEVICE
from configs.model import ModelParameters, ModelConfig
from loss.FocalLoss import FocalLoss
from loss.NaiveLoss import NaiveLoss
from metrics.metrics import FoldResults, FlopsMetrics, TrainingMetrics
from models.category_net import CategoryPoiNet
from data.create_fold import SuperInputData
from models.mtl_poi import MTLnet
from models.next_poi_net import NextPoiNet
from optmizer.pcgrad import PCGrad


# Evaluation Functions
def evaluate_model(model, dataloader, loss_function, reshape_method, foward_method, device, num_classes=None):
    """
    Unified evaluation function for both validation and testing.

    Args:
        model: The MTLPOI model
        dataloader: DataLoader with evaluation data
        loss_functions: Dictionary of loss functions
        reshape_method: Method to reshape output and target
        foward_method: Method to forward data through the model
        device: Device to run evaluation on
        num_classes: Number of POI classes

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            x, y = data['x'].to(device), data['y'].to(device)

            # Forward pass
            out = foward_method(x)
            if num_classes is not None:
                pred, truth = reshape_method(out, y, num_classes)
            else:
                pred, truth = reshape_method(out, y)

            # Calculate loss
            loss = loss_function(pred, truth).item()
            total_loss += loss * truth.size(0)  # Weight by batch size

            # Calculate accuracy
            pred_class = torch.argmax(pred, dim=1)
            total_correct += (pred_class == truth).sum().item()
            total_samples += truth.size(0)

    # Return average accuracy and loss
    return total_correct / total_samples, total_loss / total_samples


# Training Function
def train_model(model,
                optimizer,
                scheduler,
                dataloader_next: SuperInputData,
                dataloader_category: SuperInputData,
                task_loss_fn,
                mtl_loss_fn,
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
        mtl_loss_fn: Multi-task loss function
        num_epochs: Number of training epochs
        num_classes: Number of POI classes
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Returns:
        FoldResults object with training metrics
    """
    # Initialize fold results and training components
    fold_results = FoldResults()
    pcgrad = PCGrad(n_tasks=2, device=DEVICE, reduction="sum")
    scaler = GradScaler()

    # Get FLOPS metrics
    fold_results.flops = calculate_model_flops(dataloader_category, dataloader_next, model)
    fold_results.flops.display()

    # Set gradient clipping norm
    max_grad_norm = 1.0

    # Create progress bar that extends tqdm
    progress = TrainingProgressBar(
        num_epochs,
        [dataloader_next.train.dataloader,
         dataloader_category.train.dataloader],
        desc="Training"
    )

    # Main training loop - iterate directly over progress bar for epochs
    for _ in progress:
        fold_results.start_epoch()
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
            x_next = data_next['x'].to(DEVICE)
            y_next = data_next['y'].to(DEVICE)
            x_category = data_category['x'].to(DEVICE)
            y_category = data_category['y'].to(DEVICE)

            # Forward pass for both tasks with mixed precision
            out_category, out_next = model(x_category, x_next)

            # Process predictions
            pred_next, truth_next = NextPoiNet.reshape_output(out_next, y_next, num_classes)
            pred_category, truth_category = CategoryPoiNet.reshape_output(out_category, y_category)

            # Calculate losses (normalized by accumulation steps)
            loss_next = task_loss_fn(pred_next, truth_next) / gradient_accumulation_steps
            loss_category = task_loss_fn(pred_category, truth_category) / gradient_accumulation_steps
            loss_mtl = mtl_loss_fn.compute_loss(loss_next, loss_category)

            # Backpropagate scaled loss using PCGrad for multi-task optimization
            scaler.scale(loss_next).backward(retain_graph=True)
            scaler.scale(loss_category).backward(retain_graph=True)

            # Apply PCGrad to manage gradient conflicts between tasks
            pcgrad.backward(
                losses=torch.stack([loss_next, loss_category]),
                shared_parameters=list(model.shared_layers.parameters()),
            )

            # Apply optimizer step and scaler update after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step and scaler update
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Update batch index
            batch_idx += 1

            # Track metrics (use non-scaled loss for metrics)
            batch_loss = loss_mtl.item() * gradient_accumulation_steps
            batch_loss_next = loss_next.item() * gradient_accumulation_steps
            batch_loss_category = loss_category.item() * gradient_accumulation_steps

            running_loss += batch_loss
            next_running_loss += batch_loss_next
            category_running_loss += batch_loss_category

            # Calculate accuracy for monitoring
            with torch.no_grad():
                pred_next_cls = torch.argmax(pred_next, dim=1)
                pred_category_cls = torch.argmax(pred_category, dim=1)

                next_acc = accuracy_score(truth_next.cpu().numpy(), pred_next_cls.cpu().numpy())
                category_acc = accuracy_score(truth_category.cpu().numpy(), pred_category_cls.cpu().numpy())

                next_running_acc += next_acc
                category_running_acc += category_acc

            # Update metrics on progress bar (it automatically handles the progress update)
            progress.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'next': f'{batch_loss_next:.4f}({next_acc:.4f})',
                'cat': f'{batch_loss_category:.4f}({category_acc:.4f})'
            })
            steps += 1

        # Calculate epoch metrics
        epoch_loss = running_loss / steps
        fold_results.mtl.add_loss(epoch_loss)
        fold_results.next.add_loss(next_running_loss / steps)
        fold_results.category.add_loss(category_running_loss / steps)
        fold_results.next.add_accuracy(next_running_acc / steps)
        fold_results.category.add_accuracy(category_running_acc / steps)

        # Validation phase with progress tracking
        with progress.validation():
            acc_val_category, loss_val_category = evaluate_model(
                model, dataloader_category.val.dataloader, task_loss_fn,
                reshape_method=CategoryPoiNet.reshape_output,
                foward_method=model.forward_categorypoi,
                device=DEVICE,
            )

            acc_val_next, loss_val_next = evaluate_model(
                model, dataloader_next.val.dataloader, task_loss_fn,
                reshape_method=NextPoiNet.reshape_output,
                foward_method=model.forward_nextpoi,
                device=DEVICE,
                num_classes=num_classes,
            )

            # Store validation metrics
            fold_results.category.add_val_loss(loss_val_category)
            fold_results.next.add_val_loss(loss_val_next)
            fold_results.category.add_val_accuracy(acc_val_category)
            fold_results.next.add_val_accuracy(acc_val_next)

        # Update metrics on progress bar with validation results
        progress.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'next_val': f'{loss_val_next:.4f}({acc_val_next:.4f})',
            'cat_val': f'{loss_val_category:.4f}({acc_val_category:.4f})'
        })

        # End epoch timing and update scheduler
        fold_results.end_epoch()
        scheduler.step((loss_val_next + loss_val_category) / 2)

    return fold_results


def calculate_model_flops(dataloader_category, dataloader_next, model):
    sample_category = next(iter(dataloader_category.train.dataloader))['x'].to(DEVICE)
    sample_next = next(iter(dataloader_next.train.dataloader))['x'].to(DEVICE)
    flops, macs, params = calculate_flops(model=model,
                                          kwargs={
                                              'x1': sample_category,
                                              'x2': sample_next
                                          },
                                          print_results=False)
    return FlopsMetrics(flops=flops, macs=macs, params=params)


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
        dataloader_next = dataloader['next']
        dataloader_category = dataloader['category']

        # Use AdamW with weight decay for better convergence
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler with patience
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=num_epochs,
            steps_per_epoch=len(dataloader_next.train.dataloader),
            pct_start=0.3,
            div_factor=25
        )

        # Initialize loss functions
        task_loss_fn = FocalLoss(alpha=1, gamma=2, reduction='mean')
        mtl_loss_fn = NaiveLoss()

        # Train the model with gradient accumulation
        results = train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            task_loss_fn, mtl_loss_fn, num_epochs, num_classes,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Run final validation
        print("\nRunning final validation...")
        model.eval()
        report_next, report_category = validation_model(dataloader_category, dataloader_next, model, num_classes)
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

    return training_metric


def validation_model(dataloader_category, dataloader_next, model, num_classes):
    """
    Perform final validation and add classification reports to training metrics.
    
    Args:
        dataloader_category: Dataloader for category prediction
        dataloader_next: Dataloader for next POI prediction
        model: The trained model
        num_classes: Number of POI classes
        training_metric: Training metrics object to update
    """
    with torch.no_grad():
        # Move validation data to device in batches to avoid OOM errors
        all_pred_next = []
        all_truth_next = []
        all_pred_category = []
        all_truth_category = []

        # Evaluate next POI task in batches
        for batch in dataloader_next.val.dataloader:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            out = model.forward_nextpoi(x)
            pred, truth = NextPoiNet.reshape_output(out, y, num_classes)
            pred_class = torch.argmax(pred, dim=1)

            all_pred_next.append(pred_class.cpu())
            all_truth_next.append(truth.cpu())

        # Evaluate category task in batches
        for batch in dataloader_category.val.dataloader:
            x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
            out = model.forward_categorypoi(x)
            pred, truth = CategoryPoiNet.reshape_output(out, y)
            pred_class = torch.argmax(pred, dim=1)

            all_pred_category.append(pred_class.cpu())
            all_truth_category.append(truth.cpu())

            # Concatenate results
        pred_next = torch.cat(all_pred_next)
        truth_next = torch.cat(all_truth_next)
        pred_category = torch.cat(all_pred_category)
        truth_category = torch.cat(all_truth_category)

        # Generate classification reports
        report_next = classification_report(truth_next, pred_next, output_dict=True)
        report_category = classification_report(truth_category, pred_category, output_dict=True)

        return report_next, report_category
