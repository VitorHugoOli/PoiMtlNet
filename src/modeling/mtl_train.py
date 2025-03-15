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
from sklearn.utils.class_weight import compute_class_weight
from torch import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import cycle

from configs.globals import DEVICE
from configs.model import ModelParameters, ModelConfig
from metrics.metrics import FoldResults, FlopsMetrics, TrainingMetrics
from models.category_net import CategoryPoiNet
from data.create_fold import SuperInputData
from models.mtl_poi import MTLnet
from models.next_poi_net import NextPoiNet
from optmizer.pcgrad import PCGrad


# Evaluation Functions
def evaluate_model(model, dataloader, loss_functions, reshape_method, foward_method, device, num_classes=None):
    """
    Unified evaluation function for both validation and testing.

    Args:
        model: The MTLPOI model
        dataloader: DataLoader with evaluation data
        loss_functions: Dictionary of loss functions
        one_hot: One-hot encoding tensor
        id_to_name: Dictionary mapping IDs to names
        task_name: Task name ('next' or 'category')
        num_classes: Number of POI classes
        mode: Evaluation mode ('val' or 'test')

    Returns:
        Tuple of (y_true, y_pred, average_loss) - loss is None for test mode
    """
    accuracy = 0
    running_loss = 0.0
    steps = 0

    model.eval()

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating", leave=False):
            x, y = data['x'], data['y']
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            out = foward_method(x)
            if num_classes is not None:
                pred, truth = reshape_method(out, y, num_classes)
            else:
                pred, truth = reshape_method(out, y)

            running_loss += loss_functions['category'](pred, truth).item()
            pred = torch.argmax(pred, dim=1)
            accuracy += accuracy_score(truth, pred)

            steps += 1

    running_loss /= steps
    accuracy /= steps

    return accuracy, running_loss


# Training Function
def train_model(model,
                optimizer,
                scheduler,
                dataloader_next: SuperInputData, dataloader_category: SuperInputData,
                loss_functions,
                num_epochs,
                num_classes,
                print_interval=1):
    """
    Train the model with multi-task learning.

    Args:
        model: The MTLPOI model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloader_next: SuperInputData
        dataloader_category: SuperInputData
        loss_functions: Dictionary of loss functions
        one_hot: One-hot encoding tensor
        num_epochs: Number of training epochs
        id_to_name: Dictionary mapping IDs to names
        num_classes: Number of POI classes
        print_interval: Interval for printing results

    Returns:
        Dictionary of training results

    """
    fold_results = FoldResults()
    pcgrad = PCGrad(n_tasks=2, device=DEVICE, reduction="sum")

    scaler = GradScaler()

    sample_category = next(iter(dataloader_category.train.dataloader))['x'].to(DEVICE)
    sample_next = next(iter(dataloader_next.train.dataloader))['x'].to(DEVICE)

    flops, macs, params = calculate_flops(model=model,
                                          kwargs={
                                              'x1': sample_category,
                                              'x2': sample_next
                                          },
                                          print_results=False)

    fold_results.flops = FlopsMetrics(flops=flops, macs=macs, params=params)
    fold_results.flops.display()
    
    # Create progress bar for epochs
    epoch_progress = tqdm(range(num_epochs), desc="Training epochs", unit="epoch")

    for epoch in epoch_progress:
        # Track epoch start time using FoldResults
        fold_results.start_epoch()
        
        model.train()

        running_loss = 0.0
        next_running_loss = 0.0
        category_running_loss = 0.0
        next_running_acc = 0.0
        category_running_acc = 0.0
        max_norm = 1.0

        steps = 0
        
        # Get dataset length to create batch progress bar
        dataset_length = min(len(dataloader_next.train.dataloader), len(dataloader_category.train.dataloader))
        
        # Create a progress bar for batches within this epoch
        batch_progress = tqdm(
            zip(dataloader_next.train.dataloader, dataloader_category.train.dataloader),
            total=dataset_length,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
            unit="batch"
        )

        for data_next, data_category in batch_progress:
            optimizer.zero_grad()

            x_next, y_next = data_next['x'].to(DEVICE), data_next['y'].to(DEVICE)
            x_category, y_category = data_category['x'].to(DEVICE), data_category['y'].to(DEVICE)

            # Forward pass for both tasks
            with autocast(device_type=DEVICE.type):
                out_category, out_next = model(x_category, x_next)

                # Process next POI predictions
                pred_next, truth_next = NextPoiNet.reshape_output(out_next, y_next, num_classes)
                pred_category, truth_category = CategoryPoiNet.reshape_output(out_category, y_category)

                # Calculate losses
                loss_next = loss_functions['category'](pred_next, truth_next)
                loss_category = loss_functions['category'](pred_category, truth_category)
                loss_mtl = loss_next + loss_category

            # aplicar pcgrad
            loss_next.backward(retain_graph=True)
            loss_category.backward(retain_graph=True)

            # Scale the loss and compute scaled gradients
            scaler.scale(loss_mtl).backward(retain_graph=True)

            pcgrad.backward(
                losses=torch.stack([loss_next, loss_category]),
                shared_parameters=list(model.shared_layers.parameters()),
            )

            # Clip gradients (after unscaling to get the correct magnitudes)
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            steps += 1

            running_loss += loss_mtl.item()
            next_running_loss += loss_next.item()
            category_running_loss += loss_category.item()

            pred_next = torch.argmax(pred_next, dim=1)
            pred_category = torch.argmax(pred_category, dim=1)

            next_running_acc += accuracy_score(truth_next, pred_next)
            category_running_acc += accuracy_score(truth_category, pred_category)

            # Update batch progress bar with current losses
            batch_progress.set_postfix({
                'loss': f'{loss_mtl.item():.4f}',
                'next_loss': f'{loss_next.item():.4f}',
                'cat_loss': f'{loss_category.item():.4f}',
                'next_acc': f'{accuracy_score(truth_next, pred_next):.4f}',
                'cat_acc': f'{accuracy_score(truth_category, pred_category):.4f}'
            })

        mtls_loss = running_loss / steps

        # Store training losses
        fold_results.mtl.add_loss(mtls_loss)
        fold_results.next.add_loss(next_running_loss / steps)
        fold_results.category.add_loss(category_running_loss / steps)
        fold_results.next.add_accuracy(next_running_acc / steps)
        fold_results.category.add_accuracy(category_running_acc / steps)

        # Evaluation is done in full precision
        print("\nRunning validation...")
        acc_aval_next, loss_aval_next = evaluate_model(
            model, dataloader_category.val.dataloader, loss_functions,
            reshape_method=CategoryPoiNet.reshape_output,
            foward_method=model.forward_categorypoi,
            device=DEVICE,
        )

        acc_aval_category, loss_aval_category = evaluate_model(
            model, dataloader_next.val.dataloader, loss_functions,
            reshape_method=NextPoiNet.reshape_output,
            foward_method=model.forward_nextpoi,
            device=DEVICE,
            num_classes=num_classes,
        )

        fold_results.category.add_val_loss(loss_aval_category)
        fold_results.next.add_val_loss(loss_aval_next)

        fold_results.category.add_val_accuracy(acc_aval_category)
        fold_results.next.add_val_accuracy(acc_aval_next)
        
        # End epoch timing and calculate duration
        fold_results.end_epoch()
        
        # Update the epoch progress bar
        epoch_progress.set_postfix({
            'loss': f'{mtls_loss:.4f}', 
            'val_loss': f'{(loss_aval_next + loss_aval_category)/2:.4f}'
        })
        
        # Let FoldResults handle the detailed metrics display
        fold_results.display_training_status(epoch, num_epochs)

        scheduler.step(mtls_loss)
    
    # Display final summary
    fold_results.display_final_summary()

    return fold_results


# Main training function with cross-validation
def train_with_cross_validation(dataloaders: dict[int, dict[str, SuperInputData]],
                                num_classes: int,
                                num_epochs: int,
                                learning_rate: float
                                ):
    """
    Train the model with cross-validation.

    Args:
        dataloaders: List of dataloaders for cross-validation
        num_classes: Number of POI classes
        num_epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer


    Returns:
        Dictionary of results
    """
    training_metric: TrainingMetrics = TrainingMetrics()
    
    # Keep track of cross-validation time
    cv_start_time = time.time()
    total_folds = len(dataloaders)

    for fold_idx, (i_fold, dataloader) in enumerate(dataloaders.items()):
        fold_start_time = time.time()
        
        print(f"\n{'#' * 110}")
        print(f'FOLD {i_fold} [{fold_idx+1}/{total_folds}]:')

        # Initialize model
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

        dataloader_next = dataloader['next']
        dataloader_category = dataloader['category']

        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # TODO: CHECK IF `dataloader_next.modeling.dataloader.dataset.x` works
        # Compute class weights for balanced loss
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(dataloader_category.train.y.cpu().numpy()),
            y=dataloader_category.train.y.cpu().numpy()
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

        # Initialize loss functions
        loss_functions = {
            'npc': nn.CrossEntropyLoss(),
            'category': nn.CrossEntropyLoss(weight=class_weights),
            'reconstruction': nn.MSELoss()
        }

        # Train model
        results = train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            loss_functions, num_epochs, num_classes)

        training_metric.add_fold_results(results)

        # final evaluation
        print("\nRunning final validation...")
        model.eval()

        validation_model(dataloader_category, dataloader_next, model, num_classes, training_metric)
        
        # Display fold completion time
        fold_time = time.time() - fold_start_time
        elapsed_time = time.time() - cv_start_time
        remaining_folds = total_folds - (fold_idx + 1)
        estimated_time_remaining = (fold_time * remaining_folds) if remaining_folds > 0 else 0
        
        print(f"\nFold {i_fold} completed in {timedelta(seconds=int(fold_time))}")
        print(f"Cross-validation progress: {fold_idx+1}/{total_folds} folds")
        print(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
        print(f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}")
    
    # Calculate and display total cross-validation time
    total_cv_time = time.time() - cv_start_time
    print(f"\n{'#' * 110}")
    print(f"Cross-validation completed in {timedelta(seconds=int(total_cv_time))}")
    
    # Display summary of all metrics across folds
    training_metric.display_summary()

    return training_metric


def validation_model(dataloader_category, dataloader_next, model, num_classes, training_metric):
    with torch.no_grad():
        out_next = model.forward_nextpoi(dataloader_next.val.x.to(DEVICE))
        out_category = model.forward_categorypoi(dataloader_category.val.x.to(DEVICE))

        pred_next, truth_next = NextPoiNet.reshape_output(out_next, dataloader_next.val.y.to(DEVICE), num_classes)
        pred_category, truth_category = CategoryPoiNet.reshape_output(out_category,
                                                                      dataloader_category.val.y.to(DEVICE))

        pred_next = torch.argmax(pred_next, dim=1)
        pred_category = torch.argmax(pred_category, dim=1)

        report_next = classification_report(pred_next, truth_next, zero_division=1, output_dict=True)
        report_category = classification_report(pred_category, truth_category, zero_division=1, output_dict=True)

        training_metric.add_report(report_next)
        training_metric.add_report(report_category)
