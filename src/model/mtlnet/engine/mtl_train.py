
import torch
import torch.optim as optim
import time
from datetime import timedelta

from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import OneCycleLR

from common.flops import calculate_flops
from common.mps_support import clear_mps_cache
from common.training_progress import TrainingProgressBar
from configs.globals import DEVICE
from configs.model import ModelParameters, MTLModelConfig
from data.create_fold import SuperInputData
from criterion.nash_mtl import NashMTL
from model.mtlnet.engine.evaluate import evaluate_model
from model.mtlnet.engine.validation import validation_best_model
from model.mtlnet.modeling.mtl_poi import MTLnet
from utils.ml_history.metrics import MLHistory, FoldHistory
from utils.ml_history.parms.neural import NeuralParams


# Training Function
def train_model(model,
                optimizer,
                scheduler,
                dataloader_next: SuperInputData,
                dataloader_category: SuperInputData,
                next_criterion,
                category_criterion,
                mtl_criterion,
                num_epochs,
                num_classes,
                fold_history=FoldHistory.standalone({'next', 'category'}),
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
        next_running_f1 = 0.0
        category_running_f1 = 0.0
        steps = 0

        # Reset gradients at the beginning
        optimizer.zero_grad()

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

            optimizer.step()
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
                next_running_f1 += f1_next
                category_running_f1 += f1_category

            # Update metrics on progress bar (it automatically handles the progress update)
            progress.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'next': f'{f1_next:.4f}({next_acc:.4})',
                'cat': f'{f1_category:.4f}({category_acc:.4f})'
            })
            steps += 1

        # Calculate epoch metrics
        epoch_loss = running_loss / steps
        fold_history.model.add(loss=epoch_loss, accuracy=0)
        fold_history.to('next').add(loss=0,
                                    f1=next_running_f1 / steps,
                                    accuracy=next_running_acc / steps)
        fold_history.to('category').add(loss=0,
                                        f1=category_running_f1 / steps,
                                        accuracy=category_running_acc / steps)

        # Validation phase with progress tracking
        with progress.validation():
            acc_val_next, f1_val_next, acc_val_category, f1_val_category, loss_val = evaluate_model(
                model,
                [dataloader_next.val.dataloader, dataloader_category.val.dataloader],
                next_criterion,
                category_criterion,
                mtl_criterion,
                DEVICE,
                num_classes=num_classes
            )

            # Store validation metrics
            fold_history.model.add_val(
                val_loss=loss_val,
                val_f1=0,
                val_accuracy=0,
            )
            fold_history.to('next').add_val(
                val_loss=0,
                val_accuracy=acc_val_next,
                val_f1=f1_val_next,
                model_state=model.state_dict(),
                best_metric='val_f1'
            )
            fold_history.to('category').add_val(
                val_loss=0,
                val_accuracy=acc_val_category,
                val_f1=f1_val_category,
                model_state=model.state_dict(),
                best_metric='val_f1'
            )

        # Update metrics on progress bar with validation results
        progress.set_postfix({
            'val_loss': f'{loss_val:.4f}',
            'next_val': f'{f1_val_next:.4f}({acc_val_next:.4f})',
            'cat_val': f'{f1_val_category:.4f}({acc_val_category:.4f})'
        })

    return fold_history


# Cross-validation function
def train_with_cross_validation(dataloaders: dict[int, dict[str, SuperInputData]],
                                history: MLHistory,
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
    for fold_idx, (i_fold, dataloader) in enumerate(dataloaders.items()):
        history.display.start_fold()
        clear_mps_cache()

        # Initialize model with enhanced weight initialization
        model = MTLnet(
            feature_size=ModelParameters.INPUT_DIM,
            shared_layer_size=ModelParameters.SHARED_LAYER_SIZE,
            num_classes=MTLModelConfig.NUM_CLASSES,
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
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )

        # Learning rate scheduler with patience
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=num_epochs,
            steps_per_epoch=len(dataloader_next.train.dataloader),

        )

        # Initialize loss functions
        next_criterion = CrossEntropyLoss(reduction='mean')
        category_criterion = CrossEntropyLoss(reduction='mean')
        mtl_criterion = NashMTL(n_tasks=2, device=DEVICE, max_norm=1.0, update_weights_every=1)
        # mtl_criterion = NaiveLoss(alpha=0.5, beta=0.5)

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=dataloader_next.train.dataloader.batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
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
            history.flops = calculate_flops(dataloader_category, dataloader_next, model)
            history.display.flops()

        # Train the model with gradient accumulation
        train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            next_criterion, category_criterion, mtl_criterion, num_epochs, num_classes,
            fold_history=history.get_curr_fold(),
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Run final validation
        print("\nRunning final validation...")
        report_next, report_category = validation_best_model(
            dataloader_next.val.dataloader,
            dataloader_category.val.dataloader,
            history.get_curr_fold().to('next').best_model,
            history.get_curr_fold().to('category').best_model,
            model
        )

        history.get_curr_fold().to('next').add_report(report_next)
        history.get_curr_fold().to('category').add_report(report_category)

        history.step()
        history.display.end_fold()

    # Display summary metrics
    history.display.end_training()
