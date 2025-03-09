import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from calflops import calculate_flops
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from itertools import cycle

from configs.globals import CATEGORIES_MAP
from configs.model import ModelConfig, ModelParameters
from models.mtl_poi import MTLPOI
from data.create_fold import SuperInputData

# Set device for PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Evaluation Functions
def evaluate_model(model, dataloader, loss_functions, task_name, num_classes, mode='val'):
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
    y_true = []
    y_pred = []
    running_loss = 0.0
    steps = 0

    model.eval()

    with torch.no_grad():
        for data in dataloader:
            x, y = data['x'], data['y']
            x = x.to(device)
            y = y.to(device)

            if task_name == 'next':
                out = model.forward_nextpoi(x)

                if mode == 'val':
                    # Process validation data
                    B, S, _ = out.shape
                    out = out.view(B * S, -1)
                    y = y.view(B * S, -1)

                    valid_samples = y < num_classes
                    expanded_mask = valid_samples.expand(-1, out.shape[1])

                    y = y[valid_samples]
                    out = out[expanded_mask].view(-1, num_classes)

                    predicted = torch.argmax(out, dim=-1)
                    loss = loss_functions['category'](out, y)
                else:
                    # Process test data
                    predicted = torch.argmax(out, dim=-1)
                    valid_samples = y < num_classes
                    y = y[valid_samples]
                    predicted = predicted[valid_samples]

                y_true.extend(y.view(-1).tolist())
                y_pred.extend(predicted.view(-1).tolist())

                if mode == 'val':
                    running_loss += loss.item()
                    steps += 1

            else:  # category task
                out, r = model.forward_categorypoi(x)
                if mode == 'val':
                    out = out.squeeze(1)

                predicted = torch.argmax(out, dim=-1)
                y_true.extend(y.tolist())
                y_pred.extend(predicted.tolist())

                if mode == 'val':
                    classifier_loss = loss_functions['category'](out, y.view(-1))
                    encoder_loss = loss_functions['reconstruction'](r, x)
                    loss = classifier_loss + encoder_loss
                    running_loss += loss.item()
                    steps += 1

    # Calculate average loss for validation mode
    average_loss = running_loss / steps if steps > 0 and mode == 'val' else None

    # Print classification report
    if mode == 'val':
        print_classification_report_val(y_true, y_pred, task_name)
    else:
        print_classification_report_test(y_true, y_pred, task_name)

    return y_true, y_pred, average_loss


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
        dataloaders: Dictionary of dataloaders
        loss_functions: Dictionary of loss functions
        one_hot: One-hot encoding tensor
        num_epochs: Number of training epochs
        id_to_name: Dictionary mapping IDs to names
        num_classes: Number of POI classes
        print_interval: Interval for printing results

    Returns:
        Dictionary of training results
    """
    mtl_train_losses = []
    next_losses = []
    category_losses = []
    next_val_losses = []
    category_val_losses = []

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        next_running_loss = 0.0
        category_running_loss = 0.0
        max_norm = 1.0

        y_true_category = []
        y_pred_category = []
        y_true_next = []
        y_pred_next = []

        steps = 0
        category_iter = cycle(dataloader_category.train.dataloader)

        for data_next in dataloader_next.train.dataloader:
            data_category = next(category_iter)
            optimizer.zero_grad()

            x_next, y_next = data_next['x'], data_next['y']
            x_next = x_next.to(device)
            y_next = y_next.to(device)

            x_category, y_category = data_category['x'], data_category['y']
            x_category = x_category.to(device)
            y_category = y_category.to(device)

            # Forward pass for both tasks
            out_category, r, out_next = model(x_category, x_next)

            # Process category predictions
            category_predicted = torch.argmax(out_category, dim=-1)
            y_true_category.extend(y_category.tolist())
            y_pred_category.extend(category_predicted.tolist())

            # Process next POI predictions
            B, S, _ = out_next.shape
            out_next = out_next.view(B * S, -1)
            y_next = y_next.view(B * S, -1)
            idx_valid = (y_next < num_classes).view(-1)
            y_next = y_next[idx_valid].view(-1)
            out_next = out_next[idx_valid]
            next_predicted = torch.argmax(out_next, dim=-1)
            y_true_next.extend(y_next.tolist())
            y_pred_next.extend(next_predicted.view(-1).tolist())
            out_next = out_next.view(-1, num_classes)

            # Calculate losses
            loss_next = loss_functions['category'](out_next, y_next)

            out_category = out_category.squeeze(1)
            loss_PCat_cl = loss_functions['category'](out_category, y_category.view(-1))
            loss_PCat_enc = loss_functions['reconstruction'](r, x_category)
            loss_category = loss_PCat_cl + loss_PCat_enc

            # Combined multi-task loss
            loss = loss_next + loss_category
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

            steps += 1
            running_loss += loss.item()
            next_running_loss += loss_next.item()
            category_running_loss += loss_category.item()

        if steps > 0 and ((epoch + 1) % print_interval == 0 or epoch == 0):
            # Calculate epoch losses
            mtl_loss_by_epoch = running_loss / steps
            next_loss_by_epoch = next_running_loss / steps
            category_loss_by_epoch = category_running_loss / steps

            # Store training losses
            mtl_train_losses.append(mtl_loss_by_epoch)
            next_losses.append(next_loss_by_epoch)
            category_losses.append(category_loss_by_epoch)

            # Print training results
            print_train_losses(epoch, num_epochs, mtl_loss_by_epoch, next_loss_by_epoch, category_loss_by_epoch)
            print_classification_report_train(y_true_category, y_pred_category, y_true_next, y_pred_next)
            print()
            print("-" * 110)
            print()

            # Validate model
            # Category validation
            val_y_true_category, val_y_pred_category, category_val_loss = evaluate_model(
                model, dataloader_category.val.dataloader, loss_functions,
                'category', num_classes, mode='val'
            )
            category_val_losses.append(category_val_loss)

            # Next POI validation
            val_y_true_next, val_y_pred_next, next_val_loss = evaluate_model(
                model, dataloader_category.val.dataloader, loss_functions,
                'next', num_classes, mode='val'
            )
            next_val_losses.append(next_val_loss)

            print("*" * 110)
            print_val_losses(next_val_loss, category_val_loss)

            # Update learning rate
            scheduler.step(mtl_loss_by_epoch)

    # Compile results
    results = {
        'mtl_train_losses': mtl_train_losses,
        'next_losses': next_losses,
        'category_losses': category_losses,
        'next_val_losses': next_val_losses,
        'category_val_losses': category_val_losses,
        'y_true_category': y_true_category,
        'y_pred_category': y_pred_category,
        'y_true_next': y_true_next,
        'y_pred_next': y_pred_next,
        'val_y_true_category': val_y_true_category,
        'val_y_pred_category': val_y_pred_category,
        'val_y_true_next': val_y_true_next,
        'val_y_pred_next': val_y_pred_next
    }

    return results


# Utility Functions
def print_classification_report_val(y_true, y_pred, task_name):
    report = classification_report(y_true, y_pred, zero_division=1, output_dict=True)

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(columns='support')
    metrics_df = metrics_df.drop(index=['weighted avg', 'accuracy'])

    metrics_df.rename(index=CATEGORIES_MAP, inplace=True)
    metrics_df = metrics_df.transpose()

    metrics_df = metrics_df.map(lambda x: f"{x * 100:.1f}")

    print(f'validação {task_name} metrics:')
    print(metrics_df.to_string() + '\n')


def print_classification_report_test(y_true, y_pred, id_to_name, task_name):
    report = classification_report(y_true, y_pred, zero_division=1, output_dict=True)

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop(columns='support')
    metrics_df = metrics_df.drop(index=['weighted avg', 'accuracy'])

    metrics_df.rename(index=id_to_name, inplace=True)
    metrics_df = metrics_df.transpose()

    metrics_df = metrics_df.map(lambda x: f"{x * 100:.1f}")

    print(f'{task_name} test metrics:')
    print(metrics_df.to_string() + '\n')


def print_classification_report_train(y_true_category, y_pred_category, y_true_next, y_pred_next):
    category_report = classification_report(y_true_category, y_pred_category, zero_division=1, output_dict=True)
    next_report = classification_report(y_true_next, y_pred_next, zero_division=1, output_dict=True)

    category_df = pd.DataFrame(category_report).transpose()
    next_df = pd.DataFrame(next_report).transpose()

    category_df = category_df.drop(columns='support')
    category_df = category_df.drop(index=['weighted avg', 'accuracy'])

    next_df = next_df.drop(columns='support')
    next_df = next_df.drop(index=['weighted avg', 'accuracy'])

    category_df.rename(index=CATEGORIES_MAP, inplace=True)
    next_df.rename(index=CATEGORIES_MAP, inplace=True)

    category_df = category_df.transpose()
    next_df = next_df.transpose()

    category_df = category_df.map(lambda x: f"{x * 100:.1f}")
    next_df = next_df.map(lambda x: f"{x * 100:.1f}")

    print("\ncategory train metrics:")
    print(category_df.to_string())

    print("\n next train metrics:")
    print(next_df.to_string())


def print_train_losses(epoch, num_epochs, mtl_loss, next_loss, category_losss):
    print(f'\nEPOCH {epoch + 1}/{num_epochs}:')
    print(f'mtl loss: {mtl_loss:.1f}')
    print(f'next loss: {next_loss:.1f}')
    print(f'category loss: {category_losss:.1f}')


def print_val_losses(next_loss, category_losss):
    print(f'\nVALIDATION LOSSES:')
    print(f'next val loss: {next_loss:.1f}')
    print(f'category val loss: {category_losss:.1f}\n')


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
    one_hot = torch.eye(num_classes).to(device)

    fold_results_test = {}
    fold_results_train = {}
    fold_losses_train = {}
    fold_results_val = {}
    fold_losses_val = {}
    fold_flops = {}

    for i_fold, dataloader in dataloaders.items():
        print("#" * 110)
        print(f'FOLD {i_fold}:')

        # Initialize model
        model = MTLPOI(
            input_dim=ModelParameters.INPUT_DIM,
            shared_layer_size=ModelParameters.SHARED_LAYER_SIZE,
            num_classes=num_classes,
            num_heads=ModelParameters.NUM_HEADS,
            num_layers=ModelParameters.NUM_LAYERS,
            seq_length=ModelParameters.SEQ_LENGTH,
            num_shared_layers=ModelParameters.NUM_SHARED_LAYERS
        )
        model = model.to(device)

        dataloader_next = dataloader['next']
        dataloader_category = dataloader['category']

        sample_category = next(iter(dataloader_category.train.dataloader))['x'].to(device)
        sample_next = next(iter(dataloader_next.train.dataloader))['x'].to(device)

        flops, macs, params = calculate_flops(model=model,
                                              kwargs={
                                                  'x1': sample_category,
                                                  'x2': sample_next
                                              },
                                              print_results=False)

        print(f"{flops} | {macs} | {params}")

        fold_flops[i_fold] = {
            'flops': flops,
            'macs': macs,
            'params': params,
        }

        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min')

        # TODO: CHECK IF `dataloader_next.train.dataloader.dataset.x` works
        # Compute class weights for balanced loss
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(dataloader_category.train.y.cpu().numpy()),
            y=dataloader_category.train.y.cpu().numpy()
        )

        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # Initialize loss functions
        loss_functions = {
            'npc': nn.CrossEntropyLoss(),
            'category': nn.CrossEntropyLoss(weight=class_weights),
            'reconstruction': nn.MSELoss()
        }

        # Initialize result dictionaries for this fold
        fold_results_test[i_fold] = {
            'test_next_true': [],
            'test_next_pred': [],
            'test_category_true': [],
            'test_category_pred': []
        }

        # Train model
        results = train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            loss_functions, num_epochs, num_classes)

        # Store training results
        fold_losses_train[i_fold]['mtl_train_losses'] = results['mtl_train_losses']
        fold_losses_train[i_fold]['next_losses'] = results['next_losses']
        fold_losses_train[i_fold]['category_losses'] = results['category_losses']

        fold_losses_val[i_fold]['next_val_losses'] = results['next_val_losses']
        fold_losses_val[i_fold]['category_val_losses'] = results['category_val_losses']

        fold_results_train[i_fold]['train_next_true'] = results['y_true_next']
        fold_results_train[i_fold]['train_next_pred'] = results['y_pred_next']
        fold_results_train[i_fold]['train_category_true'] = results['y_true_category']
        fold_results_train[i_fold]['train_category_pred'] = results['y_pred_category']

        fold_results_val[i_fold]['val_next_true'] = results['val_y_true_next']
        fold_results_val[i_fold]['val_next_pred'] = results['val_y_pred_next']
        fold_results_val[i_fold]['val_category_true'] = results['val_y_true_category']
        fold_results_val[i_fold]['val_category_pred'] = results['val_y_pred_category']

        # Test model on next POI task
        y_true_test_next, y_pred_test_next, _ = evaluate_model(
            model, dataloader_next.test.dataloader, loss_functions,
            'next', num_classes, mode='test'
        )
        fold_results_test[i_fold]['test_next_true'] = y_true_test_next
        fold_results_test[i_fold]['test_next_pred'] = y_pred_test_next

        # Test model on category task
        y_true_test_category, y_pred_test_category, _ = evaluate_model(
            model, dataloader_category.test.dataloader, loss_functions,
            'category', num_classes, mode='test'
        )
        fold_results_test[i_fold]['test_category_true'] = y_true_test_category
        fold_results_test[i_fold]['test_category_pred'] = y_pred_test_category

    return {
        'test_results': fold_results_test,
        'train_results': fold_results_train,
        'val_results': fold_results_val,
        'train_losses': fold_losses_train,
        'val_losses': fold_losses_val,
        'flops': fold_flops
    }
