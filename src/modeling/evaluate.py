import torch
from sklearn.metrics import classification_report


def evaluate_model(model, dataloaders, task_loss_fn, mtl_loss_fn, device, num_classes=None):
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
    all_predictions_next = []
    all_truths_next = []
    all_predictions_category = []
    all_truths_category = []
    batchs = 0

    with torch.no_grad():
        for data_next, data_cat in zip(*dataloaders):
            x_next, y_next = data_next['x'].to(device), data_next['y'].to(device)
            x_category, y_category = data_cat['x'].to(device), data_cat['y'].to(device)

            # Forward pass
            cat_out, next_out = model((x_category, x_next))
            pred_next, truth_next = next_out, y_next
            pred_category, truth_category = cat_out, y_category

            total_loss += mtl_loss_fn.compute_loss_no_adjustment(
                task_loss_fn(pred_next, truth_next).item(),
                task_loss_fn(pred_category, truth_category).item()
            )

            # Calculate accuracy
            pred_next_class = torch.argmax(pred_next, dim=1)
            pred_category_class = torch.argmax(pred_category, dim=1)
            all_predictions_next.extend(pred_next_class.cpu().numpy())
            all_truths_next.extend(truth_next.cpu().numpy())
            all_predictions_category.extend(pred_category_class.cpu().numpy())
            all_truths_category.extend(truth_category.cpu().numpy())
            batchs += 1

    loss_next = total_loss / batchs

    next_report = classification_report(all_truths_next, all_predictions_next, output_dict=True, zero_division=0)
    category_report = classification_report(all_truths_category, all_predictions_category, output_dict=True,zero_division=0)
    acc_next = next_report['accuracy']
    acc_category = category_report['accuracy']
    f1_next = next_report['macro avg']['f1-score']
    f1_category = category_report['macro avg']['f1-score']

    return acc_next, f1_next, acc_category, f1_category, loss_next


def evaluate_model_by_head(model, dataloader, loss_function, foward_method, device, num_classes=None):
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
            pred, truth = out, y

            # Calculate loss
            loss = loss_function(pred, truth).item()
            total_loss += loss

            # Calculate accuracy
            pred_class = torch.argmax(pred, dim=1)
            total_correct += (pred_class == truth).sum().item()
            total_samples += truth.size(0)

    # Return average accuracy and loss
    return total_correct / total_samples, total_loss / total_samples
