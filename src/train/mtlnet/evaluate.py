import torch
from sklearn.metrics import classification_report


@torch.no_grad()
def evaluate_model(model, dataloaders, next_criterion, category_criterion, mtl_creterion, device):
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
    running_loss = torch.tensor(0.0, device=device)
    preds_next_list, truths_next_list = [], []
    preds_cat_list, truths_cat_list = [], []
    batchs = 0

    for data_next, data_cat in zip(*dataloaders):
        x_next, y_next = data_next
        x_next, y_next = x_next.to(device, non_blocking=True), y_next.to(device, non_blocking=True)
        x_category, y_category = data_cat
        x_category, y_category = x_category.to(device, non_blocking=True), y_category.to(device, non_blocking=True)

        # Forward pass
        cat_out, next_out = model((x_category, x_next))
        pred_next, truth_next = next_out, y_next
        pred_category, truth_category = cat_out, y_category

        next_loss = next_criterion(pred_next, truth_next)
        category_loss = category_criterion(pred_category, truth_category)
        running_loss += (next_loss.detach() + category_loss.detach()) / 2

        # Collect predictions on-device
        preds_next_list.append(torch.argmax(pred_next, dim=1))
        truths_next_list.append(truth_next)
        preds_cat_list.append(torch.argmax(pred_category, dim=1))
        truths_cat_list.append(truth_category)
        batchs += 1

    # Single bulk GPU→CPU transfer
    loss = running_loss.item() / batchs
    all_predictions_next = torch.cat(preds_next_list).cpu().numpy()
    all_truths_next = torch.cat(truths_next_list).cpu().numpy()
    all_predictions_category = torch.cat(preds_cat_list).cpu().numpy()
    all_truths_category = torch.cat(truths_cat_list).cpu().numpy()

    next_report = classification_report(all_truths_next, all_predictions_next, output_dict=True, zero_division=0)
    category_report = classification_report(all_truths_category, all_predictions_category, output_dict=True, zero_division=0)

    f1_next = next_report['macro avg']['f1-score']
    acc_next = next_report['accuracy']

    f1_category = category_report['macro avg']['f1-score']
    acc_category = category_report['accuracy']

    return acc_next, f1_next, acc_category, f1_category, loss


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
    running_loss = torch.tensor(0.0, device=device)
    running_correct = torch.tensor(0, device=device, dtype=torch.long)
    total_samples = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Forward pass
            out = foward_method(x)
            pred, truth = out, y

            # Accumulate on-device
            running_loss += loss_function(pred, truth).detach()
            running_correct += (torch.argmax(pred, dim=1) == truth).sum()
            total_samples += truth.size(0)

    # Single MPS sync
    return running_correct.item() / total_samples, running_loss.item() / total_samples
