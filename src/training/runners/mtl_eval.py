import torch
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy

from utils.progress import zip_longest_cycle


@torch.no_grad()
def evaluate_model(model, dataloaders, next_criterion, category_criterion, mtl_creterion, device):
    """
    Unified evaluation function for both validation and testing.

    Uses zip_longest_cycle() to match training coverage — the shorter loader
    is cycled so all samples from both loaders are evaluated.

    Args:
        model: The MTLPOI model
        dataloaders: List of [next_dataloader, category_dataloader]
        next_criterion: Loss function for next task
        category_criterion: Loss function for category task
        mtl_creterion: MTL loss (unused for validation loss, kept for API compat)
        device: Device to run evaluation on

    Returns:
        Tuple of (acc_next, f1_next, acc_category, f1_category, loss)
    """
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    preds_next_list, truths_next_list = [], []
    preds_cat_list, truths_cat_list = [], []
    batchs = 0

    for data_next, data_cat in zip_longest_cycle(*dataloaders):
        x_next, y_next = data_next
        if x_next.device != device:
            x_next = x_next.to(device, non_blocking=True)
            y_next = y_next.to(device, non_blocking=True)
        x_category, y_category = data_cat
        if x_category.device != device:
            x_category = x_category.to(device, non_blocking=True)
            y_category = y_category.to(device, non_blocking=True)

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

    # Stay on device — torchmetrics functional API computes metrics
    # without bulk-transferring all predictions to CPU.
    loss = running_loss.item() / batchs
    all_predictions_next = torch.cat(preds_next_list)
    all_truths_next = torch.cat(truths_next_list)
    all_predictions_category = torch.cat(preds_cat_list)
    all_truths_category = torch.cat(truths_cat_list)

    num_classes = model.num_classes

    f1_next = multiclass_f1_score(
        all_predictions_next, all_truths_next,
        num_classes=num_classes, average='macro', zero_division=0,
    ).item()
    acc_next = multiclass_accuracy(
        all_predictions_next, all_truths_next,
        num_classes=num_classes, average='micro',
    ).item()

    f1_category = multiclass_f1_score(
        all_predictions_category, all_truths_category,
        num_classes=num_classes, average='macro', zero_division=0,
    ).item()
    acc_category = multiclass_accuracy(
        all_predictions_category, all_truths_category,
        num_classes=num_classes, average='micro',
    ).item()

    return acc_next, f1_next, acc_category, f1_category, loss
