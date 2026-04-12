import torch

from tracking.metrics import compute_classification_metrics
from utils.progress import zip_longest_cycle


@torch.no_grad()
def evaluate_model(model, dataloaders, next_criterion, category_criterion, mtl_creterion, device):
    """Unified MTL validation pass — computes loss and the full metric dict per task.

    Uses ``zip_longest_cycle`` to match training coverage — the shorter loader
    is cycled so all samples from both loaders are evaluated. Logits are kept
    on-device through ``compute_classification_metrics``, which produces the
    legacy ``f1``/``accuracy`` keys **plus** ``f1_weighted``, ``accuracy_macro``,
    ``top{k}_acc``, ``mrr`` and ``ndcg_{k}``.

    Args:
        model: The MTLPOI model.
        dataloaders: List of ``[next_dataloader, category_dataloader]``.
        next_criterion: Loss function for the next task.
        category_criterion: Loss function for the category task.
        mtl_creterion: MTL loss (unused for validation loss — kept for API compat).
        device: Device to run evaluation on.

    Returns:
        Tuple ``(metrics_next, metrics_category, loss)`` where each
        ``metrics_*`` is the unprefixed dict returned by
        ``compute_classification_metrics``.
    """
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    logits_next_list, truths_next_list = [], []
    logits_cat_list, truths_cat_list = [], []
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

        cat_out, next_out = model((x_category, x_next))

        next_loss = next_criterion(next_out, y_next)
        category_loss = category_criterion(cat_out, y_category)
        running_loss += (next_loss.detach() + category_loss.detach()) / 2

        logits_next_list.append(next_out.detach())
        truths_next_list.append(y_next)
        logits_cat_list.append(cat_out.detach())
        truths_cat_list.append(y_category)
        batchs += 1

    loss = running_loss.item() / batchs
    all_logits_next = torch.cat(logits_next_list)
    all_truths_next = torch.cat(truths_next_list)
    all_logits_category = torch.cat(logits_cat_list)
    all_truths_category = torch.cat(truths_cat_list)

    num_classes = model.num_classes

    metrics_next = compute_classification_metrics(
        all_logits_next, all_truths_next, num_classes=num_classes,
    )
    metrics_category = compute_classification_metrics(
        all_logits_category, all_truths_category, num_classes=num_classes,
    )

    return metrics_next, metrics_category, loss
