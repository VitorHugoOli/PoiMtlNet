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
    ``top{k}_acc``, ``mrr`` and ``ndcg_{k}``. Each per-task metric dict also
    carries its own ``'loss'`` key so the caller can forward the whole dict
    via ``**`` straight into ``log_val`` without hand-wiring scalars.

    Args:
        model: The MTLPOI model.
        dataloaders: List of ``[next_dataloader, category_dataloader]``.
        next_criterion: Loss function for the next task.
        category_criterion: Loss function for the category task.
        mtl_creterion: MTL loss (unused for validation loss — kept for API compat).
        device: Device to run evaluation on.

    Returns:
        Tuple ``(metrics_next, metrics_category, loss_combined)``:
            * ``metrics_next`` / ``metrics_category`` — dicts from
              ``compute_classification_metrics`` with an added
              per-task ``'loss'`` key.
            * ``loss_combined`` — scalar average of
              ``(next_loss + category_loss) / 2`` across batches, kept
              for the MTL-level ``model_task`` tracking store.
    """
    model.eval()
    combined_running_loss = torch.tensor(0.0, device=device)
    next_running_loss = torch.tensor(0.0, device=device)
    category_running_loss = torch.tensor(0.0, device=device)
    logits_next_list, truths_next_list = [], []
    logits_cat_list, truths_cat_list = [], []
    batches = 0

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
        # Accumulate on-device; single .item() sync after the loop.
        next_running_loss += next_loss.detach()
        category_running_loss += category_loss.detach()
        combined_running_loss += (next_loss.detach() + category_loss.detach()) / 2

        logits_next_list.append(next_out.detach())
        truths_next_list.append(y_next)
        logits_cat_list.append(cat_out.detach())
        truths_cat_list.append(y_category)
        batches += 1

    if batches == 0:
        empty = compute_classification_metrics(
            torch.zeros(0, model.num_classes, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            num_classes=model.num_classes,
        )
        empty['loss'] = 0.0
        return empty, dict(empty), 0.0

    loss_combined = combined_running_loss.item() / batches
    loss_next = next_running_loss.item() / batches
    loss_category = category_running_loss.item() / batches

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
    metrics_next['loss'] = loss_next
    metrics_category['loss'] = loss_category

    return metrics_next, metrics_category, loss_combined
