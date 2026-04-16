import contextlib

import torch

from tracking.metrics import compute_classification_metrics
from utils.progress import zip_longest_cycle


def _ood_restricted_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    train_label_set: set[int],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Acc@K restricted to in-distribution samples (target in train set).

    Returns keys ``top{k}_acc_indist`` for each k, plus ``n_indist``
    (count of in-distribution samples) and ``n_ood`` (count of OOD).
    If all samples are OOD, returns 0.0 for every metric.
    """
    in_dist_mask = torch.tensor(
        [int(t.item()) in train_label_set for t in targets],
        dtype=torch.bool,
        device=targets.device,
    )
    n_indist = int(in_dist_mask.sum().item())
    n_ood = len(targets) - n_indist

    result: dict[str, float] = {
        "n_indist": float(n_indist),
        "n_ood": float(n_ood),
        "ood_fraction": float(n_ood / max(len(targets), 1)),
    }

    if n_indist == 0:
        for k in ks:
            result[f"top{k}_acc_indist"] = 0.0
        result["mrr_indist"] = 0.0
        return result

    logits_id = logits[in_dist_mask]
    targets_id = targets[in_dist_mask]

    for k in ks:
        k_eff = min(k, logits_id.shape[-1])
        topk = logits_id.topk(k_eff, dim=-1).indices
        hit = (topk == targets_id.unsqueeze(-1)).any(dim=-1)
        result[f"top{k}_acc_indist"] = float(hit.float().mean().item())

    # MRR on in-distribution subset
    from tracking.metrics import _rank_of_target
    rank = _rank_of_target(logits_id, targets_id).float()
    result["mrr_indist"] = float((1.0 / rank).mean().item())

    return result


@torch.no_grad()
def evaluate_model(
    model,
    dataloaders,
    next_criterion,
    category_criterion,
    mtl_creterion,
    device,
    num_classes: int | None = None,
    task_b_num_classes: int | None = None,
    task_a_num_classes: int | None = None,
    train_labels_b: set[int] | None = None,
    train_labels_a: set[int] | None = None,
):
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

    _autocast_ctx = (
        torch.autocast(device.type, dtype=torch.float16)
        if device.type == 'cuda'
        else contextlib.nullcontext()
    )

    for data_next, data_cat in zip_longest_cycle(*dataloaders):
        x_next, y_next = data_next
        if x_next.device != device:
            x_next = x_next.to(device, non_blocking=True)
            y_next = y_next.to(device, non_blocking=True)
        x_category, y_category = data_cat
        if x_category.device != device:
            x_category = x_category.to(device, non_blocking=True)
            y_category = y_category.to(device, non_blocking=True)

        with _autocast_ctx:
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

    if num_classes is None:
        num_classes = model.num_classes
    # Default: use the same num_classes for both (legacy 2-task path).
    # For non-legacy task_sets (e.g. check2HGI: cat=7, region~1109) the
    # caller passes per-task values to avoid torchmetrics OOM at
    # num_classes^2 on the large-cardinality head.
    nc_b = task_b_num_classes if task_b_num_classes is not None else num_classes
    nc_a = task_a_num_classes if task_a_num_classes is not None else num_classes

    metrics_next = compute_classification_metrics(
        all_logits_next, all_truths_next, num_classes=nc_b,
    )
    metrics_category = compute_classification_metrics(
        all_logits_category, all_truths_category, num_classes=nc_a,
    )
    metrics_next['loss'] = loss_next
    metrics_category['loss'] = loss_category

    # OOD-restricted Acc@K for CH06. When the caller provides the set of
    # labels seen in the current training fold, we filter val samples to
    # those whose target label is in-distribution and compute ranking
    # metrics on that subset. Keys are suffixed ``_indist``. This guards
    # against the artefact where a model scores well on train-overlap
    # POIs but 0 on OOD POIs, and the raw Acc@K averages across both.
    if train_labels_b is not None:
        ood_b = _ood_restricted_topk(all_logits_next, all_truths_next, train_labels_b)
        metrics_next.update(ood_b)
    if train_labels_a is not None:
        ood_a = _ood_restricted_topk(all_logits_category, all_truths_category, train_labels_a)
        metrics_category.update(ood_a)

    return metrics_next, metrics_category, loss_combined
