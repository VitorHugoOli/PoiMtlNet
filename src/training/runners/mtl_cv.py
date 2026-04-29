import contextlib
import logging
import math
import numpy as np
import torch
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from torch.nn import CrossEntropyLoss

from tracking.metrics import compute_classification_metrics
from utils.flops import calculate_model_flops
from utils.mps import clear_mps_cache
from utils.progress import TrainingProgressBar
from configs.globals import DEVICE
from configs.experiment import ExperimentConfig
from losses.registry import create_loss
from models.registry import create_model
from training.helpers import (
    compute_class_weights,
    setup_optimizer,
    setup_per_head_optimizer,
    setup_scheduler,
)
from training.callbacks import CallbackContext, CallbackList
from data.folds import TaskFoldData, FoldResult
from tracking import MLHistory, FlopsMetrics, NeuralParams
from tracking.fold import FoldHistory, TaskHistory
from tasks import LEGACY_CATEGORY_NEXT, TaskSet
from training.runners.mtl_eval import evaluate_model
from training.runners.mtl_validation import validation_best_model


def _flatten_task_grads(
        grads: tuple[Optional[torch.Tensor], ...],
        parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    flat = []
    for grad, param in zip(grads, parameters):
        if grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
        else:
            flat.append(grad.reshape(-1))
    if not flat:
        return torch.empty(0, device=DEVICE)
    return torch.cat(flat)


def _compute_gradient_cosine(
        losses: torch.Tensor,
        shared_parameters: list[torch.nn.Parameter],
) -> tuple[float, float, float]:
    """Cosine + norms of shared-parameter gradients for the 2-task pair.

    Returns ``(cosine, task_b_norm, task_a_norm)``. Losses ordering is
    fixed: ``losses[0]`` = task B (slot "next" in legacy, "next_region"
    under CHECK2HGI_NEXT_REGION); ``losses[1]`` = task A (slot
    "category" in legacy, "next_category" under the new preset).
    """
    if not shared_parameters:
        return float("nan"), 0.0, 0.0

    task_b_grads = torch.autograd.grad(
        losses[0],
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    task_a_grads = torch.autograd.grad(
        losses[1],
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    task_b_flat = _flatten_task_grads(task_b_grads, shared_parameters)
    task_a_flat = _flatten_task_grads(task_a_grads, shared_parameters)

    task_b_norm = torch.norm(task_b_flat)
    task_a_norm = torch.norm(task_a_flat)
    denom = task_b_norm * task_a_norm
    if denom <= 0:
        return float("nan"), task_b_norm.item(), task_a_norm.item()

    cosine = torch.dot(task_b_flat, task_a_flat) / denom
    return cosine.item(), task_b_norm.item(), task_a_norm.item()


def _get_weighted_loss(
        mtl_criterion,
        losses: torch.Tensor,
        shared_parameters: list[torch.nn.Parameter],
        task_specific_parameters: list[torch.nn.Parameter],
        **context,
) -> tuple[torch.Tensor | None, dict, bool]:
    if not hasattr(mtl_criterion, "get_weighted_loss"):
        loss, extra_outputs = mtl_criterion.backward(
            losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **context,
        )
        # Gradient-surgery losses (CAGrad, Aligned-MTL) do the backward pass
        # internally and return loss=None. Fall back to the raw loss sum for
        # reporting (detached so it can't trigger another backward).
        if loss is None:
            loss = losses.sum().detach()
        return loss, extra_outputs, True
    try:
        loss, extra_outputs = mtl_criterion.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **context,
        )
        return loss, extra_outputs, False
    except NotImplementedError:
        loss, extra_outputs = mtl_criterion.backward(
            losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **context,
        )
        if loss is None:
            loss = losses.sum().detach()
        return loss, extra_outputs, True


def _class_majority_fraction(y: torch.Tensor) -> float:
    """Fraction of samples belonging to the most frequent class.

    Used to normalise per-head Acc@1 onto a "lift over majority" scale so
    the joint_lift monitor treats 7-class and 10^3-class heads
    commensurately. Returns 0.0 for empty tensors (caller clamps to 1e-6
    before division).
    """
    if y.numel() == 0:
        return 0.0
    # torch.bincount is faster than torch.unique(return_counts) for
    # contiguous int64 labels and works identically here.
    y_flat = y.detach().to(torch.int64).view(-1).cpu()
    counts = torch.bincount(y_flat)
    return float(counts.max().item() / y_flat.numel())


def _pareto_front_indices(points: list[tuple[float, float]]) -> list[int]:
    """Pareto-front indices over per-head val scores.

    Each ``points[i]`` is ``(task_b_score, task_a_score)`` at epoch i;
    the legacy slot mapping is (next_f1, category_f1).
    """
    front = []
    for i, (task_b_i, task_a_i) in enumerate(points):
        dominated = False
        for j, (task_b_j, task_a_j) in enumerate(points):
            if i == j:
                continue
            if (
                task_b_j >= task_b_i
                and task_a_j >= task_a_i
                and (task_b_j > task_b_i or task_a_j > task_a_i)
            ):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _criterion_parameters(mtl_criterion) -> list[torch.nn.Parameter]:
    parameters = getattr(mtl_criterion, "parameters", None)
    if parameters is None:
        return []
    return list(parameters())


def _fmt_metric(value: float) -> str:
    """Compact numeric formatter for progress-bar metrics."""
    if not math.isfinite(value):
        return "-"
    return f"{value * 100:.2f}"


# Training Function
def train_model(model: torch.nn.Module,
                optimizer,
                scheduler,
                dataloader_next: TaskFoldData,
                dataloader_category: TaskFoldData,
                next_criterion,
                category_criterion,
                mtl_criterion,
                num_epochs,
                num_classes,
                shared_parameters: Optional[list] = None,
                task_specific_parameters: Optional[list] = None,
                fold_history: Optional[FoldHistory] = None,
                max_grad_norm: float = 1.0,
                gradient_accumulation_steps: int = 1,
                timeout: Optional[int] = None,
                next_target_cutoff: Optional[float] = None,
                category_target_cutoff: Optional[float] = None,
                callbacks: Optional[list] = None,
                task_set: TaskSet = LEGACY_CATEGORY_NEXT,
                freeze_cat_after_epoch: Optional[int] = None,
                alternating_optimizer_step: bool = False,
                cat_specific_parameters: Optional[list] = None,
                reg_specific_parameters: Optional[list] = None,
                ):
    """
    Train the model with multi-task learning.

    ``task_set`` names the two task slots; defaults to the legacy
    ``{category, next}`` pair which keeps every metric key and
    fold_history task name bit-exact with the pre-parameterisation
    runner. Non-legacy task sets (e.g. ``CHECK2HGI_NEXT_REGION``)
    re-label the slots in every emitted metric key.

    Internal variable naming uses ``task_a_*`` / ``task_b_*`` as of
    the rename in commit (see CRITICAL_REVIEW.md §3 item 1 — the
    pre-rename ``next_*`` / ``category_*`` names lied on non-legacy
    task_sets because slot "NEXT" holds next_region labels under the
    check2HGI preset). The mapping is: slot A ↔ ``task_a_*`` ↔
    ``category_*`` in public API (``dataloader_category``,
    ``category_criterion``); slot B ↔ ``task_b_*`` ↔ ``next_*`` in
    public API (``dataloader_next``, ``next_criterion``). Public
    parameter names kept for backward compatibility with callers.

    The loss tensor ordering is unchanged: ``losses[0] = task_b``,
    ``losses[1] = task_a``.
    """
    task_a_name = task_set.task_a.name  # slot A (category-slot)
    task_b_name = task_set.task_b.name  # slot B (next-slot)
    # Per-task num_classes. Legacy preset has both==num_classes==7 so the
    # metric output is unchanged. Non-legacy check2HGI has task_b=~1109
    # (region), task_a=7 (category); passing them separately avoids the
    # MPS bincount OOM in torchmetrics when num_classes**2 blows up.
    task_a_num_classes = task_set.task_a.num_classes or num_classes
    task_b_num_classes = task_set.task_b.num_classes or num_classes

    start_time = time.time()

    # FoldHistory default is built here (not as a mutable default arg) so
    # the task-name set can depend on ``task_set`` at call time.
    if fold_history is None:
        fold_history = FoldHistory.standalone({task_a_name, task_b_name})

    # Cache parameter group lists once. The shared/task partition is fixed
    # for the model's lifetime, so generating these per batch (the previous
    # behaviour) just walks named_parameters() repeatedly to no benefit.
    if shared_parameters is None:
        shared_parameters = list(model.shared_parameters())
    if task_specific_parameters is None:
        task_specific_parameters = list(model.task_specific_parameters())

    # Create progress bar that extends tqdm
    progress = TrainingProgressBar(
        num_epochs,
        [dataloader_next.train.dataloader,
         dataloader_category.train.dataloader],
    )

    cb = CallbackList(callbacks)

    # Inject model reference for callbacks that need it (e.g. ModelCheckpoint)
    for c in cb.callbacks:
        if hasattr(c, 'set_model'):
            c.set_model(model)

    cb.on_train_begin(CallbackContext(epoch=0, epochs_total=num_epochs))

    # Mixed-precision autocast: float16 forward passes on CUDA, no-op otherwise.
    # MPS float16 autocast adds overhead for small tensors — disabled there.
    _autocast_ctx = (
        torch.autocast(DEVICE.type, dtype=torch.float16)
        if DEVICE.type == 'cuda'
        else contextlib.nullcontext()
    )

    cutoff_hits = {
        task_b_name: False,
        task_a_name: False,
    }

    # Initialize model-level tracking
    if fold_history.model_task is None:
        fold_history.model_task = TaskHistory()

    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    batches_per_epoch = max(
        len(dataloader_next.train.dataloader),
        len(dataloader_category.train.dataloader),
    )
    pareto_points: list[tuple[float, float]] = []

    # F50 P3 — track whether warmup-then-freeze has fired (idempotent).
    _cat_frozen_post_warmup = False

    # Main training loop
    for epoch_idx in progress:
        model.train()
        # F50 P3 — at the boundary epoch, freeze category_encoder + category_poi.
        # Reg + shared keep training. Tests whether continued cat-encoder
        # co-adaptation as reg-helper (F49 Layer 2) hurts reg at scale. The
        # optimizer naturally skips params with grad=None, so no optimizer
        # rebuild is needed. category_encoder.eval() also disables its dropout.
        if (freeze_cat_after_epoch is not None
                and not _cat_frozen_post_warmup
                and epoch_idx >= int(freeze_cat_after_epoch)):
            for p in model.category_encoder.parameters():
                p.requires_grad_(False)
            for p in model.category_poi.parameters():
                p.requires_grad_(False)
            model.category_encoder.eval()
            _cat_frozen_post_warmup = True
            print(
                f"[P3 freeze-cat-after-epoch] cat_encoder + category_poi frozen "
                f"at epoch {epoch_idx} (target N={freeze_cat_after_epoch})"
            )

        # F40 — scheduled-loss epoch hook. Losses without `set_epoch`
        # (i.e. all current ones except `scheduled_static`) are silently
        # skipped via getattr default.
        _set_epoch = getattr(mtl_criterion, "set_epoch", None)
        if callable(_set_epoch):
            _set_epoch(epoch_idx)

        # Initialize on-device accumulators to avoid per-batch MPS syncs
        running_loss = torch.tensor(0.0, device=DEVICE)
        task_b_running_loss = torch.tensor(0.0, device=DEVICE)
        task_a_running_loss = torch.tensor(0.0, device=DEVICE)
        steps = 0

        # Collect logits on-device so compute_classification_metrics() can
        # produce the full metric dict (Macro/Weighted F1, Top-K, MRR, NDCG)
        # in a single per-epoch call.
        all_task_b_logits, all_task_b_targets = [], []
        all_task_a_logits, all_task_a_targets = [], []

        # Per-epoch diagnostics — recomputed once per epoch on batch 0
        # (see Phase 0 §60 of plan/MTL_IMPROVEMENT_PLAN.md).
        epoch_grad_cosine: float = float("nan")
        epoch_task_b_grad_norm: float = 0.0
        epoch_task_a_grad_norm: float = 0.0
        epoch_loss_weights: Optional[torch.Tensor] = None
        accumulated_in_group: int = 0

        # Iterate over batches with automatic progress tracking.
        # zero_grad is called at the end of every optimizer step, so the
        # loop starts each epoch with clean gradients (the last batch of
        # the previous epoch is always forced to step via the
        # `(batch_idx + 1) == batches_per_epoch` branch below).
        for batch_idx, (data_task_b, data_task_a) in enumerate(progress.iter_epoch()):
            # When the dataset is pre-moved to DEVICE (item #3, MPS path with
            # num_workers=0), the .to() calls are no-ops. Keep the guards so
            # the loop still works under a CPU-side dataloader path.
            x_task_b, y_task_b = data_task_b
            if x_task_b.device != DEVICE:
                x_task_b = x_task_b.to(DEVICE, non_blocking=True)
                y_task_b = y_task_b.to(DEVICE, non_blocking=True)
            x_task_a, y_task_a = data_task_a
            if x_task_a.device != DEVICE:
                x_task_a = x_task_a.to(DEVICE, non_blocking=True)
                y_task_a = y_task_a.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx:
                task_a_output, task_b_output = model((x_task_a, x_task_b))

                pred_task_b, truth_task_b = task_b_output, y_task_b
                pred_task_a, truth_task_a = task_a_output, y_task_a

                # Calculate losses (inside autocast so CE uses float16 logits)
                task_b_loss = next_criterion(pred_task_b, truth_task_b)
                task_a_loss = category_criterion(pred_task_a, truth_task_a)

            # NashMTL backward stays outside autocast — gradients in float32
            losses = torch.stack([task_b_loss, task_a_loss])

            # Shared-gradient cosine once per epoch. torch.autograd.grad does
            # not populate .grad, so it leaves the subsequent backward path
            # untouched — but it requires retain_graph=True, which the helper
            # already sets.
            if batch_idx == 0 and shared_parameters:
                (
                    epoch_grad_cosine,
                    epoch_task_b_grad_norm,
                    epoch_task_a_grad_norm,
                ) = _compute_gradient_cosine(losses, shared_parameters)

            # F50 P4 — per-batch alternating-SGD. Even batches use L_cat only,
            # odd batches use L_reg only. Shared params receive one task's
            # gradient signal per batch (alternating). Inactive task's
            # task-specific params will have their grads zeroed before
            # optimizer.step() (see ``should_step`` block below).
            _alt_inactive_params = None
            if alternating_optimizer_step:
                if batch_idx % 2 == 0:
                    loss = task_a_loss  # cat-only batch
                    _alt_inactive_params = reg_specific_parameters
                else:
                    loss = task_b_loss  # reg-only batch
                    _alt_inactive_params = cat_specific_parameters
                extra_outputs: dict = {}
                already_backpropagated = False
            else:
                loss, extra_outputs, already_backpropagated = _get_weighted_loss(
                    mtl_criterion,
                    losses,
                    shared_parameters=shared_parameters,
                    task_specific_parameters=task_specific_parameters,
                    epoch=epoch_idx,
                )
            if already_backpropagated and gradient_accumulation_steps > 1:
                raise TypeError(
                    f"{mtl_criterion.__class__.__name__} is not compatible with "
                    "gradient accumulation; use gradient_accumulation_steps=1 "
                    "or a loss with get_weighted_loss()."
                )
            if extra_outputs and "weights" in extra_outputs:
                epoch_loss_weights = extra_outputs["weights"]
            if not already_backpropagated:
                # Scale by the accumulation group size so the effective
                # gradient magnitude matches a full-size batch. Partial
                # trailing groups are rescaled at step time below.
                (loss / gradient_accumulation_steps).backward()
            accumulated_in_group += 1

            should_step = (
                ((batch_idx + 1) % gradient_accumulation_steps) == 0
                or (batch_idx + 1) == batches_per_epoch
            )
            if should_step:
                # Compensate for partial trailing groups: if we accumulated
                # fewer batches than the nominal group size, re-scale grads
                # by gradient_accumulation_steps / accumulated_in_group so
                # the update matches an averaged mini-batch. No-op for full
                # groups and loss variants that already backpropagated.
                if (
                    not already_backpropagated
                    and accumulated_in_group != gradient_accumulation_steps
                    and accumulated_in_group > 0
                ):
                    scale = gradient_accumulation_steps / accumulated_in_group
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                # F50 P4 — zero gradients of the inactive task's task-specific
                # params so the optimizer step only updates {active task,
                # shared}. Shared params keep their gradient from the active
                # loss; inactive task params do nothing this batch.
                if _alt_inactive_params is not None:
                    for p in _alt_inactive_params:
                        if p.grad is not None:
                            p.grad.zero_()
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + _criterion_parameters(mtl_criterion),
                        max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_in_group = 0

            # Accumulate on-device — no .item() sync per batch
            running_loss += loss.detach()
            task_b_running_loss += task_b_loss.detach()
            task_a_running_loss += task_a_loss.detach()

            # Collect logits on-device for epoch-level metrics. We keep the
            # full logit tensor (not argmax) so ranking metrics are free.
            with torch.no_grad():
                all_task_b_logits.append(pred_task_b.detach())
                all_task_b_targets.append(truth_task_b)
                all_task_a_logits.append(pred_task_a.detach())
                all_task_a_targets.append(truth_task_a)

            steps += 1

        epoch_task_b_logits = torch.cat(all_task_b_logits)
        epoch_task_b_targets = torch.cat(all_task_b_targets)
        epoch_task_a_logits = torch.cat(all_task_a_logits)
        epoch_task_a_targets = torch.cat(all_task_a_targets)

        train_metrics_task_b = compute_classification_metrics(
            epoch_task_b_logits, epoch_task_b_targets, num_classes=task_b_num_classes,
        )
        train_metrics_task_a = compute_classification_metrics(
            epoch_task_a_logits, epoch_task_a_targets, num_classes=task_a_num_classes,
        )
        f1_task_b = train_metrics_task_b['f1']
        f1_task_a = train_metrics_task_a['f1']

        # Calculate epoch metrics (single sync for losses)
        epoch_loss = running_loss.item() / steps
        epoch_loss_task_b = task_b_running_loss.item() / steps
        epoch_loss_task_a = task_a_running_loss.item() / steps
        loss_ratio_task_b_to_task_a = epoch_loss_task_b / max(epoch_loss_task_a, 1e-8)

        best_task_b = fold_history.task(task_b_name).best.best_value
        best_task_a = fold_history.task(task_a_name).best.best_value
        progress.set_postfix({
            'tr': f'N{_fmt_metric(f1_task_b)}|C{_fmt_metric(f1_task_a)}',
            'val': '-',
            'best': f'N{_fmt_metric(best_task_b)}|C{_fmt_metric(best_task_a)}',
        })

        fold_history.model_task.log_train(loss=epoch_loss, accuracy=0)
        fold_history.log_train(
            task_b_name, loss=epoch_loss_task_b, **train_metrics_task_b,
        )
        fold_history.log_train(
            task_a_name, loss=epoch_loss_task_a, **train_metrics_task_a,
        )
        diagnostic_payload = {
            "grad_cosine_shared": epoch_grad_cosine,
            f"grad_norm_{task_b_name}_shared": epoch_task_b_grad_norm,
            f"grad_norm_{task_a_name}_shared": epoch_task_a_grad_norm,
            f"loss_ratio_{task_b_name}_to_{task_a_name}": loss_ratio_task_b_to_task_a,
        }
        if epoch_loss_weights is not None:
            weights_cpu = epoch_loss_weights.detach().cpu()
            if len(weights_cpu) >= 2:
                diagnostic_payload[f"loss_weight_{task_b_name}"] = float(weights_cpu[0])
                diagnostic_payload[f"loss_weight_{task_a_name}"] = float(weights_cpu[1])
        gate_stats = getattr(model, "last_gate_stats", {})
        if gate_stats:
            if "category_entropy" in gate_stats:
                diagnostic_payload["gate_entropy_category"] = float(
                    gate_stats["category_entropy"].detach().cpu()
                )
            if "next_entropy" in gate_stats:
                diagnostic_payload["gate_entropy_next"] = float(
                    gate_stats["next_entropy"].detach().cpu()
                )
        fold_history.log_diagnostic(**diagnostic_payload)

        # Validation phase with progress tracking
        with progress.validation():
            # Build train-label sets for OOD-restricted Acc@K (CH06).
            # Only populated when task_set is non-legacy (high-cardinality
            # heads where OOD filtering matters). Legacy 7-class heads
            # always have all classes in every fold → OOD is empty → skip.
            _tl_b: set[int] | None = None
            _tl_a: set[int] | None = None
            if task_b_num_classes is not None and task_b_num_classes > 256:
                _tl_b = set(dataloader_next.train.y.unique().tolist())
            if task_a_num_classes is not None and task_a_num_classes > 256:
                _tl_a = set(dataloader_category.train.y.unique().tolist())

            val_metrics_task_b, val_metrics_task_a, loss_val = evaluate_model(
                model,
                [dataloader_next.val.dataloader, dataloader_category.val.dataloader],
                next_criterion,
                category_criterion,
                mtl_criterion,
                DEVICE,
                num_classes=num_classes,
                task_b_num_classes=task_b_num_classes,
                task_a_num_classes=task_a_num_classes,
                train_labels_b=_tl_b,
                train_labels_a=_tl_a,
            )

            f1_val_task_b = val_metrics_task_b['f1']
            f1_val_task_a = val_metrics_task_a['f1']
            # Acc@1 per head (already computed by compute_classification_metrics
            # under the ``accuracy`` key). Used for the joint_lift monitor
            # recommended by docs/plans/CHECK2HGI_MTL_OVERVIEW.md §2 for the
            # check2HGI track, where next_region has ~10^3 classes and macro-F1
            # is a weak summary statistic.
            acc1_val_task_b = val_metrics_task_b.get('accuracy', 0.0)
            acc1_val_task_a = val_metrics_task_a.get('accuracy', 0.0)

            # Legacy joint score — mean of per-head F1. Scale-coherent when
            # both heads are 7-class (legacy). Kept for back-compat with the
            # {category, next} preset's existing callback monitor.
            joint_score = 0.5 * (f1_val_task_b + f1_val_task_a)
            # Naive Acc@1 mean. Reported (so CH06 can empirically compare
            # checkpoint choices) but NOT used as a default monitor on the
            # check2HGI track — ~50% category Acc@1 vs ~5% region Acc@1
            # makes the mean dominated by the easier head, biasing every
            # selected checkpoint toward category performance. See
            # CRITICAL_REVIEW.md §1 item (joint_acc1 scale-incoherence).
            joint_acc1 = 0.5 * (acc1_val_task_b + acc1_val_task_a)
            # Scale-coherent joint: each head contributes its *lift over
            # majority-class baseline*. The 2026-04-15 review-agent finding
            # showed arithmetic mean is STILL dominated by the head with the
            # smaller majority fraction when the two differ by orders of
            # magnitude (e.g. FL next_poi majority ~0.001% vs next_region
            # 22.5% → POI lift can be 1000×, region lift ~1× → arithmetic
            # mean ~500 is dominated by POI). Geometric mean forces both
            # heads to contribute multiplicatively, penalising either head
            # collapsing to majority-class behaviour.
            #
            # Compute per-head majority fractions once and cache across epochs
            # (train labels don't change mid-fold). The attribute is stored on
            # fold_history so subsequent epochs reuse the cached value.
            if not hasattr(fold_history, "_joint_lift_majority"):
                fold_history._joint_lift_majority = (
                    max(_class_majority_fraction(dataloader_next.train.y), 1e-6),
                    max(_class_majority_fraction(dataloader_category.train.y), 1e-6),
                )
            task_b_majority, task_a_majority = fold_history._joint_lift_majority
            task_b_lift = max(acc1_val_task_b / task_b_majority, 1e-8)
            task_a_lift = max(acc1_val_task_a / task_a_majority, 1e-8)
            # Geometric mean of per-head lifts (scale-coherent; the checkpoint
            # monitor for the check2HGI track).
            joint_geom_lift = math.sqrt(task_b_lift * task_a_lift)
            # Arithmetic mean kept for back-compat + side-by-side reporting
            # (paper can show it as the "naive" number in appendix).
            joint_arith_lift = 0.5 * (task_b_lift + task_a_lift)
            pareto_points.append((f1_val_task_b, f1_val_task_a))
            pareto_front = _pareto_front_indices(pareto_points)
            fold_history.add_artifact(
                "pareto_front",
                [
                    {
                        "epoch": idx,
                        f"{task_b_name}_f1": pareto_points[idx][0],
                        f"{task_a_name}_f1": pareto_points[idx][1],
                    }
                    for idx in pareto_front
                ],
            )

            # Only create state_dict when at least one task improves.
            task_b_improved = f1_val_task_b > fold_history.task(task_b_name).best.best_value
            task_a_improved = f1_val_task_a > fold_history.task(task_a_name).best.best_value
            prev_joint_best = fold_history.model_task.best.best_value if fold_history.model_task.best.best_epoch >= 0 else -1.0
            joint_improved = joint_score > prev_joint_best
            state = model.state_dict() if (joint_improved or task_b_improved or task_a_improved) else None

            # Per-task val losses now come from evaluate_model() inside the
            # metric dicts, so log_val no longer needs a hand-wired scalar.
            # model_task keeps the combined MTL loss; the f1=0/accuracy=0
            # placeholders stay as stable schema on the MTL summary store.
            fold_history.model_task.log_val(
                loss=loss_val,
                f1=joint_score,
                accuracy=0,
                model_state=state if joint_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )
            fold_history.log_val(
                task_b_name,
                **val_metrics_task_b,
                model_state=state if task_b_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )
            fold_history.log_val(
                task_a_name,
                **val_metrics_task_a,
                model_state=state if task_a_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )

        # Update compact F1-only metrics on progress bar.
        best_task_b = fold_history.task(task_b_name).best.best_value
        best_task_a = fold_history.task(task_a_name).best.best_value
        progress.set_postfix({
            'tr': f'N{_fmt_metric(f1_task_b)}|C{_fmt_metric(f1_task_a)}',
            'val': f'N{_fmt_metric(f1_val_task_b)}|C{_fmt_metric(f1_val_task_a)}',
            'best': f'N{_fmt_metric(best_task_b)}|C{_fmt_metric(best_task_a)}',
        })

        cb.on_epoch_end(CallbackContext(
            epoch=epoch_idx,
            epochs_total=num_epochs,
            metrics={
                f"val_f1_{task_b_name}": f1_val_task_b,
                f"val_f1_{task_a_name}": f1_val_task_a,
                f"val_accuracy_{task_b_name}": acc1_val_task_b,
                f"val_accuracy_{task_a_name}": acc1_val_task_a,
                "val_joint_score": joint_score,         # = mean(val_f1_*) — legacy default
                "val_joint_acc1": joint_acc1,           # = mean(val_accuracy_*) — reported, not the monitor
                "val_joint_arith_lift": joint_arith_lift,  # = mean(acc1_*/majority_*) — reported, v1 formula
                "val_joint_geom_lift": joint_geom_lift,    # = geometric mean — check2HGI monitor (fixed 2026-04-15)
                # Aliases so existing `monitor="val_joint_lift"` doesn't silently no-op:
                "val_joint_lift": joint_geom_lift,
                "val_loss": loss_val,
                "train_loss": epoch_loss,
                f"train_f1_{task_b_name}": f1_task_b,
                f"train_f1_{task_a_name}": f1_task_a,
            },
        ))

        if next_target_cutoff is not None and f1_val_task_b * 100 >= next_target_cutoff:
            cutoff_hits[task_b_name] = True

        if category_target_cutoff is not None and f1_val_task_a * 100 >= category_target_cutoff:
            cutoff_hits[task_a_name] = True

        if cutoff_hits[task_b_name] and cutoff_hits[task_a_name]:
            logger.info("Stopping early at epoch %d with validation F1 scores: "
                        "%s: %.4f, %s: %.4f.", epoch_idx + 1,
                        task_b_name, f1_val_task_b, task_a_name, f1_val_task_a)
            break

        current_time = time.time()
        if timeout is not None and (current_time - start_time) > timeout:
            logger.info("Training timed out after %.2f seconds during epoch %d.", timeout, epoch_idx + 1)
            break

        if cb.stop_training:
            logger.info("Callback requested stop at epoch %d.", epoch_idx + 1)
            break

    cb.on_train_end(CallbackContext(epoch=epoch_idx, epochs_total=num_epochs))
    return fold_history


# Cross-validation function
def train_with_cross_validation(dataloaders: dict[int, FoldResult],
                                history: MLHistory,
                                config: ExperimentConfig,
                                results_path: Optional[Path] = None,
                                callbacks: Optional[list] = None,
                                task_set: TaskSet = LEGACY_CATEGORY_NEXT):
    num_classes = config.model_params.get('num_classes', 7)
    task_a_name = task_set.task_a.name
    task_b_name = task_set.task_b.name
    # Per-task label-space sizes. Legacy preset has both == 7 so
    # compute_class_weights behaves identically. Non-legacy task_sets
    # (check2HGI: task_b = ~1109 regions) need per-task values or the
    # class-weight computation silently mis-sizes its output tensor.
    task_a_num_classes = task_set.task_a.num_classes or num_classes
    task_b_num_classes = task_set.task_b.num_classes or num_classes

    for fold_idx, (i_fold, dataloader) in enumerate(dataloaders.items()):
        clear_mps_cache()

        # Initialize model via registry
        model = create_model(config.model_name, **config.model_params).to(DEVICE)
        if config.use_torch_compile and DEVICE.type == 'cuda':
            model = torch.compile(model)

        # F49 encoder-frozen λ=0 isolation: freeze the cat encoder + cat
        # head so the cat **encoder** cannot co-adapt as a reg-helper via
        # cross-attention K/V. Block-internal cat-side processing
        # (`_CrossAttnBlock.ffn_a / ln_a*`) is intentionally NOT frozen —
        # those live in `shared_parameters()` and the reg stream consumes
        # their outputs as K/V via residuals; freezing them would corrupt
        # the reg pipeline. The optimizer (setup_per_head_optimizer /
        # setup_optimizer) filters `requires_grad=False` from every group,
        # so AdamW weight_decay does NOT decay the frozen weights.
        if getattr(config, "freeze_cat_stream", False):
            for p in model.category_encoder.parameters():
                p.requires_grad_(False)
            for p in model.category_poi.parameters():
                p.requires_grad_(False)
            model.category_encoder.eval()  # disables dropout in the cat encoder

        # Cache parameter group lists once per fold (item #2 — avoids
        # walking named_parameters() on every NashMTL backward call).
        cached_shared_params = list(model.shared_parameters())
        cached_task_params = list(model.task_specific_parameters())

        # Get dataloaders
        dataloader_next: TaskFoldData = dataloader.next
        dataloader_category: TaskFoldData = dataloader.category

        # F40 — scheduled_static needs total_epochs at construction time;
        # default from config.epochs unless the user already pinned it via
        # --mtl-loss-param.
        _loss_params = dict(config.mtl_loss_params)
        if config.mtl_loss == "scheduled_static" and "total_epochs" not in _loss_params:
            _loss_params["total_epochs"] = int(config.epochs)
        mtl_criterion = create_loss(config.mtl_loss, n_tasks=2, device=DEVICE, **_loss_params)

        # Per-head LR mode (F48-H3) — activated when all three of
        # cat_lr/reg_lr/shared_lr are set in the config. Otherwise fall
        # back to the legacy single-LR optimizer.
        _cat_lr = getattr(config, "cat_lr", None)
        _reg_lr = getattr(config, "reg_lr", None)
        _shared_lr = getattr(config, "shared_lr", None)
        _per_head = (_cat_lr is not None and _reg_lr is not None
                     and _shared_lr is not None)
        if _per_head:
            _reg_encoder_lr = getattr(config, "reg_encoder_lr", None)
            _reg_head_lr = getattr(config, "reg_head_lr", None)
            optimizer = setup_per_head_optimizer(
                model,
                cat_lr=float(_cat_lr),
                reg_lr=float(_reg_lr),
                shared_lr=float(_shared_lr),
                weight_decay=config.weight_decay,
                eps=config.optimizer_eps,
                extra_parameters=_criterion_parameters(mtl_criterion),
                reg_encoder_lr=float(_reg_encoder_lr) if _reg_encoder_lr is not None else None,
                reg_head_lr=float(_reg_head_lr) if _reg_head_lr is not None else None,
            )
        else:
            optimizer = setup_optimizer(
                model,
                config.learning_rate,
                config.weight_decay,
                eps=config.optimizer_eps,
                extra_parameters=_criterion_parameters(mtl_criterion),
            )
        # steps_per_epoch must match zip_longest_cycle() — the longer loader
        batches_per_epoch = max(
            len(dataloader_next.train.dataloader),
            len(dataloader_category.train.dataloader),
        )
        steps_per_epoch = math.ceil(
            batches_per_epoch / max(1, int(config.gradient_accumulation_steps))
        )
        scheduler = setup_scheduler(
            optimizer, config.max_lr, config.epochs,
            steps_per_epoch,
            scheduler_type=getattr(config, "scheduler_type", "onecycle"),
            pct_start=getattr(config, "pct_start", None),
        )
        # Smoke print for F48-H3: verify per-group LRs survived scheduler
        # init. Only on the first fold to keep logs clean. Also prints
        # trainable-param count per group — under F49 --freeze-cat-stream
        # the cat group must report 0 trainable params; any other count
        # means the freeze didn't take.
        if _per_head and getattr(history, "curr_i_fold", 0) == 0:
            _groups = [
                (pg.get("name", "?"),
                 float(pg["lr"]),
                 sum(p.numel() for p in pg["params"] if p.requires_grad))
                for pg in optimizer.param_groups
            ]
            print(f"[per-head-LR] optimizer groups (name, lr, trainable_params): {_groups}")
            if getattr(config, "freeze_cat_stream", False):
                cat_group = next(
                    (pg for pg in optimizer.param_groups if pg.get("name") == "cat"),
                    None,
                )
                if cat_group is None or any(
                    p.requires_grad for p in cat_group["params"]
                ):
                    raise RuntimeError(
                        "freeze_cat_stream=True but optimizer's 'cat' group "
                        "still contains trainable params; the freeze did not "
                        "propagate. Check setup_per_head_optimizer's "
                        "requires_grad filter."
                    )

        # Per-task class weights. Legacy behaviour (unchanged): weights
        # computed but not passed to CE — kept as a diagnostic. When
        # config.use_class_weights is set (see task-30 follow-up),
        # they are also passed as the ``weight`` kwarg.
        alpha_next = compute_class_weights(
            dataloader_next.train.y, task_b_num_classes, DEVICE
        )
        alpha_cat = compute_class_weights(
            dataloader_category.train.y, task_a_num_classes, DEVICE
        )

        _use_class_weights = bool(getattr(config, "use_class_weights", False))
        next_criterion = CrossEntropyLoss(
            reduction='mean',
            weight=alpha_next if _use_class_weights else None,
        )
        category_criterion = CrossEntropyLoss(
            reduction='mean',
            weight=alpha_cat if _use_class_weights else None,
        )

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=dataloader_next.train.dataloader.batch_size,
                num_epochs=config.epochs,
                learning_rate=config.learning_rate,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'mtl': mtl_criterion.__class__.__name__,
                    task_b_name: next_criterion.__class__.__name__,
                    task_a_name: category_criterion.__class__.__name__,
                },
                criterion_state={
                    'mtl': {},
                    task_b_name: next_criterion.state_dict(),
                    task_a_name: category_criterion.state_dict(),
                }
            )
        )

        if history.flops is None:
            sample_category, _ = next(iter(dataloader_category.train.dataloader))
            sample_next, _ = next(iter(dataloader_next.train.dataloader))
            sample_category = sample_category.to(DEVICE)
            sample_next = sample_next.to(DEVICE)
            result = calculate_model_flops(model, [sample_category[1:], sample_next[1:]], print_report=True, units='K')
            if 'total_flops' in result and 'params' in result:
                history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))
            else:
                # fvcore unavailable — set a sentinel so we don't retry every fold
                history.set_flops(FlopsMetrics(flops=0, params=0))

        # Train the model
        train_model(
            model, optimizer, scheduler,
            dataloader_next, dataloader_category,
            next_criterion, category_criterion, mtl_criterion,
            config.epochs, num_classes,
            shared_parameters=cached_shared_params,
            task_specific_parameters=cached_task_params,
            fold_history=history.get_curr_fold(),
            max_grad_norm=config.max_grad_norm,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            timeout=config.timeout,
            next_target_cutoff=config.target_cutoff,
            category_target_cutoff=config.target_cutoff,
            callbacks=callbacks,
            task_set=task_set,
            freeze_cat_after_epoch=getattr(config, "freeze_cat_after_epoch", None),
            alternating_optimizer_step=getattr(config, "alternating_optimizer_step", False),
            cat_specific_parameters=(
                list(model.cat_specific_parameters())
                if hasattr(model, "cat_specific_parameters") else None
            ),
            reg_specific_parameters=(
                list(model.reg_specific_parameters())
                if hasattr(model, "reg_specific_parameters") else None
            ),
        )

        # Run final validation
        logger.info("Running final validation...")
        joint_best_state = history.fold.model_task.best.best_state
        if not joint_best_state:
            logger.warning(
                "No joint best state recorded for fold %d; using current model state.",
                fold_idx,
            )
            joint_best_state = model.state_dict()
        report_next, report_category = validation_best_model(
            dataloader_next.val.dataloader,
            dataloader_category.val.dataloader,
            joint_best_state,
            joint_best_state,
            model
        )

        history.fold.task(task_b_name).report = report_next
        history.fold.task(task_a_name).report = report_category

        history.step()

    # Display summary metrics (if verbose=False; verbose step() handles it)
    if not history._verbose:
        history.display.end_training()

    # Write run manifest
    if results_path is not None:
        from configs.experiment import RunManifest
        from configs.paths import IoPaths, EmbeddingEngine
        engine = EmbeddingEngine(config.embedding_engine)
        manifest = RunManifest.from_current_env(
            config=config,
            dataset_paths={
                "category_input": IoPaths.get_category(config.state, engine),
                "next_input": IoPaths.get_next(config.state, engine),
            },
        )
        manifest.write(results_path)
