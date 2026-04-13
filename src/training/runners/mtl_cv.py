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
from training.helpers import compute_class_weights, setup_optimizer, setup_scheduler
from training.callbacks import CallbackContext, CallbackList
from data.folds import TaskFoldData, FoldResult
from tracking import MLHistory, FlopsMetrics, NeuralParams
from tracking.fold import FoldHistory, TaskHistory
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
    if not shared_parameters:
        return float("nan"), 0.0, 0.0

    next_grads = torch.autograd.grad(
        losses[0],
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    category_grads = torch.autograd.grad(
        losses[1],
        shared_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    next_flat = _flatten_task_grads(next_grads, shared_parameters)
    category_flat = _flatten_task_grads(category_grads, shared_parameters)

    next_norm = torch.norm(next_flat)
    category_norm = torch.norm(category_flat)
    denom = next_norm * category_norm
    if denom <= 0:
        return float("nan"), next_norm.item(), category_norm.item()

    cosine = torch.dot(next_flat, category_flat) / denom
    return cosine.item(), next_norm.item(), category_norm.item()


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
        return loss, extra_outputs, True


def _pareto_front_indices(points: list[tuple[float, float]]) -> list[int]:
    front = []
    for i, (next_f1, cat_f1) in enumerate(points):
        dominated = False
        for j, (other_next, other_cat) in enumerate(points):
            if i == j:
                continue
            if (
                other_next >= next_f1
                and other_cat >= cat_f1
                and (other_next > next_f1 or other_cat > cat_f1)
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
                fold_history=FoldHistory.standalone({'next', 'category'}),
                max_grad_norm: float = 1.0,
                gradient_accumulation_steps: int = 1,
                timeout: Optional[int] = None,
                next_target_cutoff: Optional[float] = None,
                category_target_cutoff: Optional[float] = None,
                callbacks: Optional[list] = None,
                ):
    """
    Train the model with multi-task learning.
    """
    start_time = time.time()

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
        'next': False,
        'category': False
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

    # Main training loop
    for epoch_idx in progress:
        model.train()

        # Initialize on-device accumulators to avoid per-batch MPS syncs
        running_loss = torch.tensor(0.0, device=DEVICE)
        next_running_loss = torch.tensor(0.0, device=DEVICE)
        category_running_loss = torch.tensor(0.0, device=DEVICE)
        steps = 0

        # Collect logits on-device so compute_classification_metrics() can
        # produce the full metric dict (Macro/Weighted F1, Top-K, MRR, NDCG)
        # in a single per-epoch call.
        all_next_logits, all_next_targets = [], []
        all_cat_logits, all_cat_targets = [], []

        # Per-epoch diagnostics — recomputed once per epoch on batch 0
        # (see Phase 0 §60 of plan/MTL_IMPROVEMENT_PLAN.md).
        epoch_grad_cosine: float = float("nan")
        epoch_next_grad_norm: float = 0.0
        epoch_category_grad_norm: float = 0.0
        epoch_loss_weights: Optional[torch.Tensor] = None
        accumulated_in_group: int = 0

        # Iterate over batches with automatic progress tracking.
        # zero_grad is called at the end of every optimizer step, so the
        # loop starts each epoch with clean gradients (the last batch of
        # the previous epoch is always forced to step via the
        # `(batch_idx + 1) == batches_per_epoch` branch below).
        for batch_idx, (data_next, data_category) in enumerate(progress.iter_epoch()):
            # When the dataset is pre-moved to DEVICE (item #3, MPS path with
            # num_workers=0), the .to() calls are no-ops. Keep the guards so
            # the loop still works under a CPU-side dataloader path.
            x_next, y_next = data_next
            if x_next.device != DEVICE:
                x_next = x_next.to(DEVICE, non_blocking=True)
                y_next = y_next.to(DEVICE, non_blocking=True)
            x_category, y_category = data_category
            if x_category.device != DEVICE:
                x_category = x_category.to(DEVICE, non_blocking=True)
                y_category = y_category.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_ctx:
                category_output, next_poi_output = model((x_category, x_next))

                pred_next, truth_next = next_poi_output, y_next
                pred_category, truth_category = category_output, y_category

                # Calculate losses (inside autocast so CE uses float16 logits)
                next_loss = next_criterion(pred_next, truth_next)
                category_loss = category_criterion(pred_category, truth_category)

            # NashMTL backward stays outside autocast — gradients in float32
            losses = torch.stack([next_loss, category_loss])

            # Shared-gradient cosine once per epoch. torch.autograd.grad does
            # not populate .grad, so it leaves the subsequent backward path
            # untouched — but it requires retain_graph=True, which the helper
            # already sets.
            if batch_idx == 0 and shared_parameters:
                (
                    epoch_grad_cosine,
                    epoch_next_grad_norm,
                    epoch_category_grad_norm,
                ) = _compute_gradient_cosine(losses, shared_parameters)

            loss, extra_outputs, already_backpropagated = _get_weighted_loss(
                mtl_criterion,
                losses,
                shared_parameters=shared_parameters,
                task_specific_parameters=task_specific_parameters,
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
            next_running_loss += next_loss.detach()
            category_running_loss += category_loss.detach()

            # Collect logits on-device for epoch-level metrics. We keep the
            # full logit tensor (not argmax) so ranking metrics are free.
            with torch.no_grad():
                all_next_logits.append(pred_next.detach())
                all_next_targets.append(truth_next)
                all_cat_logits.append(pred_category.detach())
                all_cat_targets.append(truth_category)

            steps += 1

        epoch_next_logits = torch.cat(all_next_logits)
        epoch_next_targets = torch.cat(all_next_targets)
        epoch_cat_logits = torch.cat(all_cat_logits)
        epoch_cat_targets = torch.cat(all_cat_targets)

        train_metrics_next = compute_classification_metrics(
            epoch_next_logits, epoch_next_targets, num_classes=num_classes,
        )
        train_metrics_cat = compute_classification_metrics(
            epoch_cat_logits, epoch_cat_targets, num_classes=num_classes,
        )
        f1_next = train_metrics_next['f1']
        next_acc = train_metrics_next['accuracy']
        f1_category = train_metrics_cat['f1']
        category_acc = train_metrics_cat['accuracy']

        # Calculate epoch metrics (single sync for losses)
        epoch_loss = running_loss.item() / steps
        epoch_loss_next = next_running_loss.item() / steps
        epoch_loss_category = category_running_loss.item() / steps
        loss_ratio_next_to_category = epoch_loss_next / max(epoch_loss_category, 1e-8)

        progress.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'next': f'{f1_next:.4f}({next_acc:.4f})',
            'cat': f'{f1_category:.4f}({category_acc:.4f})'
        })

        fold_history.model_task.log_train(loss=epoch_loss, accuracy=0)
        fold_history.log_train(
            'next', loss=epoch_loss_next, **train_metrics_next,
        )
        fold_history.log_train(
            'category', loss=epoch_loss_category, **train_metrics_cat,
        )
        diagnostic_payload = {
            "grad_cosine_shared": epoch_grad_cosine,
            "grad_norm_next_shared": epoch_next_grad_norm,
            "grad_norm_category_shared": epoch_category_grad_norm,
            "loss_ratio_next_to_category": loss_ratio_next_to_category,
        }
        if epoch_loss_weights is not None:
            weights_cpu = epoch_loss_weights.detach().cpu()
            if len(weights_cpu) >= 2:
                diagnostic_payload["loss_weight_next"] = float(weights_cpu[0])
                diagnostic_payload["loss_weight_category"] = float(weights_cpu[1])
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
            val_metrics_next, val_metrics_cat, loss_val = evaluate_model(
                model,
                [dataloader_next.val.dataloader, dataloader_category.val.dataloader],
                next_criterion,
                category_criterion,
                mtl_criterion,
                DEVICE,
                num_classes=num_classes,
            )

            f1_val_next = val_metrics_next['f1']
            f1_val_category = val_metrics_cat['f1']
            acc_val_next = val_metrics_next['accuracy']
            acc_val_category = val_metrics_cat['accuracy']

            joint_score = 0.5 * (f1_val_next + f1_val_category)
            pareto_points.append((f1_val_next, f1_val_category))
            pareto_front = _pareto_front_indices(pareto_points)
            fold_history.add_artifact(
                "pareto_front",
                [
                    {
                        "epoch": idx,
                        "next_f1": pareto_points[idx][0],
                        "category_f1": pareto_points[idx][1],
                    }
                    for idx in pareto_front
                ],
            )

            # Only create state_dict when at least one task improves.
            next_improved = f1_val_next > fold_history.task('next').best.best_value
            cat_improved = f1_val_category > fold_history.task('category').best.best_value
            prev_joint_best = fold_history.model_task.best.best_value if fold_history.model_task.best.best_epoch >= 0 else -1.0
            joint_improved = joint_score > prev_joint_best
            state = model.state_dict() if (joint_improved or next_improved or cat_improved) else None

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
                'next',
                **val_metrics_next,
                model_state=state if next_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )
            fold_history.log_val(
                'category',
                **val_metrics_cat,
                model_state=state if cat_improved else None,
                elapsed_time=fold_history.timer.timer(),
            )

        # Update metrics on progress bar with validation results
        progress.set_postfix({
            'val_loss': f'{loss_val:.4f}',
            'next_val': f'{f1_val_next:.4f}({acc_val_next:.4f})',
            'cat_val': f'{f1_val_category:.4f}({acc_val_category:.4f})'
        })

        cb.on_epoch_end(CallbackContext(
            epoch=epoch_idx,
            epochs_total=num_epochs,
            metrics={
                "val_f1_next": f1_val_next,
                "val_f1_category": f1_val_category,
                "val_joint_score": joint_score,
                "val_loss": loss_val,
                "train_loss": epoch_loss,
                "train_f1_next": f1_next,
                "train_f1_category": f1_category,
            },
        ))

        if next_target_cutoff is not None and f1_val_next * 100 >= next_target_cutoff:
            cutoff_hits['next'] = True

        if category_target_cutoff is not None and f1_val_category * 100 >= category_target_cutoff:
            cutoff_hits['category'] = True

        if cutoff_hits['next'] and cutoff_hits['category']:
            logger.info("Stopping early at epoch %d with validation F1 scores: "
                        "Next: %.4f, Category: %.4f.", epoch_idx + 1, f1_val_next, f1_val_category)
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
                                callbacks: Optional[list] = None):
    num_classes = config.model_params.get('num_classes', 7)

    for fold_idx, (i_fold, dataloader) in enumerate(dataloaders.items()):
        clear_mps_cache()

        # Initialize model via registry
        model = create_model(config.model_name, **config.model_params).to(DEVICE)
        if config.use_torch_compile and DEVICE.type == 'cuda':
            model = torch.compile(model)

        # Cache parameter group lists once per fold (item #2 — avoids
        # walking named_parameters() on every NashMTL backward call).
        cached_shared_params = list(model.shared_parameters())
        cached_task_params = list(model.task_specific_parameters())

        # Get dataloaders
        dataloader_next: TaskFoldData = dataloader.next
        dataloader_category: TaskFoldData = dataloader.category

        mtl_criterion = create_loss(config.mtl_loss, n_tasks=2, device=DEVICE, **config.mtl_loss_params)

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
        )

        alpha_next = compute_class_weights(
            dataloader_next.train.y, num_classes, DEVICE
        )
        alpha_cat = compute_class_weights(
            dataloader_category.train.y, num_classes, DEVICE
        )

        next_criterion = CrossEntropyLoss(reduction='mean')
        category_criterion = CrossEntropyLoss(reduction='mean')

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
            sample_category, _ = next(iter(dataloader_category.train.dataloader))
            sample_next, _ = next(iter(dataloader_next.train.dataloader))
            sample_category = sample_category.to(DEVICE)
            sample_next = sample_next.to(DEVICE)
            result = calculate_model_flops(model, [sample_category[1:], sample_next[1:]], print_report=True, units='K')
            history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))

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

        history.fold.task('next').report = report_next
        history.fold.task('category').report = report_category

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
