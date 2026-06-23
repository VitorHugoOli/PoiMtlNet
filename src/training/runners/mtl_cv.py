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

from tracking.metrics import (
    compute_classification_metrics,
    _handrolled_cls_metrics,
    _rank_of_target,
    _mrr_from_rank,
    _ndcg_from_rank,
)
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
                alpha_frozen_until_epoch: Optional[int] = None,
                cat_specific_parameters: Optional[list] = None,
                reg_specific_parameters: Optional[list] = None,
                joint_loader_strategy: str = "max_size_cycle",
                reg_freeze_at_epoch: Optional[int] = None,
                task_best_tracker=None,
                task_best_save_dir: Optional[Path] = None,
                log_t_kd_weight: float = 0.0,
                log_t_kd_tau: float = 1.0,
                log_t_kd_gate: str = "none",
                log_c_kd_weight: float = 0.0,
                log_c_kd_tau: float = 1.0,
                log_c_kd_warmup_epochs: int = 0,
                log_c_kd_ec_lambda: float = 0.0,
                cat_kd_weight: float = 0.0,
                cat_kd_tau: float = 1.0,
                loss_scale_norm: bool = False,
                checkpoint_selector: str = "geom_simple",
                joint_min_epoch: int = 0,
                joint_train_loader=None,
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

    # Create progress bar that extends tqdm.
    # G0.1 aligned-pairing: when a joint train loader is provided, drive the
    # epoch from that SINGLE loader (it yields ((x_reg,y_reg),(x_cat,y_cat))
    # under one shared permutation → cat-window k paired with reg-window k).
    # zip_longest_cycle with a 1-element list passes batches straight through,
    # so the loop's ``(data_task_b, data_task_a)`` unpacking is unchanged.
    if joint_train_loader is not None:
        progress = TrainingProgressBar(
            num_epochs,
            [joint_train_loader],
            joint_loader_strategy=joint_loader_strategy,
        )
    else:
        progress = TrainingProgressBar(
            num_epochs,
            [dataloader_next.train.dataloader,
             dataloader_category.train.dataloader],
            joint_loader_strategy=joint_loader_strategy,
        )

    cb = CallbackList(callbacks)

    # Inject model reference for callbacks that need it (e.g. ModelCheckpoint)
    for c in cb.callbacks:
        if hasattr(c, 'set_model'):
            c.set_model(model)

    cb.on_train_begin(CallbackContext(epoch=0, epochs_total=num_epochs))

    # Mixed-precision autocast: float16 forward passes on CUDA, no-op otherwise.
    # MPS float16 autocast adds overhead for small tensors — disabled there.
    # Diagnostic escape hatch (mtl_improvement T2P.0 harness-confound control):
    # set MTL_DISABLE_AMP=1 to force the full fp32 path (no autocast), matching
    # the p1-STL ceiling harness which never autocasts. NB the CUDA trainer runs
    # fp16 autocast with NO GradScaler — this env var isolates that precision
    # delta when comparing MTL-reg vs the fp32 STL ceiling. Default (unset) keeps
    # the canonical fp16 behaviour untouched.
    import os as _os
    _disable_amp = _os.environ.get("MTL_DISABLE_AMP") == "1"
    _autocast_ctx = (
        torch.autocast(DEVICE.type, dtype=torch.float16)
        if DEVICE.type == 'cuda' and not _disable_amp
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
    if joint_loader_strategy == "min_size_truncate":
        batches_per_epoch = min(
            len(dataloader_next.train.dataloader),
            len(dataloader_category.train.dataloader),
        )
    else:
        batches_per_epoch = max(
            len(dataloader_next.train.dataloader),
            len(dataloader_category.train.dataloader),
        )
    pareto_points: list[tuple[float, float]] = []

    # F50 D5 — encoder weight-trajectory diagnostic. Snapshot the initial
    # `next_encoder` and `category_encoder` parameter vectors so per-epoch
    # Frobenius drift can be logged below. The hypothesis (F50 T3 §5.5):
    # MTL's reg-best epoch is structurally pinned at ~ep 5 because the
    # `next_encoder` stops updating early under joint loss while
    # `category_encoder` keeps drifting. Per-epoch L2 norm and drift-from-init
    # log directly tests this. Silent no-op if encoders are absent.
    def _flatten_encoder(encoder):
        if encoder is None:
            return None
        params = [p.detach().reshape(-1) for p in encoder.parameters() if p.requires_grad]
        if not params:
            return None
        return torch.cat(params).clone()

    next_enc_init = _flatten_encoder(getattr(model, "next_encoder", None))
    cat_enc_init = _flatten_encoder(getattr(model, "category_encoder", None))
    next_enc_prev = next_enc_init.clone() if next_enc_init is not None else None
    cat_enc_prev = cat_enc_init.clone() if cat_enc_init is not None else None

    # F50 P3 — track whether warmup-then-freeze has fired (idempotent).
    _cat_frozen_post_warmup = False
    # substrate-protocol-cleanup Tier C2 — track whether reg-freeze has fired.
    # Once True, task_b_loss is zeroed before forming the MTL loss tensor
    # and next_encoder.* / next_poi.* params have requires_grad=False, so
    # the optimizer step is naturally a no-op on them.
    _reg_frozen_post_peak = False
    # F50 B4 — α-freeze warmup. If alpha_frozen_until_epoch is set, lock
    # α at its init value for ep 0..N-1, then unfreeze. Pre-freeze the
    # parameter HERE (before training) so the first epoch already sees
    # frozen α; the unfreeze happens at epoch N inside the loop below.
    _alpha_unfrozen = False
    if alpha_frozen_until_epoch is not None and int(alpha_frozen_until_epoch) > 0:
        next_head = getattr(model, "next_poi", None)
        head_alpha = getattr(next_head, "alpha", None)
        if isinstance(head_alpha, torch.nn.Parameter):
            head_alpha.requires_grad_(False)
            print(
                f"[B4 alpha-frozen-until-epoch] α frozen at "
                f"{float(head_alpha):.4f} until epoch {alpha_frozen_until_epoch}"
            )

    # Main training loop
    import os as _os_ng
    _os_nanguard = _os_ng.environ.get("MTL_NAN_GUARD") == "1"
    for epoch_idx in progress:
        model.train()
        # F50 B4 — at the boundary epoch, unfreeze α so it can grow.
        if (alpha_frozen_until_epoch is not None
                and not _alpha_unfrozen
                and epoch_idx >= int(alpha_frozen_until_epoch)):
            next_head = getattr(model, "next_poi", None)
            head_alpha = getattr(next_head, "alpha", None)
            if isinstance(head_alpha, torch.nn.Parameter):
                head_alpha.requires_grad_(True)
                _alpha_unfrozen = True
                print(
                    f"[B4 alpha-frozen-until-epoch] α unfrozen at epoch "
                    f"{epoch_idx} (target N={alpha_frozen_until_epoch}); "
                    f"current α = {float(head_alpha):.4f}"
                )
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

        # substrate-protocol-cleanup Tier C2 — at boundary epoch N, freeze
        # reg-side params (next_encoder + next_poi a.k.a. task_b_encoder +
        # next_head per the task_set slot mapping) and from this epoch
        # onward zero ``task_b_loss`` before the MTL combiner sees it.
        # Mirror of P3 but for the reg side. The optimizer's
        # ``requires_grad`` filter at step time naturally skips frozen
        # params — no optimizer rebuild needed. The runner's per-head
        # optimizer (setup_per_head_optimizer) also pre-filters by
        # requires_grad so the reg-encoder/reg-head groups get pruned.
        if (reg_freeze_at_epoch is not None
                and not _reg_frozen_post_peak
                and epoch_idx >= int(reg_freeze_at_epoch)):
            # Reg encoder & head naming convention in this codebase:
            # ``next_encoder`` (slot B encoder) + ``next_poi`` (reg head).
            # These map to the docstring's task_b_encoder.* / next_head.*.
            for attr in ("next_encoder", "next_poi"):
                sub = getattr(model, attr, None)
                if sub is None:
                    continue
                for p in sub.parameters():
                    p.requires_grad_(False)
                if hasattr(sub, "eval"):
                    sub.eval()
            _reg_frozen_post_peak = True
            print(
                f"[C2 reg-freeze-at-epoch] next_encoder + next_poi frozen "
                f"at epoch {epoch_idx} (target N={reg_freeze_at_epoch}); "
                f"task_b_loss zeroed for remaining epochs"
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

        # S1 (perf-audit) — STREAMING TRAIN-metric for the high-cardinality reg head.
        # Retaining the full epoch's [N, n_regions] reg logits to compute train metrics
        # OOMs the GPU and (after the CPU move) costs ~tens of GB host RAM at large/overlap
        # scale. Every train metric is a per-ROW reduction (accuracy / rank→MRR/NDCG /
        # top-k hit) or an additive per-CLASS count (bincount via _handrolled_cls_metrics),
        # so we accumulate only tiny [N]/[C] vectors per batch and reconstruct the IDENTICAL
        # metric dict at epoch end via the SAME metrics.py helpers — byte-identical to the
        # full-logit path (verified), O(N·C)→O(N+C) memory. Reg only (cat C=7 is trivial and
        # uses the torchmetrics low-card path). Disable with MTL_STREAM_TRAIN_METRIC=0.
        import os as _os_s1
        _stream_train_b = (
            _os_s1.environ.get("MTL_STREAM_TRAIN_METRIC", "1").strip() in ("1", "true", "True")
            and task_b_num_classes is not None and task_b_num_classes > 256
        )
        _S1_TOPK = (3, 5)  # matches compute_classification_metrics default top_k
        s1_preds_b, s1_targets_b, s1_rank_b = [], [], []
        s1_hit_b = {k: [] for k in _S1_TOPK}

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

                # T4.0a (mtl_improvement) loss-scale normalization — divide each
                # task's CE by log(num_classes) BEFORE the MTL combiner, so the
                # built-in ~4.7x CE-magnitude gap (ln(n_regions) vs ln(7)) is
                # decoupled from the inter-task weight. Gated; default no-op.
                if loss_scale_norm:
                    import math as _math
                    _nb = pred_task_b.shape[-1]
                    _na = pred_task_a.shape[-1]
                    if _nb > 1:
                        task_b_loss = task_b_loss / _math.log(_nb)
                    if _na > 1:
                        task_a_loss = task_a_loss / _math.log(_na)

                # substrate-protocol-cleanup Tier A1 / mtl-protocol-fix Phase 3 §4.5 —
                # log_T KL distillation supervisory signal. For each sample with
                # a valid ``last_region_idx`` r, the teacher distribution is
                # softmax(log_T[r] / τ) over n_regions; the student is
                # softmax(reg_logits / τ). KD term = τ² · KL(student || teacher),
                # added to task_b_loss with weight ``log_t_kd_weight``. Strict
                # no-op fast path when weight == 0.0 (the W=0.0 baseline). Padding
                # rows (last_region_idx < 0 or >= num_classes) are excluded so
                # they don't contaminate the gradient. Requires a log_T-aware
                # reg head (next_getnext_hard / next_stan_flow / next_getnext)
                # whose forward registers a ``log_T`` buffer of shape
                # [num_classes, num_classes]. See
                # docs/results/mtl_protocol_fix/phase3_rank1_findings.md.
                if log_t_kd_weight > 0.0:
                    from data.aux_side_channel import get_current_aux
                    _aux = get_current_aux()
                    _reg_head = getattr(model, "next_poi", None)
                    _log_T = getattr(_reg_head, "log_T", None) if _reg_head is not None else None
                    if _aux is not None and _log_T is not None:
                        _nc = pred_task_b.shape[-1]
                        # log_T may have been built with num_regions > num_classes
                        # of the current task slot; the head slices it down to
                        # [num_classes, num_classes] at init. Defensive re-slice
                        # here in case a head variant skips that step.
                        if _log_T.shape[0] >= _nc and _log_T.shape[1] >= _nc:
                            _log_T_use = _log_T[:_nc, :_nc]
                        else:
                            _log_T_use = None
                        if _log_T_use is not None:
                            if _aux.device != pred_task_b.device:
                                _aux = _aux.to(pred_task_b.device)
                            _pad = (_aux < 0) | (_aux >= _nc)
                            _valid = ~_pad
                            if _valid.any():
                                _safe = _aux.clamp(min=0, max=_nc - 1)
                                _tau = float(log_t_kd_tau)
                                # Teacher: softmax of the per-sample log_T row at τ.
                                # log_T is already in log-prob space; softmax
                                # re-normalises (it may not be a strict prob
                                # distribution numerically) — standard form.
                                _teacher_logits = _log_T_use.index_select(0, _safe).float() / _tau
                                _teacher = torch.softmax(_teacher_logits, dim=-1)
                                # Student: log_softmax of reg logits at τ.
                                _student_log = torch.log_softmax(
                                    pred_task_b.float() / _tau, dim=-1
                                )
                                # KL(student || teacher) per-sample, with τ² scaling
                                # (standard Hinton-distillation gradient preservation).
                                # F.kl_div expects input=log-probs, target=probs and
                                # computes sum_j target_j * (log(target_j) - input_j)
                                # which is KL(target || input) = KL(teacher || student).
                                # phase3_rank1_findings.md writes
                                # ``KL(softmax(reg_logits/τ) ‖ exp(log_T[...]))`` i.e.
                                # KL(student || teacher). Implement that direction
                                # explicitly: sum_j student * (log_student - log_teacher).
                                _log_teacher = torch.log(_teacher.clamp_min(1e-12))
                                _student = _student_log.exp()
                                _kld_per_sample = (
                                    _student * (_student_log - _log_teacher)
                                ).sum(dim=-1)
                                # R5 (mtl_frontier) — per-instance KD gating. Redistribute
                                # the (batch-mean-fixed) KD weight by Markov-coverage of the
                                # teacher row: peaked row (Markov-1 binds) → upweight KD; flat
                                # row → downweight. Mean-1 over valid samples ⇒ the TOTAL KD
                                # budget equals global-W (tests redistribution, not strength).
                                # gate=="none" → multiply by _valid only (bit-identical).
                                if log_t_kd_gate != "none":
                                    import math as _math
                                    if log_t_kd_gate == "coverage_max":
                                        _cov = _teacher.max(dim=-1).values
                                    else:  # coverage_entropy
                                        _ent = -(_teacher * _log_teacher).sum(dim=-1)
                                        _cov = 1.0 - _ent / _math.log(_teacher.shape[-1])
                                    _cov = _cov * _valid.float()
                                    _gmean = (_cov.sum() / _valid.sum().clamp_min(1)).clamp_min(1e-6)
                                    _gate = _cov / _gmean  # mean-1 over valid, 0 on pad
                                    # C28 fires-diagnostic: spread of the per-sample gate
                                    # (0 ⇒ uniform ⇒ ≡ global-W; >0 ⇒ genuinely redistributing).
                                    if _valid.any():
                                        model._r5_gate_std = float(_gate[_valid].std().detach())
                                    _kld_per_sample = _kld_per_sample * _gate
                                else:
                                    _kld_per_sample = _kld_per_sample * _valid.float()
                                _denom = _valid.sum().clamp_min(1).float()
                                _kd_loss = (_kld_per_sample.sum() / _denom) * (_tau * _tau)
                                task_b_loss = task_b_loss + log_t_kd_weight * _kd_loss

                # R1 (mtl_frontier) — log_C co-location KD (ESMM probability-chain).
                # A SECOND distillation term whose per-sample teacher is the
                # cat-marginalized region prior:
                #   prior(reg) = Σ_c P(reg|c) · P̂(c),  P̂ = softmax(cat_logits).detach()
                #   teacher    = softmax(log(prior)/τ);  student = softmax(reg_logits/τ)
                #   L_reg += W · τ² · KL(student || teacher)
                # P(reg|c) = exp(log_C) is the train-only per-fold/seed matrix
                # buffered on the reg head. Stacks on top of log_T-KD. Strict
                # no-op fast path at weight 0.0. C28 dead-codepath guard: a one-shot
                # diagnostic asserts the teacher is non-trivial (non-uniform) and
                # logs its mean max-prob the first time it fires.
                # R3 (mtl_frontier) — CrossDistil warm-up gate (both arms): the
                # synchronous teacher is noisy early; apply only from epoch N.
                _logc_active = epoch_idx >= int(log_c_kd_warmup_epochs)
                _ec = float(log_c_kd_ec_lambda)
                if log_c_kd_weight > 0.0 and _logc_active:
                    _reg_head_c = getattr(model, "next_poi", None)
                    _log_C = getattr(_reg_head_c, "log_C", None) if _reg_head_c is not None else None
                    if _log_C is not None:
                        _ncr = pred_task_b.shape[-1]   # n_regions
                        _nca = pred_task_a.shape[-1]   # n_cats
                        if _log_C.shape[0] >= _ncr and _log_C.shape[1] == _nca:
                            _tau_c = float(log_c_kd_tau)
                            # P̂(c) — detached cat posterior (teacher factor only).
                            _phat = torch.softmax(pred_task_a.float(), dim=-1).detach()  # [B, n_cats]
                            _P_reg_c = _log_C[:_ncr, :].float().exp()                     # [n_regions, n_cats]
                            # prior(reg) = Σ_c P(reg|c)·P̂(c) → [B, n_regions]
                            _prior = _phat @ _P_reg_c.transpose(0, 1)                     # [B, n_regions]
                            _prior = _prior.clamp_min(1e-12)
                            _teacher_c = torch.softmax(torch.log(_prior) / _tau_c, dim=-1)
                            # R3 CrossDistil error-correction: blend the soft teacher
                            # with the reg ground-truth one-hot (corrects teacher errors).
                            if _ec > 0.0:
                                _oh = torch.zeros_like(_teacher_c)
                                _yb = truth_task_b.clamp(0, _ncr - 1).long().unsqueeze(-1)
                                _oh.scatter_(1, _yb, 1.0)
                                _teacher_c = (1.0 - _ec) * _teacher_c + _ec * _oh
                            _student_c_log = torch.log_softmax(pred_task_b.float() / _tau_c, dim=-1)
                            _student_c = _student_c_log.exp()
                            _log_teacher_c = torch.log(_teacher_c.clamp_min(1e-12))
                            _kldc = (_student_c * (_student_c_log - _log_teacher_c)).sum(dim=-1)
                            _kdc_loss = _kldc.mean() * (_tau_c * _tau_c)
                            task_b_loss = task_b_loss + log_c_kd_weight * _kdc_loss
                            if not globals().get("_LOGC_FIRED", False):
                                globals()["_LOGC_FIRED"] = True
                                _tmax = float(_teacher_c.max(dim=-1).values.mean())
                                _unif = 1.0 / float(_ncr)
                                logger.info(
                                    "[R1/R3 log_C-KD fwd FIRED] W=%.3g τ=%.3g warmup=%d "
                                    "ec=%.2g teacher max-prob=%.4f (%.0f× uniform) "
                                    "mean KL=%.4f",
                                    log_c_kd_weight, _tau_c, log_c_kd_warmup_epochs, _ec,
                                    _tmax, _tmax / _unif, float(_kldc.mean()),
                                )

                # R3 reverse arm — distill the reg-implied category prior into the
                # CAT head: prior(cat)=Σ_r P(cat|r)·P̂_reg(r), P̂_reg detached.
                #   L_cat += W · τ² · KL(softmax(cat_logits/τ) || softmax(log(prior)/τ))
                if cat_kd_weight > 0.0 and _logc_active:
                    _reg_head_r = getattr(model, "next_poi", None)
                    _log_Crev = getattr(_reg_head_r, "log_C_rev", None) if _reg_head_r is not None else None
                    if _log_Crev is not None:
                        _ncr = pred_task_b.shape[-1]; _nca = pred_task_a.shape[-1]
                        if _log_Crev.shape[0] >= _ncr and _log_Crev.shape[1] == _nca:
                            _tau_k = float(cat_kd_tau)
                            _phat_reg = torch.softmax(pred_task_b.float(), dim=-1).detach()   # [B, n_regions]
                            _P_cat_r = _log_Crev[:_ncr, :].float().exp()                       # [n_regions, n_cats]
                            _prior_cat = (_phat_reg @ _P_cat_r).clamp_min(1e-12)               # [B, n_cats]
                            _teacher_k = torch.softmax(torch.log(_prior_cat) / _tau_k, dim=-1)
                            if _ec > 0.0:
                                _ohc = torch.zeros_like(_teacher_k)
                                _ya = truth_task_a.clamp(0, _nca - 1).long().unsqueeze(-1)
                                _ohc.scatter_(1, _ya, 1.0)
                                _teacher_k = (1.0 - _ec) * _teacher_k + _ec * _ohc
                            _student_k_log = torch.log_softmax(pred_task_a.float() / _tau_k, dim=-1)
                            _kldk = (_student_k_log.exp() * (_student_k_log - torch.log(_teacher_k.clamp_min(1e-12)))).sum(dim=-1)
                            task_a_loss = task_a_loss + cat_kd_weight * (_kldk.mean() * (_tau_k * _tau_k))
                            if not globals().get("_CATKD_FIRED", False):
                                globals()["_CATKD_FIRED"] = True
                                _tmk = float(_teacher_k.max(dim=-1).values.mean())
                                logger.info(
                                    "[R3 reverse cat-KD FIRED] W=%.3g τ=%.3g warmup=%d "
                                    "ec=%.2g teacher max-prob=%.4f (%.1f× uniform) mean KL=%.4f",
                                    cat_kd_weight, _tau_k, log_c_kd_warmup_epochs, _ec,
                                    _tmk, _tmk * _nca, float(_kldk.mean()),
                                )

            # substrate-protocol-cleanup Tier C2 — after the reg-freeze
            # boundary, the reg loss is contributed at weight 0 so the MTL
            # combiner (static_weight, NashMTL, etc.) sees a zero on the
            # reg slot. Coupled with the frozen ``requires_grad=False`` on
            # next_encoder/next_poi, this is a complete reg-side stop:
            # no reg loss, no reg param update. The reg head still emits
            # logits (we keep them for val metric tracking each epoch).
            if _reg_frozen_post_peak:
                task_b_loss = task_b_loss * 0.0

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
                    _gn = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + _criterion_parameters(mtl_criterion),
                        max_grad_norm,
                    )
                    # Env-gated NaN/grad-norm diagnostic (MTL_NAN_GUARD=1). No-op otherwise
                    # (byte-identical). Catches the first non-finite loss/grad and traces the
                    # pre-NaN grad-norm trajectory — for the CA ep30-collapse audit.
                    if _os_nanguard:
                        _lf = float(loss.detach()); _gnf = float(_gn)
                        if (not math.isfinite(_lf)) or (not math.isfinite(_gnf)):
                            logger.warning("[NAN_GUARD] NON-FINITE epoch=%d batch=%d loss=%s grad_norm=%s",
                                           epoch_idx, batch_idx, _lf, _gnf)
                        elif (batch_idx % 100) == 0:
                            logger.info("[NAN_GUARD] epoch=%d batch=%d loss=%.4f grad_norm=%.3f",
                                        epoch_idx, batch_idx, _lf, _gnf)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_in_group = 0

            # Accumulate on-device — no .item() sync per batch
            running_loss += loss.detach()
            task_b_running_loss += task_b_loss.detach()
            task_a_running_loss += task_a_loss.detach()

            # Collect logits for epoch-level TRAIN metrics. We keep the full logit
            # tensor (not argmax) so ranking metrics are free. Accumulate on CPU:
            # at large/overlap scale (FL stride-1 ~1.1M train rows, CA/TX ~4.7-8.5k
            # regions) the on-GPU torch.cat of a full epoch's [N, n_regions] logits
            # OOMs the 44 GB A40 (dies on the last batch's cat). These are
            # DIAGNOSTIC TRAIN metrics only — they feed logging/progress, NOT
            # selection/early-stopping/checkpointing (all of which key off VAL
            # metrics), so moving the accumulation to host RAM is quality-neutral:
            # the model trajectory (post-step, no_grad, detached) and the val-driven
            # scored results are bitwise unchanged (advisor-vetted + AL A/B verified).
            # NB: do NOT apply this to the VAL path (mtl_eval.py) — val IS the scored
            # metric and its fp16-CUDA tie-breaking defines the canonical Acc@10.
            with torch.no_grad():
                # cat (task_a) — low-cardinality, keep the full-logit path (trivial size).
                all_task_a_logits.append(pred_task_a.detach().cpu())
                all_task_a_targets.append(truth_task_a.cpu())
                if _stream_train_b:
                    # reg (task_b) — STREAM per-row/per-class reductions on CPU (matches the
                    # CPU compute path byte-for-byte), discarding the [batch, C] logits.
                    _lb = pred_task_b.detach().cpu()
                    _tb = truth_task_b.cpu()
                    s1_preds_b.append(_lb.argmax(dim=-1))
                    s1_targets_b.append(_tb)
                    s1_rank_b.append(_rank_of_target(_lb, _tb))
                    for _k in _S1_TOPK:
                        _ke = min(_k, _lb.shape[-1])
                        _topk = _lb.topk(_ke, dim=-1).indices
                        s1_hit_b[_k].append((_topk == _tb.unsqueeze(-1)).any(dim=-1))
                    del _lb
                else:
                    all_task_b_logits.append(pred_task_b.detach().cpu())
                    all_task_b_targets.append(truth_task_b.cpu())

            steps += 1

        epoch_task_a_logits = torch.cat(all_task_a_logits)
        epoch_task_a_targets = torch.cat(all_task_a_targets)

        if _stream_train_b:
            # Reconstruct the reg train-metric dict from the streamed [N]/[C] accumulators.
            # Byte-identical to compute_classification_metrics(full reg logits, top_k=(3,5))
            # on the C>256 handrolled path: same helpers, same keys, same order. preds=cat of
            # per-batch argmax == full argmax (per-row); rank/hit are per-row; bincounts are
            # additive — so every metric matches the full-logit computation exactly.
            _preds = torch.cat(s1_preds_b)
            _tgts = torch.cat(s1_targets_b)
            _rank = torch.cat(s1_rank_b)
            _accm, _accM, _f1m, _f1w = _handrolled_cls_metrics(_preds, _tgts, task_b_num_classes)
            train_metrics_task_b = {
                "accuracy": _accm,
                "accuracy_macro": _accM,
                "f1": _f1m,
                "f1_weighted": _f1w,
                "mrr": _mrr_from_rank(_rank),
            }
            for _k in _S1_TOPK:
                train_metrics_task_b[f"top{_k}_acc"] = torch.cat(s1_hit_b[_k]).float().mean().item()
                train_metrics_task_b[f"ndcg_{_k}"] = _ndcg_from_rank(_rank, _k)
        else:
            epoch_task_b_logits = torch.cat(all_task_b_logits)
            epoch_task_b_targets = torch.cat(all_task_b_targets)
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
        # F50 F63 — α trajectory logging. The next_getnext{,_hard} heads
        # carry a learnable scalar `alpha` that scales the graph prior
        # (`stan_logits + α · log_T[last_region]`). T3 hypothesised α
        # growth is the temporal mechanism behind STL's late reg-best
        # epoch (16-20). Logging it per-epoch lets us correlate growth
        # rate vs reg metric trajectory directly. Silent no-op for heads
        # without an `alpha` attribute.
        next_head = getattr(model, "next_poi", None)
        head_alpha = getattr(next_head, "alpha", None)
        if isinstance(head_alpha, torch.Tensor) and head_alpha.numel() == 1:
            diagnostic_payload["head_alpha"] = float(head_alpha.detach().cpu())

        # 2026-06-12 (HANDOFF_AUDIT X3 / CODE_AUDIT P1-C) — β trajectory logging.
        # The dual-tower head fuses the shared pathway via
        # `priv_feat + β · aux_proj(shared_feat)` (next_stan_flow_dualtower,
        # β init 0.1). β sits in the reg param group and is weight-decayed at
        # wd=0.05 (only α is peeled into the zero-WD group) — the same AdamW
        # decay-to-zero mechanism F50 diagnosed for α. Logging β per epoch makes
        # that drift visible (was invisible — only head_alpha was logged). Silent
        # no-op for heads without a `beta` attribute.
        head_beta = getattr(next_head, "beta", None)
        if isinstance(head_beta, torch.Tensor) and head_beta.numel() == 1:
            diagnostic_payload["head_beta"] = float(head_beta.detach().cpu())

        # Idea 2 — input-dependent aux-fusion gate γ (fusion_mode=aux_gated).
        # Mean γ over the last batch; logged so the "aux_gated≡aux" null is
        # checkable (did the input-conditioned gate move off its ≈0.12 init?).
        _aux_gamma = getattr(next_head, "last_aux_gamma", None)
        if _aux_gamma is not None:
            diagnostic_payload["aux_gamma"] = float(_aux_gamma)

        # Conditional coupling — mean ‖cat-condition contribution‖ into the reg
        # feature (0 at init via zero-init cond_proj; >0 means the reg head
        # learned to use the predicted category — the C28 fires-check).
        _cond_norm = getattr(next_head, "last_cond_norm", None)
        if _cond_norm is not None:
            diagnostic_payload["cond_norm"] = float(_cond_norm)

        # R5 — per-instance log_T-KD gate spread (0 ⇒ uniform ≡ global-W; >0 ⇒ live).
        _r5_gate_std = getattr(model, "_r5_gate_std", None)
        if _r5_gate_std is not None:
            diagnostic_payload["r5_gate_std"] = float(_r5_gate_std)

        # R10 — trained GRM gate trajectory. Mean γ_a/γ_b over the cross-attn
        # blocks (last batch of the epoch). init≈0.88; logged so the "GRM≡G" null
        # is checkable (did the gate move, and where did it settle?).
        _blocks = getattr(model, "crossattn_blocks", None)
        if _blocks is not None:
            _ga = [b.last_gamma_a for b in _blocks if getattr(b, "last_gamma_a", None) is not None]
            _gb = [b.last_gamma_b for b in _blocks if getattr(b, "last_gamma_b", None) is not None]
            if _ga:
                diagnostic_payload["grm_gamma_a"] = sum(_ga) / len(_ga)
            if _gb:
                diagnostic_payload["grm_gamma_b"] = sum(_gb) / len(_gb)

        # F50 D5 — encoder trajectory diagnostic. Frobenius norm of current
        # encoder weights, drift from epoch-0 init, and step drift from the
        # previous epoch. Cheap (one cat + two diffs of an O(64×256×L) flat
        # vector). The smoking-gun comparison is reg-side drift saturating
        # earlier than cat-side under joint training.
        next_enc_now = _flatten_encoder(getattr(model, "next_encoder", None))
        if next_enc_now is not None and next_enc_init is not None:
            diagnostic_payload["reg_encoder_l2norm"] = float(next_enc_now.norm().cpu())
            diagnostic_payload["reg_encoder_drift_from_init"] = float(
                (next_enc_now - next_enc_init).norm().cpu()
            )
            if next_enc_prev is not None:
                diagnostic_payload["reg_encoder_step_drift"] = float(
                    (next_enc_now - next_enc_prev).norm().cpu()
                )
            next_enc_prev = next_enc_now
        cat_enc_now = _flatten_encoder(getattr(model, "category_encoder", None))
        if cat_enc_now is not None and cat_enc_init is not None:
            diagnostic_payload["cat_encoder_l2norm"] = float(cat_enc_now.norm().cpu())
            diagnostic_payload["cat_encoder_drift_from_init"] = float(
                (cat_enc_now - cat_enc_init).norm().cpu()
            )
            if cat_enc_prev is not None:
                diagnostic_payload["cat_encoder_step_drift"] = float(
                    (cat_enc_now - cat_enc_prev).norm().cpu()
                )
            cat_enc_prev = cat_enc_now

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
            # Geometric mean of per-head lifts (scale-coherent; the interim
            # 2026-04-15 monitor — superseded as default by geom_simple below).
            joint_geom_lift = math.sqrt(task_b_lift * task_a_lift)
            # Arithmetic mean kept for back-compat + side-by-side reporting
            # (paper can show it as the "naive" number in appendix).
            joint_arith_lift = 0.5 * (task_b_lift + task_a_lift)
            # C21 (mtl-protocol-fix, 2026-05-24) — the VALIDATED, headline-aligned
            # joint selector and the code DEFAULT. Geometric mean of the metrics
            # each head is actually REPORTED on: category macro-F1 (``f1``) and
            # region Acc@10 (``top10_acc_indist``). Both are bounded [0,1] and on
            # comparable scales, so NO majority normalization is applied (the lift
            # form was an acc1-only workaround; reusing majority_fraction as an
            # Acc@10 baseline would be cardinality-wrong). For non-region task_b
            # (no top10 key, e.g. the {category,next} preset) reg falls back to its
            # own ``f1`` → sqrt(cat_f1 * task_b_f1). Recovered +5.62pp deployable
            # reg Acc@10 vs the v11 joint_score at FL multi-seed (docs/CONCERNS §C21).
            reg_acc10_val = val_metrics_task_b.get('top10_acc_indist', f1_val_task_b)
            joint_geom_simple = math.sqrt(
                max(f1_val_task_a, 0.0) * max(reg_acc10_val, 0.0)
            )
            # Select the scalar that gates the single joint checkpoint
            # (``model_task.best``). Default geom_simple; legacy/interim opt-in.
            if checkpoint_selector == "joint_f1_mean":
                joint_selector_value = joint_score          # v11 paper-canon LEGACY (broken)
            elif checkpoint_selector == "geom_lift":
                joint_selector_value = joint_geom_lift       # interim acc1-lift form
            else:                                            # "geom_simple" (default, correct)
                joint_selector_value = joint_geom_simple
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
            # AUDIT-C2: read the per-task monitor key so the improvement
            # check matches whichever metric the BestModelTracker is
            # actually watching (F1, accuracy, mrr, ...). Pre-C2 this
            # hardcoded F1 even when the tracker watched accuracy →
            # state_dict was occasionally not produced when it should
            # have been. Falls back to F1 for legacy paths.
            _mon_b = fold_history.task(task_b_name).best.monitor
            _mon_a = fold_history.task(task_a_name).best.monitor
            _val_b_mon = val_metrics_task_b.get(_mon_b, f1_val_task_b)
            _val_a_mon = val_metrics_task_a.get(_mon_a, f1_val_task_a)
            task_b_improved = _val_b_mon > fold_history.task(task_b_name).best.best_value
            task_a_improved = _val_a_mon > fold_history.task(task_a_name).best.best_value
            prev_joint_best = fold_history.model_task.best.best_value if fold_history.model_task.best.best_epoch >= 0 else -1.0
            # C21: gate the joint checkpoint on the configured selector
            # (default geom_simple), respecting min_best_epoch (skip the
            # init-artifact window, as the per-task trackers do).
            joint_eligible = epoch_idx >= joint_min_epoch
            joint_improved = joint_eligible and (joint_selector_value > prev_joint_best)
            state = model.state_dict() if (joint_improved or task_b_improved or task_a_improved) else None

            # Per-task val losses now come from evaluate_model() inside the
            # metric dicts, so log_val no longer needs a hand-wired scalar.
            # model_task keeps the combined MTL loss; the f1=0/accuracy=0
            # placeholders stay as stable schema on the MTL summary store.
            fold_history.model_task.log_val(
                loss=loss_val,
                f1=joint_selector_value,  # C21: the configured joint selector (default geom_simple)
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

            # substrate-protocol-cleanup Tier C1 — three-snapshot routing.
            # Update the side-channel ``MultiTaskBestTracker`` in lockstep
            # with the existing single-best path. Uses the same per-epoch
            # state_dict so each slot's snapshot is internally consistent
            # (head + backbone come from the same epoch). The cat slot
            # tracks task_a's monitored metric (F1 by default); reg slot
            # tracks task_b's monitored metric (typically Acc@10 via
            # ``accuracy`` key on check2HGI); joint slot tracks
            # ``joint_geom_lift`` — the scale-coherent geometric-mean
            # selector. Single-best ``model_task.best`` is left untouched.
            if task_best_tracker is not None:
                full_state = state if state is not None else model.state_dict()
                cat_metric_val = val_metrics_task_a.get(
                    task_best_tracker.cat_best.monitor, f1_val_task_a,
                )
                reg_metric_val = val_metrics_task_b.get(
                    task_best_tracker.reg_best.monitor, f1_val_task_b,
                )
                task_best_tracker.update(
                    epoch=fold_history.task(task_b_name).val.num_epochs() - 1,
                    model_state=full_state,
                    cat_metric=cat_metric_val,
                    reg_metric=reg_metric_val,
                    joint_metric=joint_selector_value,  # C21: matches the primary selector (default geom_simple)
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
                "val_joint_geom_lift": joint_geom_lift,    # = geometric mean of acc1-lifts (interim 2026-04-15)
                "val_joint_geom_simple": joint_geom_simple,  # = sqrt(cat_f1 * reg_top10) — C21 DEFAULT selector
                # Alias points at the ACTIVE default selector (geom_simple) so a
                # monitor="val_joint_lift" callback tracks the shipped selector:
                "val_joint_lift": joint_geom_simple,
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

        # AUDIT-C4 fix — per-fold transition prior. When
        # ``config.per_fold_transition_dir`` is set, swap the static
        # ``transition_path`` in task_b.head_params for the fold-specific
        # file ``region_transition_log_seed{S}_fold{N}.pt``. The seed
        # MUST match the trainer's ``--seed S`` because the per-fold
        # log_T is built from train rows under the same fold split;
        # using a file built at a different seed silently leaks
        # ~80% of val transitions into the prior at every other seed
        # (caught 2026-04-30 — F51 multi-seed sweep with seed=42
        # log_T applied at seeds 0/1/7/100). N is 1-indexed because
        # that's what ``compute_region_transition.py --per-fold``
        # writes. ``i_fold`` here is 0-indexed (FoldCreator dict keys).
        # Default None preserves the legacy single-prior behaviour, so
        # this is a no-op for the running tier-A queue.
        per_fold_dir = getattr(config, "per_fold_transition_dir", None)
        per_fold_model_params = config.model_params
        if per_fold_dir is not None:
            import dataclasses as _dataclasses
            from pathlib import Path as _Path
            ts = config.model_params.get("task_set")
            if ts is not None and getattr(ts, "task_b", None) is not None:
                seed = int(getattr(config, "seed", 42))
                pf_path = (
                    _Path(per_fold_dir)
                    / f"region_transition_log_seed{seed}_fold{i_fold + 1}.pt"
                )
                if not pf_path.exists():
                    legacy_path = (
                        _Path(per_fold_dir)
                        / f"region_transition_log_fold{i_fold + 1}.pt"
                    )
                    if legacy_path.exists():
                        raise FileNotFoundError(
                            f"per-fold log_T at expected seed-tagged path "
                            f"{pf_path} missing. A legacy unseeded file at "
                            f"{legacy_path} was found, but using it would "
                            f"leak val transitions if its build seed != "
                            f"current --seed {seed}. Migrate by either "
                            f"renaming the legacy file (if you know the seed "
                            f"it was built at) or rebuilding: python "
                            f"scripts/compute_region_transition.py --state "
                            f"{config.state} --per-fold --seed {seed}"
                        )
                    raise FileNotFoundError(
                        f"per_fold_transition_dir set but {pf_path} missing. "
                        f"Build with: python scripts/compute_region_transition.py "
                        f"--state {config.state} --per-fold --seed {seed}"
                    )
                # C22 stale-log_T guard (substrate-protocol-cleanup Tier C4,
                # added 2026-05-28): refuse to start if log_T mtime predates
                # the substrate parquet it was built from. Previously
                # runbook-enforced only; silently survived regens and
                # inflated reg Acc@10 by +8 to +12 pp (mtl_protocol_fix
                # Phase 2 P5 FL seed=42 case).
                parquet_path = _Path(per_fold_dir) / "input" / "next_region.parquet"
                if parquet_path.exists():
                    if pf_path.stat().st_mtime < parquet_path.stat().st_mtime:
                        raise ValueError(
                            f"Stale per-fold log_T detected: {pf_path} mtime "
                            f"is older than {parquet_path} mtime. The substrate "
                            f"parquet has been regenerated since this log_T was "
                            f"built; running would silently leak ~+8 to +12 pp "
                            f"into reg Acc@10. Rebuild: python "
                            f"scripts/compute_region_transition.py --state "
                            f"{config.state} --per-fold --seed {seed}"
                        )
                # 2026-05-15: hard-fail when the per-fold log_T's ``n_splits``
                # does not match the trainer's ``config.k_folds``. The
                # ``--folds N`` flag overrides ``config.k_folds`` to
                # ``max(2, N)``, so a 1-fold smoke against a 5-fold-built
                # log_T silently leaks ~30-40% of val transitions into the
                # prior (the α scalar amplifies this through training,
                # inflating reg ``top10_acc_indist`` by 13-23 pp). See
                # ``docs/studies/mtl-exploration/LEAK_BLAST_RADIUS_AUDIT.md``
                # for the discovery + per-state magnitudes, and
                # ``docs/findings/MTL_FLAWS_AND_FIXES.md §2.13`` for the
                # catalog entry. Files written by post-2026-05-15
                # ``compute_region_transition.py`` stash ``n_splits`` in
                # the payload; legacy files (pre-2026-05-15) lack the key
                # and are accepted only at the canonical n_splits=5
                # because that's the historical default they were built
                # under.
                trainer_n_splits = int(config.k_folds)
                pf_payload = torch.load(pf_path, map_location="cpu", weights_only=False)
                pf_n_splits = (
                    pf_payload.get("n_splits") if isinstance(pf_payload, dict) else None
                )
                if pf_n_splits is None:
                    if trainer_n_splits != 5:
                        raise ValueError(
                            f"Per-fold log_T at {pf_path} is a legacy file "
                            f"(no 'n_splits' field in payload) and the trainer "
                            f"is running at n_splits={trainer_n_splits} (not the "
                            f"canonical 5). Legacy files were always built at "
                            f"n_splits=5; running at a different n_splits "
                            f"silently leaks ~30-80% of val transitions into "
                            f"the prior (depending on overlap). Rebuild the "
                            f"prior at the trainer's n_splits: python "
                            f"scripts/compute_region_transition.py --state "
                            f"{config.state} --per-fold --n-splits "
                            f"{trainer_n_splits} --seed {seed}"
                        )
                    logger.warning(
                        "[C4 per-fold log_T] legacy file %s has no n_splits "
                        "field; trainer is at canonical n_splits=5 so "
                        "accepting; rebuild to silence this warning.",
                        pf_path,
                    )
                elif int(pf_n_splits) != trainer_n_splits:
                    raise ValueError(
                        f"Per-fold log_T at {pf_path} was built with "
                        f"n_splits={pf_n_splits}, but the trainer is running "
                        f"at n_splits={trainer_n_splits} (set via --folds; "
                        f"max(2, N)). Mismatch silently leaks val transitions "
                        f"into the prior. Rebuild for the trainer's n_splits: "
                        f"python scripts/compute_region_transition.py --state "
                        f"{config.state} --per-fold --n-splits "
                        f"{trainer_n_splits} --seed {seed}"
                    )
                tb_head_params = dict(ts.task_b.head_params or {})
                tb_head_params["transition_path"] = str(pf_path)

                # R1 (mtl_frontier) — per-fold log_C co-location prior, swapped
                # in beside log_T when --log-c-kd-weight > 0. Same dir, same
                # seed/fold/n_splits + stale-mtime leak guards as log_T (a leak
                # here would contaminate the reg KD teacher exactly as a stale
                # log_T contaminates the Markov prior).
                if (float(getattr(config, "log_c_kd_weight", 0.0) or 0.0) > 0.0
                        or float(getattr(config, "cat_kd_weight", 0.0) or 0.0) > 0.0):
                    pc_path = (
                        _Path(per_fold_dir)
                        / f"region_colocation_log_seed{seed}_fold{i_fold + 1}.pt"
                    )
                    if not pc_path.exists():
                        raise FileNotFoundError(
                            f"--log-c-kd-weight>0 but per-fold log_C {pc_path} "
                            f"missing. Build: python "
                            f"scripts/compute_region_colocation.py --state "
                            f"{config.state} --per-fold --seed {seed} --engine "
                            f"{getattr(config, 'embedding_engine', '<engine>')}"
                        )
                    if parquet_path.exists() and (
                        pc_path.stat().st_mtime < parquet_path.stat().st_mtime
                    ):
                        raise ValueError(
                            f"Stale per-fold log_C: {pc_path} mtime predates "
                            f"{parquet_path}; rebuild via "
                            f"scripts/compute_region_colocation.py --per-fold "
                            f"--seed {seed}."
                        )
                    pc_payload = torch.load(pc_path, map_location="cpu", weights_only=False)
                    pc_n_splits = (
                        pc_payload.get("n_splits") if isinstance(pc_payload, dict) else None
                    )
                    if pc_n_splits is not None and int(pc_n_splits) != trainer_n_splits:
                        raise ValueError(
                            f"Per-fold log_C at {pc_path} built with "
                            f"n_splits={pc_n_splits} != trainer {trainer_n_splits}; "
                            f"rebuild (leaks val co-locations into the KD teacher)."
                        )
                    pc_seed = pc_payload.get("seed") if isinstance(pc_payload, dict) else None
                    if pc_seed is not None and int(pc_seed) != seed:
                        raise ValueError(
                            f"Per-fold log_C at {pc_path} built at seed={pc_seed} "
                            f"!= trainer seed={seed}."
                        )
                    tb_head_params["colocation_path"] = str(pc_path)
                    logger.info(
                        "[R1 per-fold log_C] fold %d seed %d using %s",
                        i_fold + 1, seed, pc_path,
                    )

                new_task_b = _dataclasses.replace(ts.task_b, head_params=tb_head_params)
                new_task_set = _dataclasses.replace(ts, task_b=new_task_b)
                per_fold_model_params = dict(config.model_params)
                per_fold_model_params["task_set"] = new_task_set
                logger.info(
                    "[C4 per-fold log_T] fold %d seed %d using %s",
                    i_fold + 1, seed, pf_path,
                )

        # Initialize model via registry
        model = create_model(config.model_name, **per_fold_model_params).to(DEVICE)
        if config.use_torch_compile and DEVICE.type == 'cuda':
            # Opt-in compile tuning (default unchanged → eager-equivalent compile).
            # MTL_COMPILE_DYNAMIC=1 → one symbolic-shape graph instead of recompiling
            # per batch shape (the mixed-batch zip_longest_cycle + partial-last-batch
            # produce many shapes → a recompile each → the ~32-min FL warmup). This
            # collapses that storm. MTL_COMPILE_MODE overrides the inductor mode.
            # Pair with a persistent shared TORCHINDUCTOR_CACHE_DIR across board cells
            # so the (one-time) compile is reused for every seed/state → ~0 warmup.
            import os as _os
            # Raise the dynamo recompile cache limit (default 8). The MTL forward is
            # compiled in a TRAIN (grad) and an EVAL (no-grad) variant — genuinely
            # different graphs — plus a couple of shape variants; that can exceed 8
            # and make dynamo silently FALL BACK TO EAGER (catastrophic — the original
            # "minutes-long compiled fold" was this fallback, not a real warmup). A
            # higher limit keeps every variant compiled. Pure safety: no numeric change.
            try:
                import torch._dynamo as _dyn
                _lim = int(_os.environ.get("MTL_COMPILE_CACHE_LIMIT", "64"))
                _dyn.config.cache_size_limit = max(_dyn.config.cache_size_limit, _lim)
                if hasattr(_dyn.config, "recompile_limit"):
                    _dyn.config.recompile_limit = max(_dyn.config.recompile_limit, _lim)
            except Exception:
                pass
            _ckw = {}
            if _os.environ.get("MTL_COMPILE_DYNAMIC") == "1":
                _ckw["dynamic"] = True
            _cmode = _os.environ.get("MTL_COMPILE_MODE")
            if _cmode:
                _ckw["mode"] = _cmode
            model = torch.compile(model, **_ckw)

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
                alpha_no_weight_decay=bool(getattr(config, "alpha_no_weight_decay", False)),
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
            reg_head_warmup_decay_peak_mult=getattr(
                config, "reg_head_warmup_decay_peak_mult", 10.0),
            reg_head_warmup_decay_warmup_epochs=getattr(
                config, "reg_head_warmup_decay_warmup_epochs", 5),
            reg_head_warmup_decay_plateau_epochs=getattr(
                config, "reg_head_warmup_decay_plateau_epochs", 15),
            eta_min=float(getattr(config, "eta_min", 0.0)),
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

        # C25 (2026-06-05) — PER-TASK class weighting. The reg head (task_b,
        # next_criterion) is reported by top-K Acc@10, which class-balancing HURTS;
        # the cat head (task_a, category_criterion) is reported by macro-F1, which
        # balancing HELPS. ``use_class_weights_{reg,cat}`` override the legacy
        # single ``use_class_weights``; ``None`` inherits it (back-compat).
        _base_cw = bool(getattr(config, "use_class_weights", False))
        _cw_reg_override = getattr(config, "use_class_weights_reg", None)
        _cw_cat_override = getattr(config, "use_class_weights_cat", None)
        _use_cw_reg = _base_cw if _cw_reg_override is None else bool(_cw_reg_override)
        _use_cw_cat = _base_cw if _cw_cat_override is None else bool(_cw_cat_override)
        next_criterion = CrossEntropyLoss(
            reduction='mean',
            weight=alpha_next if _use_cw_reg else None,
        )
        # T2V.7 (2026-06-06) — CAT loss-calibration lever for MTL. When
        # ``config.loss_calibration`` is non-empty (set by --logit-adjust-tau /
        # --focal-gamma / --cat-label-smoothing / --tail-loss), build the CAT
        # criterion via the train-only-stats calibrated loss (Menon ICLR'21
        # logit-adjustment etc.) — the SAME lever the (c) STL cat ceiling used
        # (la τ=0.5). Leak guard: ``dataloader_category.train.y`` is the TRAIN
        # split. Calibration is the cat-loss axis; it is mutually exclusive with
        # cat class-weighting (the T1.4 finding: logit-adjust ALONE wins; stacking
        # with class weights over-corrects) → when calibrated, cat is NOT also
        # class-weighted. Empty dict (the default) keeps the plain-CE path
        # bit-identical, so non-calibrated runs are unaffected.
        _cat_lc = dict(getattr(config, "loss_calibration", {}) or {})
        if _cat_lc:
            from losses.calibrated import build_calibrated_loss
            category_criterion = build_calibrated_loss(
                task_a_num_classes,
                dataloader_category.train.y,
                label_smoothing=_cat_lc.get("label_smoothing", 0.0),
                focal_gamma=_cat_lc.get("focal_gamma", 0.0),
                logit_adjust_tau=_cat_lc.get("logit_adjust_tau", 0.0),
                tail_mode=_cat_lc.get("tail_mode", None),
                cb_beta=_cat_lc.get("cb_beta", 0.999),
                ldam_max_m=_cat_lc.get("ldam_max_m", 0.5),
                ldam_scale=_cat_lc.get("ldam_scale", 30.0),
                device=DEVICE,
            )
        else:
            category_criterion = CrossEntropyLoss(
                reduction='mean',
                weight=alpha_cat if _use_cw_cat else None,
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

        # substrate-protocol-cleanup Tier C1 — opt-in three-snapshot
        # routing. Per-fold MultiTaskBestTracker; cat slot watches the
        # task_a monitor, reg slot watches task_b monitor (typically
        # ``accuracy`` for high-cardinality region heads), joint slot
        # watches the geometric-mean joint lift. The trainer updates all
        # three slots in lockstep each epoch alongside the existing
        # single-best path.
        task_best_tracker = None
        task_best_save_dir = None
        if getattr(config, "save_task_best_snapshots", False):
            from tracking.best_tracker import MultiTaskBestTracker
            _curr_fold_hist = history.get_curr_fold()
            _cat_mon = _curr_fold_hist.task(task_a_name).best.monitor
            _reg_mon = _curr_fold_hist.task(task_b_name).best.monitor
            task_best_tracker = MultiTaskBestTracker(
                cat_monitor=_cat_mon,
                reg_monitor=_reg_mon,
                joint_monitor="joint_geom_simple",  # C21: matches the default primary selector
                mode="max",
                min_epoch=int(getattr(config, "min_best_epoch", 0) or 0),
            )
            if results_path is not None:
                task_best_save_dir = (
                    Path(results_path) / "task_best_snapshots"
                )
                task_best_save_dir.mkdir(parents=True, exist_ok=True)

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
            alpha_frozen_until_epoch=getattr(config, "alpha_frozen_until_epoch", None),
            cat_specific_parameters=(
                list(model.cat_specific_parameters())
                if hasattr(model, "cat_specific_parameters") else None
            ),
            reg_specific_parameters=(
                list(model.reg_specific_parameters())
                if hasattr(model, "reg_specific_parameters") else None
            ),
            joint_loader_strategy=getattr(
                config, "joint_loader_strategy", "max_size_cycle"),
            reg_freeze_at_epoch=getattr(config, "reg_freeze_at_epoch", None),
            task_best_tracker=task_best_tracker,
            task_best_save_dir=task_best_save_dir,
            log_t_kd_weight=float(getattr(config, "log_t_kd_weight", 0.0) or 0.0),
            log_t_kd_tau=float(getattr(config, "log_t_kd_tau", 1.0) or 1.0),
            log_t_kd_gate=str(getattr(config, "log_t_kd_gate", "none") or "none"),
            log_c_kd_weight=float(getattr(config, "log_c_kd_weight", 0.0) or 0.0),
            log_c_kd_tau=float(getattr(config, "log_c_kd_tau", 1.0) or 1.0),
            log_c_kd_warmup_epochs=int(getattr(config, "log_c_kd_warmup_epochs", 0) or 0),
            log_c_kd_ec_lambda=float(getattr(config, "log_c_kd_ec_lambda", 0.0) or 0.0),
            cat_kd_weight=float(getattr(config, "cat_kd_weight", 0.0) or 0.0),
            cat_kd_tau=float(getattr(config, "cat_kd_tau", 1.0) or 1.0),
            loss_scale_norm=bool(getattr(config, "loss_scale_norm", False)),
            checkpoint_selector=str(getattr(config, "checkpoint_selector", "geom_simple")),
            joint_min_epoch=int(getattr(config, "min_best_epoch", 0) or 0),
            joint_train_loader=getattr(dataloader, "joint_train_loader", None),
        )

        # substrate-protocol-cleanup Tier C1 — write the three best
        # snapshots to disk at fold end. Slot names match the CLI flag
        # documentation: ``fold{N}_cat_best.pt``, ``fold{N}_reg_best.pt``,
        # ``fold{N}_joint_best.pt``. ``fold_idx`` is 0-indexed for the
        # dict iteration but the on-disk name is 1-indexed for parity
        # with the per-fold log_T filename convention
        # (``region_transition_log_seed{S}_fold{N}.pt`` is also 1-indexed).
        if task_best_tracker is not None and task_best_save_dir is not None:
            snaps = task_best_tracker.snapshots()
            for slot, state_dict in snaps.items():
                snap_path = (
                    task_best_save_dir / f"fold{fold_idx + 1}_{slot}_best.pt"
                )
                torch.save(state_dict, snap_path)
                logger.info(
                    "[C1 task-best snapshots] fold %d slot=%s saved to %s "
                    "(best_value=%.4f at epoch %d)",
                    fold_idx + 1, slot, snap_path,
                    getattr(task_best_tracker, f"{slot}_best").best_value,
                    getattr(task_best_tracker, f"{slot}_best").best_epoch,
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
