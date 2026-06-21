import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

from tracking.metrics import (
    compute_classification_metrics,
    _handrolled_cls_metrics,
    _rank_of_target,
    _mrr_from_rank,
    _ndcg_from_rank,
)
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
    # Vectorised membership test: building the mask via Python iteration with
    # `.item()` per target forces an N-element host↔device sync — on FL val
    # (~31K rows) that's ~31K syncs per validation pass, ≈1–2s wasted per
    # epoch × 50 epochs × 5 folds. `torch.isin` does it in one kernel.
    train_labels_t = torch.as_tensor(
        sorted(train_label_set), dtype=targets.dtype, device=targets.device,
    )
    in_dist_mask = torch.isin(targets, train_labels_t)
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

    # S2 (perf-audit) — CHUNKED/STREAMING val metric for the high-cardinality reg head.
    # The full path cats the whole val epoch's [N, n_regions] reg logits ON GPU and runs
    # compute_classification_metrics + _ood_restricted_topk over it — which OOMs the A40 at
    # CA/TX scale (4.7k-8.5k regions). Every val reg metric (incl. the SCORED top10_acc_indist)
    # is a per-ROW reduction or additive per-CLASS count, so we stream per-batch reductions on
    # GPU (matching the canonical fp16-CUDA topk tie-breaking — NEVER CPU-moved) and assemble
    # the IDENTICAL metric dicts + ood dict from the accumulators. Reg only; cat (C=7) keeps the
    # full path. SCORED PATH → behind MTL_CHUNK_VAL_METRIC (default OFF) until A40 A/B-verified.
    _nc_b_gate = task_b_num_classes if task_b_num_classes is not None else (
        num_classes if num_classes is not None else getattr(model, "num_classes", 0)
    )
    _s2_env = os.environ.get("MTL_CHUNK_VAL_METRIC", "").strip() in ("1", "true", "True")
    # FOOTGUN GUARD (2026-06-20): the full path cats the WHOLE val epoch's
    # [N_val, n_regions] reg logits on GPU. At overlap scale (8.5× rows) with a
    # high region count this is ~20 GB for CA (586k × 8501 × 4 B) and instantly
    # OOMs the A40 mid-validation — even after the host-RAM fix. S2 is byte-identical
    # (assembles the same metric dicts from streamed per-batch reductions), so when
    # the full logit would exceed a GPU-safe budget we AUTO-enable chunking (with a
    # one-time WARN) instead of OOMing. Non-overlap region MTL (small N_val ⇒ ~2 GB
    # at CA) stays on the full path untouched, preserving the frozen §0.1 numbers.
    _full_logit_gb = 0.0
    try:
        _n_val = len(dataloaders[0].dataset)
        _full_logit_gb = _n_val * int(_nc_b_gate or 0) * 4 / (1024 ** 3)
    except Exception:
        pass
    _S2_BUDGET_GB = float(os.environ.get("MTL_S2_AUTO_BUDGET_GB", "4"))
    _auto_chunk = _full_logit_gb > _S2_BUDGET_GB
    _chunk_val = (
        (_s2_env or _auto_chunk)
        and _nc_b_gate is not None and _nc_b_gate > 256  # only the handrolled path
    )
    if _auto_chunk and not _s2_env:
        logger.warning(
            "[S2 auto] full val reg-logit ≈ %.1f GB (N_val=%d × C=%d) exceeds the "
            "%.1f GB budget — auto-enabling the chunked val metric (byte-identical) "
            "to avoid a GPU OOM. Set MTL_CHUNK_VAL_METRIC=1 to silence, or "
            "MTL_S2_AUTO_BUDGET_GB to tune.",
            _full_logit_gb, locals().get("_n_val", -1), int(_nc_b_gate or 0), _S2_BUDGET_GB,
        )
    _S2_KS = (1, 3, 5, 10)  # union of metrics_next top_k=(3,5) and ood ks=(1,5,10)
    sv_preds, sv_tgts, sv_rank = [], [], []
    sv_hit = {k: [] for k in _S2_KS}

    # 2026-06-12 (HANDOFF_AUDIT X4 / CODE_AUDIT P1-D) — eval-precision escape hatch.
    # MTL eval autocasts fp16 on CUDA unconditionally, while the p1-STL ceiling
    # harness evaluates in fp32. Over ~5-9k region logits, fp16 quantisation
    # creates more exact ties → _rank_of_target (strictly-higher count) scores
    # MTL ranks tie-optimistically vs the fp32 ceiling. The headline Δreg is
    # decided at −0.09…−0.31pp, within that delta. Set MTL_DISABLE_AMP_EVAL=1
    # (or the training hatch MTL_DISABLE_AMP=1) to force fp32 eval and measure the
    # precision-clean number. Default (unset) keeps canonical fp16 eval untouched.
    _disable_amp_eval = (
        os.environ.get("MTL_DISABLE_AMP_EVAL") == "1"
        or os.environ.get("MTL_DISABLE_AMP") == "1"
    )
    _autocast_ctx = (
        torch.autocast(device.type, dtype=torch.float16)
        if device.type == 'cuda' and not _disable_amp_eval
        else contextlib.nullcontext()
    )

    # 2026-06-12 (HANDOFF_AUDIT X1 / CODE_AUDIT P0-A) — cross-attn pairing roll probe.
    # The two MTL train loaders draw independent shuffles → the cross-attn block mixes
    # row i (cat) ↔ row i (reg) of RANDOMLY-paired windows at train time, while val is
    # aligned. Probe: roll the task-b (reg / `x_next`) stream by 1 along the batch dim at
    # EVAL only → cat row i now cross-attends reg row i−1. If cat-F1 is unchanged
    # (Δ≈0), the model ignores per-sample pairing → "K/V mixing is dead" is confirmed
    # clean. (reg metric becomes meaningless under the roll — read cat-F1 only.) Gate:
    # MTL_ROLL_TASKB_EVAL=1. Default unset → no roll.
    _roll_taskb = os.environ.get("MTL_ROLL_TASKB_EVAL") == "1"

    for data_next, data_cat in zip_longest_cycle(*dataloaders):
        x_next, y_next = data_next
        if x_next.device != device:
            x_next = x_next.to(device, non_blocking=True)
            y_next = y_next.to(device, non_blocking=True)
        x_category, y_category = data_cat
        if x_category.device != device:
            x_category = x_category.to(device, non_blocking=True)
            y_category = y_category.to(device, non_blocking=True)

        if _roll_taskb and x_next.shape[0] > 1:
            x_next = torch.roll(x_next, shifts=1, dims=0)

        with _autocast_ctx:
            cat_out, next_out = model((x_category, x_next))
            next_loss = next_criterion(next_out, y_next)
            category_loss = category_criterion(cat_out, y_category)
        # Accumulate on-device; single .item() sync after the loop.
        next_running_loss += next_loss.detach()
        category_running_loss += category_loss.detach()
        combined_running_loss += (next_loss.detach() + category_loss.detach()) / 2

        # cat (task_a) — low-card, always full path.
        logits_cat_list.append(cat_out.detach())
        truths_cat_list.append(y_category)
        if _chunk_val:
            # reg (task_b) — STREAM per-row reductions ON GPU (canonical fp16 tie-break);
            # discard the [batch, n_regions] logits so the full [N, C] is never materialized.
            _no = next_out.detach()
            sv_preds.append(_no.argmax(dim=-1))
            sv_tgts.append(y_next)
            sv_rank.append(_rank_of_target(_no, y_next))
            for _k in _S2_KS:
                _ke = min(_k, _no.shape[-1])
                sv_hit[_k].append((_no.topk(_ke, dim=-1).indices == y_next.unsqueeze(-1)).any(dim=-1))
            del _no
        else:
            logits_next_list.append(next_out.detach())
            truths_next_list.append(y_next)
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

    if _chunk_val:
        # Reg metrics_next from the streamed accumulators (byte-identical to
        # compute_classification_metrics(full reg logits, top_k=(3,5)) on the C>256
        # handrolled path: same helpers, same keys/order; preds/rank/hit per-row, on-GPU).
        _P = torch.cat(sv_preds)
        _T = torch.cat(sv_tgts)
        _R = torch.cat(sv_rank)
        _HIT = {k: torch.cat(sv_hit[k]) for k in _S2_KS}
        _am, _aM, _fm, _fw = _handrolled_cls_metrics(_P, _T, nc_b)
        metrics_next = {
            "accuracy": _am, "accuracy_macro": _aM, "f1": _fm,
            "f1_weighted": _fw, "mrr": _mrr_from_rank(_R),
        }
        for _k in (3, 5):
            metrics_next[f"top{_k}_acc"] = _HIT[_k].float().mean().item()
            metrics_next[f"ndcg_{_k}"] = _ndcg_from_rank(_R, _k)
    else:
        all_logits_next = torch.cat(logits_next_list)
        all_truths_next = torch.cat(truths_next_list)
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
        if _chunk_val:
            # Byte-identical to _ood_restricted_topk (ks=(1,5,10)) from the same
            # streamed accumulators + the in-dist mask: hit/rank are per-row, so
            # hit_all[mask] == hit on logits[mask]; one .float().mean() (NOT a running
            # int count — that drifts at the last ULP).
            _train_t = torch.as_tensor(sorted(train_labels_b), dtype=_T.dtype, device=_T.device)
            _mask = torch.isin(_T, _train_t)
            _ni = int(_mask.sum().item())
            _noo = len(_T) - _ni
            ood_b = {
                "n_indist": float(_ni), "n_ood": float(_noo),
                "ood_fraction": float(_noo / max(len(_T), 1)),
            }
            if _ni == 0:
                for _k in (1, 5, 10):
                    ood_b[f"top{_k}_acc_indist"] = 0.0
                ood_b["mrr_indist"] = 0.0
            else:
                for _k in (1, 5, 10):
                    ood_b[f"top{_k}_acc_indist"] = float(_HIT[_k][_mask].float().mean().item())
                ood_b["mrr_indist"] = float((1.0 / _R[_mask].float()).mean().item())
        else:
            ood_b = _ood_restricted_topk(all_logits_next, all_truths_next, train_labels_b)
        metrics_next.update(ood_b)
    if train_labels_a is not None:
        ood_a = _ood_restricted_topk(all_logits_category, all_truths_category, train_labels_a)
        metrics_category.update(ood_a)

    return metrics_next, metrics_category, loss_combined
