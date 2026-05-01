"""P1 - Region-head ablation: single-task next_region on Check2HGI.

Tests N head variants at configurable folds x epochs.
Reports Acc@{1,5,10}, MRR, macro-F1 per head.

This version respects each head's module-default hyperparameters; we
only pass `embed_dim`, `num_classes`, and `seq_length` (for the
transformer head). Everything else (hidden_dim, dropout, num_layers)
falls back to the head's __init__ defaults unless explicitly overridden
via --override-hparams. This fixes a sandbagging bug where the earlier
revision forced hidden_dim=emb_dim*2 and dropout=0.1 on every head,
catastrophically mis-tuning the transformer and halving GRU/LSTM capacity.

Usage:
    # E1 fair re-baseline (all 5 heads at per-head defaults)
    python scripts/p1_region_head_ablation.py --folds 1 --epochs 30

    # E2 winner scaling (example)
    python scripts/p1_region_head_ablation.py --folds 1 --epochs 30 \
        --heads next_gru --override-hparams hidden_dim=384 num_layers=3 \
        --label-smoothing 0.1 --input-layernorm

    # E3 transformer rescue (per-head LR via --max-lr-transformer)
    python scripts/p1_region_head_ablation.py --folds 1 --epochs 30 \
        --heads next_mtl --max-lr 5e-4

    # E-region: region-embedding input variant
    python scripts/p1_region_head_ablation.py --folds 1 --epochs 30 \
        --heads next_gru --input-type region

    # E-concat: check-in + region concat input
    python scripts/p1_region_head_ablation.py --folds 1 --epochs 30 \
        --heads next_gru --input-type concat
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle as pkl
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from configs.globals import DEVICE
from configs.model import InputsConfig
from configs.paths import EmbeddingEngine, IoPaths, OUTPUT_DIR
from data.aux_side_channel import AuxPublishingLoader
from data.folds import (
    POIDataset,
    POIDatasetWithAux,
    _convert_to_tensors,
    TaskType,
    load_next_data,
)
from models.registry import create_model
from tracking.metrics import compute_classification_metrics
from utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ALL_HEADS = [
    "next_mtl", "next_gru", "next_lstm", "next_tcn_residual", "next_temporal_cnn",
    "next_stan", "next_getnext", "next_getnext_hard", "next_getnext_hard_hsm",
]
# Per-head default max_lr for OneCycleLR. Transformer converges at much
# lower LRs; RNN/CNN heads tolerate 3e-3. Override via --max-lr.
_HEAD_MAX_LR = {
    "next_mtl": 5e-4,
    "next_gru": 3e-3,
    "next_lstm": 3e-3,
    "next_tcn_residual": 3e-3,
    "next_temporal_cnn": 3e-3,
    "next_stan": 3e-3,
    "next_getnext": 3e-3,
    "next_getnext_hard": 3e-3,
    "next_getnext_hard_hsm": 3e-3,
}

# Heads that require ``last_region_idx`` delivered via aux_side_channel.
# For single-task ablation we wire POIDatasetWithAux + AuxPublishingLoader
# around train/val DataLoaders when the head is in this set.
# Note: next_getnext/next_tgstan/next_stahyper also consume log_T but
# read last-step embedding from x[..., last_idx] internally — they
# don't need the aux side channel.
_HEADS_REQUIRING_AUX = {"next_getnext_hard", "next_getnext_hard_hsm"}


def _load_region_embeddings(state: str, source: str = "check2hgi") -> tuple[np.ndarray, int]:
    """Load region_embeddings.parquet as [n_regions, D] array.

    ``source`` switches which engine's region embeddings are consulted.
    Only the region-embedding *lookup* changes; region labels, placeid→
    region map, and sequences are always taken from check2hgi for a
    like-for-like task definition (CH15 / P1.5).
    """
    if source == "check2hgi":
        path = IoPaths.CHECK2HGI.get_state_dir(state) / "region_embeddings.parquet"
    elif source == "hgi":
        path = IoPaths.HGI.get_state_dir(state) / "region_embeddings.parquet"
    else:
        raise ValueError(f"Unknown region-emb source: {source} (expected 'check2hgi' or 'hgi').")
    df = pd.read_parquet(path)
    emb_cols = [c for c in df.columns if c.startswith("reg_")]
    df = df.sort_values("region_id").reset_index(drop=True)
    emb = df[emb_cols].to_numpy(dtype=np.float32)
    return emb, emb.shape[1]


def _load_graph_maps(state: str) -> tuple[dict, np.ndarray]:
    graph_path = IoPaths.CHECK2HGI.get_graph_data_file(state)
    with open(graph_path, "rb") as f:
        graph = pkl.load(f)
    placeid_to_idx = graph["placeid_to_idx"]
    poi_to_region = graph["poi_to_region"]
    if hasattr(poi_to_region, "cpu"):
        poi_to_region = poi_to_region.cpu().numpy()
    return placeid_to_idx, np.asarray(poi_to_region, dtype=np.int64)


def _build_region_sequence_tensor(
    state: str,
    region_emb_dim: int,
    region_emb_source: str = "check2hgi",
) -> torch.Tensor:
    """Build [N, 9, D] tensor where each step is the region embedding of
    the region the user was in at that check-in.

    Padded positions (poi_k == -1) map to the zero vector so the heads'
    `x.abs().sum(dim=-1) == 0` padding mask logic works identically to
    the check-in path.

    ``region_emb_source`` selects which engine's region embeddings are
    used for the lookup (see ``_load_region_embeddings``). Default
    ``check2hgi`` preserves prior behaviour; ``hgi`` is used for the
    P1.5 substrate comparison (CH15).
    """
    seq_path = IoPaths.CHECK2HGI.get_temp_dir(state) / "sequences_next.parquet"
    seq_df = pd.read_parquet(seq_path)
    placeid_to_idx, poi_to_region = _load_graph_maps(state)
    region_emb, _ = _load_region_embeddings(state, source=region_emb_source)

    n = len(seq_df)
    seq_len = 9
    out = np.zeros((n, seq_len, region_emb_dim), dtype=np.float32)

    for i in range(seq_len):
        col = f"poi_{i}"
        placeids = seq_df[col].astype(np.int64).to_numpy()
        mask = placeids != -1
        valid = placeids[mask]
        # Vectorised placeid -> poi_idx lookup
        poi_idx = pd.Series(valid).map(placeid_to_idx).to_numpy(dtype=np.int64)
        region_idx = poi_to_region[poi_idx]
        out[np.where(mask)[0], i, :] = region_emb[region_idx]

    return torch.from_numpy(out)


def _load_checkin_region_data(state: str):
    """Load check-in embedding tensor + region labels + stratification info.

    Returns a 7-tuple adding ``last_region_tensor`` to the historical
    6-tuple. ``last_region_tensor`` is None when the parquet lacks the
    ``last_region_idx`` column (older schema, pre-commit ``6a2f808``).
    """
    engine = EmbeddingEngine.CHECK2HGI
    X, y_cat, userids, emb_dim = load_next_data(state, engine)

    region_df = IoPaths.load_next_region(state, engine)
    y_region = region_df["region_idx"].to_numpy(dtype=np.int64)
    n_regions = int(y_region.max()) + 1

    # ``last_region_idx`` is only present in the post-6a2f808 schema. Older
    # parquets built before the B5 feature landed won't have it, so we
    # return None and let the caller decide how to handle that.
    last_region_tensor = None
    if "last_region_idx" in region_df.columns:
        last_region_np = region_df["last_region_idx"].to_numpy(dtype=np.int64)
        last_region_tensor = torch.from_numpy(np.ascontiguousarray(last_region_np))

    slide_window = InputsConfig.SLIDE_WINDOW
    x_tensor, _ = _convert_to_tensors(X, y_cat, TaskType.NEXT, embedding_dim=emb_dim, slide_window=slide_window)
    y_region_tensor = torch.from_numpy(np.ascontiguousarray(y_region, dtype=np.int64))

    return x_tensor, y_region_tensor, y_cat, userids, emb_dim, n_regions, last_region_tensor


def _load_data(state: str, input_type: str, region_emb_source: str = "check2hgi"):
    """Route to the correct input loader. Returns a 7-tuple:
    ``(x_tensor, y_region_tensor, y_cat, userids, emb_dim, n_regions, last_region_tensor)``.

    ``last_region_tensor`` is None when the underlying parquet lacks the
    ``last_region_idx`` column (pre-``6a2f808`` schema). Heads in
    ``_HEADS_REQUIRING_AUX`` refuse to run when it is None — that's a
    data-prep gap the user must resolve via ``scripts/regenerate_next_region.py``.
    """
    x_checkin, y_region_tensor, y_cat, userids, checkin_dim, n_regions, last_region_tensor = (
        _load_checkin_region_data(state)
    )

    if input_type == "checkin":
        return x_checkin, y_region_tensor, y_cat, userids, checkin_dim, n_regions, last_region_tensor

    # Build region-emb sequence and align row-for-row with the check-in tensor.
    x_region = _build_region_sequence_tensor(state, region_emb_dim=checkin_dim, region_emb_source=region_emb_source)
    # Row counts must match (sequences_next.parquet is authoritative upstream).
    if x_region.shape[0] != x_checkin.shape[0]:
        raise RuntimeError(
            f"Row mismatch: region tensor has {x_region.shape[0]} rows, "
            f"checkin tensor has {x_checkin.shape[0]}. Regenerate inputs."
        )

    if input_type == "region":
        return x_region, y_region_tensor, y_cat, userids, checkin_dim, n_regions, last_region_tensor
    if input_type == "concat":
        x_concat = torch.cat([x_checkin, x_region], dim=-1)  # [N, 9, 128]
        return x_concat, y_region_tensor, y_cat, userids, checkin_dim * 2, n_regions, last_region_tensor
    raise ValueError(f"Unknown input_type: {input_type}")


def _dataloader(x, y, batch_size, shuffle, aux=None):
    """Build a DataLoader; if ``aux`` is provided, wrap with
    ``POIDatasetWithAux`` + ``AuxPublishingLoader`` so the head can read
    the auxiliary tensor via the thread-local side-channel. The training
    loop still sees ``(x, y)`` 2-tuples — the wrapper publishes + strips
    ``aux`` transparently.
    """
    if aux is None:
        ds = POIDataset(x, y, device=DEVICE)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    ds = POIDatasetWithAux(x, y, aux, device=DEVICE)
    base = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return AuxPublishingLoader(base)


def _parse_overrides(raw: list[str] | None) -> dict:
    """Parse --override-hparams name=value pairs into a dict with type coercion."""
    if not raw:
        return {}
    out = {}
    for kv in raw:
        if "=" not in kv:
            raise ValueError(f"Override must be key=value, got: {kv}")
        k, v = kv.split("=", 1)
        # Coerce: int > float > str
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _build_head(head_name, emb_dim, n_classes, seq_length, overrides):
    """Instantiate a head passing only embed_dim + num_classes (+ seq_length
    for heads that require it), plus any explicit overrides the user passed.
    Everything else falls back to the head's __init__ defaults.
    """
    import inspect
    from models.registry import _MODEL_REGISTRY, _ensure_registered
    _ensure_registered()
    target_cls = _MODEL_REGISTRY[head_name]
    sig = inspect.signature(target_cls.__init__)
    accepted = {p.name for p in sig.parameters.values()
                if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}

    kwargs = {"embed_dim": emb_dim, "num_classes": n_classes}
    # next_mtl is the only head that needs seq_length / num_heads to be
    # set; its defaults are reasonable except num_heads has no default.
    if head_name == "next_mtl":
        kwargs["seq_length"] = seq_length
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("num_layers", 4)

    # Apply user overrides (only keys the target accepts)
    for k, v in overrides.items():
        if k in accepted:
            kwargs[k] = v
    return target_cls(**kwargs)


class _InputLN(torch.nn.Module):
    """Wraps a head with input LayerNorm over the embedding dimension."""
    def __init__(self, head, emb_dim):
        super().__init__()
        self.ln = torch.nn.LayerNorm(emb_dim)
        self.head = head

    def forward(self, x):
        # Preserve zero-padding: don't LayerNorm positions that are all-zero
        # (they would become non-zero after LN, breaking downstream padding
        # masks that use x.abs().sum(dim=-1) == 0).
        pad = (x.abs().sum(dim=-1, keepdim=True) == 0).float()
        x_ln = self.ln(x)
        x_out = x_ln * (1.0 - pad) + x * pad
        return self.head(x_out)


class _MTLPreencoder(torch.nn.Module):
    """Wrap a head with MTLnet's next_encoder stack as pre-processor.

    Mirrors ``MTLnet._build_encoder`` (Linear+ReLU+LayerNorm+Dropout stack)
    so single-task evaluation sees the same input distribution the MTL
    head would see — minus the cross-attn / shared-backbone blocks.
    Used by F41 (Exp D) to isolate the *upstream encoder* contribution of
    the CH18 STL-vs-MTL gap.

    Padding semantics are preserved: positions that were zero in the
    input are re-zeroed after the encoder so the head's padding mask
    (``x.abs().sum(-1) == 0``) still fires. In MTL this is enforced by
    the downstream cross-attn `mask.unsqueeze(-1)` step; we replicate it
    directly here.
    """

    def __init__(self, in_size, hidden_size, out_size, num_layers=2,
                 dropout=0.1, head=None):
        super().__init__()
        layers = [
            torch.nn.Linear(in_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(num_layers - 1):
            layers += [
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Dropout(dropout),
            ]
        layers += [
            torch.nn.Linear(hidden_size, out_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(out_size),
        ]
        self.encoder = torch.nn.Sequential(*layers)
        self.head = head

    def forward(self, x):
        pad_mask = (x.abs().sum(dim=-1, keepdim=True) == 0)  # [B, T, 1] bool
        x_enc = self.encoder(x)
        x_enc = x_enc.masked_fill(pad_mask, 0.0)
        return self.head(x_enc)


CANONICAL_METRICS: tuple[str, ...] = (
    "accuracy", "top5_acc", "top10_acc", "mrr", "f1",
)
"""Metrics tracked by ``_new_per_metric_tracker``. Same set used in MTL
``storage.py`` ``CANONICAL_BEST_METRICS`` for parity."""


def _new_per_metric_tracker() -> dict:
    """Empty per-metric best tracker for ``_update_per_metric_best``."""
    return {m: {"value": -1.0, "snapshot": {}} for m in CANONICAL_METRICS}


def _update_per_metric_best(tracker: dict, metrics: dict, epoch: int) -> None:
    """Update each metric's best snapshot if this epoch's value is higher.

    AUDIT-C1: emits a per-metric snapshot so the STL output JSON has
    ``per_metric_best.f1.f1`` (the metric *at its own best epoch*) instead
    of the legacy single-snapshot scheme that reported every metric at
    top10's best epoch.
    """
    for m in CANONICAL_METRICS:
        v = float(metrics.get(m, 0.0))
        if v > tracker[m]["value"]:
            tracker[m]["value"] = v
            tracker[m]["snapshot"] = dict(metrics)
            tracker[m]["snapshot"]["best_epoch"] = int(epoch)


def _train_single_task(head_name, x_tensor, y_tensor, train_idx, val_idx,
                       emb_dim, n_classes, epochs, batch_size, seed,
                       overrides, max_lr, label_smoothing, input_ln,
                       aux_tensor=None, mtl_preencoder=False,
                       preenc_hidden=256, preenc_layers=2,
                       preenc_dropout=0.1):
    """Train a single-task model and return val metrics.

    ``aux_tensor`` — optional per-sample auxiliary int64 tensor (e.g.
    ``last_region_idx``). When the head is in ``_HEADS_REQUIRING_AUX``
    and ``aux_tensor`` is None, we raise because the head's forward pass
    would silently fall back to pure STAN and the comparison would be
    meaningless.
    """
    seed_everything(seed)

    if head_name in _HEADS_REQUIRING_AUX:
        if aux_tensor is None:
            raise RuntimeError(
                f"Head '{head_name}' requires 'last_region_idx' aux but the "
                f"parquet didn't contain it. Regenerate via "
                f"`scripts/regenerate_next_region.py --state <state>`."
            )

    x_train, y_train = x_tensor[train_idx], y_tensor[train_idx]
    x_val, y_val = x_tensor[val_idx], y_tensor[val_idx]

    if head_name in _HEADS_REQUIRING_AUX and aux_tensor is not None:
        aux_train = aux_tensor[train_idx]
        aux_val = aux_tensor[val_idx]
        train_dl = _dataloader(x_train, y_train, batch_size, True, aux=aux_train)
        val_dl = _dataloader(x_val, y_val, batch_size, False, aux=aux_val)
    else:
        train_dl = _dataloader(x_train, y_train, batch_size, True)
        val_dl = _dataloader(x_val, y_val, batch_size, False)

    seq_length = InputsConfig.SLIDE_WINDOW
    # Under --mtl-preencoder, the head's `embed_dim` must match the
    # preencoder's output dim (e.g. 256), not the raw input dim (64).
    # This mirrors MTL where the head receives the next_encoder output.
    head_embed_dim = preenc_hidden if mtl_preencoder else emb_dim
    head = _build_head(head_name, head_embed_dim, n_classes, seq_length, overrides)
    model = head
    if input_ln:
        model = _InputLN(head, head_embed_dim)
    if mtl_preencoder:
        model = _MTLPreencoder(
            in_size=emb_dim,
            hidden_size=preenc_hidden,
            out_size=preenc_hidden,
            num_layers=preenc_layers,
            dropout=preenc_dropout,
            head=model,
        )
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    steps_per_epoch = len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
    )
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)

    # AUDIT-C1 fix: track per-metric best across all epochs, not just
    # top10. The legacy code picked top10-best then reported F1/MRR/Acc@1
    # at THAT epoch — mirroring (in opposite direction) the MTL F1-vs-
    # top10 mismatch. With per_metric_best we emit a separate
    # snapshot at each canonical metric's best epoch so downstream
    # comparisons can be apples-to-apples.
    per_metric_best: dict = _new_per_metric_tracker()

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Val
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_dl:
                out = model(x_batch)
                all_logits.append(out.detach())
                all_targets.append(y_batch)

        logits = torch.cat(all_logits)
        targets = torch.cat(all_targets)
        metrics = compute_classification_metrics(logits, targets, num_classes=n_classes, top_k=(5, 10))

        _update_per_metric_best(per_metric_best, metrics, epoch + 1)

    # Backward-compatible primary return: top10-best snapshot (same as
    # before). Downstream callers can read per_metric_best for clean
    # cross-metric reporting.
    best_metrics = dict(per_metric_best["top10_acc"]["snapshot"])
    best_metrics["per_metric_best"] = {
        m: per_metric_best[m]["snapshot"] for m in CANONICAL_METRICS
    }
    return best_metrics


def _checkpoint_path(out_dir: Path, state: str, input_type: str,
                     folds: int, epochs: int, tag: str | None) -> Path:
    """Checkpoint path keyed on the full run config.

    Uses the same filename discriminants as the final JSON output (state,
    input_type, folds, epochs, tag) so resume is automatic when the user
    re-invokes with identical CLI args.
    """
    tag_s = f"_{tag}" if tag else ""
    return out_dir / f"region_head_{state}_{input_type}_{folds}f_{epochs}ep{tag_s}.checkpoint.json"


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Checkpoint at %s unreadable (%s) — starting fresh.", path, e)
        return {}


def _save_checkpoint(path: Path, payload: dict) -> None:
    """Atomic checkpoint write: serialise to a sibling temp file, then rename.

    Important for resume safety — if the process is killed mid-write, the
    old checkpoint is still valid rather than left half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def run_ablation(state: str, heads: list[str], folds: int, epochs: int,
                 batch_size: int, seed: int, input_type: str,
                 overrides: dict, max_lr: float | None,
                 label_smoothing: float, input_ln: bool,
                 tag: str | None, resume: bool = True,
                 region_emb_source: str = "check2hgi",
                 mtl_preencoder: bool = False,
                 preenc_hidden: int = 256,
                 preenc_layers: int = 2,
                 preenc_dropout: float = 0.1,
                 per_fold_transition_dir: str | None = None):
    logger.info("Loading data for %s (input_type=%s, region_emb=%s)...", state, input_type, region_emb_source)
    x_tensor, y_region, y_cat, userids, emb_dim, n_regions, last_region_tensor = _load_data(
        state, input_type, region_emb_source,
    )
    aux_hint = "present" if last_region_tensor is not None else "missing"
    logger.info("x=%s, emb_dim=%d, n_regions=%d, n_seqs=%d, last_region_idx=%s",
                x_tensor.shape, emb_dim, n_regions, len(y_region), aux_hint)

    sgkf = StratifiedGroupKFold(n_splits=max(2, folds), shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.zeros(len(y_cat)), y_cat, groups=userids))[:folds]

    out_dir = Path("docs/studies/check2hgi/results/P1")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = _checkpoint_path(out_dir, state, input_type, folds, epochs, tag)
    ckpt = _load_checkpoint(ckpt_path) if resume else {}
    if ckpt:
        logger.info(
            "Resuming from checkpoint %s — existing heads: %s",
            ckpt_path,
            {h: len(ckpt.get("heads", {}).get(h, {}).get("per_fold", [])) for h in ckpt.get("heads", {})},
        )
    else:
        logger.info("No checkpoint — starting fresh. Will save to %s after each fold.", ckpt_path)

    # Seed the checkpoint shell with run-level metadata so a kill between
    # --resume invocations still carries seed / n_regions forward.
    results = dict(ckpt.get("heads", {}))
    for head_name in heads:
        head_max_lr = max_lr if max_lr is not None else _HEAD_MAX_LR.get(head_name, 3e-3)
        logger.info("=" * 60)
        logger.info("HEAD: %s (%d folds x %d epochs, max_lr=%.0e, ls=%.2f, input_ln=%s)",
                    head_name, folds, epochs, head_max_lr, label_smoothing, input_ln)
        logger.info("=" * 60)

        # Resume: start from where this head left off.
        head_state = results.get(head_name, {})
        fold_metrics: list = list(head_state.get("per_fold", []))
        completed = len(fold_metrics)
        if completed >= folds:
            logger.info(
                "Head %s already has %d/%d folds — skipping (delete %s to re-run).",
                head_name, completed, folds, ckpt_path,
            )
            # Make sure aggregate is recomputed + emitted even if this head
            # was fully complete in a prior session. The final-table print
            # below reads ``aggregate`` unconditionally.
            if "aggregate" not in head_state:
                agg = {}
                for key in ["accuracy", "top5_acc", "top10_acc", "mrr", "f1"]:
                    vals = [m.get(key, 0.0) for m in fold_metrics]
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    # AUDIT-C6: ddof=1 (sample std) matches MTL-side
                    # ``statistics.stdev`` so cross-pipeline σ comparisons
                    # use the same convention. For n=5 this is ~12% larger
                    # than population std (np.std default).
                    agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                results[head_name] = {**head_state, "per_fold": fold_metrics, "aggregate": agg}
            continue
        if completed > 0:
            logger.info(
                "Head %s resuming at fold %d/%d (%d folds already in checkpoint).",
                head_name, completed, folds, completed,
            )

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            if fold_idx < completed:
                continue
            # AUDIT-C4 — when per_fold_transition_dir is set, override the
            # head's transition_path with the leak-free fold-specific file.
            # The fold split here uses the same StratifiedGroupKFold(seed=42)
            # as compute_region_transition.py --per-fold, so fold N here
            # matches region_transition_log_fold{N+1}.pt by construction.
            fold_overrides = dict(overrides)
            if per_fold_transition_dir is not None:
                # F51 seeded naming: region_transition_log_seed{S}_fold{N}.pt.
                # Trainer hard-fails on legacy unseeded files; STL is now
                # consistent. Falls back to legacy only if seeded missing
                # AND legacy present (loud warning) — for backwards compat
                # with old experiments still on the legacy path.
                pf_seeded = Path(per_fold_transition_dir) / f"region_transition_log_seed{seed}_fold{fold_idx + 1}.pt"
                pf_legacy = Path(per_fold_transition_dir) / f"region_transition_log_fold{fold_idx + 1}.pt"
                if pf_seeded.exists():
                    pf_path = pf_seeded
                elif pf_legacy.exists():
                    logger.warning(
                        "[C4 STL] using LEGACY unseeded log_T at %s — only valid "
                        "for seed=42; non-42 seeds leak val transitions. Build "
                        "seeded version with: python scripts/compute_region_transition.py "
                        "--state %s --per-fold --seed %d",
                        pf_legacy, state, seed,
                    )
                    pf_path = pf_legacy
                else:
                    raise FileNotFoundError(
                        f"per_fold_transition_dir set but neither {pf_seeded} "
                        f"nor legacy {pf_legacy} exists. Build seeded with: "
                        f"python scripts/compute_region_transition.py "
                        f"--state {state} --per-fold --seed {seed}"
                    )
                fold_overrides["transition_path"] = str(pf_path)
                logger.info("[C4 STL] fold %d using per-fold log_T %s", fold_idx, pf_path)
            t0 = time.time()
            metrics = _train_single_task(
                head_name, x_tensor, y_region, train_idx, val_idx,
                emb_dim, n_regions, epochs, batch_size, seed + fold_idx,
                fold_overrides, head_max_lr, label_smoothing, input_ln,
                aux_tensor=last_region_tensor,
                mtl_preencoder=mtl_preencoder,
                preenc_hidden=preenc_hidden,
                preenc_layers=preenc_layers,
                preenc_dropout=preenc_dropout,
            )
            elapsed = time.time() - t0
            fold_metrics.append(metrics)
            logger.info(
                "  fold %d: Acc@1=%.4f Acc@5=%.4f Acc@10=%.4f MRR=%.4f F1=%.4f (%.1fs, best_ep=%d)",
                fold_idx,
                metrics.get("accuracy", 0),
                metrics.get("top5_acc", 0),
                metrics.get("top10_acc", 0),
                metrics.get("mrr", 0),
                metrics.get("f1", 0),
                elapsed,
                metrics.get("best_epoch", 0),
            )

            # Checkpoint after every fold so a kill at any point loses at
            # most the in-progress fold. Atomic write via temp-rename keeps
            # the on-disk copy valid even if Python dies during save.
            results[head_name] = {
                "per_fold": fold_metrics,
                "config": {
                    "max_lr": head_max_lr,
                    "label_smoothing": label_smoothing,
                    "input_layernorm": input_ln,
                    "overrides": overrides,
                    "input_type": input_type,
                },
            }
            _save_checkpoint(ckpt_path, {
                "state": state, "folds": folds, "epochs": epochs, "seed": seed,
                "input_type": input_type, "n_regions": int(n_regions),
                "heads": results,
            })

        # Aggregate for this head now that all folds are done.
        agg = {}
        for key in ["accuracy", "top5_acc", "top10_acc", "mrr", "f1"]:
            vals = [m.get(key, 0.0) for m in fold_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            # AUDIT-C6: ddof=1 to match MTL-side ``statistics.stdev``
            agg[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # AUDIT-C1: per-metric aggregate. For each canonical metric, take
        # its value at ITS OWN best epoch in every fold and aggregate.
        # Pre-fix the only aggregate was at top10-best epoch — biased.
        agg_per_metric: dict = {}
        for selector in CANONICAL_METRICS:
            for reported in CANONICAL_METRICS:
                vals = [
                    m.get("per_metric_best", {}).get(selector, {}).get(reported, 0.0)
                    for m in fold_metrics
                ]
                agg_per_metric[f"{reported}_at_{selector}_best_mean"] = (
                    float(np.mean(vals)) if vals else 0.0
                )
                agg_per_metric[f"{reported}_at_{selector}_best_std"] = (
                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                )

        results[head_name] = {
            "per_fold": fold_metrics,
            "aggregate": agg,
            "aggregate_per_metric_best": agg_per_metric,
            "config": {
                "max_lr": head_max_lr,
                "label_smoothing": label_smoothing,
                "input_layernorm": input_ln,
                "overrides": overrides,
                "input_type": input_type,
            },
        }
        logger.info(
            "  AGGREGATE: Acc@1=%.4f+/-%.4f  Acc@10=%.4f+/-%.4f  MRR=%.4f+/-%.4f  F1=%.4f+/-%.4f",
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["top10_acc_mean"], agg["top10_acc_std"],
            agg["mrr_mean"], agg["mrr_std"],
            agg["f1_mean"], agg["f1_std"],
        )

    # Summary table
    print("\n" + "=" * 80)
    print(f"P1 REGION-HEAD ABLATION - {state} - {folds}f x {epochs}ep - input={input_type}")
    print("=" * 80)
    print(f"{'Head':<25} {'Acc@1':>8} {'Acc@5':>8} {'Acc@10':>8} {'MRR':>8} {'F1':>8}")
    print("-" * 80)
    for head_name in heads:
        a = results[head_name]["aggregate"]
        print(f"{head_name:<25} {a['accuracy_mean']*100:>7.2f}% {a['top5_acc_mean']*100:>7.2f}% "
              f"{a['top10_acc_mean']*100:>7.2f}% {a['mrr_mean']*100:>7.2f}% {a['f1_mean']*100:>7.2f}%")

    # Markov baseline reference (per-fold Markov on region sequences — same
    # for all input types since Markov uses the region-transition graph, not
    # the embedding view). See docs/studies/check2hgi/results/P0/...
    print("-" * 80)
    markov = {"alabama": "~21.3%", "florida": "~45.9%"}.get(state, "?")
    print(f"{'Markov 1-step (floor)':<25} {'~12%':>8} {'—':>8} {markov:>8} {'—':>8} {'—':>8}")
    print("=" * 80)

    # Save the final consolidated JSON (same schema as pre-resume version).
    tag_s = f"_{tag}" if tag else ""
    out_path = out_dir / f"region_head_{state}_{input_type}_{folds}f_{epochs}ep{tag_s}.json"
    with open(out_path, "w") as f:
        json.dump({"state": state, "folds": folds, "epochs": epochs, "seed": seed,
                    "input_type": input_type, "n_regions": int(n_regions),
                    "heads": results}, f, indent=2, default=str)
    logger.info("Saved: %s", out_path)

    # Remove the checkpoint once the final file is written — it served its
    # purpose. Leave the checkpoint in place if the aggregate JSON write
    # fails so the next invocation can retry without losing compute.
    try:
        ckpt_path.unlink(missing_ok=True)
        logger.info("Removed checkpoint %s (run complete).", ckpt_path)
    except OSError as e:
        logger.warning("Could not remove checkpoint %s: %s", ckpt_path, e)

    return results


def main():
    parser = argparse.ArgumentParser(description="P1 region-head ablation")
    parser.add_argument("--state", default="alabama")
    parser.add_argument("--heads", nargs="+", default=ALL_HEADS)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-type", choices=["checkin", "region", "concat"], default="checkin")
    parser.add_argument("--override-hparams", nargs="*", default=None,
                        help="Override head defaults: key=value (e.g. hidden_dim=384)")
    parser.add_argument("--max-lr", type=float, default=None,
                        help="Override OneCycleLR max_lr (default: per-head)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--input-layernorm", action="store_true")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffix for result file (e.g. E1, E2_scale, E_region)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="Resume from checkpoint if one exists (default: on).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any existing checkpoint and start fresh.")
    parser.add_argument("--region-emb-source", choices=["check2hgi", "hgi"], default="check2hgi",
                        help="Which engine's region_embeddings.parquet to use for region/concat input_type. "
                             "Labels + sequences always come from check2hgi — only the embedding lookup changes. "
                             "Used for P1.5 embedding-substrate comparison (CH15).")
    parser.add_argument("--mtl-preencoder", action="store_true",
                        help="Wrap the head with MTLnet's next_encoder stack (Linear+ReLU+LayerNorm+Dropout) "
                             "as a pre-processor, mirroring the MTL pipeline's upstream encoder without the "
                             "cross-attn / shared-backbone. Used by F41 (Exp D) to isolate the upstream "
                             "encoder contribution to the CH18 STL-vs-MTL region gap. Forces the head's "
                             "embed_dim to match --preenc-hidden.")
    parser.add_argument("--preenc-hidden", type=int, default=256,
                        help="Hidden / output dim of --mtl-preencoder (default 256, matches MTL shared_layer_size).")
    parser.add_argument("--preenc-layers", type=int, default=2,
                        help="Number of Linear blocks in --mtl-preencoder (default 2, matches MTL num_encoder_layers).")
    parser.add_argument(
        "--per-fold-transition-dir",
        dest="per_fold_transition_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "AUDIT-C4 fix: directory containing per-fold transition matrices "
            "(``region_transition_log_fold{1..k}.pt``). When set, the trainer "
            "overrides ``transition_path`` per fold to the matching file, "
            "eliminating val→train leakage in the GETNext graph prior. "
            "Build with: python scripts/compute_region_transition.py "
            "--state STATE --per-fold"
        ),
    )
    parser.add_argument("--preenc-dropout", type=float, default=0.1,
                        help="Dropout in --mtl-preencoder (default 0.1, matches MTL encoder_dropout).")
    args = parser.parse_args()

    overrides = _parse_overrides(args.override_hparams)
    run_ablation(
        args.state, args.heads, args.folds, args.epochs, args.batch_size, args.seed,
        args.input_type, overrides, args.max_lr, args.label_smoothing,
        args.input_layernorm, args.tag, resume=args.resume,
        region_emb_source=args.region_emb_source,
        mtl_preencoder=args.mtl_preencoder,
        preenc_hidden=args.preenc_hidden,
        preenc_layers=args.preenc_layers,
        preenc_dropout=args.preenc_dropout,
        per_fold_transition_dir=args.per_fold_transition_dir,
    )


if __name__ == "__main__":
    main()
