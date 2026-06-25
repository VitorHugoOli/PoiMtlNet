"""Faithful STAN baseline trainer (single-task next_region, 5-fold CV).

Mirrors the protocol used by ``scripts/p1_region_head_ablation.py`` so the
results are directly comparable to in-house STL region heads:

    - StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
      stratified on the target check-in's category and grouped by userid.
    - AdamW(lr=1e-4, wd=0.01) + OneCycleLR(max_lr=3e-3) + grad-clip 1.0.
    - 50 epochs default, batch 2048.
    - Best-epoch selection on val Acc@10.
    - Output JSON layout matches the in-house ``region_head_*`` JSONs.

Usage::

    PYTHONPATH=src DATA_ROOT=... OUTPUT_DIR=... \\
        python -m research.baselines.stan.train \\
            --state alabama --folds 5 --epochs 50 \\
            --tag FAITHFUL_STAN_al_5f50ep
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

_root = Path(__file__).resolve().parents[3]
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.globals import DEVICE  # noqa: E402
from tracking.metrics import compute_classification_metrics  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

from research.baselines.stan.etl import (  # noqa: E402
    centroids_path as etl_centroids_path,
    out_path as etl_out_path,
)
from research.baselines.stan.model import FaithfulSTAN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("faithful_stan")

def load_tensors(state: str):
    df = pd.read_parquet(etl_out_path(state))
    # Infer the sequence length (STAN CONTEXT_LEN) from the prefix-expansion columns,
    # so the trainer follows whatever --context-len the ETL produced.
    seq_len = sum(1 for c in df.columns if c.startswith("poi_idx_"))
    poi = np.stack([df[f"poi_idx_{k}"].to_numpy(np.int64) for k in range(seq_len)], axis=1)
    lat = np.stack([df[f"lat_{k}"].to_numpy(np.float32) for k in range(seq_len)], axis=1)
    lon = np.stack([df[f"lon_{k}"].to_numpy(np.float32) for k in range(seq_len)], axis=1)
    tmin = np.stack([df[f"t_minutes_{k}"].to_numpy(np.int64) for k in range(seq_len)], axis=1)
    hour = np.stack([df[f"hour_of_week_{k}"].to_numpy(np.int64) for k in range(seq_len)], axis=1)
    y = df["target_region_idx"].to_numpy(np.int64)
    cat = df["target_category"].astype("category").cat.codes.to_numpy(np.int64)
    uid = df["userid"].to_numpy(np.int64)
    n_pois = int(poi[poi >= 0].max()) + 1

    centroids_df = pd.read_parquet(etl_centroids_path(state)).sort_values("region_idx")
    centroids = centroids_df[["centroid_lat", "centroid_lon"]].to_numpy(np.float32)
    n_regions = int(centroids.shape[0])
    # Review fix (2026-06-26): region_emb [R,D] and region_centroids [R,2] MUST share R,
    # else the matching layer broadcast crashes. Assert rather than silently bumping
    # n_regions above the centroid count (which would desync them).
    assert int(y.max()) + 1 <= n_regions, (
        f"target region idx {int(y.max())} >= n_regions(centroids) {n_regions} — "
        f"centroid/region index desync; rebuild ETL")

    return (
        torch.from_numpy(poi),
        torch.from_numpy(hour),
        torch.from_numpy(lat),
        torch.from_numpy(lon),
        torch.from_numpy(tmin),
        torch.from_numpy(y),
        torch.from_numpy(centroids),
        cat, uid, n_pois, n_regions, seq_len,
    )


def train_one_fold(poi, hour, lat, lon, tmin, y, centroids, train_idx, val_idx,
                   n_pois, n_regions, *, epochs, batch_size, seed,
                   d_model, dropout, lr, seq_length, patience) -> dict:
    seed_everything(seed)
    # Determinism off here (warn_only=True earlier just emitted spam without
    # actual determinism since CuBLAS isn't deterministic by default).
    torch.use_deterministic_algorithms(False)

    # Pre-move all fold tensors to GPU once. Eliminates DataLoader IPC, fork
    # pickling, and per-batch H2D copies — the dominant overhead on small
    # datasets (AL: 12K rows). For CA/TX (~360–460K rows × 9 windows) the
    # tensor footprint is ~150 MB total, trivially fitting on the 80 GB H100.
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long)

    def _gather(t, idx):
        return t.index_select(0, idx).to(DEVICE, non_blocking=torch.cuda.is_available())

    poi_tr = _gather(poi, train_idx_t); hour_tr = _gather(hour, train_idx_t)
    lat_tr = _gather(lat, train_idx_t); lon_tr = _gather(lon, train_idx_t)
    tmin_tr = _gather(tmin, train_idx_t); y_tr = _gather(y, train_idx_t)
    poi_va = _gather(poi, val_idx_t); hour_va = _gather(hour, val_idx_t)
    lat_va = _gather(lat, val_idx_t); lon_va = _gather(lon, val_idx_t)
    tmin_va = _gather(tmin, val_idx_t); y_va = _gather(y, val_idx_t)

    n_train = poi_tr.shape[0]
    steps_per_epoch = max(1, (n_train + batch_size - 1) // batch_size)

    model = FaithfulSTAN(
        n_pois=n_pois, n_regions=n_regions,
        d_model=d_model, dropout=dropout, seq_length=seq_length,
    ).to(DEVICE)
    centroids_dev = centroids.to(DEVICE)

    # Audit fix #5 (2026-06-26): STAN's reference trains at a CONSTANT LR (StepLR γ=1)
    # with early-stopping on a real val plateau, NOT OneCycle (which annealed the LR to
    # ~0 by ep50 so the prior run's "best at 49/50" never plateaued -> under-trained).
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()

    # Audit fix #6: fp32 (no bf16 autocast); the board protocol forbids AMP here.
    g = torch.Generator(device=DEVICE if torch.cuda.is_available() else "cpu")
    g.manual_seed(seed)

    best_acc10 = -1.0
    best_epoch = -1
    best_logits = None
    epochs_since_improve = 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=g, device=poi_tr.device)
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            poi_b = poi_tr[idx]; hour_b = hour_tr[idx]
            lat_b = lat_tr[idx]; lon_b = lon_tr[idx]
            t_b = tmin_tr[idx]; y_b = y_tr[idx]
            optim.zero_grad(set_to_none=True)
            out = model(poi_b, hour_b, lat_b, lon_b, t_b, centroids_dev)
            loss = crit(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # Cheap epoch validation: only Acc@10 (no sklearn F1) for selection.
        model.eval()
        n_val = poi_va.shape[0]
        top10_correct = 0
        val_logits_chunks = []
        with torch.no_grad():
            for s in range(0, n_val, batch_size):
                e = s + batch_size
                out = model(poi_va[s:e], hour_va[s:e], lat_va[s:e],
                            lon_va[s:e], tmin_va[s:e], centroids_dev).float()
                top10 = out.topk(min(10, out.shape[-1]), dim=-1).indices  # [b, 10]
                top10_correct += (top10 == y_va[s:e].unsqueeze(1)).any(-1).sum().item()
                # Review fix (2026-06-26): offload each val chunk to CPU immediately —
                # the audit's O(N_val·C) cache (FL: 4703 classes × ~300K val rows ≈ 15 GB)
                # must not accumulate on the GPU.
                val_logits_chunks.append(out.cpu())
        acc10 = top10_correct / n_val
        if acc10 > best_acc10 + 1e-5:
            best_acc10 = acc10
            best_epoch = epoch + 1
            best_logits = torch.cat(val_logits_chunks, dim=0)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                logger.info("    early-stop at epoch %d (best Acc@10=%.4f @ep%d, patience=%d)",
                            epoch + 1, best_acc10, best_epoch, patience)
                break

    # Full metrics ONCE at the best epoch's logits.
    m = compute_classification_metrics(best_logits, y_va.cpu(),
                                       num_classes=n_regions, top_k=(5, 10))
    return dict(m, best_epoch=best_epoch, n_epochs_ran=epoch + 1)


def run(state: str, folds: int, epochs: int, batch_size: int, seed: int,
        d_model: int, dropout: float, lr: float, patience: int,
        tag: str | None) -> None:
    # Audit fix #6: true fp32 — disable TF32 too (board protocol).
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    poi, hour, lat, lon, tmin, y, centroids, cat, uid, n_pois, n_regions, seq_len = load_tensors(state)
    logger.info("Loaded %s: rows=%d  n_pois=%d  n_regions=%d  seq_len=%d  centroids=%s",
                state, poi.shape[0], n_pois, n_regions, seq_len, tuple(centroids.shape))

    sgkf = StratifiedGroupKFold(n_splits=max(2, folds), shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.zeros(len(cat)), cat, groups=uid))[:folds]

    out_dir = Path("docs/results/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_s = f"_{tag}" if tag else ""
    out_file = out_dir / f"faithful_stan_{state}_{folds}f_{epochs}ep{tag_s}.json"

    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        t0 = time.time()
        m = train_one_fold(
            poi, hour, lat, lon, tmin, y, centroids, train_idx, val_idx,
            n_pois=n_pois, n_regions=n_regions,
            epochs=epochs, batch_size=batch_size, seed=seed + fold_idx,
            d_model=d_model, dropout=dropout, lr=lr,
            seq_length=seq_len, patience=patience,
        )
        elapsed = time.time() - t0
        fold_metrics.append(m)
        logger.info("  fold %d: Acc@1=%.4f Acc@5=%.4f Acc@10=%.4f MRR=%.4f F1=%.4f (%.1fs, best_ep=%d)",
                    fold_idx, m.get("accuracy", 0), m.get("top5_acc", 0), m.get("top10_acc", 0),
                    m.get("mrr", 0), m.get("f1", 0), elapsed, m.get("best_epoch", 0))

    agg = {}
    for k in ["accuracy", "top5_acc", "top10_acc", "mrr", "f1"]:
        vals = [m.get(k, 0.0) for m in fold_metrics]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    logger.info("AGGREGATE: " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()))

    payload = {
        "state": state, "folds": folds, "epochs": epochs, "seed": seed,
        "n_regions": n_regions, "n_pois": n_pois, "seq_len": seq_len,
        "config": {"d_model": d_model, "dropout": dropout,
                   "lr": lr, "patience": patience, "batch_size": batch_size,
                   "schedule": "constant", "precision": "fp32",
                   "sequence": "STAN prefix-expansion"},
        "per_fold": fold_metrics, "aggregate": agg,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    logger.info("Saved: %s", out_file)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200,
                   help="epoch CAP; early-stopping ends training on a real plateau")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="CONSTANT learning rate (STAN-faithful; no OneCycle)")
    p.add_argument("--patience", type=int, default=20,
                   help="early-stop patience on val Acc@10")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()
    run(args.state, args.folds, args.epochs, args.batch_size, args.seed,
        args.d_model, args.dropout, args.lr, args.patience, args.tag)


if __name__ == "__main__":
    main()
