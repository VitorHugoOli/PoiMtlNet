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
from torch.utils.data import DataLoader, TensorDataset

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

WINDOW_SIZE = 9


def load_tensors(state: str):
    df = pd.read_parquet(etl_out_path(state))
    poi = np.stack([df[f"poi_idx_{k}"].to_numpy(np.int64) for k in range(WINDOW_SIZE)], axis=1)
    lat = np.stack([df[f"lat_{k}"].to_numpy(np.float32) for k in range(WINDOW_SIZE)], axis=1)
    lon = np.stack([df[f"lon_{k}"].to_numpy(np.float32) for k in range(WINDOW_SIZE)], axis=1)
    tmin = np.stack([df[f"t_minutes_{k}"].to_numpy(np.int64) for k in range(WINDOW_SIZE)], axis=1)
    hour = np.stack([df[f"hour_of_week_{k}"].to_numpy(np.int64) for k in range(WINDOW_SIZE)], axis=1)
    y = df["target_region_idx"].to_numpy(np.int64)
    cat = df["target_category"].astype("category").cat.codes.to_numpy(np.int64)
    uid = df["userid"].to_numpy(np.int64)
    n_pois = int(poi[poi >= 0].max()) + 1

    centroids_df = pd.read_parquet(etl_centroids_path(state)).sort_values("region_idx")
    centroids = centroids_df[["centroid_lat", "centroid_lon"]].to_numpy(np.float32)
    n_regions = int(centroids.shape[0])
    # Cap n_regions to max-observed if any centroid index is out-of-range
    n_regions = max(n_regions, int(y.max()) + 1)

    return (
        torch.from_numpy(poi),
        torch.from_numpy(hour),
        torch.from_numpy(lat),
        torch.from_numpy(lon),
        torch.from_numpy(tmin),
        torch.from_numpy(y),
        torch.from_numpy(centroids),
        cat, uid, n_pois, n_regions,
    )


def _make_loader(idx, poi, hour, lat, lon, tmin, y, batch_size, shuffle):
    ds = TensorDataset(poi[idx], hour[idx], lat[idx], lon[idx], tmin[idx], y[idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one_fold(poi, hour, lat, lon, tmin, y, centroids, train_idx, val_idx,
                   n_pois, n_regions, *, epochs, batch_size, seed,
                   d_model, dropout, max_lr) -> dict:
    seed_everything(seed)
    train_dl = _make_loader(train_idx, poi, hour, lat, lon, tmin, y, batch_size, True)
    val_dl = _make_loader(val_idx, poi, hour, lat, lon, tmin, y, batch_size, False)

    model = FaithfulSTAN(
        n_pois=n_pois, n_regions=n_regions,
        d_model=d_model, dropout=dropout, seq_length=WINDOW_SIZE,
    ).to(DEVICE)
    centroids_dev = centroids.to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_dl)
    )
    crit = nn.CrossEntropyLoss()

    best_acc10 = -1.0
    best = {}
    for epoch in range(epochs):
        model.train()
        for poi_b, hour_b, lat_b, lon_b, t_b, y_b in train_dl:
            poi_b, hour_b, lat_b, lon_b, t_b, y_b = (
                poi_b.to(DEVICE), hour_b.to(DEVICE),
                lat_b.to(DEVICE), lon_b.to(DEVICE),
                t_b.to(DEVICE), y_b.to(DEVICE),
            )
            optim.zero_grad(set_to_none=True)
            out = model(poi_b, hour_b, lat_b, lon_b, t_b, centroids_dev)
            loss = crit(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

        model.eval()
        logits, targets = [], []
        with torch.no_grad():
            for poi_b, hour_b, lat_b, lon_b, t_b, y_b in val_dl:
                poi_b, hour_b, lat_b, lon_b, t_b = (
                    poi_b.to(DEVICE), hour_b.to(DEVICE),
                    lat_b.to(DEVICE), lon_b.to(DEVICE), t_b.to(DEVICE),
                )
                logits.append(model(poi_b, hour_b, lat_b, lon_b, t_b, centroids_dev).cpu())
                targets.append(y_b)
        logits = torch.cat(logits)
        targets = torch.cat(targets)
        m = compute_classification_metrics(logits, targets, num_classes=n_regions, top_k=(5, 10))
        if m.get("top10_acc", 0) > best_acc10:
            best_acc10 = m["top10_acc"]
            best = dict(m, best_epoch=epoch + 1)
    return best


def run(state: str, folds: int, epochs: int, batch_size: int, seed: int,
        d_model: int, dropout: float, max_lr: float,
        tag: str | None) -> None:
    poi, hour, lat, lon, tmin, y, centroids, cat, uid, n_pois, n_regions = load_tensors(state)
    logger.info("Loaded %s: rows=%d  n_pois=%d  n_regions=%d  centroids=%s",
                state, poi.shape[0], n_pois, n_regions, tuple(centroids.shape))

    sgkf = StratifiedGroupKFold(n_splits=max(2, folds), shuffle=True, random_state=seed)
    splits = list(sgkf.split(np.zeros(len(cat)), cat, groups=uid))[:folds]

    out_dir = Path("docs/studies/check2hgi/results/baselines")
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
            d_model=d_model, dropout=dropout, max_lr=max_lr,
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
        "n_regions": n_regions, "n_pois": n_pois,
        "config": {"d_model": d_model, "dropout": dropout,
                   "max_lr": max_lr, "batch_size": batch_size},
        "per_fold": fold_metrics, "aggregate": agg,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    logger.info("Saved: %s", out_file)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-lr", type=float, default=3e-3)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()
    run(args.state, args.folds, args.epochs, args.batch_size, args.seed,
        args.d_model, args.dropout, args.max_lr, args.tag)


if __name__ == "__main__":
    main()
