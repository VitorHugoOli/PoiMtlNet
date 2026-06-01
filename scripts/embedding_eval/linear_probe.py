"""L1 — linear probe for embedding evaluation.

A single ``Linear`` softmax on frozen embeddings, fixed protocol, multi-seed.
Measures linearly-accessible task signal. Reuses ``compute_classification_metrics``
(which has the O(N+C) high-cardinality path needed for the ~5k-region head).

Fairness controls (from the methodology review):
  * features standardized on the TRAIN split only (no leakage), so engines with
    different vector scales are compared on equal footing;
  * minibatch AdamW with early stopping on a carved-out train-val slice, so the
    7-class and ~5k-class probes are each trained *to convergence* rather than a
    fixed step budget that underfits the high-cardinality head;
  * train accuracy + test-class coverage reported so underfit / unseen-class
    effects are visible rather than silently depressing the score.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from tracking.metrics import compute_classification_metrics


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stratifiable(y: np.ndarray) -> bool:
    """train_test_split(stratify=) requires every class to have >=2 members."""
    _, cnt = np.unique(y, return_counts=True)
    return cnt.min() >= 2


def _fit_one(
    emb: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    seed: int,
    epochs: int = 300,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    test_size: float = 0.2,
    batch_size: int = 8192,
    patience: int = 20,
) -> Dict[str, float]:
    """One train/test split + linear probe fit to convergence. Returns val
    metrics plus ``train_acc`` and ``test_class_coverage`` diagnostics."""
    strat = labels if _stratifiable(labels) else None
    xtr, xte, ytr, yte = train_test_split(
        emb, labels, test_size=test_size, random_state=seed, stratify=strat
    )
    # carve an early-stopping slice out of train (never touches test)
    estrat = ytr if _stratifiable(ytr) else None
    xtr, xes, ytr, yes = train_test_split(
        xtr, ytr, test_size=0.15, random_state=seed, stratify=estrat
    )

    # standardize on TRAIN only, apply to all splits (M3: scale fairness)
    mu = xtr.mean(0, keepdims=True)
    sd = xtr.std(0, keepdims=True) + 1e-6
    xtr, xes, xte = (xtr - mu) / sd, (xes - mu) / sd, (xte - mu) / sd

    dev = _device()
    xtr_t = torch.from_numpy(xtr).to(dev, torch.float32)
    ytr_t = torch.from_numpy(ytr).to(dev, torch.long)
    xes_t = torch.from_numpy(xes).to(dev, torch.float32)
    yes_t = torch.from_numpy(yes).to(dev, torch.long)
    xte_t = torch.from_numpy(xte).to(dev, torch.float32)
    yte_t = torch.from_numpy(yte).to(dev, torch.long)

    torch.manual_seed(seed)
    clf = torch.nn.Linear(emb.shape[1], num_classes).to(dev)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

    n = xtr_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_es, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        clf.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for s in range(0, n, batch_size):
            bidx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(clf(xtr_t[bidx]), ytr_t[bidx])
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            es_loss = F.cross_entropy(clf(xes_t), yes_t).item()
        if es_loss < best_es - 1e-4:
            best_es, bad = es_loss, 0
            best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        clf.load_state_dict(best_state)

    clf.eval()
    with torch.no_grad():
        logits = clf(xte_t)
        train_logits = clf(xtr_t)
    res = compute_classification_metrics(logits, yte_t, num_classes=num_classes)
    res["train_acc"] = float((train_logits.argmax(1) == ytr_t).float().mean().item())
    seen = np.unique(ytr)
    res["test_class_coverage"] = float(np.isin(np.unique(yte), seen).mean())
    return res


def linear_probe(
    emb: np.ndarray,
    labels: np.ndarray,
    seeds: List[int] = (0, 1, 7, 100),
    **kwargs,
) -> Dict[str, Dict[str, float]]:
    """Multi-seed linear probe. Returns ``{metric: {mean, sd, n}}`` (sd = sample
    SD, ddof=1; with n=4 seeds this is a spread estimate, NOT a 95% CI).

    Labels are densified to ``[0, C)`` first so ``num_classes`` is exact even
    when the raw label space is sparse (region indices).
    """
    uniq, dense = np.unique(labels, return_inverse=True)
    num_classes = len(uniq)
    dense = dense.astype(np.int64)

    runs = [_fit_one(emb, dense, num_classes, s, **kwargs) for s in seeds]
    out: Dict[str, Dict[str, float]] = {}
    for k in runs[0]:
        vals = np.array([r[k] for r in runs], dtype=np.float64)
        out[k] = {"mean": float(vals.mean()),
                  "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                  "n": len(vals)}
    return out
