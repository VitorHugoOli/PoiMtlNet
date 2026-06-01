"""Corrected L0/L1 for next-reg using REGION embeddings (the actual task input).

The category probe (run.py) feeds the per-item *final* embedding — correct for
next-cat. But next-reg's real input is the **region embedding**
(`region_embeddings.parquet`, cols reg_0..reg_D), looked up per sequence step via
placeid->region (the `--task-b-input-type region` modality; the docs show region
input gives ~53% Acc@10 vs ~20% on checkin input). So next-reg must be screened
on the region-embedding artifact, NOT the final embedding.

Region ids are the SHARED census-tract partition (check2hgi poi_to_region); both
HGI and Check2HGI region_embeddings.parquet are indexed by the same region_id,
which is exactly how p1_region_head_ablation.py looks them up for either engine.

L0 (train-free): adjacency coherence — fraction of each region's cosine-kNN that
  are graph-adjacent (region_adjacency). A good region embedding places spatially
  adjacent regions near each other. + linear CKA across engines.
L1 (probe): 1-step transition probe — given the CURRENT region's embedding,
  linearly predict the NEXT region (pairs = last_region_idx -> region_idx from
  next_region.parquet). Acc@k mirrors the real next-reg Acc@10. This is a
  1-step-Markov reduction of the 9-window+log_T task: a cheap proxy whose RANKING
  (not absolute value) we check against the real STL §0.3.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
for p in (_root, _root / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from configs.paths import EmbeddingEngine, IoPaths
from tracking.metrics import compute_classification_metrics


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_region_emb(state: str, engine: EmbeddingEngine):
    """Return (emb [R,D] indexed by region_id, n_regions). Missing ids -> zeros."""
    import pandas as pd
    path = IoPaths.get_embedd(state, engine).parent / "region_embeddings.parquet"
    if not path.exists():
        return None, 0
    df = pd.read_parquet(path)
    reg_cols = [c for c in df.columns if str(c).startswith("reg_")]
    rid = df["region_id"].to_numpy(dtype=np.int64)
    R = int(rid.max()) + 1
    emb = np.zeros((R, len(reg_cols)), dtype=np.float32)
    emb[rid] = df[reg_cols].to_numpy(dtype=np.float32)
    return emb, R


def load_pairs(state: str):
    """(current_region, next_region) from the shared check2hgi next_region.parquet."""
    nr = IoPaths.load_next_region(state, EmbeddingEngine.CHECK2HGI)
    last = nr["last_region_idx"].to_numpy(dtype=np.int64)
    nxt = nr["region_idx"].to_numpy(dtype=np.int64)
    m = last >= 0
    return last[m], nxt[m]


def adjacency_coherence(emb: np.ndarray, state: str, k: int = 10,
                        n_folds: int = 5, seed: int = 42):
    """Per-region fraction of cosine top-k NN that are graph-adjacent. kNN runs
    over the full (fixed) region set; variance comes from averaging the per-region
    coherence over 5 disjoint region-folds. Returns (mean, sd)."""
    from sklearn.model_selection import KFold
    g = pickle.load(open(IoPaths.CHECK2HGI.get_graph_data_file(state), "rb"))
    ei = np.asarray(g["region_adjacency"])
    R = emb.shape[0]
    adj = [set() for _ in range(R)]
    for a, b in zip(ei[0], ei[1]):
        if a < R and b < R:
            adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    has = (np.linalg.norm(emb, axis=1) > 0)
    dev = _dev()
    x = F.normalize(torch.from_numpy(emb).to(dev, torch.float32), dim=1)
    sims = x @ x.T
    sims.fill_diagonal_(float("-inf"))
    nn = sims.topk(min(k, R - 1), dim=1).indices.cpu().numpy()
    scored = [r for r in range(R) if has[r] and adj[r]]
    per_region = np.array([np.mean([n in adj[r] for n in nn[r]]) for r in scored])
    if len(per_region) < n_folds:
        return float(per_region.mean()) if len(per_region) else float("nan"), 0.0
    fold_means = [per_region[te].mean() for _, te in
                  KFold(n_folds, shuffle=True, random_state=seed).split(per_region)]
    return float(np.mean(fold_means)), float(np.std(fold_means, ddof=1))


def transition_probe(emb, last, nxt, n_folds=5, seed=42, epochs=300, lr=1e-2):
    """5-fold CV linear probe: region_emb[current] -> next region. Returns
    {metric:(mean,sd)} over the 5 folds (mirrors L2's 5-fold protocol)."""
    from sklearn.model_selection import KFold
    uniq, nxt = np.unique(nxt, return_inverse=True)  # densify to [0,C)
    nxt = nxt.astype(np.int64)
    n_classes = len(uniq)
    X = emb[last]                       # [N, D] current region's embedding
    dev = _dev()
    runs = []
    for tr_idx, te_idx in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        xtr, xte, ytr, yte = X[tr_idx], X[te_idx], nxt[tr_idx], nxt[te_idx]
        xtr, xes, ytr, yes = train_test_split(xtr, ytr, test_size=0.15, random_state=seed)
        mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
        xtr, xes, xte = (xtr - mu) / sd, (xes - mu) / sd, (xte - mu) / sd
        xtr_t = torch.from_numpy(xtr).to(dev, torch.float32)
        ytr_t = torch.from_numpy(ytr).to(dev, torch.long)
        xes_t = torch.from_numpy(xes).to(dev, torch.float32)
        yes_t = torch.from_numpy(yes).to(dev, torch.long)
        xte_t = torch.from_numpy(xte).to(dev, torch.float32)
        yte_t = torch.from_numpy(yte).to(dev, torch.long)
        torch.manual_seed(seed)
        clf = torch.nn.Linear(X.shape[1], n_classes).to(dev)
        opt = torch.optim.AdamW(clf.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(seed)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(epochs):
            clf.train()
            perm = torch.randperm(xtr_t.shape[0], generator=gen).to(dev)
            for i in range(0, len(perm), 8192):
                bidx = perm[i:i + 8192]
                opt.zero_grad()
                F.cross_entropy(clf(xtr_t[bidx]), ytr_t[bidx]).backward()
                opt.step()
            clf.eval()
            with torch.no_grad():
                esl = F.cross_entropy(clf(xes_t), yes_t).item()
            if esl < best - 1e-4:
                best, bad = esl, 0
                best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        if best_state:
            clf.load_state_dict(best_state)
        clf.eval()
        with torch.no_grad():
            logits = clf(xte_t)
        runs.append(compute_classification_metrics(logits, yte_t, num_classes=n_classes, top_k=(5, 10)))
    out = {}
    for key in runs[0]:
        v = np.array([r[key] for r in runs])
        out[key] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Corrected next-reg eval on region embeddings")
    ap.add_argument("--engines", nargs="+", required=True)
    ap.add_argument("--state", default="florida")
    ap.add_argument("--seed", type=int, default=42, help="5-fold shuffle seed")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    last, nxt = load_pairs(args.state)
    print(f"[{args.state}] {len(last)} transition pairs, {int(nxt.max())+1} regions, "
          f"self-transition rate={np.mean(last==nxt):.3f}  (5-fold CV, mean±SD)\n")
    print(f"{'engine':30s} {'adj_coh@10':>16s} {'probe_acc':>16s} {'acc@5':>16s} {'acc@10':>16s}")
    for name in args.engines:
        emb, R = load_region_emb(args.state, EmbeddingEngine(name))
        if emb is None:
            print(f"{name:30s}  region_embeddings.parquet MISSING")
            continue
        am, asd = adjacency_coherence(emb, args.state, n_folds=args.n_folds, seed=args.seed)
        pr = transition_probe(emb, last, nxt, n_folds=args.n_folds, seed=args.seed)
        def cell(k): m, s = pr[k]; return f"{m:.4f}±{s:.4f}"
        print(f"{name:30s} {f'{am:.4f}±{asd:.4f}':>16s} {cell('accuracy'):>16s} "
              f"{cell('top5_acc'):>16s} {cell('top10_acc'):>16s}")


if __name__ == "__main__":
    main()
