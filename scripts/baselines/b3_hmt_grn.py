#!/usr/bin/env python
"""B3 — HMT-GRN-STYLE external MTL baseline (class-(B) E2E trainer).

Faithful-to-spirit re-implementation of HMT-GRN (Lim, Hooi, Ng, Goh, Weng, Tan;
"Hierarchical Multi-Task Graph Recurrent Network for Next POI Recommendation",
SIGIR 2022, pp. 1133-1143; https://dl.acm.org/doi/abs/10.1145/3477495.3531989 ;
official code https://github.com/poi-rec/HMT-GRN).

WHAT HMT-GRN IS (paper): a shared Graph Recurrent Network (GRN) feature learner
over a user's POI trajectory, feeding a MULTI-TASK head that jointly predicts the
next *region* (a coarse spatial cell) and the next *POI*. The User-Region matrix
is denser than the sparse User-POI matrix, so the auxiliary region task
regularises the POI task. At inference HMT-GRN runs a Hierarchical Beam Search
(HBS) that uses the region distribution to prune the POI search space, plus a
graph-based selectivity layer.

WHAT THIS BASELINE IS ("HMT-GRN-STYLE"): the E2E shared-recurrent + per-task
softmax MTL skeleton, matched to OUR board's two HEADLINE tasks
(next-CATEGORY + next-REGION), trained with equal-weight cross-entropy. It is
the SOLE external MTL baseline. The label "HMT-GRN-STYLE" (not "HMT-GRN")
flags the documented deviations below.

=== DOCUMENTED DEVIATIONS (faithfulness ledger) ===
1. DROP Hierarchical Beam Search (HBS) + graph selectivity. Our headline is the
   next-REGION ranking metric (top10_acc_indist); there is NO next-POI head on
   the board, so the region->POI pruning machinery has nothing to prune. We keep
   only the shared-recurrent + per-task-softmax MTL core.
2. Region definition: HMT-GRN's G@P geohash grid -> our TIGER-tract region
   partition (the SHARED check2hgi geographic partition, poi_to_region in the
   graph maps). The label space is therefore identical to the champion's reg
   task, making the comparison apples-to-apples.
3. Tasks: HMT-GRN's {next-region, next-POI} -> our {next-CATEGORY, next-REGION}.
   Region stays as the dense auxiliary; category (7 root classes) replaces the
   sparse POI task as the second headline. Equal-weight CE (HMT-GRN uses an
   equal-weight multi-task sum of the per-task losses).
4. GRN -> a shared LSTM recurrent encoder over learned POI-id embeddings. The
   "graph" enrichment of GRN is dropped (our champion's graph signal lives in
   the Check2HGI substrate, which this baseline deliberately does NOT consume —
   B3 learns its embeddings end-to-end from raw POI ids, so it is a genuinely
   INDEPENDENT architecture, not a substrate probe). LSTM is the spec's stated
   "LSTM/GRN" option.
5. Split: HMT-GRN's native per-user 80/20 chronological split -> our
   user-disjoint StratifiedGroupKFold(groups=userid, y=next_category), seed,
   5 folds — BIT-IDENTICAL to FoldCreator._create_check2hgi_mtl_folds and to
   compute_region_transition._build_per_fold (same algorithm/groups/y/seed).
6. Per-fold train-only priors: the region head adds a log-transition prior
   P(next_region | last_region) estimated from TRAIN ROWS ONLY of the current
   fold (mirrors the champion's seeded per-fold log_T; never the full corpus).

=== HARD REQUIREMENTS satisfied ===
* LEAK-SAFE: POI-id embeddings + recurrent weights + the region transition prior
  are all fit on the fold's TRAIN PORTION ONLY (train_idx of the user-disjoint
  StratifiedGroupKFold). Val users are disjoint from train users (asserted in
  smoke). Row order follows next.parquet/sequences_next.parquet (sorted
  ['userid','datetime'] upstream); the three parquets are length-asserted equal.
* MATCHED PROTOCOL: same folds/seeds/labels/metric as the champion. Reg metric is
  ``top10_acc_indist`` via the imported ``_ood_restricted_topk`` with the train
  label set; cat metric is macro-``f1`` via the imported
  ``compute_classification_metrics``. Joint selector = geom_simple.
* WINDOWING: uses the CURRENT (stride-9 / non-overlapping) sequences_next.parquet.
  The paper-grade n=20 runs are POST-FREEZE at stride-1 (P3) — at which point the
  same code reads a stride-1 sequences_next.parquet built by the overlap builder
  (no code change needed; B3 consumes whatever windowing is on disk). NOW we ship
  CODE + a tiny smoke; we do NOT run n=20.
* NON-CONFLICTING: this is ONE new file under scripts/baselines/. It IMPORTS from
  src/ + scripts/ but edits nothing. No EmbeddingEngine enum member is added (B3
  is not a probe engine — it reads raw POI-id sequences directly). Artifacts go
  under results/baseline_b3_hmt_grn_style/<state>/.

Usage:
    # tiny smoke (alabama, 1 fold, 2 epochs) — proves plumbing + leak-safety
    PYTHONPATH=src .venv/bin/python scripts/baselines/b3_hmt_grn.py --smoke

    # one full state x seed (5 folds) — POST-FREEZE only; do NOT run n=20 here
    PYTHONPATH=src .venv/bin/python scripts/baselines/b3_hmt_grn.py \
        --state alabama --seed 0 --folds 5 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

# --- repo src on path (mirrors compute_region_transition.py) ---
_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = str(_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from configs.globals import CATEGORIES_MAP, DEVICE  # noqa: E402
from configs.paths import EmbeddingEngine, IoPaths, OUTPUT_DIR  # noqa: E402
from data.folds import load_next_data  # noqa: E402  (bit-identical fold inputs)
from tracking.metrics import compute_classification_metrics  # noqa: E402
from training.runners.mtl_eval import _ood_restricted_topk  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("b3_hmt_grn")

ENGINE = EmbeddingEngine.CHECK2HGI_DK_OVL  # DEFAULT board base: GATED STRIDE-1 OVERLAP
# (AL=96,326 rows) — the base the board + Check2HGI + the SC baselines run on. Pass
# --engine check2hgi for the legacy CANONICAL stride-9 base (AL=12,709; back-compat).
BASELINE_TAG = "baseline_b3_hmt_grn_style"
N_CAT = len(CATEGORIES_MAP)  # 8 slots (0..6 + 'None')
PAD = -1


# ============================================================
# DATA — raw POI-id sequences + aligned cat/region labels
# ============================================================
def _map_categories(next_category: pd.Series) -> np.ndarray:
    """Same mapping FoldCreator/load_next_data uses (label str -> int id)."""
    inv = {v: k for k, v in CATEGORIES_MAP.items()}
    return next_category.map(inv).to_numpy(dtype=np.int64)


def load_b3_data(state: str, engine: EmbeddingEngine = ENGINE):
    """Return raw-POI-id sequences + aligned (cat, region, last_region, userid).

    Row order is the SHARED order of next.parquet/next_region.parquet/
    sequences_next.parquet. The three are asserted equal length, matching the
    matched-head row-alignment contract.
    """
    seq_path = IoPaths.get_seq_next(state, engine)
    seq_df = pd.read_parquet(seq_path)
    next_df = IoPaths.load_next(state, engine)
    region_df = IoPaths.load_next_region(state, engine)

    n = len(seq_df)
    assert len(next_df) == n, (len(next_df), n)
    assert len(region_df) == n, (len(region_df), n)

    # [AUDIT-FIX B3] Guard the latent NaN-misalignment: build_fold_split derives indices on
    # load_next_data(), which DROPS NaN next_category rows, but these poi_windows/y_* arrays are
    # NOT NaN-filtered. If a state ever carries NaN labels, the fold indices would index a LONGER
    # array than the split was built on -> silent row-misalignment AND a user-disjointness break.
    # 0 NaN across all 6 board states today, so this is dormant; fail loud if that ever changes.
    _n_nan = int(next_df["next_category"].isna().sum())
    assert _n_nan == 0, (
        f"{state}: {_n_nan} NaN next_category rows. load_next_data drops these but B3's poi_windows/"
        f"y_* arrays do not -> fold-index misalignment. Apply the same NaN filter to B3 arrays first."
    )

    poi_cols = [f"poi_{i}" for i in range(9)]
    poi_windows = seq_df[poi_cols].astype(np.int64).to_numpy()  # [N, 9], PAD=-1
    target_poi = seq_df["target_poi"].astype(np.int64).to_numpy()
    userids = seq_df["userid"].astype(np.int64).to_numpy()

    # category label aligned with next.parquet (same as load_next_data's y)
    y_cat = _map_categories(next_df["next_category"])
    # region labels (target region + last observed region), from next_region
    y_region = region_df["region_idx"].astype(np.int64).to_numpy()
    last_region = region_df["last_region_idx"].astype(np.int64).to_numpy()

    # cross-check userid alignment between seq and next.parquet
    next_users = next_df["userid"].astype(np.int64).to_numpy()
    assert np.array_equal(userids, next_users), "userid order mismatch seq vs next"

    n_regions = int(y_region.max()) + 1
    return {
        "poi_windows": poi_windows,
        "target_poi": target_poi,
        "userids": userids,
        "y_cat": y_cat,
        "y_region": y_region,
        "last_region": last_region,
        "n_regions": n_regions,
    }


def build_fold_split(state: str, seed: int, n_splits: int, split_k: int = 5,
                     engine: EmbeddingEngine = ENGINE):
    """BIT-IDENTICAL fold split to FoldCreator._create_check2hgi_mtl_folds.

    Reuses load_next_data(state, CHECK2HGI) for (X, y_cat, userids) and the same
    StratifiedGroupKFold(split_k=5, shuffle=True, random_state=seed).split(
    X, y_cat, groups=userids). The board ALWAYS uses 5 splits, so the partition
    must be generated with ``split_k=5`` to stay bit-identical; ``n_splits`` then
    selects how many of those 5 folds we actually RUN (smoke uses 1). Returns
    (list of (train_idx, val_idx), userids).
    """
    X, y_cat, userids, _ = load_next_data(state, engine)
    sgkf = StratifiedGroupKFold(n_splits=split_k, shuffle=True, random_state=seed)
    splits = [tuple(s) for s in sgkf.split(X, y_cat, groups=userids)]
    return splits[:n_splits], userids


# ============================================================
# LEAK-SAFE per-fold train-only region transition prior
# ============================================================
def build_train_region_prior(
    last_region: np.ndarray,
    y_region: np.ndarray,
    train_idx: np.ndarray,
    n_regions: int,
    eps: float = 0.01,
) -> torch.Tensor:
    """log P(next_region | last_region) from TRAIN ROWS ONLY (Laplace-smoothed).

    [AUDIT-FIX B3] Analogous to compute_region_transition._log_probs_from_rows
    restricted to the fold's train_idx (user-disjoint train portion), but NOT
    bit-identical: this conditions on next_region.last_region_idx (the region of
    the last NON-pad POI, never -1 in the data), whereas the champion conditions
    on poi_8's region with valid_mask=(poi_8!=-1) and so SKIPS the ~1471 AL rows
    whose last window slot is pad. B3 therefore RETAINS those short-sequence rows.
    Both are train-only and leak-safe (last_region_idx is input-derived, not from
    the target region_idx). Returns an [n_regions, n_regions] log-prob tensor; row
    i is the distribution over next regions given last region i. Rows with
    last_region == -1 are skipped (none in practice; defensive).
    """
    lr = last_region[train_idx]
    tr = y_region[train_idx]
    valid = lr >= 0
    lr, tr = lr[valid], tr[valid]
    counts = np.full((n_regions, n_regions), eps, dtype=np.float64)
    np.add.at(counts, (lr, tr), 1.0)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return torch.from_numpy(np.log(probs).astype(np.float32))


# ============================================================
# MODEL — shared LSTM (GRN substitute) + per-task softmax heads
# ============================================================
class HMTGRNStyle(nn.Module):
    """Shared recurrent encoder over learned POI-id embeddings -> two softmax
    heads (next-category, next-region), equal-weight CE.

    The POI-id embedding table is learned END-TO-END (leak-safe: only train
    rows update it within a fold). ``poi_vocab`` is the number of distinct POI
    ids seen in the FOLD'S TRAIN sequences, +1 for the OOV/pad slot at index 0.
    Val POIs unseen in train map to the OOV slot — the standard way to avoid a
    val->train vocab leak.
    """

    def __init__(
        self,
        poi_vocab: int,
        n_regions: int,
        n_cat: int = N_CAT,
        emb_dim: int = 64,
        hidden: int = 128,
        layers: int = 1,
        dropout: float = 0.2,
        alpha_prior: float = 1.0,
    ):
        super().__init__()
        # index 0 reserved for OOV/pad (padding_idx so its grad stays 0)
        self.poi_emb = nn.Embedding(poi_vocab + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.cat_head = nn.Linear(hidden, n_cat)
        self.region_head = nn.Linear(hidden, n_regions)
        # additive region transition prior (train-only), weight alpha_prior
        self.alpha_prior = alpha_prior
        self.register_buffer("log_T", torch.zeros(n_regions, n_regions))

    def set_prior(self, log_T: torch.Tensor):
        self.log_T = log_T.to(self.log_T.device)

    def forward(self, poi_seq: torch.Tensor, last_region: torch.Tensor):
        """poi_seq: [B, 9] LongTensor of remapped train-vocab ids (0=OOV/pad).
        last_region: [B] LongTensor (or -1) for the additive region prior.
        """
        emb = self.poi_emb(poi_seq)              # [B, 9, E]
        out, (h, _) = self.lstm(emb)             # h: [layers, B, H]
        feat = self.drop(h[-1])                  # [B, H]  last-layer hidden state
        cat_logits = self.cat_head(feat)         # [B, n_cat]
        region_logits = self.region_head(feat)   # [B, n_regions]
        # add the train-only transition prior P(next|last) (GETNext/HMT-GRN-style
        # spatial prior); padded last_region (-1) contributes nothing.
        if self.alpha_prior > 0:
            valid = last_region >= 0
            if valid.any():
                bias = torch.zeros_like(region_logits)
                bias[valid] = self.log_T[last_region[valid]]
                region_logits = region_logits + self.alpha_prior * bias
        return cat_logits, region_logits


# ============================================================
# VOCAB remap (train-only) — leak-safe POI id -> dense train index
# ============================================================
def build_train_vocab(poi_windows: np.ndarray, target_poi: np.ndarray,
                      train_idx: np.ndarray) -> dict:
    """Distinct POI ids in the FOLD'S TRAIN rows -> dense ids starting at 1
    (0 = OOV/pad). Built from train rows only; val rows with unseen POIs map to 0.
    """
    train_pois = set()
    for r in train_idx:
        for p in poi_windows[r]:
            if p >= 0:
                train_pois.add(int(p))
        # target_poi itself is NOT an input feature for the next-region/cat
        # heads, so it is intentionally excluded from the input vocab.
    return {p: i + 1 for i, p in enumerate(sorted(train_pois))}


def remap_windows(poi_windows: np.ndarray, vocab: dict) -> np.ndarray:
    """Vectorised remap; PAD(-1) and OOV -> 0.

    Uses a pandas Series.map over the flattened array (single pass) instead of a
    per-element Python loop so full-state runs stay feasible.
    """
    flat = pd.Series(poi_windows.reshape(-1))
    mapped = flat.map(vocab).fillna(0).to_numpy(dtype=np.int64)
    return mapped.reshape(poi_windows.shape)


# ============================================================
# TRAIN / EVAL one fold
# ============================================================
def run_fold(data, train_idx, val_idx, seed, epochs, device, alpha_prior=1.0,
             batch_size=2048, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_regions = data["n_regions"]
    vocab = build_train_vocab(data["poi_windows"], data["target_poi"], train_idx)

    poi_remapped = remap_windows(data["poi_windows"], vocab)
    poi_t = torch.from_numpy(poi_remapped).long()
    last_region_t = torch.from_numpy(data["last_region"]).long()
    y_cat_t = torch.from_numpy(data["y_cat"]).long()
    y_reg_t = torch.from_numpy(data["y_region"]).long()

    log_T = build_train_region_prior(
        data["last_region"], data["y_region"], train_idx, n_regions
    )

    model = HMTGRNStyle(
        poi_vocab=len(vocab), n_regions=n_regions, alpha_prior=alpha_prior,
    ).to(device)
    model.set_prior(log_T)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    tr = torch.from_numpy(np.asarray(train_idx)).long()
    train_label_set_b = set(int(r) for r in data["y_region"][train_idx])

    def batches(idx_tensor, shuffle):
        order = torch.randperm(len(idx_tensor)) if shuffle else torch.arange(len(idx_tensor))
        for s in range(0, len(idx_tensor), batch_size):
            yield idx_tensor[order[s:s + batch_size]]

    for ep in range(epochs):
        model.train()
        running = 0.0
        for b in batches(tr, shuffle=True):
            xb = poi_t[b].to(device)
            lrb = last_region_t[b].to(device)
            yc = y_cat_t[b].to(device)
            yr = y_reg_t[b].to(device)
            cat_logits, reg_logits = model(xb, lrb)
            loss = ce(cat_logits, yc) + ce(reg_logits, yr)  # equal-weight CE
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(b)
        logger.info("  epoch %d/%d  train_loss=%.4f", ep + 1, epochs,
                    running / max(len(tr), 1))

    # --- eval (matched metrics) ---
    model.eval()
    va = torch.from_numpy(np.asarray(val_idx)).long()
    cat_logits_all, reg_logits_all = [], []
    yc_all, yr_all = [], []
    with torch.no_grad():
        for b in batches(va, shuffle=False):
            xb = poi_t[b].to(device)
            lrb = last_region_t[b].to(device)
            cl, rl = model(xb, lrb)
            cat_logits_all.append(cl)
            reg_logits_all.append(rl)
            yc_all.append(y_cat_t[b].to(device))
            yr_all.append(y_reg_t[b].to(device))
    cat_logits = torch.cat(cat_logits_all)
    reg_logits = torch.cat(reg_logits_all)
    yc = torch.cat(yc_all)
    yr = torch.cat(yr_all)

    cat_metrics = compute_classification_metrics(cat_logits, yc, num_classes=N_CAT)
    reg_ood = _ood_restricted_topk(reg_logits, yr, train_label_set_b)

    cat_f1 = float(cat_metrics["f1"])
    reg_top10 = float(reg_ood["top10_acc_indist"])
    geom = float(np.sqrt(max(cat_f1, 0.0) * max(reg_top10, 0.0)))
    return {
        "cat_f1": cat_f1,
        "reg_top10_acc_indist": reg_top10,
        "reg_top1_acc_indist": float(reg_ood["top1_acc_indist"]),
        "reg_top5_acc_indist": float(reg_ood["top5_acc_indist"]),
        "geom_simple": geom,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_train_vocab": int(len(vocab)),
    }


# ============================================================
# DRIVER
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="B3 HMT-GRN-STYLE MTL baseline")
    ap.add_argument("--state", default="alabama")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--alpha-prior", type=float, default=1.0,
                    help="weight on the train-only region transition prior")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None,
                    help="override device (cpu/cuda); default = repo DEVICE")
    ap.add_argument("--engine", "--base", dest="engine", default="check2hgi_dk_ovl",
                    help="board base engine that supplies the fold inputs + region/cat labels. "
                         "DEFAULT 'check2hgi_dk_ovl' = the GATED STRIDE-1 OVERLAP base the board + "
                         "Check2HGI + the SC baselines run on (AL=96,326 rows). Pass 'check2hgi' for "
                         "the legacy CANONICAL stride-9 base (AL=12,709 rows; back-compat only).")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny AL run: 1 fold, 2 epochs, asserts leak-safety")
    ap.add_argument("--no-write", action="store_true",
                    help="do not persist results JSON (smoke default)")
    args = ap.parse_args()

    if args.smoke:
        args.state = "alabama"
        args.folds = 1
        args.epochs = 2
        args.no_write = True

    engine = EmbeddingEngine(args.engine)

    device = torch.device(args.device) if args.device else DEVICE
    logger.info("B3 HMT-GRN-STYLE | state=%s seed=%d folds=%d epochs=%d device=%s engine=%s",
                args.state, args.seed, args.folds, args.epochs, device, engine.value)

    data = load_b3_data(args.state, engine)
    logger.info("rows=%d  n_regions=%d  cat_classes=%d",
                len(data["userids"]), data["n_regions"], N_CAT)

    splits, userids = build_fold_split(args.state, args.seed, args.folds, engine=engine)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if args.smoke and fold_idx > 0:
            break
        # --- LEAK-SAFETY ASSERT: val users disjoint from train users ---
        train_users = set(int(u) for u in userids[train_idx])
        val_users = set(int(u) for u in userids[val_idx])
        assert val_users.isdisjoint(train_users), (
            "LEAK: val users overlap train users in fold %d" % fold_idx
        )
        logger.info(
            "Fold %d/%d  train=%d val=%d  train_users=%d val_users=%d "
            "(disjoint OK)",
            fold_idx + 1, args.folds, len(train_idx), len(val_idx),
            len(train_users), len(val_users),
        )
        t0 = time.time()
        res = run_fold(
            data, train_idx, val_idx, args.seed, args.epochs, device,
            alpha_prior=args.alpha_prior, batch_size=args.batch_size, lr=args.lr,
        )
        res["fold_idx"] = fold_idx
        res["elapsed_s"] = round(time.time() - t0, 1)
        fold_results.append(res)
        logger.info(
            "Fold %d  cat_f1=%.4f  reg_top10_indist=%.4f  geom=%.4f  (%.1fs)",
            fold_idx, res["cat_f1"], res["reg_top10_acc_indist"],
            res["geom_simple"], res["elapsed_s"],
        )

    agg = {
        "cat_f1_mean": float(np.mean([r["cat_f1"] for r in fold_results])),
        "reg_top10_acc_indist_mean": float(
            np.mean([r["reg_top10_acc_indist"] for r in fold_results])
        ),
        "geom_simple_mean": float(np.mean([r["geom_simple"] for r in fold_results])),
    }
    summary = {
        "baseline": BASELINE_TAG,
        "label": "HMT-GRN-STYLE",
        "paper": "Lim et al., HMT-GRN, SIGIR 2022",
        "deviations": [
            "drop Hierarchical Beam Search + graph selectivity (no next-POI head)",
            "G@P geohash grid -> TIGER tract region partition",
            "tasks {next-region, next-POI} -> {next-CATEGORY, next-REGION}",
            "GRN -> shared LSTM over end-to-end-learned POI-id embeddings",
            "native per-user 80/20 -> StratifiedGroupKFold(userid) board folds",
            "per-fold train-only region transition prior",
        ],
        "state": args.state,
        "base_engine": engine.value,
        "seed": args.seed,
        "folds": args.folds,
        "epochs": args.epochs,
        "alpha_prior": args.alpha_prior,
        "fold_results": fold_results,
        "aggregate": agg,
        "windowing": "stride-9 (current); paper-grade n=20 is POST-FREEZE stride-1 (P3)",
    }
    logger.info("AGGREGATE  cat_f1=%.4f  reg_top10_indist=%.4f  geom=%.4f",
                agg["cat_f1_mean"], agg["reg_top10_acc_indist_mean"],
                agg["geom_simple_mean"])

    if not args.no_write:
        out_dir = OUTPUT_DIR.parent / "results" / BASELINE_TAG / args.state.lower()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"b3_seed{args.seed}_folds{args.folds}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("wrote %s", out_path)
    else:
        print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
