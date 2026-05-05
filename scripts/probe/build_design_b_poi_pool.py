"""Design B — Check2HGI with POI2Vec injected at the POI-pool boundary.

Canonical 11-dim input (NOT POI2Vec-augmented). Encoder is identical to
canonical c2hgi. POI2Vec enters AFTER ``Checkin2POI`` as a detached additive
residual that only the reg path sees.

Critical property: cat path is byte-identical in gradient-flow to canonical
c2hgi. The encoder receives gradient only from L_c2p (computed against the
non-detached canonical poi_emb). The L_p2r and L_r2c gradients flow through
the POI2Vec residual + POI2Region but stop at the detach() before reaching
encoder weights.

Design discriminator (vs Design E which failed): input features are NOT
augmented. The 75-dim mix is what killed Design E's cat — POI-static
features in input propagated through GCN regardless of gradient routing.

Architecture::

    checkin_emb        = CheckinEncoder(canonical_11_dim, ...)        # active grad
    poi_emb_canonical  = Checkin2POI(checkin_emb, ...)                # active grad
    poi_emb_for_reg    = poi_emb_canonical.detach() + γ·Linear(POI2Vec_for_poi)
    region_emb         = POI2Region(poi_emb_for_reg, ...)
    city_emb           = region2city(region_emb)

Loss boundaries::

    L_c2p uses (checkin_emb, poi_emb_canonical)   ← exactly canonical c2hgi
    L_p2r uses (poi_emb_for_reg, region_emb_exp)  ← reg path only
    L_r2c uses (region_emb, city_emb)             ← reg path only

POI2Vec is loaded frozen from HGI's pre-trained
``poi2vec_poi_embeddings_<State>.csv`` (64-dim per placeid).

Outputs to ``output/check2hgi_design_b/<state>/``:
  - embeddings.parquet       (per-check-in, from canonical encoder — cat-grade)
  - poi_embeddings.parquet   (per-POI from poi_emb_for_reg — fclass-aware)
  - region_embeddings.parquet (reg-grade)

Usage::

    python scripts/probe/build_design_b_poi_pool.py --state alabama --epochs 500
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from tqdm import trange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "research"))

from embeddings.check2hgi.model.Check2HGIModule import Check2HGI, corruption
from embeddings.check2hgi.model.CheckinEncoder import CheckinEncoder
from embeddings.check2hgi.model.Checkin2POI import Checkin2POI
from embeddings.hgi.model.RegionEncoder import POI2Region


POI2VEC_DIM = 64


class Check2HGI_DesignB(Check2HGI):
    """Adds a POI2Vec residual at the POI-pool boundary for the reg path only.

    The cat path (encoder + L_c2p) operates on the canonical poi_emb. The
    reg path (POI2Region, L_p2r, L_r2c) operates on poi_emb_for_reg, which
    is the canonical POI vector with `.detach()` applied + a learnable
    Linear projection of frozen POI2Vec features.
    """

    def __init__(self, *args, poi2vec_table: torch.Tensor, gamma_init: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.poi2vec_proj = nn.Linear(POI2VEC_DIM, self.hidden_channels, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.register_buffer("poi2vec_table", poi2vec_table.float())

    def forward(self, data):
        num_pois = data.num_pois
        num_regions = data.num_regions

        # Active encoder pass (canonical)
        pos_checkin_emb = self.checkin_encoder(data.x, data.edge_index, data.edge_weight)
        cor_x = self.corruption(data.x)
        neg_checkin_emb = self.checkin_encoder(cor_x, data.edge_index, data.edge_weight)

        # Canonical POI pool (active grad — used by L_c2p)
        pos_poi_emb_canonical = self.checkin2poi(pos_checkin_emb, data.checkin_to_poi, num_pois)
        neg_poi_emb_canonical = self.checkin2poi(neg_checkin_emb, data.checkin_to_poi, num_pois)

        # POI2Vec-enriched POI emb for reg path: detach the canonical pool
        # output so gradients from L_p2r / L_r2c never reach the encoder.
        poi2vec_residual = self.poi2vec_proj(self.poi2vec_table)  # [N_pois, D]
        pos_poi_emb_for_reg = pos_poi_emb_canonical.detach() + self.gamma * poi2vec_residual
        neg_poi_emb_for_reg = neg_poi_emb_canonical.detach() + self.gamma * poi2vec_residual

        # Region path
        pos_region_emb = self.poi2region(pos_poi_emb_for_reg, data.poi_to_region, data.region_adjacency)
        neg_region_emb = self.poi2region(neg_poi_emb_for_reg, data.poi_to_region, data.region_adjacency)
        city_emb = self.region2city(pos_region_emb, data.region_area)

        # Stored embeddings:
        #   checkin_embedding ← active encoder output (cat-grade)
        #   poi_embedding     ← reg-side enriched POI vectors (fclass-aware)
        #   region_embedding  ← reg-grade
        self.checkin_embedding = pos_checkin_emb
        self.poi_embedding = pos_poi_emb_for_reg
        self.region_embedding = pos_region_emb

        # ---- L_c2p inputs: canonical c2hgi (no POI2Vec contamination) ----
        pos_poi_expanded = pos_poi_emb_canonical[data.checkin_to_poi]
        neg_poi_indices = self._sample_negative_indices(
            data.checkin_to_poi, num_pois, data.x.device
        )
        neg_poi_expanded = pos_poi_emb_canonical[neg_poi_indices]

        # ---- L_p2r inputs: reg-side POI vectors ----
        pos_region_expanded = pos_region_emb[data.poi_to_region]
        neg_region_indices = self._sample_negative_indices_with_similarity(
            data.poi_to_region, num_regions,
            data.coarse_region_similarity, data.x.device
        )
        neg_region_expanded = pos_region_emb[neg_region_indices]

        return (
            pos_checkin_emb, pos_poi_expanded, neg_poi_expanded,
            pos_poi_emb_for_reg, pos_region_expanded, neg_region_expanded,
            pos_region_emb, neg_region_emb, city_emb,
        )


def load_poi2vec_table(state: str, num_pois: int, placeid_to_idx: dict) -> torch.Tensor:
    state_lc = state.lower()
    state_cap = state.capitalize()
    csv = REPO / f"output/hgi/{state_lc}/poi2vec_poi_embeddings_{state_cap}.csv"
    df = pd.read_csv(csv)
    emb_cols = [str(i) for i in range(POI2VEC_DIM)]
    arr = np.zeros((num_pois, POI2VEC_DIM), dtype=np.float32)
    seen = np.zeros(num_pois, dtype=bool)
    for placeid, vec in zip(df["placeid"].astype(int).tolist(), df[emb_cols].to_numpy(np.float32)):
        idx = placeid_to_idx.get(placeid)
        if idx is None:
            continue
        arr[idx] = vec
        seen[idx] = True
    n_unmapped = int((~seen).sum())
    if n_unmapped > 0:
        print(f"  WARN {n_unmapped} POIs missing from POI2Vec — left zero")
    return torch.from_numpy(arr)


def train_design_b(state: str, args: argparse.Namespace) -> None:
    state_lc = state.lower()
    out_dir = REPO / "output" / "check2hgi_design_b" / state_lc
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the CANONICAL c2hgi graph (11-dim features), NOT the POI2Vec-augmented one.
    graph_path = REPO / "output" / "check2hgi" / state_lc / "temp" / "checkin_graph.pt"
    with open(graph_path, "rb") as f:
        d = pickle.load(f)

    in_channels = d["node_features"].shape[1]
    num_checkins = d["num_checkins"]
    num_pois = d["num_pois"]
    num_regions = d["num_regions"]
    print(f"[{state_lc}] checkins={num_checkins} pois={num_pois} regions={num_regions} "
          f"feature_dim={in_channels} (canonical 11-dim — NOT augmented)")

    device = torch.device(args.device)

    poi2vec_table = load_poi2vec_table(state, num_pois, d["placeid_to_idx"])

    data = Data(
        x=torch.tensor(d["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(d["edge_index"], dtype=torch.int64),
        edge_weight=torch.tensor(d["edge_weight"], dtype=torch.float32),
        checkin_to_poi=torch.tensor(d["checkin_to_poi"], dtype=torch.int64),
        poi_to_region=torch.tensor(d["poi_to_region"], dtype=torch.int64),
        region_adjacency=torch.tensor(d["region_adjacency"], dtype=torch.int64),
        region_area=torch.tensor(d["region_area"], dtype=torch.float32),
        coarse_region_similarity=torch.tensor(d["coarse_region_similarity"], dtype=torch.float32),
        num_pois=num_pois,
        num_regions=num_regions,
    ).to(device)

    metadata = d["metadata"]

    checkin_encoder = CheckinEncoder(in_channels, args.dim, num_layers=args.num_layers)
    checkin2poi = Checkin2POI(args.dim, args.attention_head)
    poi2region = POI2Region(args.dim, args.attention_head)

    def region2city(z, area):
        return torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1))

    model = Check2HGI_DesignB(
        hidden_channels=args.dim,
        checkin_encoder=checkin_encoder,
        checkin2poi=checkin2poi,
        poi2region=poi2region,
        region2city=region2city,
        corruption=corruption,
        alpha_c2p=args.alpha_c2p,
        alpha_p2r=args.alpha_p2r,
        alpha_r2c=args.alpha_r2c,
        poi2vec_table=poi2vec_table,
        gamma_init=args.gamma_init,
    ).to(device)
    print(f"[{state_lc}] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    t = trange(1, args.epochs + 1, desc=f"Train Design B [{state_lc}]")
    lowest = math.inf
    best_epoch = 0
    best_state = None
    for epoch in t:
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss(*outputs)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        scheduler.step()
        l = loss.item()
        if l < lowest:
            lowest = l
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        t.set_postfix(loss=f"{l:.4f}", best=f"{lowest:.4f}", best_ep=best_epoch,
                      gamma=f"{model.gamma.item():.3f}")

    print(f"[{state_lc}] best_epoch={best_epoch} loss={lowest:.4f} gamma={model.gamma.item():.3f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _ = model(data)
        checkin_emb, poi_emb, region_emb = model.get_embeddings()

    emb_np = checkin_emb.numpy()
    df = pd.DataFrame(emb_np, columns=[f"{i}" for i in range(emb_np.shape[1])])
    df.insert(0, "datetime", metadata["datetime"].values)
    df.insert(0, "category", metadata["category"].values)
    df.insert(0, "placeid", metadata["placeid"].values)
    df.insert(0, "userid", metadata["userid"].values)
    df.to_parquet(out_dir / "embeddings.parquet", index=False)
    print(f"[{state_lc}] wrote embeddings.parquet shape={emb_np.shape}")

    poi_np = poi_emb.numpy()
    poi_df = pd.DataFrame(poi_np, columns=[f"{i}" for i in range(poi_np.shape[1])])
    placeid_to_idx = d["placeid_to_idx"]
    idx_to_placeid = {v: k for k, v in placeid_to_idx.items()}
    poi_df.insert(0, "placeid", [idx_to_placeid.get(i, i) for i in range(len(poi_df))])
    poi_df.to_parquet(out_dir / "poi_embeddings.parquet", index=False)
    print(f"[{state_lc}] wrote poi_embeddings.parquet shape={poi_np.shape}")

    reg_np = region_emb.numpy()
    reg_df = pd.DataFrame(reg_np, columns=[f"reg_{i}" for i in range(reg_np.shape[1])])
    reg_df.insert(0, "region_id", range(num_regions))
    reg_df.to_parquet(out_dir / "region_embeddings.parquet", index=False)
    print(f"[{state_lc}] wrote region_embeddings.parquet shape={reg_np.shape}")

    (out_dir / "temp").mkdir(exist_ok=True)
    import shutil
    shutil.copy(graph_path, out_dir / "temp" / "checkin_graph.pt")
    src_seq = REPO / f"output/check2hgi/{state_lc}/temp/sequences_next.parquet"
    if src_seq.exists():
        shutil.copy(src_seq, out_dir / "temp" / "sequences_next.parquet")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--num-layers", dest="num_layers", type=int, default=2)
    ap.add_argument("--attention-head", dest="attention_head", type=int, default=4)
    ap.add_argument("--alpha-c2p", dest="alpha_c2p", type=float, default=0.4)
    ap.add_argument("--alpha-p2r", dest="alpha_p2r", type=float, default=0.3)
    ap.add_argument("--alpha-r2c", dest="alpha_r2c", type=float, default=0.3)
    ap.add_argument("--gamma-init", dest="gamma_init", type=float, default=1.0,
                    help="Initial value of γ (POI2Vec residual scale).")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max-norm", dest="max_norm", type=float, default=0.9)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    train_design_b(args.state, args)


if __name__ == "__main__":
    main()
