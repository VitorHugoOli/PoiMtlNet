"""Design K — Design J + HGI's Delaunay POI-POI edges as a POI-level GCN.

Same anchor regularizer as J (λ pulls learnable table to POI2Vec). The new
piece: between Checkin2POI and POI2Region, run a GCN over the POIs using
HGI's Delaunay triangulation edges (edge weights from
`output/hgi/<state>/temp/edges.csv`). This injects HGI's spatial-functional
graph structure — the load-bearing piece HGI has and c2hgi's merge family
lacks — directly into the reg path while leaving the cat path detached.

    poi_emb_canonical    = Checkin2POI(checkin_emb)              # cat path
    poi_residual         = γ · poi_table[i]                       # learnable
    poi_emb_for_reg_in   = poi_emb_canonical.detach() + poi_residual
    poi_emb_for_reg      = poi_gcn(poi_emb_for_reg_in,           # NEW: Delaunay GCN
                                   delaunay_edge_index, delaunay_edge_weight)
    region_emb           = POI2Region(poi_emb_for_reg, ...)      # reg path

Loss:
    L_total = L_c2hgi + λ · ‖ poi_table − POI2Vec ‖²

Usage::

    python scripts/probe/build_design_k_delaunay.py --state alabama --epochs 500 \\
        --device mps --anchor-lambda 0.5 --out-suffix l0_5
"""
from __future__ import annotations
import argparse, math, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import trange

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "research"))

from embeddings.check2hgi.model.Check2HGIModule import Check2HGI, corruption
from embeddings.check2hgi.model.CheckinEncoder import CheckinEncoder
from embeddings.check2hgi.model.variants import ResidualLNEncoder
from embeddings.check2hgi.model.Checkin2POI import Checkin2POI
from embeddings.hgi.model.RegionEncoder import POI2Region

POI2VEC_DIM = 64


class Check2HGI_DesignK(Check2HGI):
    """Design J + Delaunay POI-POI GCN before POI2Region."""

    def __init__(self, *args, num_pois, poi2vec_anchor: torch.Tensor,
                 gamma_init=1.0, side_features: torch.Tensor | None = None,
                 side_feature_hidden: int = 16,
                 hgi_poi_target: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.poi_table = nn.Embedding(num_pois, self.hidden_channels)
        with torch.no_grad():
            self.poi_table.weight.copy_(poi2vec_anchor.float())
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.register_buffer("poi2vec_anchor", poi2vec_anchor.float())
        self.poi_gcn = GCNConv(self.hidden_channels, self.hidden_channels,
                               cached=True, bias=True)
        self.poi_gcn_act = nn.PReLU(self.hidden_channels)

        # design_k+sidefeat (CORRECTED, pre-GCN): inject side-features as additional
        # POI node features BEFORE the Delaunay GCN so they PROPAGATE through the
        # spatial graph — mirroring how HGI consumes POI2Vec features (data.x →
        # POIEncoder/Delaunay GCN, HGIModule.py:112). The earlier post-GCN concat+
        # LayerNorm version washed the spatial structure and never diffused the
        # features (falsified: FL 0.7320 < base 0.7341). side_proj now outputs
        # hidden_channels for the additive pre-GCN injection; no post-GCN LayerNorm.
        if side_features is not None:
            self.register_buffer("side_features", side_features.float())
            self.side_proj = nn.Sequential(
                nn.Linear(int(side_features.shape[1]), self.hidden_channels), nn.PReLU())
        else:
            self.side_features = None
            self.side_proj = None

        # #5 HGI-POI-decoder distill: a small MLP reconstructs HGI's learned POI
        # embedding (256-d) from design_k's reg-path POI vector. Loss adds
        # γ·‖Dec(poi_for_reg) − hgi_poi.detach()‖² → transfers HGI's hierarchical
        # geometry (a DIFFERENT axis than design_k's spatial). future_works memo §4.8.
        if hgi_poi_target is not None:
            self.register_buffer("hgi_poi_target", hgi_poi_target.float())
            self.register_buffer("hgi_poi_mask", (hgi_poi_target.abs().sum(1) > 0).float())
            self.hgi_decoder = nn.Sequential(
                nn.Linear(self.hidden_channels, 128), nn.PReLU(),
                nn.Linear(128, int(hgi_poi_target.shape[1])))
        else:
            self.hgi_poi_target = None
            self.hgi_decoder = None
        self._poi_for_reg_cache = None

    def forward(self, data):
        num_pois = data.num_pois
        num_regions = data.num_regions

        pos_checkin_emb = self.checkin_encoder(data.x, data.edge_index, data.edge_weight)
        cor_x = self.corruption(data.x)
        neg_checkin_emb = self.checkin_encoder(cor_x, data.edge_index, data.edge_weight)

        pos_poi_emb_canonical = self.checkin2poi(pos_checkin_emb, data.checkin_to_poi, num_pois)
        neg_poi_emb_canonical = self.checkin2poi(neg_checkin_emb, data.checkin_to_poi, num_pois)

        # speed: nn.Embedding(N).weight is identical to embedding(arange(N))
        poi_residual = self.poi_table.weight

        # detach() severs cat-path encoder from reg-axis loss (matches J)
        pos_pre_gcn = pos_poi_emb_canonical.detach() + self.gamma * poi_residual
        neg_pre_gcn = neg_poi_emb_canonical.detach() + self.gamma * poi_residual

        # design_k+sidefeat (CORRECTED): add side-features as POI node features
        # BEFORE the Delaunay GCN so they diffuse through the spatial graph.
        if self.side_proj is not None:
            side_h = self.side_proj(self.side_features)
            pos_pre_gcn = pos_pre_gcn + side_h
            neg_pre_gcn = neg_pre_gcn + side_h

        # Delaunay POI-POI GCN — HGI's spatial sauce
        pos_poi_emb_for_reg = self.poi_gcn_act(
            self.poi_gcn(pos_pre_gcn, data.delaunay_edge_index, data.delaunay_edge_weight))
        neg_poi_emb_for_reg = self.poi_gcn_act(
            self.poi_gcn(neg_pre_gcn, data.delaunay_edge_index, data.delaunay_edge_weight))

        pos_region_emb = self.poi2region(pos_poi_emb_for_reg, data.poi_to_region, data.region_adjacency)
        neg_region_emb = self.poi2region(neg_poi_emb_for_reg, data.poi_to_region, data.region_adjacency)
        city_emb = self.region2city(pos_region_emb, data.region_area)

        self.checkin_embedding = pos_checkin_emb
        self.poi_embedding = pos_poi_emb_for_reg
        self.region_embedding = pos_region_emb
        self._poi_for_reg_cache = pos_poi_emb_for_reg

        pos_poi_expanded = pos_poi_emb_canonical[data.checkin_to_poi]
        neg_poi_indices = self._sample_negative_indices(data.checkin_to_poi, num_pois, data.x.device)
        neg_poi_expanded = pos_poi_emb_canonical[neg_poi_indices]
        pos_region_expanded = pos_region_emb[data.poi_to_region]
        neg_region_indices = self._sample_negative_indices_with_similarity(
            data.poi_to_region, num_regions, data.coarse_region_similarity, data.x.device)
        neg_region_expanded = pos_region_emb[neg_region_indices]

        return (
            pos_checkin_emb, pos_poi_expanded, neg_poi_expanded,
            pos_poi_emb_for_reg, pos_region_expanded, neg_region_expanded,
            pos_region_emb, neg_region_emb, city_emb,
        )

    def anchor_loss(self):
        return ((self.poi_table.weight - self.poi2vec_anchor) ** 2).mean()

    def decoder_loss(self):
        if self.hgi_decoder is None:
            return torch.tensor(0.0, device=self.poi_table.weight.device)
        pred = self.hgi_decoder(self._poi_for_reg_cache)        # [N_pois, 256]
        sq = ((pred - self.hgi_poi_target) ** 2).mean(1)        # per-POI
        m = self.hgi_poi_mask
        return (sq * m).sum() / m.sum().clamp(min=1)            # mean over mapped POIs


def load_poi2vec(state: str, num_pois: int, placeid_to_idx: dict) -> torch.Tensor:
    state_lc = state.lower(); state_cap = state.capitalize()
    csv = REPO / f"output/hgi/{state_lc}/poi2vec_poi_embeddings_{state_cap}.csv"
    df = pd.read_csv(csv)
    emb_cols = [str(i) for i in range(POI2VEC_DIM)]
    arr = np.zeros((num_pois, POI2VEC_DIM), dtype=np.float32)
    for placeid, vec in zip(df["placeid"].astype(int).tolist(), df[emb_cols].to_numpy(np.float32)):
        idx = placeid_to_idx.get(placeid)
        if idx is not None:
            arr[idx] = vec
    return torch.from_numpy(arr)


def load_hgi_poi_target(state: str, num_pois: int, placeid_to_idx: dict):
    """Load HGI's learned POI embedding (poi_embeddings.parquet) mapped to c2hgi idx.
    Returns [num_pois, hgi_dim] with zero rows for unmapped POIs (#5 decoder target)."""
    f = REPO / "output" / "hgi" / state.lower() / "poi_embeddings.parquet"
    if not f.exists():
        f = REPO / "output" / "hgi" / state.lower() / "embeddings.parquet"  # HGI POI-level emb
    df = pd.read_parquet(f)
    dim = [c for c in df.columns if c not in ("placeid", "category") and not c.startswith("reg_")]
    arr = np.zeros((num_pois, len(dim)), dtype=np.float32)
    for pid, vec in zip(df["placeid"].astype(int).tolist(), df[dim].to_numpy(np.float32)):
        idx = placeid_to_idx.get(int(pid))
        if idx is not None:
            arr[idx] = vec
    print(f"[#5 decoder] HGI POI target dim={len(dim)} mapped={int((arr.any(1)).sum())}/{num_pois}")
    return torch.from_numpy(arr)


def load_delaunay_edges(state: str, placeid_to_idx: dict, num_pois: int,
                        poi_to_region=None, cross_region_weight: float = 1.0,
                        edge_power: float = 1.0):
    """Load HGI's Delaunay edges and remap to c2hgi's POI index space.

    HGI's edges.csv has (source, target, weight) where source/target are
    row indices into HGI's pois.csv. Map row → placeid → c2hgi POI idx.

    T6.2 edge-weight re-tune (cross_region_weight<1 / edge_power>1): multiply
    CROSS-region edge weights by ``cross_region_weight`` (focuses GCN smoothing
    WITHIN regions → raises region cohesion) and sharpen via ``weight**edge_power``.
    Defaults (1.0/1.0) reproduce the base design_k edge load byte-for-byte.
    """
    state_lc = state.lower()
    pois_path = REPO / "output" / "hgi" / state_lc / "temp" / "pois.csv"
    edges_path = REPO / "output" / "hgi" / state_lc / "temp" / "edges.csv"
    hgi_pois = pd.read_csv(pois_path)
    edges = pd.read_csv(edges_path)

    # row index in hgi_pois → c2hgi POI idx
    row_to_c2hgi = {}
    for row_idx, pid in enumerate(hgi_pois["placeid"].tolist()):
        c2_idx = placeid_to_idx.get(int(pid))
        if c2_idx is not None:
            row_to_c2hgi[row_idx] = c2_idx

    src = []
    tgt = []
    w = []
    n_skip = 0
    for s, t, weight in zip(edges["source"].tolist(), edges["target"].tolist(),
                             edges["weight"].tolist()):
        s_c = row_to_c2hgi.get(int(s))
        t_c = row_to_c2hgi.get(int(t))
        if s_c is None or t_c is None:
            n_skip += 1
            continue
        wv = float(weight) ** edge_power
        # T6.2: down-weight cross-region edges
        if poi_to_region is not None and cross_region_weight != 1.0:
            if int(poi_to_region[s_c]) != int(poi_to_region[t_c]):
                wv *= cross_region_weight
        # symmetrise
        src.append(s_c); tgt.append(t_c); w.append(wv)
        src.append(t_c); tgt.append(s_c); w.append(wv)

    edge_index = torch.tensor([src, tgt], dtype=torch.int64)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    print(f"[delaunay] state={state} loaded {edge_index.shape[1]} edges "
          f"(skipped {n_skip} unmapped from {len(edges)} raw)")
    return edge_index, edge_weight


def train(state: str, args):
    state_lc = state.lower()
    # Reproducibility (unsupervised build is seed-dependent). Pinned by default.
    seed = int(getattr(args, "seed", 42))
    torch.manual_seed(seed); np.random.seed(seed)
    suffix = getattr(args, "out_suffix", "") or ""
    base = "check2hgi_design_k" + (f"_{suffix}" if suffix else "")
    out_dir = REPO / "output" / base / state_lc
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_path = REPO / "output" / "check2hgi" / state_lc / "temp" / "checkin_graph.pt"
    with open(graph_path, "rb") as f:
        d = pickle.load(f)

    in_channels = d["node_features"].shape[1]
    num_pois = d["num_pois"]; num_regions = d["num_regions"]
    print(f"[{state_lc}] pois={num_pois} regions={num_regions} feat={in_channels} "
          f"λ={args.anchor_lambda} out={out_dir.name}")

    device = torch.device(args.device)
    poi2vec = load_poi2vec(state, num_pois, d["placeid_to_idx"])
    del_ei, del_ew = load_delaunay_edges(
        state, d["placeid_to_idx"], num_pois,
        poi_to_region=np.asarray(d["poi_to_region"]),
        cross_region_weight=getattr(args, "cross_region_weight", 1.0),
        edge_power=getattr(args, "edge_power", 1.0))

    # design_k+sidefeat: optionally load the T4.3 32-d POI side-feature tensor.
    side_features = None
    if getattr(args, "use_side_features", False):
        sf_path = REPO / "output" / "check2hgi" / state_lc / "poi_side_features.pt"
        if not sf_path.exists():
            import subprocess, os as _os
            r = subprocess.run(
                [sys.executable, str(REPO / "scripts/canonical_improvement/compute_poi_side_features.py"),
                 "--state", state], cwd=str(REPO),
                env={**_os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}/research"})
            if r.returncode != 0:
                raise RuntimeError(f"compute_poi_side_features.py failed rc={r.returncode}")
        sf = torch.load(sf_path)
        side_features = (sf["features"] if isinstance(sf, dict) else sf)
        if side_features.shape[0] != num_pois:
            raise ValueError(f"side_features rows {side_features.shape[0]} != num_pois {num_pois}")
        print(f"[{state_lc}] side_features attached shape={tuple(side_features.shape)}")

    data = Data(
        x=torch.tensor(d["node_features"], dtype=torch.float32),
        edge_index=torch.tensor(d["edge_index"], dtype=torch.int64),
        edge_weight=torch.tensor(d["edge_weight"], dtype=torch.float32),
        checkin_to_poi=torch.tensor(d["checkin_to_poi"], dtype=torch.int64),
        poi_to_region=torch.tensor(d["poi_to_region"], dtype=torch.int64),
        region_adjacency=torch.tensor(d["region_adjacency"], dtype=torch.int64),
        region_area=torch.tensor(d["region_area"], dtype=torch.float32),
        coarse_region_similarity=torch.tensor(d["coarse_region_similarity"], dtype=torch.float32),
        delaunay_edge_index=del_ei,
        delaunay_edge_weight=del_ew,
        num_pois=num_pois, num_regions=num_regions,
    ).to(device)
    metadata = d["metadata"]

    _encoder = getattr(args, "encoder", "gcn") or "gcn"
    if _encoder == "resln":
        checkin_encoder = ResidualLNEncoder(in_channels, args.dim, num_layers=args.num_layers)
        print(f"[{state_lc}] encoder=ResidualLNEncoder num_layers={args.num_layers}")
    else:
        checkin_encoder = CheckinEncoder(in_channels, args.dim, num_layers=args.num_layers)
        print(f"[{state_lc}] encoder=CheckinEncoder (canonical GCN) num_layers={args.num_layers}")
    checkin2poi = Checkin2POI(args.dim, args.attention_head)
    poi2region = POI2Region(args.dim, args.attention_head)

    def region2city(z, area):
        return torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1))

    model = Check2HGI_DesignK(
        hidden_channels=args.dim, checkin_encoder=checkin_encoder,
        checkin2poi=checkin2poi, poi2region=poi2region,
        region2city=region2city, corruption=corruption,
        alpha_c2p=args.alpha_c2p, alpha_p2r=args.alpha_p2r, alpha_r2c=args.alpha_r2c,
        num_pois=num_pois, poi2vec_anchor=poi2vec, gamma_init=args.gamma_init,
        side_features=side_features, side_feature_hidden=args.side_feature_hidden,
        hgi_poi_target=(load_hgi_poi_target(state, num_pois, d["placeid_to_idx"])
                        if getattr(args, "hgi_decoder_gamma", 0.0) > 0.0 else None),
    ).to(device)
    print(f"[{state_lc}] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=getattr(args, "weight_decay", 0.0))
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) if args.gamma != 1.0 else None

    t = trange(1, args.epochs + 1, desc=f"Train K[{state_lc}]")
    lowest = math.inf; best_epoch = 0; best_state = None
    POSTFIX_EVERY = 25
    for epoch in t:
        model.train(); optimizer.zero_grad()
        outputs = model(data)
        loss_main = model.loss(*outputs)
        loss_anchor = model.anchor_loss()
        loss = loss_main + args.anchor_lambda * loss_anchor
        if getattr(args, "hgi_decoder_gamma", 0.0) > 0.0:
            loss = loss + args.hgi_decoder_gamma * model.decoder_loss()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        l = loss.item()
        if l < lowest:
            lowest = l; best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % POSTFIX_EVERY == 0 or epoch == args.epochs:
            t.set_postfix(loss=f"{l:.4f}", best_ep=best_epoch, refresh=False)
            t.refresh()

    print(f"[{state_lc}] best_epoch={best_epoch} loss={lowest:.4f}")
    model.load_state_dict(best_state); model.eval()
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

    poi_np = poi_emb.numpy()
    poi_df = pd.DataFrame(poi_np, columns=[f"{i}" for i in range(poi_np.shape[1])])
    placeid_to_idx = d["placeid_to_idx"]
    idx_to_placeid = {v: k for k, v in placeid_to_idx.items()}
    poi_df.insert(0, "placeid", [idx_to_placeid.get(i, i) for i in range(len(poi_df))])
    poi_df.to_parquet(out_dir / "poi_embeddings.parquet", index=False)

    reg_np = region_emb.numpy()
    reg_df = pd.DataFrame(reg_np, columns=[f"reg_{i}" for i in range(reg_np.shape[1])])
    reg_df.insert(0, "region_id", range(num_regions))
    reg_df.to_parquet(out_dir / "region_embeddings.parquet", index=False)

    (out_dir / "temp").mkdir(exist_ok=True)
    import shutil
    shutil.copy(graph_path, out_dir / "temp" / "checkin_graph.pt")
    src_seq = REPO / f"output/check2hgi/{state_lc}/temp/sequences_next.parquet"
    if src_seq.exists():
        shutil.copy(src_seq, out_dir / "temp" / "sequences_next.parquet")
    print(f"[{state_lc}] wrote {out_dir}")


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
    ap.add_argument("--gamma-init", dest="gamma_init", type=float, default=1.0)
    ap.add_argument("--anchor-lambda", dest="anchor_lambda", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max-norm", dest="max_norm", type=float, default=0.9)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42,
                    help="Reproducibility seed (encoder init + negatives). Pinned by default.")
    ap.add_argument("--encoder", type=str, default="gcn", choices=["gcn", "resln"],
                    help="Check-in encoder. 'gcn' (canonical) or 'resln' (ResidualLNEncoder).")
    ap.add_argument("--use-side-features", dest="use_side_features", action="store_true",
                    help="Stack T4.3 POI side-features (32d) after the Delaunay GCN (reg path).")
    ap.add_argument("--side-feature-hidden", dest="side_feature_hidden", type=int, default=16)
    ap.add_argument("--cross-region-weight", dest="cross_region_weight", type=float, default=1.0,
                    help="T6.2: multiply cross-region Delaunay edge weights by this (<1 focuses "
                         "intra-region smoothing, raises region cohesion). 1.0 = base design_k.")
    ap.add_argument("--edge-power", dest="edge_power", type=float, default=1.0,
                    help="T6.2: sharpen Delaunay edge weights via weight**power. 1.0 = base.")
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0,
                    help="v3c re-screen: AdamW-style weight decay on the build optimizer.")
    ap.add_argument("--hgi-decoder-gamma", dest="hgi_decoder_gamma", type=float, default=0.0,
                    help="#5: coefficient on the HGI-POI-embedding distillation decoder loss "
                         "(memo §4.8). 0.0 = off. Sweep {0.05,0.1,0.3}.")
    ap.add_argument("--out-suffix", dest="out_suffix", type=str, default="",
                    help="Append suffix to output dir (e.g. 'l0_5' → output/check2hgi_design_k_l0_5/)")
    args = ap.parse_args()
    train(args.state, args)


if __name__ == "__main__":
    main()
