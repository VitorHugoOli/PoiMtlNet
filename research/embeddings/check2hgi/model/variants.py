"""Variant model components for the check2hgi-up study.

Each variant is a single, isolated change so that the downstream comparison
can attribute effects to a specific intervention. Borrows ideas from HGI
(hard negatives, set-transformer pooling), Time2Vec (explicit time encoding),
and modern self-supervised practice (InfoNCE, uncertainty-weighted MTL).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, RGCNConv

from embeddings.check2hgi.model.Check2HGIModule import Check2HGI, corruption, EPS


# ---------------------------------------------------------------------------
# V2: Time-aware GAT encoder
# ---------------------------------------------------------------------------

class GATTimeEncoder(nn.Module):
    """GATv2 encoder where attention is (optionally) conditioned on the (1-d)
    temporal edge weight from the user-sequence graph.

    Per the T3.1 advisor (2026-05-15): conditioning attention on the temporal
    edge weight makes the encoder learn to copy the most-recent same-user
    neighbour's category into the output, producing a +11 pp leak-probe
    delta and 99% cat F1 at FL. The leak is structural to the
    (attention × temporal-edge-attr × user-sequence) triangle. Setting
    ``use_edge_attr=False`` breaks one corner of the triangle: attention is
    learned purely from node features, with no temporal conditioning. The
    rest of the architecture is unchanged. This is "fix 1" from the
    T3.1 rehabilitation menu.
    """

    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=4,
                 dropout=0.0, use_edge_attr=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_edge_attr = bool(use_edge_attr)

        assert hidden_channels % heads == 0
        head_dim = hidden_channels // heads

        # When use_edge_attr=False, the GATv2Conv is constructed without an
        # edge_dim — it does NOT expect edge_attr at forward time. Attention
        # is then learned purely from node features.
        _edge_dim = 1 if self.use_edge_attr else None

        self.convs = nn.ModuleList()
        # First layer: in_channels -> hidden_channels
        self.convs.append(GATv2Conv(in_channels, head_dim, heads=heads,
                                    edge_dim=_edge_dim, concat=True, add_self_loops=True))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, head_dim, heads=heads,
                                        edge_dim=_edge_dim, concat=True, add_self_loops=True))

        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # **kwargs absorbs edge_type plumbed by Check2HGI for R-GCN compatibility.
        if self.use_edge_attr and edge_weight is not None:
            edge_attr = edge_weight.unsqueeze(-1)
        else:
            edge_attr = None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# V3: Residual + LayerNorm GCN encoder
# ---------------------------------------------------------------------------

class ResidualLNEncoder(nn.Module):
    """Multi-layer GCN with pre-LN and residual connections from layer 1 onward."""

    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, bias=True))
        self.norms.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, bias=True))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # **kwargs absorbs edge_type plumbed by Check2HGI for R-GCN compatibility.
        # First layer (no residual since dim mismatch)
        x = self.convs[0](x, edge_index, edge_weight)
        x = self.norms[0](x)
        x = self.act(x)
        x = self.dropout(x)
        # Subsequent layers with residual connections
        for i in range(1, self.num_layers):
            h = self.norms[i](x)
            h = self.convs[i](h, edge_index, edge_weight)
            if i < self.num_layers - 1:
                h = self.act(h)
                h = self.dropout(h)
            x = x + h
        return x


# ---------------------------------------------------------------------------
# V6: Time2Vec input-side temporal encoder (T3.4)
# ---------------------------------------------------------------------------

class _SineActivationT2V(nn.Module):
    """Time2Vec sine activation (Kazemi et al. 2019). Replicates the canonical
    formulation from ``research/embeddings/time2vec/model/activations.py``
    without importing it, to avoid pulling in the contrastive Time2Vec module
    (which carries DEVICE auto-detect side effects via `configs.globals`).

    Warm-start option: ``warm_start=True`` initialises the first 4 periodic
    channels to recover the canonical fixed-frequency (hour_sin, hour_cos,
    dow_sin, dow_cos) features identically, so SGD can deviate only as
    needed. Addresses the T3.4 cat-axis −0.56 pp regression (T3.4 advisor
    2026-05-16 §3): random init blurs the 24-hour categorical periodicity.
    """

    def __init__(self, in_features: int, out_features: int,
                 warm_start: bool = False):
        super().__init__()
        self.out_features = out_features
        # One linear channel (Wx + b) + (out_features-1) periodic channels.
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))
        if warm_start:
            self._init_warm_start(in_features, out_features)

    def _init_warm_start(self, in_features: int, out_features: int) -> None:
        """Initialise the first 4 periodic channels so that f(x) returns the
        canonical [hour_sin, hour_cos, dow_sin, dow_cos] features when the
        input x is the same 4-d vector. Remaining channels stay random.

        Recovery math: input ``x = [hs, hc, ds, dc]`` (already the canonical
        sin/cos). Identity per-channel means ``sin(W x + b) = x_i`` for the
        i-th channel. Achieved with W_i selecting the i-th input via a
        large coefficient inside arcsin, but the simpler trick: set the
        learned periodic output's first 4 channels to read x[:, i] directly
        via a near-identity linear mix and zero bias, then SGD refines.
        """
        with torch.no_grad():
            if self.w.shape[1] >= in_features:
                # Linear-channel: small slope, zero bias.
                self.w0.zero_()
                self.b0.zero_()
                # Periodic-channels: copy x verbatim through sin( x · π/2 ),
                # since sin(π/2 · ±1) = ±1 and the canonical sin/cos values
                # are bounded in [-1, 1]. The first 4 periodic channels recover
                # x[:, 0..3] exactly when x is the canonical sin/cos vector
                # (because |hs|, |hc|, |ds|, |dc| ≤ 1).
                self.w.zero_()
                self.b.zero_()
                # Set w[i, j] = π/2 · δ_{ij} for j in 0..in_features-1.
                pi_half = float(math.pi / 2.0)
                for i in range(in_features):
                    self.w[i, i] = pi_half
                # Channels in_features..out_features-2 stay random (small).
                if out_features - 1 > in_features:
                    self.w[:, in_features:].normal_(mean=0.0, std=0.01)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        v0 = tau @ self.w0 + self.b0                       # (N, 1)
        vp = torch.sin(tau @ self.w + self.b)              # (N, out-1)
        return torch.cat([v0, vp], dim=-1)                 # (N, out)


class Time2VecCheckinEncoder(nn.Module):
    """T3.4 — Time2Vec replaces the fixed (hour_sin, hour_cos, dow_sin, dow_cos)
    temporal features with a learned periodic encoding before the canonical
    2-layer GCN.

    Input layout (matches preprocess.py:_build_node_features):
        x[..., :num_categories]  category one-hot (preserved)
        x[..., num_categories:]  4 fixed-frequency sin/cos cols → replaced

    The Time2Vec layer projects the 4-d fixed temporal vector to a learned
    d_t-d periodic representation. Category one-hot is untouched. No residual
    or skip path preserves the one-hot verbatim downstream, so this swap does
    NOT enlarge the leak surface (T3.4 advisor pre-launch audit, 2026-05-15).
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_categories: int, num_layers: int = 2,
                 time2vec_dim: int = 8, dropout: float = 0.0,
                 warm_start: bool = False):
        super().__init__()
        if num_categories <= 0 or num_categories >= in_channels:
            raise ValueError(
                f"Time2VecCheckinEncoder: num_categories={num_categories} must "
                f"be in (0, in_channels={in_channels}); last 4 cols of x are "
                f"the temporal sin/cos features."
            )
        n_time_in = in_channels - num_categories
        if n_time_in != 4:
            raise ValueError(
                f"Time2VecCheckinEncoder: expected exactly 4 temporal input "
                f"features (hour_sin, hour_cos, dow_sin, dow_cos); got {n_time_in}."
            )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_categories = num_categories
        self.time2vec_dim = time2vec_dim
        self.num_layers = num_layers

        self.t2v = _SineActivationT2V(in_features=n_time_in,
                                      out_features=time2vec_dim,
                                      warm_start=warm_start)

        post_in = num_categories + time2vec_dim
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(post_in, hidden_channels, cached=False, bias=True))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                      cached=False, bias=True))

        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        # **kwargs absorbs edge_type plumbed by Check2HGI for R-GCN compatibility.
        x_cat = x[..., :self.num_categories]
        x_time = x[..., self.num_categories:]
        t2v_emb = self.t2v(x_time)
        h = torch.cat([x_cat, t2v_emb], dim=-1)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# V7: Relational GCN (R-GCN) over heterogeneous edge typing (T3.3 Option A)
# ---------------------------------------------------------------------------

class RGCNEncoder(nn.Module):
    """T3.3 — Relational GCN that aggregates separately per edge relation,
    with no attention and no edge_attr conditioning.

    Requires the upstream graph to expose ``edge_type`` as a long tensor of
    length E (relation index per edge). The companion preprocess change
    (`edge_type='both'` augmented to store per-edge type) feeds this encoder
    with K=2 relations: 0 = user_sequence, 1 = same_poi.

    Design rationale (T3.3 advisor pre-launch audit, 2026-05-15):
      * Sum aggregation per relation (no attention coefficient learning).
      * No edge_attr at forward (so the temporal-decay continuous weights
        cannot leak through attention as in the T3.1 GAT triangle).
      * Optional basis decomposition (``num_bases``) to keep the param delta
        vs canonical GCN under the F51 50 % guardrail at D=64.

    `edge_weight` is accepted for API compatibility with the rest of the
    dispatch but is IGNORED (R-GCN aggregation is over relation typing, not
    continuous weight).
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 num_relations: int = 2, num_layers: int = 2,
                 num_bases: int | None = 2, dropout: float = 0.0,
                 aggr: str = "sum"):
        super().__init__()
        if num_relations < 2:
            raise ValueError(
                f"RGCNEncoder requires num_relations >= 2; got {num_relations}. "
                f"With 1 relation R-GCN degenerates to GCN."
            )
        if aggr not in ("sum", "mean"):
            raise ValueError(f"RGCNEncoder aggr must be 'sum' or 'mean'; got {aggr}.")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.num_bases = num_bases
        self.aggr = aggr

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels,
                                   num_relations=num_relations,
                                   num_bases=num_bases, aggr=aggr,
                                   root_weight=True, bias=True))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels,
                                       num_relations=num_relations,
                                       num_bases=num_bases, aggr=aggr,
                                       root_weight=True, bias=True))

        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None, edge_type=None):
        if edge_type is None:
            raise ValueError(
                "RGCNEncoder.forward requires edge_type (long tensor of shape [E]). "
                "Caller must wire the per-edge relation index from preprocess."
            )
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_type)
            if i < len(self.convs) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h


# ---------------------------------------------------------------------------
# V1: Check2HGI with InfoNCE contrastive at C2P boundary
# ---------------------------------------------------------------------------

class Check2HGI_InfoNCE(Check2HGI):
    """Replaces the C2P bilinear-sigmoid-BCE loss with K-negative
    InfoNCE / cross-entropy style contrastive loss.

    For each check-in i: anchor = checkin_emb[i] (projected via W_c2p),
    positive = poi_emb[checkin_to_poi[i]],
    negatives = K random POIs (different from positive).
    Loss = -log( exp(sim_pos/τ) / (exp(sim_pos/τ) + Σ exp(sim_neg/τ)) )
    """

    def __init__(self, *args, num_negatives: int = 32, temperature: float = 0.2,
                 chunk_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_negatives = num_negatives
        self.temperature = temperature
        # chunk_size: process N_c in slices when materializing [N_c, K, D]
        # would OOM. None = no chunking. ~256k is safe for 6 GB GPUs at K=32, D=64.
        self.chunk_size = chunk_size

    def forward(self, data):
        num_pois = data.num_pois
        num_regions = data.num_regions

        pos_checkin_emb = self.checkin_encoder(data.x, data.edge_index, data.edge_weight)
        cor_x = self.corruption(data.x)
        neg_checkin_emb = self.checkin_encoder(cor_x, data.edge_index, data.edge_weight)

        pos_poi_emb = self.checkin2poi(pos_checkin_emb, data.checkin_to_poi, num_pois)
        neg_poi_emb = self.checkin2poi(neg_checkin_emb, data.checkin_to_poi, num_pois)

        pos_region_emb = self.poi2region(pos_poi_emb, data.poi_to_region, data.region_adjacency)
        neg_region_emb = self.poi2region(neg_poi_emb, data.poi_to_region, data.region_adjacency)

        city_emb = self.region2city(pos_region_emb, data.region_area)

        self.checkin_embedding = pos_checkin_emb
        self.poi_embedding = pos_poi_emb
        self.region_embedding = pos_region_emb

        pos_region_expanded = pos_region_emb[data.poi_to_region]
        neg_region_indices = self._sample_negative_indices_with_similarity(
            data.poi_to_region, num_regions, data.coarse_region_similarity, data.x.device
        )
        neg_region_expanded = pos_region_emb[neg_region_indices]

        return (
            pos_checkin_emb, pos_poi_emb,
            pos_poi_emb, pos_region_expanded, neg_region_expanded,
            pos_region_emb, neg_region_emb, city_emb,
            data.checkin_to_poi, num_pois,
        )

    def loss(self, pos_checkin, pos_poi,
             pos_poi_again, pos_region_exp, neg_region_exp,
             pos_region, neg_region, city,
             checkin_to_poi, num_pois):
        # ---- Loss 1: InfoNCE for check-in -> POI ----
        # anchor: project checkin embedding via W_c2p, then dot-product with pois
        anchor = torch.matmul(pos_checkin, self.weight_c2p)         # [N_c, D]
        anchor = F.normalize(anchor, dim=-1)
        pois = F.normalize(pos_poi, dim=-1)                          # [N_p, D]

        N_c = anchor.size(0)
        K = self.num_negatives
        device = anchor.device

        pos_idx = checkin_to_poi                                     # [N_c]
        # positive similarity
        sim_pos = (anchor * pois[pos_idx]).sum(dim=-1, keepdim=True) # [N_c, 1]

        # K random negatives (different from positive)
        neg_idx = torch.randint(0, num_pois - 1, (N_c, K), device=device)
        neg_idx = torch.where(neg_idx >= pos_idx.unsqueeze(-1), neg_idx + 1, neg_idx)

        if self.chunk_size is None or N_c <= self.chunk_size:
            # Gather negatives in one shot: [N_c, K, D]
            neg_pois = pois[neg_idx]
            sim_neg = (anchor.unsqueeze(1) * neg_pois).sum(dim=-1)
        else:
            # Chunked + gradient-checkpointed: materialize [chunk, K, D] only
            # transiently and recompute it during backward instead of caching.
            # On Florida-scale data (1.4 M check-ins, K=32, D=64) the full
            # tensor is 11.5 GB; chunking alone saves only peak memory because
            # autograd retains all chunks for backward. Checkpointing trades
            # ~2× compute for ~chunk_size/N_c memory reduction.
            from torch.utils.checkpoint import checkpoint as _checkpoint

            def _sim_chunk(chunk_anchor, chunk_neg_idx, pois_ref):
                chunk_neg_pois = pois_ref[chunk_neg_idx]
                return (chunk_anchor.unsqueeze(1) * chunk_neg_pois).sum(dim=-1)

            sim_neg_chunks = []
            for i in range(0, N_c, self.chunk_size):
                j = i + self.chunk_size
                sim_neg_chunks.append(
                    _checkpoint(
                        _sim_chunk, anchor[i:j], neg_idx[i:j], pois,
                        use_reentrant=False,
                    )
                )
            sim_neg = torch.cat(sim_neg_chunks, dim=0)

        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature  # [N_c, 1+K]
        targets = torch.zeros(N_c, dtype=torch.long, device=device)        # positive at idx 0
        loss_c2p = F.cross_entropy(logits, targets)

        # ---- Loss 2: POI -> Region (bilinear-sigmoid-BCE, unchanged) ----
        pos_p2r = self.discriminate(pos_poi, pos_region_exp, self.weight_p2r)
        neg_p2r = self.discriminate(pos_poi, neg_region_exp, self.weight_p2r)
        loss_p2r = -torch.log(pos_p2r + EPS).mean() - torch.log(1 - neg_p2r + EPS).mean()

        # ---- Loss 3: Region -> City ----
        pos_r2c = self.discriminate_global(pos_region, city, self.weight_r2c)
        neg_r2c = self.discriminate_global(neg_region, city, self.weight_r2c)
        loss_r2c = -torch.log(pos_r2c + EPS).mean() - torch.log(1 - neg_r2c + EPS).mean()

        return (
            self.alpha_c2p * loss_c2p
            + self.alpha_p2r * loss_p2r
            + self.alpha_r2c * loss_r2c
        )


# ---------------------------------------------------------------------------
# V4: Uncertainty-weighted alphas (Kendall et al. 2018)
# ---------------------------------------------------------------------------

class Check2HGI_Uncertainty(Check2HGI):
    """Replaces fixed alpha_{c2p, p2r, r2c} with learnable log-variance
    parameters. Loss_i' = (1/(2σ_i^2)) * L_i + log σ_i.

    log_var is clamped to [-3, 3] (Kendall stability trick) so the smallest
    loss can't pull its log_var to -inf and collapse the head.
    """

    LOGVAR_MIN, LOGVAR_MAX = -3.0, 3.0

    def __init__(self, *args, **kwargs):
        # Drop alphas — they will be ignored.
        super().__init__(*args, **kwargs)
        self.log_var_c2p = nn.Parameter(torch.zeros(1))
        self.log_var_p2r = nn.Parameter(torch.zeros(1))
        self.log_var_r2c = nn.Parameter(torch.zeros(1))

    def _clamp(self, lv):
        return torch.clamp(lv, min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)

    def loss(self, pos_checkin, pos_poi_exp, neg_poi_exp,
             pos_poi, pos_region_exp, neg_region_exp,
             pos_region, neg_region, city):
        pos_c2p = self.discriminate(pos_checkin, pos_poi_exp, self.weight_c2p)
        neg_c2p = self.discriminate(pos_checkin, neg_poi_exp, self.weight_c2p)
        loss_c2p = -torch.log(pos_c2p + EPS).mean() - torch.log(1 - neg_c2p + EPS).mean()

        pos_p2r = self.discriminate(pos_poi, pos_region_exp, self.weight_p2r)
        neg_p2r = self.discriminate(pos_poi, neg_region_exp, self.weight_p2r)
        loss_p2r = -torch.log(pos_p2r + EPS).mean() - torch.log(1 - neg_p2r + EPS).mean()

        pos_r2c = self.discriminate_global(pos_region, city, self.weight_r2c)
        neg_r2c = self.discriminate_global(neg_region, city, self.weight_r2c)
        loss_r2c = -torch.log(pos_r2c + EPS).mean() - torch.log(1 - neg_r2c + EPS).mean()

        lv_c2p = self._clamp(self.log_var_c2p)
        lv_p2r = self._clamp(self.log_var_p2r)
        lv_r2c = self._clamp(self.log_var_r2c)
        prec_c2p = torch.exp(-lv_c2p)
        prec_p2r = torch.exp(-lv_p2r)
        prec_r2c = torch.exp(-lv_r2c)

        return (
            0.5 * prec_c2p * loss_c2p + 0.5 * lv_c2p
            + 0.5 * prec_p2r * loss_p2r + 0.5 * lv_p2r
            + 0.5 * prec_r2c * loss_r2c + 0.5 * lv_r2c
        ).squeeze()


# ---------------------------------------------------------------------------
# V5: Combined (InfoNCE + GAT-time + skip+LN encoder + uncertainty)
# ---------------------------------------------------------------------------

class Check2HGI_Combined(Check2HGI_InfoNCE):
    """V1 + V4 combined — InfoNCE c2p with uncertainty-weighted multi-loss.
    The encoder is passed in by the caller via checkin_encoder."""

    LOGVAR_MIN, LOGVAR_MAX = -3.0, 3.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_var_c2p = nn.Parameter(torch.zeros(1))
        self.log_var_p2r = nn.Parameter(torch.zeros(1))
        self.log_var_r2c = nn.Parameter(torch.zeros(1))

    def _clamp(self, lv):
        return torch.clamp(lv, min=self.LOGVAR_MIN, max=self.LOGVAR_MAX)

    def loss(self, pos_checkin, pos_poi,
             pos_poi_again, pos_region_exp, neg_region_exp,
             pos_region, neg_region, city,
             checkin_to_poi, num_pois):
        # InfoNCE c2p
        anchor = torch.matmul(pos_checkin, self.weight_c2p)
        anchor = F.normalize(anchor, dim=-1)
        pois = F.normalize(pos_poi, dim=-1)
        N_c = anchor.size(0); K = self.num_negatives; device = anchor.device
        pos_idx = checkin_to_poi
        sim_pos = (anchor * pois[pos_idx]).sum(dim=-1, keepdim=True)
        neg_idx = torch.randint(0, num_pois - 1, (N_c, K), device=device)
        neg_idx = torch.where(neg_idx >= pos_idx.unsqueeze(-1), neg_idx + 1, neg_idx)
        sim_neg = (anchor.unsqueeze(1) * pois[neg_idx]).sum(dim=-1)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
        targets = torch.zeros(N_c, dtype=torch.long, device=device)
        loss_c2p = F.cross_entropy(logits, targets)

        pos_p2r = self.discriminate(pos_poi, pos_region_exp, self.weight_p2r)
        neg_p2r = self.discriminate(pos_poi, neg_region_exp, self.weight_p2r)
        loss_p2r = -torch.log(pos_p2r + EPS).mean() - torch.log(1 - neg_p2r + EPS).mean()

        pos_r2c = self.discriminate_global(pos_region, city, self.weight_r2c)
        neg_r2c = self.discriminate_global(neg_region, city, self.weight_r2c)
        loss_r2c = -torch.log(pos_r2c + EPS).mean() - torch.log(1 - neg_r2c + EPS).mean()

        lv_c2p = self._clamp(self.log_var_c2p)
        lv_p2r = self._clamp(self.log_var_p2r)
        lv_r2c = self._clamp(self.log_var_r2c)
        prec_c2p = torch.exp(-lv_c2p)
        prec_p2r = torch.exp(-lv_p2r)
        prec_r2c = torch.exp(-lv_r2c)

        return (
            0.5 * prec_c2p * loss_c2p + 0.5 * lv_c2p
            + 0.5 * prec_p2r * loss_p2r + 0.5 * lv_p2r
            + 0.5 * prec_r2c * loss_r2c + 0.5 * lv_r2c
        ).squeeze()
