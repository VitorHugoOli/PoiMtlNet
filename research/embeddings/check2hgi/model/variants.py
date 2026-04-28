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
from torch_geometric.nn import GCNConv, GATv2Conv

from embeddings.check2hgi.model.Check2HGIModule import Check2HGI, corruption, EPS


# ---------------------------------------------------------------------------
# V2: Time-aware GAT encoder
# ---------------------------------------------------------------------------

class GATTimeEncoder(nn.Module):
    """GATv2 encoder where attention is conditioned on the (1-d) temporal
    edge weight from the user-sequence graph."""

    def __init__(self, in_channels, hidden_channels, num_layers=2, heads=4, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        assert hidden_channels % heads == 0
        head_dim = hidden_channels // heads

        self.convs = nn.ModuleList()
        # First layer: in_channels -> hidden_channels
        self.convs.append(GATv2Conv(in_channels, head_dim, heads=heads,
                                    edge_dim=1, concat=True, add_self_loops=True))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, head_dim, heads=heads,
                                        edge_dim=1, concat=True, add_self_loops=True))

        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight is not None else None
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

    def forward(self, x, edge_index, edge_weight=None):
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
