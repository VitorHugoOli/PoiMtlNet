"""STL ReHDM — single-task variant consuming precomputed embeddings.

Same architecture as `model.ReHDM` (POI-level Transformer + dual-level
hypergraph + region classifier). The 6-ID embedding lookup (paper §4.2) is
replaced by a `Linear(emb_dim, d_model)` projection over the per-check-in
embedding column. Everything downstream is identical to the faithful module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from research.baselines.rehdm.model import (
    HGTransformerLayer, POILevelEncoder,
)


@dataclass
class ReHDMSTLConfig:
    n_regions: int
    emb_dim: int = 64
    d_model: int = 192
    n_heads: int = 4
    d_ff: int = 256
    n_hg_layers: int = 2
    dropout: float = 0.2
    beta: float = 0.5


class ReHDMSTL(nn.Module):
    """STL ReHDM: input is a [B, T, emb_dim] tensor of precomputed embeddings."""

    def __init__(self, cfg: ReHDMSTLConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.emb_dim, cfg.d_model)
        self.poi_block = POILevelEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)

        self.theta = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.trunc_normal_(self.theta, std=0.02)

        self.v2e = HGTransformerLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.v2e_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.v2e_ln = nn.LayerNorm(cfg.d_model)
        self.v2e_post = nn.LayerNorm(cfg.d_model)

        L_e2e = max(0, cfg.n_hg_layers - 1)
        self.e2e_layers = nn.ModuleList(
            [HGTransformerLayer(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(L_e2e)]
        )
        self.e2e_mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(),
                           nn.Linear(cfg.d_model, cfg.d_model))
             for _ in range(L_e2e)]
        )
        self.e2e_align = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in range(L_e2e)]
        )
        self.e2e_post = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(L_e2e)])

        self.classifier = nn.Linear(cfg.d_model, cfg.n_regions)

    def encode_pois(self, feats: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(feats)
        return self.poi_block(x, key_padding_mask=key_padding_mask)

    def initial_trajectory_rep(self, vertex_reps, vertex_mask):
        B, T, d = vertex_reps.shape
        edge_types = torch.zeros(B, T, dtype=torch.long, device=vertex_reps.device)
        targets = self.theta.unsqueeze(0).expand(B, d)
        h = self.v2e.attn_local(self.theta, vertex_reps, vertex_mask, edge_types)
        h = self.v2e_ln(targets + h)
        h = self.v2e_post(F.relu(self.v2e_mlp(h)))
        return h

    def forward(self, target_feats, target_mask,
                collab_feats=None, collab_mask=None,
                adjacency=None, edge_types=None):
        kpm = target_mask == 0
        target_rep = self.encode_pois(target_feats, kpm)
        h_target = self.initial_trajectory_rep(target_rep, target_mask)

        if collab_feats is None or adjacency is None or adjacency.numel() == 0:
            h_final = h_target
        else:
            kpm_c = collab_mask == 0
            collab_rep = self.encode_pois(collab_feats, kpm_c)
            h_collab = self.initial_trajectory_rep(collab_rep, collab_mask)

            h_l = h_target
            for layer, mlp, align, post in zip(
                self.e2e_layers, self.e2e_mlp, self.e2e_align, self.e2e_post
            ):
                g = layer(h_l, h_collab, adjacency, edge_types)
                gated = self.cfg.beta * align(h_l) + (1 - self.cfg.beta) * g
                h_l = post(F.relu(mlp(gated)))
            h_final = h_l

        return self.classifier(h_final)
