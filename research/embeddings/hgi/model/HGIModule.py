"""
Hierarchical Graph Infomax (HGI) module.

Based on the implementation by Weiming Huang, Daokun Zhang, and Gengchen Mai
"""

import random
import torch
import torch.nn as nn
from torch_geometric.nn.inits import reset, uniform

from embeddings.hgi.model.POIEncoder import POIEncoder
from embeddings.hgi.model.RegionEncoder import POI2Region


EPS = 1e-7


def corruption(x):
    """
    Corruption function to generate negative POIs through random permutation.

    Args:
        x: POI features [num_pois, dim]

    Returns:
        Corrupted features (randomly permuted)
    """
    return x[torch.randperm(x.size(0))]


class HierarchicalGraphInfomax(nn.Module):
    """
    The Hierarchical Graph Infomax Module for learning region representations.

    This model learns hierarchical representations across three levels:
    1. POI level: Individual points of interest
    2. Region level: Aggregated POIs within geographic regions
    3. City level: Aggregated regions for the entire city

    The model maximizes mutual information across these hierarchical levels.
    """

    def __init__(
        self,
        hidden_channels,
        poi_encoder,
        poi2region,
        region2city,
        corruption,
        alpha
    ):
        """
        Initialize HGI module.

        Args:
            hidden_channels: Hidden dimension
            poi_encoder: POI-level encoder
            poi2region: POI-to-region aggregation module
            region2city: Region-to-city aggregation function
            corruption: Corruption function for negative sampling
            alpha: Balance parameter between POI-region and region-city loss (0 to 1)
        """
        super(HierarchicalGraphInfomax, self).__init__()

        self.hidden_channels = hidden_channels
        self.poi_encoder = poi_encoder
        self.poi2region = poi2region
        self.region2city = region2city
        self.corruption = corruption
        self.alpha = alpha

        # Bilinear transformation weights for discrimination
        self.weight_poi2region = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        self.weight_region2city = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))

        # Store embeddings for extraction
        self.region_embedding = torch.tensor(0)
        self.poi_embedding = torch.tensor(0)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset model parameters."""
        reset(self.poi_encoder)
        reset(self.poi2region)
        reset(self.region2city)
        uniform(self.hidden_channels, self.weight_poi2region)
        uniform(self.hidden_channels, self.weight_region2city)

    def forward(self, data):
        """
        Forward pass to generate POI, region, and city representations.

        Returns a 7-tuple matching the original reference semantics:
            (pos_poi_idx, pos_target_region, neg_poi_idx, neg_target_region,
             pos_poi_emb, region_emb, neg_region_emb, city_emb)

        The negative POI-region pairs follow the ORIGINAL formulation:
            For each region R, sample another region R' (with hard-negative
            strategy), then take ALL POIs from R' and pair them with R's
            embedding. This is equivalent to the loop in reference hgi.py
            lines 83-112.
        """
        # POI-level encoding (positive samples)
        pos_poi_emb = self.poi_encoder(data.x, data.edge_index, data.edge_weight)

        # POI-level encoding (negative samples with corrupted features)
        cor_x = self.corruption(data.x)
        neg_poi_emb = self.poi_encoder(cor_x, data.edge_index, data.edge_weight)

        # Region-level encoding (positive)
        region_emb = self.poi2region(pos_poi_emb, data.region_id, data.region_adjacency)

        # Store for later extraction
        self.poi_embedding = pos_poi_emb
        self.region_embedding = region_emb

        # Region-level encoding (negative, from corrupted POIs)
        neg_region_emb = self.poi2region(neg_poi_emb, data.region_id, data.region_adjacency)

        # City-level encoding (area-weighted aggregation of regions)
        city_emb = self.region2city(region_emb, data.region_area)

        # ----- Hard negative sampling (matches reference hgi.py:86-107) -----
        # Original behavior: 25% probability for hard negatives (similarity 0.6-0.8)
        #                    75% probability for random negatives
        num_regions = int(torch.max(data.region_id).item()) + 1
        hard_neg_prob = 0.25
        device = data.x.device

        # For each region R, pick a foreign region neg_R (hard or random)
        neg_region_indices = torch.zeros(num_regions, dtype=torch.long, device=device)
        all_regions = list(range(num_regions))
        for region in range(num_regions):
            hard_negative_choice = random.random()
            if hard_negative_choice < hard_neg_prob:
                # Hard negatives: regions with similarity in (0.6, 0.8)
                sim = data.coarse_region_similarity[region]
                hard_mask = (sim > 0.6) & (sim < 0.8)
                hard_candidates = hard_mask.nonzero(as_tuple=True)[0].tolist()
                candidates = [r for r in hard_candidates if r != region]
                if not candidates:
                    candidates = [r for r in all_regions if r != region]
            else:
                candidates = [r for r in all_regions if r != region]
            neg_region_indices[region] = random.choice(candidates)

        # ----- Build (poi_idx, target_region) pairs -----
        # Positive pairs: every POI i scored against region region_id[i]
        #     pos_poi_idx[i]    = i
        #     pos_target[i]     = region_id[i]
        # Negative pairs: for every region R, ALL POIs from neg_R scored against R
        #     For each pair (R, neg_R = neg_region_indices[R]):
        #         For each POI j with region_id[j] == neg_R:
        #             emit a pair (j, R)
        N = pos_poi_emb.size(0)
        pos_poi_idx = torch.arange(N, device=device)
        pos_target_region = data.region_id  # alias

        neg_poi_idx_chunks = []
        neg_target_region_chunks = []
        for region in range(num_regions):
            neg_r = int(neg_region_indices[region].item())
            poi_in_neg_r = (data.region_id == neg_r).nonzero(as_tuple=True)[0]
            if poi_in_neg_r.numel() == 0:
                continue
            neg_poi_idx_chunks.append(poi_in_neg_r)
            neg_target_region_chunks.append(
                torch.full_like(poi_in_neg_r, region)
            )

        if neg_poi_idx_chunks:
            neg_poi_idx = torch.cat(neg_poi_idx_chunks)
            neg_target_region = torch.cat(neg_target_region_chunks)
        else:
            neg_poi_idx = torch.empty(0, dtype=torch.long, device=device)
            neg_target_region = torch.empty(0, dtype=torch.long, device=device)

        return (
            pos_poi_idx, pos_target_region,
            neg_poi_idx, neg_target_region,
            pos_poi_emb, region_emb, neg_region_emb, city_emb,
        )

    def discriminate_poi2region(self, poi_emb, region_emb, sigmoid=True):
        """
        Vectorized bilinear discriminator: score_i = poi_i @ W @ region_i.

        Args:
            poi_emb:    [N, D] POI embeddings
            region_emb: [N, D] Region embeddings (one per POI, aligned)
        """
        projected_poi = torch.matmul(poi_emb, self.weight_poi2region)  # [N, D]
        scores = (projected_poi * region_emb).sum(dim=1)               # [N]
        return torch.sigmoid(scores) if sigmoid else scores

    def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
        """Bilinear: region @ W @ city  →  [num_regions]."""
        value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
        return torch.sigmoid(value) if sigmoid else value

    def loss(
        self,
        pos_poi_idx, pos_target_region,
        neg_poi_idx, neg_target_region,
        pos_poi_emb, region_emb, neg_region_emb, city_emb,
    ):
        """
        Hierarchical mutual information maximization objective.

        Matches the original reference formulation exactly:
            pos_loss = -log(σ(POI_i @ W @ region[region_id[i]]))   for each POI
            neg_loss = -log(1 - σ(POI_j @ W @ region[R]))
                       for each (R, j) where region_id[j] == neg_R[R]
        """
        # Gather embeddings for the pairs
        pos_pois = pos_poi_emb[pos_poi_idx]              # [N, D]
        pos_regs = region_emb[pos_target_region]         # [N, D]
        neg_pois = pos_poi_emb[neg_poi_idx]              # [M, D]
        neg_regs = region_emb[neg_target_region]         # [M, D]

        # POI-to-Region contrastive loss
        pos_scores = self.discriminate_poi2region(pos_pois, pos_regs, sigmoid=True)
        pos_loss_region = -torch.log(pos_scores + EPS).mean()

        neg_scores = self.discriminate_poi2region(neg_pois, neg_regs, sigmoid=True)
        neg_loss_region = -torch.log(1 - neg_scores + EPS).mean()

        loss_poi2region = pos_loss_region + neg_loss_region

        # Region-to-City contrastive loss
        pos_loss_city = -torch.log(
            self.discriminate_region2city(region_emb, city_emb, sigmoid=True) + EPS
        ).mean()

        neg_loss_city = -torch.log(
            1 - self.discriminate_region2city(neg_region_emb, city_emb, sigmoid=True) + EPS
        ).mean()

        loss_region2city = pos_loss_city + neg_loss_city

        # Combined loss with alpha balancing
        return loss_poi2region * self.alpha + loss_region2city * (1 - self.alpha)

    def get_region_emb(self):
        """
        Get the current region and POI embeddings.

        Returns:
            Tuple of (region_embedding, poi_embedding) on CPU
        """
        return (
            self.region_embedding.clone().cpu().detach(),
            self.poi_embedding.clone().cpu().detach()
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self.hidden_channels})'
