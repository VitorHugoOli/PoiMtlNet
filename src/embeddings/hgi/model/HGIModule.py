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

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features
                - edge_index: Graph connectivity
                - edge_weight: Edge weights
                - region_id: Region assignment for each POI
                - region_adjacency: Region adjacency graph
                - region_area: Area of each region
                - coarse_region_similarity: Region similarity matrix

        Returns:
            Tuple of (pos_poi_emb_list, neg_poi_emb_list, region_emb, neg_region_emb, city_emb)
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

        # Region-level encoding (negative)
        neg_region_emb = self.poi2region(neg_poi_emb, data.region_id, data.region_adjacency)

        # City-level encoding (area-weighted aggregation of regions)
        city_emb = self.region2city(region_emb, data.region_area)

        # Hard negative sampling procedure (Vectorized with original hard negative logic)
        # Original behavior: 25% probability for hard negatives (similarity 0.6-0.8)
        #                    75% probability for random negatives

        num_regions = int(torch.max(data.region_id).item()) + 1
        hard_neg_prob = 0.25

        # Determine negative region for each region using original hard negative strategy
        neg_region_indices = torch.zeros(num_regions, dtype=torch.long, device=data.x.device)

        for region in range(num_regions):
            hard_negative_choice = random.random()
            all_regions = list(range(num_regions))

            if hard_negative_choice < hard_neg_prob:
                # Hard negatives: regions with similarity in (0.6, 0.8)
                sim = data.coarse_region_similarity[region]
                hard_mask = (sim > 0.6) & (sim < 0.8)
                hard_candidates = hard_mask.nonzero(as_tuple=True)[0].tolist()
                candidates = [r for r in hard_candidates if r != region]
                # Fallback to random if no hard candidates available
                if not candidates:
                    candidates = [r for r in all_regions if r != region]
            else:
                # Random negatives from all other regions
                candidates = [r for r in all_regions if r != region]

            neg_region_indices[region] = random.choice(candidates)

        # Map each POI to its negative region embedding
        # Positive pair: (POI, its own region)
        # Negative pair: (POI, a different region selected by hard negative strategy)
        neg_region_indices_per_poi = neg_region_indices[data.region_id]

        # Return tensors for vectorized loss computation:
        # 1. pos_poi_emb - POI embeddings
        # 2. region_emb[data.region_id] - correct region for each POI (positive)
        # 3. region_emb[neg_region_indices_per_poi] - wrong region for each POI (negative)
        # 4. region_emb - all region embeddings (for city-level)
        # 5. neg_region_emb - corrupted region embeddings (for city-level negative)
        # 6. city_emb - city-level embedding

        return pos_poi_emb, region_emb[data.region_id], region_emb[neg_region_indices_per_poi], region_emb, neg_region_emb, city_emb

    def discriminate_poi2region(self, poi_emb, region_emb, sigmoid=True):
        """
        Discriminate between POIs and their corresponding regions (Vectorized).

        Args:
            poi_emb: POI embeddings [num_pois, dim]
            region_emb: Region embeddings aligned with POIs [num_pois, dim]
            sigmoid: Whether to apply sigmoid activation
        """
        # Bilinear: poi @ W @ region.T ? No, elementwise pair score.
        # We want score S_i = poi_i @ W @ region_i
        
        # Projected POI: [N, D] @ [D, D] = [N, D]
        projected_poi = torch.matmul(poi_emb, self.weight_poi2region)
        
        # Dot product with region: ([N, D] * [N, D]).sum(1) = [N]
        scores = (projected_poi * region_emb).sum(dim=1)
        
        return torch.sigmoid(scores) if sigmoid else scores

    def discriminate_region2city(self, region_emb, city_emb, sigmoid=True):
        """
        Discriminate between regions and the city summary.

        Args:
            region_emb: Region embeddings [num_regions, hidden_channels]
            city_emb: City embedding [hidden_channels]
        """
        # Bilinear: region @ W @ city
        value = torch.matmul(region_emb, torch.matmul(self.weight_region2city, city_emb))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_poi_emb, pos_region_emb_expanded, neg_region_emb_expanded, region_emb, neg_region_emb, city_emb):
        """
        Compute the hierarchical mutual information maximization objective (Vectorized).
        
        Args:
            pos_poi_emb: [N, D]
            pos_region_emb_expanded: [N, D] (Correct regions for POIs)
            neg_region_emb_expanded: [N, D] (Incorrect regions for POIs)
            region_emb: [R, D] (All regions)
            neg_region_emb: [R, D] (All corrupted regions)
            city_emb: [D]
        """
        # POI-to-Region contrastive loss
        # Positive: POI matches its Region
        pos_scores = self.discriminate_poi2region(pos_poi_emb, pos_region_emb_expanded, sigmoid=True)
        pos_loss_region = -torch.log(pos_scores + EPS).mean()

        # Negative: POI mismatch with Negative Region
        # (Original code: POI vs Region of another, swapped logic to be same)
        neg_scores = self.discriminate_poi2region(pos_poi_emb, neg_region_emb_expanded, sigmoid=True)
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
