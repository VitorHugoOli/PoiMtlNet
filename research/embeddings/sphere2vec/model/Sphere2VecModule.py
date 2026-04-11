"""
Sphere2Vec-sphereM model components.

Verbatim port from the source notebook
"A General-Purpose Location Representation Learning over a Spherical Surface
for Large-Scale Geospatial Predictions (Sphere2Vec-sphereM).ipynb".

Deliberate divergences from the notebook:

1. Random seeds: the notebook sets none. The buffers in `SpherePositionEncoder`
   (256 RBF centroids initialized with `torch.randn`) are non-deterministic in
   the original. This module does NOT seed internally — seeding is the caller's
   responsibility (`create_embedding` and the equivalence test do it). This
   leaves the math identical to the notebook code.

2. `device` is an explicit constructor argument with no fallback to CUDA;
   `create_embedding` defaults it from `configs.globals.DEVICE`.

3. The unused `LayerNorm` class from the notebook is dropped (the actual
   pipeline uses `nn.LayerNorm`).

Equivalence with the notebook is enforced by
`tests/test_embeddings/test_sphere2vec.py` against the frozen snapshot at
`tests/test_embeddings/_sphere2vec_reference.py`.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_function(activation: str, context_str: str) -> nn.Module:
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
    Single fully-connected feed-forward layer with optional activation,
    dropout, layer-normalization and skip connection.

    Verbatim port from the notebook. Order of operations is:
        Linear → activation → dropout → skip → layernorm
    The skip connection is only enabled if input_dim == output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout_rate: Optional[float] = None,
        activation: str = "sigmoid",
        use_layernormalize: bool = False,
        skip_connection: bool = False,
        context_str: str = "",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # Skip connection only if dims match
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False

        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert input_tensor.size()[-1] == self.input_dim
        output = self.linear(input_tensor)
        output = self.act(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.skip_connection:
            output = output + input_tensor
        if self.layernorm is not None:
            output = self.layernorm(output)
        return output


class MultiLayerFeedForwardNN(nn.Module):
    """
    Stack of `SingleFeedForwardNN` layers. The first layer maps `input_dim →
    hidden_dim`, the middle layers map `hidden_dim → hidden_dim`, and the
    final layer maps `hidden_dim → output_dim` with no activation, dropout,
    layernorm or skip applied.

    If `num_hidden_layers <= 0`, a single `input_dim → output_dim` layer is
    used.

    Verbatim port from the notebook.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: int = 0,
        dropout_rate: Optional[float] = None,
        hidden_dim: int = -1,
        activation: str = "sigmoid",
        use_layernormalize: bool = False,
        skip_connection: bool = False,
        context_str: Optional[str] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=self.context_str,
                )
            )
        else:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.input_dim,
                    output_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=self.use_layernormalize,
                    skip_connection=self.skip_connection,
                    context_str=self.context_str,
                )
            )

            for _ in range(self.num_hidden_layers - 1):
                self.layers.append(
                    SingleFeedForwardNN(
                        input_dim=self.hidden_dim,
                        output_dim=self.hidden_dim,
                        dropout_rate=self.dropout_rate,
                        activation=self.activation,
                        use_layernormalize=self.use_layernormalize,
                        skip_connection=self.skip_connection,
                        context_str=self.context_str,
                    )
                )

            self.layers.append(
                SingleFeedForwardNN(
                    input_dim=self.hidden_dim,
                    output_dim=self.output_dim,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_layernormalize=False,
                    skip_connection=False,
                    context_str=self.context_str,
                )
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for layer in self.layers:
            output = layer(output)
        return output


class SpherePositionEncoder(nn.Module):
    """
    Multi-scale RBF position encoder on the unit 3-sphere.

    Steps:
        1. Convert (lat, lon) degrees to (x, y, z) on the unit sphere.
        2. Compute cosine similarity against `num_centroids` random unit
           centroids on the sphere → "distance" = 1 - cosine.
        3. Apply `num_scales` log-spaced RBF kernels:
                rbf_feat[c, s] = exp(-distance[c] * scales[s])
        4. Flatten to `[batch, num_centroids * num_scales]`.

    Both `scales` and `centroids` are stored as buffers, so they are NOT
    learned during training (this matches the notebook).

    NOTE on `device`: the buffers (`scales`, `centroids`) are created on CPU
    inside `__init__` regardless of the `device` argument — `torch.randn`
    and `torch.logspace` allocate on CPU and `register_buffer` does not move
    them. The forward pass moves the input `coords` to `self.device`, so if
    you construct with `device='mps'`/`'cuda'` but never call `.to(device)`
    on the model, you will hit a cross-device matmul error inside `forward`.
    Always call `model.to(device)` after construction (`create_embedding`
    does this automatically).
    """

    def __init__(
        self,
        min_scale: float = 1,
        max_scale: float = 1000,
        num_scales: int = 16,
        num_centroids: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.num_scales = num_scales
        self.num_centroids = num_centroids

        scales = torch.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
        self.register_buffer("scales", scales)

        centroids = torch.randn(num_centroids, 3)
        centroids = torch.nn.functional.normalize(centroids, dim=-1)
        self.register_buffer("centroids", centroids)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32).to(self.device)
        else:
            coords = coords.float().to(self.device)

        lat = coords[..., 0]
        lon = coords[..., 1]
        lat_rad = torch.deg2rad(lat)
        lon_rad = torch.deg2rad(lon)

        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        input_vec = torch.stack([x, y, z], dim=-1)

        dot_product = torch.matmul(input_vec, self.centroids.t())

        distance = 1.0 - dot_product
        weighted_dist = distance.unsqueeze(-1) * self.scales.view(1, 1, -1)

        rbf_feat = torch.exp(-weighted_dist)

        emb = rbf_feat.flatten(1)

        return emb


class SphereLocationEncoder(nn.Module):
    """
    Wraps `SpherePositionEncoder` with a learnable input projection and a
    `MultiLayerFeedForwardNN` head. Output dimensionality is `spa_embed_dim`.

    The unused `extent` and `interval` arguments are kept for signature
    compatibility with the notebook source.
    """

    def __init__(
        self,
        spa_embed_dim: int = 64,
        num_scales: int = 16,
        min_scale: float = 1,
        max_scale: float = 1000,
        num_centroids: int = 128,
        ffn_hidden_dim: int = 256,
        device: str = "cpu",
        extent=None,
        interval=None,
        ffn_num_hidden_layers: int = 1,
        ffn_dropout_rate: float = 0.5,
        ffn_act: str = "relu",
        ffn_use_layernormalize: bool = True,
        ffn_skip_connection: bool = True,
    ):
        super().__init__()
        self.device = device

        self.position_encoder = SpherePositionEncoder(
            min_scale=min_scale,
            max_scale=max_scale,
            num_scales=num_scales,
            num_centroids=num_centroids,
            device=device,
        )

        input_dim = num_centroids * num_scales

        self.input_projector = nn.Linear(input_dim, ffn_hidden_dim)

        self.ffn = MultiLayerFeedForwardNN(
            input_dim=ffn_hidden_dim,
            output_dim=spa_embed_dim,
            num_hidden_layers=ffn_num_hidden_layers,
            dropout_rate=ffn_dropout_rate,
            hidden_dim=ffn_hidden_dim,
            activation=ffn_act,
            use_layernormalize=ffn_use_layernormalize,
            skip_connection=ffn_skip_connection,
            context_str="Sphere2Vec",
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feat = self.position_encoder(coords)
        feat = self.input_projector(feat)
        out = self.ffn(feat)
        return out


class SphereLocationContrastiveModel(nn.Module):
    """
    Top-level model: `SphereLocationEncoder` followed by a 128 → `embed_dim`
    projection and final L2 normalization.

    Notebook defaults are reproduced exactly:
        spa_embed_dim=128, num_scales=32, min_scale=10, max_scale=1e7,
        num_centroids=256, ffn_hidden_dim=512.
    The `device` argument defaults to "cpu" (the notebook used CUDA when
    available, but `create_embedding` passes `args.device` explicitly).
    """

    def __init__(
        self,
        embed_dim: int = 64,
        spa_embed_dim: int = 128,
        num_scales: int = 32,
        min_scale: float = 10,
        max_scale: float = 1e7,
        num_centroids: int = 256,
        ffn_hidden_dim: int = 512,
        ffn_num_hidden_layers: int = 1,
        ffn_dropout_rate: float = 0.5,
        ffn_act: str = "relu",
        ffn_use_layernormalize: bool = True,
        ffn_skip_connection: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder = SphereLocationEncoder(
            spa_embed_dim=spa_embed_dim,
            num_scales=num_scales,
            min_scale=min_scale,
            max_scale=max_scale,
            num_centroids=num_centroids,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_hidden_layers=ffn_num_hidden_layers,
            ffn_dropout_rate=ffn_dropout_rate,
            ffn_act=ffn_act,
            ffn_use_layernormalize=ffn_use_layernormalize,
            ffn_skip_connection=ffn_skip_connection,
            device=device,
        )
        self.projector = nn.Linear(spa_embed_dim, embed_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        z = self.encoder(coords)
        z = self.projector(z)
        return torch.nn.functional.normalize(z, dim=-1)


def contrastive_bce(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    label: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Binary cross-entropy on cosine-similarity logits scaled by temperature.

    Verbatim port from the notebook.
    """
    sim = F.cosine_similarity(z_i, z_j)
    logits = sim / tau
    targets = label.float().to(z_i.device)
    return F.binary_cross_entropy_with_logits(logits, targets)
