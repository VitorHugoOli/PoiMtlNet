"""Check2HGI (Check-in Hierarchical Graph Infomax) embedding pipeline.

OPTIMIZED for CPU, CUDA, and MPS:
- Single encoder pass (embedding-level corruption)
- Mixed precision training (automatic device detection)
- Vectorized negative sampling
- torch.compile support for PyTorch 2.0+
"""

import argparse
import math
import pickle as pkl

import pandas as pd
import torch
import os
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import trange, tqdm

from configs.paths import IoPaths, EmbeddingEngine, Resources
from embeddings.check2hgi.model.Check2HGIModule import Check2HGI, corruption
from embeddings.check2hgi.model.CheckinEncoder import CheckinEncoder
from embeddings.check2hgi.model.variants import (
    GATTimeEncoder, ResidualLNEncoder,
    Time2VecCheckinEncoder, RGCNEncoder,
)
from embeddings.check2hgi.model.Checkin2POI import Checkin2POI
from embeddings.check2hgi.preprocess import preprocess_check2hgi

# Reuse POI2Region from HGI
from embeddings.hgi.model.RegionEncoder import POI2Region


def get_autocast_device_type(device):
    """Get the appropriate device type string for torch.autocast."""
    if isinstance(device, str):
        device_str = device
    else:
        device_str = str(device)

    if 'cuda' in device_str:
        return 'cuda'
    elif 'mps' in device_str:
        return 'mps'
    else:
        return 'cpu'


def supports_mixed_precision(device):
    device_type = get_autocast_device_type(device)
    if device_type == 'cuda':
        return True
    # MPS float16 can cause NaN issues with scatter/softmax operations
    # Disable by default for stability
    return False


def train_epoch_full_batch(data, model, optimizer, scheduler, args, use_amp=False, device_type='cpu'):
    """Train Check2HGI model for one epoch (full batch mode).

    OPTIMIZED: Supports mixed precision training for faster execution.

    T2.4 DropEdge: if args.drop_edge_rate > 0, a fresh random mask is applied
    every epoch to the check-in edge_index / edge_weight before the forward
    pass. The mask drops a fraction `drop_edge_rate` of edges uniformly.
    Original `data` is left untouched (we wrap a temporary view).
    """
    model.train()
    optimizer.zero_grad()

    drop_rate = float(getattr(args, 'drop_edge_rate', 0.0) or 0.0)
    if drop_rate > 0.0 and hasattr(data, 'edge_index'):
        num_edges = data.edge_index.size(1)
        if bool(getattr(args, 'symmetric_drop_edge', False)):
            # T2.4 audit fix (2026-05-16 §14:45): user-sequence edges are
            # stored as paired (src→tgt, tgt→src) rows. The asymmetric
            # per-row Bernoulli (legacy below) drops the two directions
            # INDEPENDENTLY, leaving ~drop_rate × 2 (1 - drop_rate)
            # unidirectional edges — NOT the textbook DropEdge (Rong et al.
            # 2020) which drops unique undirected edges symmetrically.
            # Canonicalize each edge to (min, max), generate ONE Bernoulli
            # per unique undirected key, apply same decision to both rows.
            src_idx = data.edge_index[0]
            tgt_idx = data.edge_index[1]
            edge_key_min = torch.minimum(src_idx, tgt_idx).to(torch.int64)
            edge_key_max = torch.maximum(src_idx, tgt_idx).to(torch.int64)
            # Hash unique undirected key: min * (N+1) + max. N is conservative.
            N = int(max(src_idx.max().item(), tgt_idx.max().item())) + 1
            edge_key = edge_key_min * N + edge_key_max
            _, inverse = torch.unique(edge_key, return_inverse=True)
            n_unique = int(inverse.max().item()) + 1
            unique_keep = (
                torch.rand(n_unique, device=data.edge_index.device) >= drop_rate
            )
            keep_mask = unique_keep[inverse]
        else:
            keep_mask = torch.rand(num_edges, device=data.edge_index.device) >= drop_rate
        # Wrap data with masked edges. PyG Data objects accept attr access;
        # use a shallow copy + override only edge_* attrs.
        import copy as _copy
        data_view = _copy.copy(data)
        data_view.edge_index = data.edge_index[:, keep_mask]
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            data_view.edge_weight = data.edge_weight[keep_mask]
        active = data_view
    else:
        active = data

    if use_amp and device_type != 'cpu':
        # Mixed precision for CUDA/MPS
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            outputs = model(active)
            loss = model.loss(*outputs)
    else:
        outputs = model(active)
        loss = model.loss(*outputs)

    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_epoch_mini_batch(data, loader, model, optimizer, args, use_amp=False, device_type='cpu'):
    """Train Check2HGI model for one epoch (mini-batch mode).

    OPTIMIZED: Supports mixed precision training for faster execution.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(args.device)
        optimizer.zero_grad()

        if use_amp and device_type != 'cpu':
            # Mixed precision for CUDA/MPS
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                outputs = model(batch)
                loss = model.loss(*outputs)
        else:
            outputs = model(batch)
            loss = model.loss(*outputs)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_check2hgi(city, args):
    """Train Check2HGI model and generate embeddings.

    OPTIMIZED:
    - Single encoder pass (2x speedup on encoder)
    - Mixed precision training (1.5-2x speedup on GPU/MPS)
    - torch.compile for additional optimization
    - Embedding extraction only at end (reduced CPU transfers)
    """
    # Bit-reproducibility (Tier-3 audit 2026-05-16, FINDING 1):
    # Encoder construction (nn.Parameter(torch.randn(...)) at variants.py and
    # CheckinEncoder) consumes from the global PRNG. Without an explicit seed
    # here, the encoder init depends on the OS-level entropy at process
    # start, making per-seed runs not bit-reproducible (the sign-test across
    # seeds remains statistically valid — each seed's INDEPENDENT random
    # init captures real init variance — but the headline number is not
    # reproducible from a re-run). Honour ``args.seed`` (defaults to 42).
    _ssl_seed = int(getattr(args, "seed", 42))
    import random as _py_random
    import numpy as _np
    _py_random.seed(_ssl_seed)
    _np.random.seed(_ssl_seed)
    torch.manual_seed(_ssl_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_ssl_seed)

    output_folder = IoPaths.CHECK2HGI.get_state_dir(city)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    data_path = IoPaths.CHECK2HGI.get_graph_data_file(city)
    print(f"Loading data: {data_path}")

    with open(data_path, 'rb') as handle:
        city_dict = pkl.load(handle)

    in_channels = city_dict['node_features'].shape[1]
    num_checkins = city_dict['num_checkins']
    num_pois = city_dict['num_pois']
    num_regions = city_dict['num_regions']

    print(f"Check-ins: {num_checkins}, POIs: {num_pois}, Regions: {num_regions}, Features: {in_channels}")

    # Create PyTorch Geometric Data object
    # Keep on CPU initially for DataLoader compatibility (especially MPS)
    _data_kwargs = dict(
        x=torch.tensor(city_dict['node_features'], dtype=torch.float32),
        edge_index=torch.tensor(city_dict['edge_index'], dtype=torch.int64),
        edge_weight=torch.tensor(city_dict['edge_weight'], dtype=torch.float32),
        checkin_to_poi=torch.tensor(city_dict['checkin_to_poi'], dtype=torch.int64),
        poi_to_region=torch.tensor(city_dict['poi_to_region'], dtype=torch.int64),
        region_adjacency=torch.tensor(city_dict['region_adjacency'], dtype=torch.int64),
        region_area=torch.tensor(city_dict['region_area'], dtype=torch.float32),
        coarse_region_similarity=torch.tensor(city_dict['coarse_region_similarity'], dtype=torch.float32),
        num_pois=num_pois,
        num_regions=num_regions,
    )
    # T3.3 plumbing: expose per-edge relation index for R-GCN (only present
    # on freshly-preprocessed graphs; legacy cached graphs omit it).
    if 'edge_type' in city_dict:
        _data_kwargs['edge_type'] = torch.tensor(city_dict['edge_type'], dtype=torch.int64)
    data = Data(**_data_kwargs)

    metadata = city_dict['metadata']

    # Initialize model components
    # T3 encoder swap. `args.encoder` defaults to "gcn" (canonical CheckinEncoder).
    # Other choices:
    #   "gat"     — GATTimeEncoder, attention with optional edge-weight conditioning (T3.1).
    #   "resln"   — ResidualLNEncoder, GCN-family with residual + LayerNorm (T3.2).
    #   "time2vec"— Time2VecCheckinEncoder, learned-frequency replacement for the
    #               4 fixed sin/cos temporal cols (T3.4).
    #   "rgcn"    — RGCNEncoder, relation-typed aggregation; requires K≥2 relations
    #               i.e. a graph preprocessed with edge_type='both' (T3.3 Option A).
    _encoder_name = getattr(args, 'encoder', 'gcn') or 'gcn'
    if _encoder_name == 'gat':
        _gat_heads = int(getattr(args, 'gat_heads', 4))
        _gat_dropout = float(getattr(args, 'encoder_dropout', 0.0) or 0.0)
        _gat_use_edge_attr = bool(getattr(args, 'gat_use_edge_attr', True))
        checkin_encoder = GATTimeEncoder(in_channels, args.dim, num_layers=args.num_layers,
                                          heads=_gat_heads, dropout=_gat_dropout,
                                          use_edge_attr=_gat_use_edge_attr)
        print(f"[encoder] GATTimeEncoder heads={_gat_heads} dropout={_gat_dropout} use_edge_attr={_gat_use_edge_attr}")
    elif _encoder_name == 'resln':
        _resln_dropout = float(getattr(args, 'encoder_dropout', 0.0) or 0.0)
        checkin_encoder = ResidualLNEncoder(in_channels, args.dim, num_layers=args.num_layers,
                                             dropout=_resln_dropout)
        print(f"[encoder] ResidualLNEncoder dropout={_resln_dropout}")
    elif _encoder_name == 'time2vec':
        _t2v_dim = int(getattr(args, 'time2vec_dim', 8))
        _t2v_dropout = float(getattr(args, 'encoder_dropout', 0.0) or 0.0)
        _t2v_warm = bool(getattr(args, 'time2vec_warm_start', False))
        # The preprocess output is [category_onehot (num_cat), temporal (4)].
        # Number of categories = in_channels - 4.
        _num_cat = in_channels - 4
        checkin_encoder = Time2VecCheckinEncoder(
            in_channels, args.dim, num_categories=_num_cat,
            num_layers=args.num_layers, time2vec_dim=_t2v_dim,
            dropout=_t2v_dropout, warm_start=_t2v_warm,
        )
        print(f"[encoder] Time2VecCheckinEncoder d_t={_t2v_dim} num_cat={_num_cat} dropout={_t2v_dropout} warm_start={_t2v_warm}")
    elif _encoder_name == 'rgcn':
        if not hasattr(data, 'edge_type'):
            raise ValueError(
                "encoder='rgcn' requires the preprocessed graph to expose "
                "per-edge relation index. Re-preprocess with edge_type='both' "
                "(force_preprocess=True) and try again."
            )
        _num_relations = int(getattr(args, 'rgcn_num_relations', 2))
        _num_bases = getattr(args, 'rgcn_num_bases', 2)
        if isinstance(_num_bases, str) and _num_bases.lower() == 'none':
            _num_bases = None
        if _num_bases is not None:
            _num_bases = int(_num_bases)
        _rgcn_aggr = str(getattr(args, 'rgcn_aggr', 'sum'))
        _rgcn_dropout = float(getattr(args, 'encoder_dropout', 0.0) or 0.0)
        checkin_encoder = RGCNEncoder(
            in_channels, args.dim, num_relations=_num_relations,
            num_layers=args.num_layers, num_bases=_num_bases,
            dropout=_rgcn_dropout, aggr=_rgcn_aggr,
        )
        print(f"[encoder] RGCNEncoder K={_num_relations} bases={_num_bases} aggr={_rgcn_aggr} dropout={_rgcn_dropout}")
    else:
        checkin_encoder = CheckinEncoder(in_channels, args.dim, num_layers=args.num_layers)
        print(f"[encoder] CheckinEncoder (canonical GCN)")
    checkin2poi = Checkin2POI(args.dim, args.attention_head)
    poi2region = POI2Region(args.dim, args.attention_head)

    def region2city(z, area):
        return torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1))

    # T4.3 — side-feature plumbing. If --use-side-features is set, load the
    # precomputed (num_pois, side_feature_dim) tensor from disk and attach to
    # the Data object so the forward pass can inject it post-pool.
    _use_side_features = bool(getattr(args, 'use_side_features', False))
    _side_feature_dim = 0
    _side_feature_hidden = int(getattr(args, 'side_feature_hidden', 16))
    if _use_side_features:
        from pathlib import Path as _Path
        _sf_path = _Path(IoPaths.CHECK2HGI.get_state_dir(city)) / "poi_side_features.pt"
        if not _sf_path.exists():
            raise FileNotFoundError(
                f"--use-side-features requires precomputed {_sf_path}. Run "
                f"`python scripts/compute_poi_side_features.py --state {city}` first."
            )
        _sf_payload = torch.load(_sf_path, weights_only=False)
        _sf_tensor = _sf_payload["features"].to(dtype=torch.float32)
        if _sf_tensor.shape[0] != int(num_pois):
            raise ValueError(
                f"side_features rows {_sf_tensor.shape[0]} != num_pois {num_pois}; "
                f"recompute side features (preprocess cache changed)."
            )
        # Attach to data (moved to device with the rest below).
        data.side_features = _sf_tensor
        _side_feature_dim = int(_sf_tensor.shape[1])
        print(f"[T4.3] side_features attached shape={tuple(_sf_tensor.shape)} "
              f"hidden={_side_feature_hidden}")

    # T4.1 — GraphMAE plumbing.
    _mae_lambda = float(getattr(args, 'mae_lambda', 0.0) or 0.0)
    _mae_mask_rate = float(getattr(args, 'mae_mask_rate', 0.5) or 0.5)
    _mae_gamma = float(getattr(args, 'mae_gamma', 3.0) or 3.0)
    if _mae_lambda > 0.0:
        print(f"[T4.1] GraphMAE enabled λ={_mae_lambda} mask_rate={_mae_mask_rate} γ={_mae_gamma}")

    model = Check2HGI(
        hidden_channels=args.dim,
        checkin_encoder=checkin_encoder,
        checkin2poi=checkin2poi,
        poi2region=poi2region,
        region2city=region2city,
        corruption=corruption,
        alpha_c2p=args.alpha_c2p,
        alpha_p2r=args.alpha_p2r,
        alpha_r2c=args.alpha_r2c,
        c2p_hard_neg_prob=getattr(args, 'c2p_hard_neg_prob', 0.0),
        c2p_corrupted_neg=getattr(args, 'c2p_corrupted_neg', False),
        p2r_hard_neg_prob=getattr(args, 'p2r_hard_neg_prob', None),
        p2r_hard_neg_min_batch=getattr(args, 'p2r_hard_neg_min_batch', 50000),
        p2r_hard_neg_sim_range=getattr(args, 'p2r_hard_neg_sim_range', (0.6, 0.8)),
        side_feature_dim=_side_feature_dim,
        side_feature_hidden=_side_feature_hidden,
        mae_lambda=_mae_lambda,
        mae_mask_rate=_mae_mask_rate,
        mae_gamma=_mae_gamma,
        mae_in_channels=in_channels if _mae_lambda > 0.0 else None,
    ).to(args.device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Note: torch.compile is NOT compatible with PyTorch Geometric's dynamic
    # scatter operations used in this model (POI2Region uses pyg_softmax with
    # dynamic tensor sizes). Skipping compilation.
    if args.use_compile:
        print("Warning: torch.compile disabled - incompatible with PyG scatter operations")

    # Setup mixed precision training
    device_type = get_autocast_device_type(args.device)
    use_amp = args.use_amp and supports_mixed_precision(args.device)
    if use_amp:
        print(f"Using mixed precision training (device: {device_type})")
    else:
        print(f"Using full precision training (device: {device_type})")

    # T1.5 optimizer hygiene knobs (default = canonical Adam + constant LR; opt-in).
    weight_decay = float(getattr(args, 'weight_decay', 0.0) or 0.0)
    scheduler_type = getattr(args, 'scheduler', 'step') or 'step'
    warmup_pct = float(getattr(args, 'warmup_pct', 0.0) or 0.0)
    eta_min_ratio = float(getattr(args, 'eta_min_ratio', 0.01) or 0.01)
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        print(f"[opt] AdamW lr={args.lr} wd={weight_decay}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"[opt] Adam lr={args.lr} (no WD)")

    total_steps = int(args.epoch)
    if scheduler_type == 'cosine':
        if warmup_pct > 0:
            warmup_steps = max(1, int(warmup_pct * total_steps))

            def _lr_lambda(step):
                if step < warmup_steps:
                    return float(step + 1) / float(warmup_steps)
                progress = (step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
            print(f"[sched] warmup-cosine warmup_pct={warmup_pct} eta_min_ratio={eta_min_ratio}")
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=args.lr * eta_min_ratio
            )
            print(f"[sched] cosine T_max={total_steps} eta_min_ratio={eta_min_ratio}")
    elif scheduler_type == 'warmup_constant':
        warmup_steps = max(1, int(warmup_pct * total_steps))

        def _lr_lambda_wc(step):
            return min(1.0, float(step + 1) / float(warmup_steps))

        scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda_wc)
        print(f"[sched] warmup-constant warmup_pct={warmup_pct}")
    else:
        # Default — canonical StepLR (constant when gamma=1.0). Bit-equivalent.
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Choose training mode based on data size
    use_mini_batch = num_checkins > args.mini_batch_threshold

    if use_mini_batch:
        print(f"Using mini-batch training (threshold: {args.mini_batch_threshold})")
        # Keep data on CPU for DataLoader (MPS/CUDA tensors can't be shared across workers)
        # Batches will be moved to device during training
        num_workers = min(8, os.cpu_count() or 1)
        loader = NeighborLoader(
            data,
            num_neighbors=[args.num_neighbors] * args.num_layers,
            batch_size=args.batch_size,
            shuffle=True,
            input_nodes=None,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            prefetch_factor=5 if num_workers > 0 else None,
            persistent_workers = num_workers > 0,
        )
    else:
        print("Using full-batch training")
        # Move entire dataset to device for full-batch mode
        data = data.to(args.device)
        loader = None

    # Training loop
    # OPTIMIZED: Track best epoch, only extract embeddings at the end
    t = trange(1, args.epoch + 1, desc="Training Check2HGI")
    lowest_loss = math.inf
    best_epoch = 0

    for epoch in t:
        if use_mini_batch:
            loss = train_epoch_mini_batch(data, loader, model, optimizer, args, use_amp, device_type)
        else:
            loss = train_epoch_full_batch(data, model, optimizer, scheduler, args, use_amp, device_type)

        if loss < lowest_loss:
            lowest_loss = loss
            best_epoch = epoch
            # Save model state instead of extracting embeddings every time
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        t.set_postfix(loss=f'{loss:.4f}', best=f'{lowest_loss:.4f}', best_epoch=best_epoch)

    # Load best model and extract embeddings only once at the end
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(best_state)

    # Final forward pass to get embeddings
    # Move data to device if needed (for mini-batch mode where data stayed on CPU).
    # Use a generic parameter-iteration check (encoder-agnostic; GATv2Conv has
    # `lin_l/lin_r` rather than `lin`).
    _enc_param = next(model.checkin_encoder.parameters())
    if data.x.device != _enc_param.device:
        data = data.to(args.device)

    model.eval()
    with torch.no_grad():
        _ = model(data)
        checkin_emb, poi_emb, region_emb = model.get_embeddings()

    # Save check-in embeddings
    output_path = IoPaths.get_embedd(city, EmbeddingEngine.CHECK2HGI)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_np = checkin_emb.numpy()

    df = pd.DataFrame(embeddings_np, columns=[f'{i}' for i in range(embeddings_np.shape[1])])
    df.insert(0, 'datetime', metadata['datetime'].values)
    df.insert(0, 'category', metadata['category'].values)
    df.insert(0, 'placeid', metadata['placeid'].values)
    df.insert(0, 'userid', metadata['userid'].values)
    df.to_parquet(output_path, index=False)
    print(f"Check-in embeddings: {output_path} {embeddings_np.shape}")

    # Also save POI embeddings for reference
    poi_output_path = output_folder / "poi_embeddings.parquet"
    poi_emb_np = poi_emb.numpy()
    poi_df = pd.DataFrame(poi_emb_np, columns=[f'{i}' for i in range(poi_emb_np.shape[1])])

    # Get placeid mapping
    placeid_to_idx = city_dict['placeid_to_idx']
    idx_to_placeid = {v: k for k, v in placeid_to_idx.items()}
    poi_df.insert(0, 'placeid', [idx_to_placeid.get(i, i) for i in range(len(poi_df))])
    poi_df.to_parquet(poi_output_path, index=False)
    print(f"POI embeddings: {poi_output_path} {poi_emb_np.shape}")

    # Save region embeddings
    region_output_path = output_folder / "region_embeddings.parquet"
    region_emb_np = region_emb.numpy()
    region_df = pd.DataFrame(region_emb_np, columns=[f'reg_{i}' for i in range(region_emb_np.shape[1])])
    region_df.insert(0, 'region_id', range(num_regions))
    region_df.to_parquet(region_output_path, index=False)
    print(f"Region embeddings: {region_output_path} {region_emb_np.shape}")


def create_embedding(state: str, args):
    """Run full Check2HGI pipeline: Preprocess -> Train."""
    city = state
    shapefile_path = args.shapefile
    graph_data_file = IoPaths.CHECK2HGI.get_graph_data_file(city)

    if graph_data_file.exists() and not args.force_preprocess:
        print(f"Using existing graph data: {graph_data_file}")
    else:
        print("Preprocessing...")
        preprocess_check2hgi(
            city=city,
            city_shapefile=str(shapefile_path),
            edge_type=args.edge_type,
            temporal_decay=args.temporal_decay,
        )

    print("Training Check2HGI...")
    train_check2hgi(city, args)


run_pipeline = create_embedding  # Backwards compatibility


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check2HGI Embedding Pipeline')

    # Data
    parser.add_argument('--city', type=str, default='Texas')
    parser.add_argument('--shapefile', type=str, default=str(Resources.TL_AL))

    # Pipeline
    parser.add_argument('--force_preprocess', action='store_true')
    parser.add_argument('--edge_type', type=str, default='user_sequence',
                        choices=['user_sequence', 'same_poi', 'both'])
    parser.add_argument('--temporal_decay', type=float, default=3600.0)

    # Model
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attention_head', type=int, default=4)
    parser.add_argument('--alpha_c2p', type=float, default=0.4)
    parser.add_argument('--alpha_p2r', type=float, default=0.3)
    parser.add_argument('--alpha_r2c', type=float, default=0.3)
    parser.add_argument('--c2p_hard_neg_prob', type=float, default=0.0,
                        help='Phase-11 S1: hard-negative probability at the '
                             'check-in↔POI boundary (same-region different-POI). '
                             'Default 0.0 reproduces canonical c2hgi.')
    parser.add_argument('--c2p_corrupted_neg', action='store_true', default=False,
                        help='Phase-11 S4: use corrupted-feature same-identity '
                             'c2p negatives (DGI-style). Mutually exclusive with '
                             'c2p_hard_neg_prob > 0. Default False reproduces canonical c2hgi.')

    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_norm', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=500)

    # Mini-batch settings
    parser.add_argument('--mini_batch_threshold', type=int, default=5_000_000,
                        help='Use mini-batch training if num_checkins > threshold')
    parser.add_argument('--batch_size', type=int, default=2**13)
    parser.add_argument('--num_neighbors', type=int, default=10)

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu, cuda, cuda:0, mps')

    # Performance optimizations
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (AMP) for faster training on CUDA')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable automatic mixed precision')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='(Not recommended) torch.compile is incompatible with PyG scatter ops')

    args = parser.parse_args()

    print(f"Check2HGI Pipeline | City: {args.city} | Device: {args.device}")
    print(f"Edge type: {args.edge_type} | Temporal decay: {args.temporal_decay}")
    print(f"AMP: {args.use_amp} | Compile: {args.use_compile}")
    create_embedding(state=args.city, args=args)
    print("Done!")
