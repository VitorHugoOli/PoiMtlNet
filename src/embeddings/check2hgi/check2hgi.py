"""Check2HGI (Check-in Hierarchical Graph Infomax) embedding pipeline."""

import argparse
import math
import pickle as pkl

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import trange, tqdm

from configs.paths import IoPaths, EmbeddingEngine, Resources
from embeddings.check2hgi.model.Check2HGIModule import Check2HGI, corruption
from embeddings.check2hgi.model.CheckinEncoder import CheckinEncoder
from embeddings.check2hgi.model.Checkin2POI import Checkin2POI
from embeddings.check2hgi.preprocess import preprocess_check2hgi

# Reuse POI2Region from HGI
from embeddings.hgi.model.RegionEncoder import POI2Region


def train_epoch_full_batch(data, model, optimizer, scheduler, args):
    """Train Check2HGI model for one epoch (full batch mode)."""
    model.train()
    optimizer.zero_grad()

    outputs = model(data)
    loss = model.loss(*outputs)

    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
    optimizer.step()
    scheduler.step()

    return loss.item()


def train_epoch_mini_batch(data, loader, model, optimizer, args):
    """Train Check2HGI model for one epoch (mini-batch mode)."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(args.device)
        optimizer.zero_grad()

        # For mini-batch, we need to handle the hierarchical structure carefully
        # Here we use a simplified approach: train on sampled subgraph
        outputs = model(batch)
        loss = model.loss(*outputs)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_check2hgi(city, args):
    """Train Check2HGI model and generate embeddings."""
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
    data = Data(
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
    ).to(args.device)

    metadata = city_dict['metadata']

    # Initialize model components
    checkin_encoder = CheckinEncoder(in_channels, args.dim, num_layers=args.num_layers)
    checkin2poi = Checkin2POI(args.dim, args.attention_head)
    poi2region = POI2Region(args.dim, args.attention_head)

    def region2city(z, area):
        return torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1))

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
    ).to(args.device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Choose training mode based on data size
    use_mini_batch = num_checkins > args.mini_batch_threshold

    if use_mini_batch:
        print(f"Using mini-batch training (threshold: {args.mini_batch_threshold})")
        loader = NeighborLoader(
            data,
            num_neighbors=[args.num_neighbors] * args.num_layers,
            batch_size=args.batch_size,
            shuffle=True,
            input_nodes=None,
        )
    else:
        print("Using full-batch training")
        loader = None

    # Training loop
    t = trange(1, args.epoch + 1, desc="Training Check2HGI")
    lowest_loss = math.inf
    best_checkin_emb = None

    for epoch in t:
        if use_mini_batch:
            loss = train_epoch_mini_batch(data, loader, model, optimizer, args)
        else:
            loss = train_epoch_full_batch(data, model, optimizer, scheduler, args)

        if loss < lowest_loss:
            checkin_emb, poi_emb, region_emb = model.get_embeddings()
            best_checkin_emb = checkin_emb
            lowest_loss = loss

        t.set_postfix(loss=f'{loss:.4f}', best=f'{lowest_loss:.4f}')

    # Save check-in embeddings
    output_path = IoPaths.get_embedd(city, EmbeddingEngine.CHECK2HGI)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_np = best_checkin_emb.numpy()

    df = pd.DataFrame(embeddings_np, columns=[f'{i}' for i in range(embeddings_np.shape[1])])
    df.insert(0, 'datetime', metadata['datetime'].values)
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
    parser.add_argument('--city', type=str, default='Alabama')
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

    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_norm', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=500)

    # Mini-batch settings
    parser.add_argument('--mini_batch_threshold', type=int, default=100000,
                        help='Use mini-batch training if num_checkins > threshold')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_neighbors', type=int, default=10)

    # Device
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    print(f"Check2HGI Pipeline | City: {args.city} | Device: {args.device}")
    print(f"Edge type: {args.edge_type} | Temporal decay: {args.temporal_decay}")
    create_embedding(state=args.city, args=args)
    print("Done!")
